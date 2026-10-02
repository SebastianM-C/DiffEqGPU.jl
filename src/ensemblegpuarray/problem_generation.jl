# A single column is returned unchanged. Array columns are concatenated; other columns form a row.
# Ordinary scalar parameters must not use this helper: keep them as a 1-D vector so kernels can
# pass `p[i]` (see `ensemble_param` and `pack_ordinary_parameters`).
function _hcat_batch(cols)
    length(cols) == 1 && return cols[1]
    return cols[1] isa AbstractArray ? reduce(hcat, cols) : reshape(cols, 1, :)
end

# Pack per-trajectory `prob.p` values for `EnsembleGPUArray` kernels.
#
# Scalar parameters become a 1-D vector so `ensemble_param` returns `p[i]` as a `Number`.
# Array parameters (including a singleton trajectory or a length-1 vector) become an
# `nparam × ntraj` matrix so `ensemble_param` returns the full column `p[:, i]`.
function pack_ordinary_parameters(probs)
    cols = [prob.p isa AbstractArray ? Array(prob.p) : prob.p for prob in probs]
    if cols[1] isa AbstractArray
        # Always a matrix — including n = 1 — so singleton array packs are not mistaken
        # for scalar batches by `ensemble_param`'s 1-D branch.
        return length(cols) == 1 ? hcat(cols[1]) : reduce(hcat, cols)
    else
        return cols
    end
end

# Factorize the batched iteration matrix: pivoted when the caller supplied pivot storage
# (which it must then hand to `LinSolveGPUSplitFactorize`), unpivoted otherwise.
batched_lufact!(backend, W, ::Nothing) = lufact!(backend, W)
batched_lufact!(backend, W, ipiv) = lufact!(backend, W, ipiv)

# The mass matrix of one trajectory, as the vector of its diagonal, or `nothing` for the
# identity. `EnsembleGPUArray` only supports mass matrices it can apply per trajectory without
# coupling the columns of the batched state, which means diagonal ones.
_mass_matrix_diagonal(M::UniformScaling, N) = isone(M.λ) ? nothing : fill(M.λ, N)
function _mass_matrix_diagonal(M::Diagonal, N)
    if length(M.diag) != N
        throw(
            DimensionMismatch(
                "the mass matrix has size $(size(M)), but the state has $N components"
            )
        )
    end
    d = Array(M.diag)
    return all(isone, d) ? nothing : d
end
function _mass_matrix_diagonal(M::AbstractMatrix, N)
    isdiag(M) && return _mass_matrix_diagonal(Diagonal(Array(diag(M))), N)
    return _unsupported_mass_matrix(M)
end
_mass_matrix_diagonal(M, N) = _unsupported_mass_matrix(M)

function _unsupported_mass_matrix(M)
    throw(
        ArgumentError(
            "`EnsembleGPUArray` supports diagonal mass matrices only (a `UniformScaling`, a `Diagonal`, or a matrix whose off-diagonal entries are zero), but the problem has a mass matrix of type `$(typeof(M))`."
        )
    )
end

"""
    batched_mass_matrix(M, u0)

The mass matrix of the batched problem whose state `u0` holds one trajectory per column, and
the diagonal of `M` that the W kernels subtract per trajectory.

OrdinaryDiffEq applies the mass matrix to the vectorized `N × ntraj` state, so the batched
matrix is block diagonal with one copy of `M` per trajectory. Both returned arrays live on
the same backend as `u0` and have its element type. An identity mass matrix returns
`(I, nothing)`, so the solver keeps its mass-matrix-free code paths.
"""
function batched_mass_matrix(M, u0)
    N = size(u0, 1)
    ntraj = size(u0, 2)
    d = _mass_matrix_diagonal(M, N)
    d === nothing && return I, nothing
    T = eltype(u0)
    host_diag = convert(Vector{T}, d)
    mass_diag = similar(u0, T, N)
    copyto!(mass_diag, host_diag)
    batched_diag = similar(u0, T, N * ntraj)
    copyto!(batched_diag, repeat(host_diag, ntraj))
    return Diagonal(batched_diag), mass_diag
end

# A mass matrix with a zero on its diagonal makes the problem a DAE. Throws for a mass
# matrix `EnsembleGPUArray` does not support.
function is_singular_mass_matrix(M, N)
    d = _mass_matrix_diagonal(M, N)
    return d !== nothing && any(iszero, d)
end

# Fills `W[:, :, i]` with trajectory `i`'s Jacobian from the problem's own `jac`.
struct AnalyticBatchedJacobian{IIP, J}
    jac::J
end
AnalyticBatchedJacobian{IIP}(jac::J) where {IIP, J} = AnalyticBatchedJacobian{IIP, J}(jac)

function (J::AnalyticBatchedJacobian{IIP})(W, u, p, t) where {IIP}
    version = get_backend(u)
    wgs = workgroupsize(version, size(u, 2))
    kernel = IIP ? batched_jac_kernel : batched_jac_kernel_oop
    kernel(version)(J.jac, W, u, p, t; ndrange = size(u, 2), workgroupsize = wgs)
    return nothing
end

# Tag of the dual numbers of the batched forward-mode Jacobian.
struct DiffEqGPUJacobianTag end

"""
    ADBatchedJacobian(f, u0; chunksize)

Fills `W[:, :, i]` with trajectory `i`'s Jacobian by forward-mode automatic
differentiation of the batched right-hand side `f(du, u, p, t)`.

The Jacobian of the batched problem is block diagonal, so one evaluation of `f` on dual
numbers seeded in states `j, …, j + chunksize - 1` of every trajectory at once gives those
columns of all the per-trajectory Jacobians: `cld(N, chunksize)` evaluations per Jacobian.
The dual state and derivative are preallocated, `2 N ntraj (chunksize + 1)` numbers of the
element type of `u0`. Only the state is differentiated; the parameters and `t` stay plain,
so the right-hand side must accept dual-number states.
"""
struct ADBatchedJacobian{F, UD}
    f::F
    ud::UD
    dud::UD
end

function ADBatchedJacobian(
        f, u0; chunksize = ForwardDiff.pickchunksize(size(u0, 1), 8)
    )
    D = ForwardDiff.Dual{DiffEqGPUJacobianTag, eltype(u0), chunksize}
    return ADBatchedJacobian(f, similar(u0, D), similar(u0, D))
end

function (J::ADBatchedJacobian)(W, u, p, t)
    version = get_backend(u)
    wgs = workgroupsize(version, size(u, 2))
    N = size(u, 1)
    for j0 in 1:ForwardDiff.npartials(eltype(J.ud)):N
        seed_jacobian_duals_kernel(version)(
            J.ud, u, j0; ndrange = size(u, 2), workgroupsize = wgs
        )
        J.f(J.dud, J.ud, p, t)
        scatter_jacobian_partials_kernel(version)(
            W, J.dud, j0; ndrange = size(u, 2), workgroupsize = wgs
        )
    end
    return nothing
end

"""
    FDBatchedJacobian(f, u0; central = false)

Fills `W[:, :, i]` with trajectory `i`'s Jacobian by finite differences of the batched
right-hand side `f(du, u, p, t)`, perturbing state `j` of every trajectory at once.

It reuses the kernel of `f` as it is, so no kernel is compiled for the Jacobian, which
matters for large right-hand sides whose dual-number kernel is too costly to compile. A
Jacobian takes `N + 1` evaluations of `f` with forward differences and `2N` with central
ones, which are more accurate: about `sqrt(eps)` and `eps^(2/3)` relative error.
"""
struct FDBatchedJacobian{F, U, H}
    f::F
    up::U
    f0::U
    f1::U
    h::H
    central::Bool
end

function FDBatchedJacobian(f, u0; central = false)
    return FDBatchedJacobian(
        f, similar(u0), similar(u0), similar(u0), similar(u0, eltype(u0), size(u0, 2)),
        central
    )
end

function (J::FDBatchedJacobian)(W, u, p, t)
    version = get_backend(u)
    wgs = workgroupsize(version, size(u, 2))
    launch = (; ndrange = size(u, 2), workgroupsize = wgs)
    T = real(eltype(u))
    rel = J.central ? cbrt(eps(T)) : sqrt(eps(T))
    copyto!(J.up, u)
    J.central || J.f(J.f0, u, p, t)
    for j in 1:size(u, 1)
        fd_perturb_kernel(version)(J.up, u, J.h, j, rel, true; launch...)
        J.f(J.f1, J.up, p, t)
        if J.central
            fd_perturb_kernel(version)(J.up, u, J.h, j, rel, false; launch...)
            J.f(J.f0, J.up, p, t)
        end
        fd_scatter_kernel(version)(W, J.f1, J.f0, J.h, j, J.central ? 2 : 1; launch...)
        fd_restore_kernel(version)(J.up, u, j; launch...)
    end
    return nothing
end

_ad_chunksize(::ADTypes.AutoForwardDiff{C}) where {C} = C

"""
    batched_jacobian(f, u0, alg)

The filler of the per-trajectory Jacobians for a stiff solve of a problem without `jac`,
as the algorithm's `autodiff` asks: `AutoForwardDiff` (whose `chunksize` sets the dual
chunk, `ForwardDiff.pickchunksize(N, 8)` by default) or `AutoFiniteDiff` (`:forward` or
`:central` differences). Other automatic differentiation backends throw an `ArgumentError`.
"""
function batched_jacobian(f, u0, alg)
    ad = hasproperty(alg, :autodiff) ? alg.autodiff : ADTypes.AutoForwardDiff()
    ad isa ADTypes.AutoSparse && (ad = ADTypes.dense_ad(ad))
    if ad isa ADTypes.AutoForwardDiff
        c = _ad_chunksize(ad)
        return c === nothing ? ADBatchedJacobian(f, u0) :
            ADBatchedJacobian(f, u0; chunksize = c)
    elseif ad isa ADTypes.AutoFiniteDiff
        fdtype = ad.fdjtype
        fdtype isa Union{Val{:forward}, Val{:central}} || throw(
            ArgumentError(
                "`EnsembleGPUArray` computes the Jacobian with forward or central finite differences; got `fdjtype = $fdtype`."
            )
        )
        return FDBatchedJacobian(f, u0; central = fdtype isa Val{:central})
    end
    throw(
        ArgumentError(
            "`EnsembleGPUArray` computes the Jacobian of a problem without `jac` with `AutoForwardDiff` or `AutoFiniteDiff`; the algorithm asks for `$(nameof(typeof(ad)))`. Pass `autodiff = AutoForwardDiff()` or `AutoFiniteDiff()` to the algorithm, or give the problem a `jac`."
        )
    )
end

# `Wfact_t(W, u, p, gamma, t)` for the batched problem: per trajectory,
# `W[:, :, i] = J_i - M / gamma`, factorized in place (pivoted when `ipiv` is given), as
# OrdinaryDiffEq expects. `fill_jacobian!(W, u, p, t)` writes the per-trajectory Jacobians
# into `W`.
function batched_Wfact_t(fill_jacobian!, mass_diag, ipiv)
    return function (W, u, p, gamma, t)
        version = get_backend(u)
        wgs = workgroupsize(version, size(u, 2))
        fill_jacobian!(W, u, p, t)
        subtract_mass_kernel(version)(
            W, mass_diag, gamma; ndrange = size(u, 2), workgroupsize = wgs
        )
        return batched_lufact!(version, W, ipiv)
    end
end
# Pack the per-trajectory parameters of `probs` into the batched parameter object of an
# `EnsembleGPUArray` solve. `T` is the floating-point type of the batched state. Extensions
# add methods dispatching on the parameter type of the first problem:
# `ModelingToolkitBaseExt` batches `MTKParameters`.
pack_parameters(probs, ::Type{T}) where {T} = pack_parameters(first(probs).p, probs, T)
pack_parameters(p, probs, ::Type) = pack_ordinary_parameters(probs)

# The parameters each trajectory ends the solve with, read back from the batched parameter
# object `p` after the solve (callbacks may have changed them), as a vector with one entry
# per problem in `probs`. By default the parameters are the ones the trajectories started
# with.
final_parameters(p, probs) = map(prob -> prob.p, probs)

# `prob` with its parameters replaced by `p`, without the initialization `remake` would run.
with_parameters(prob, p) = p === prob.p ? prob : @set prob.p = p

# Callbacks write into the batched `nparam × ntraj` parameter matrix in place, so the
# trajectories end with the columns read back here. Only that layout exposes mutable
# per-trajectory parameters to affects; scalar and non-numeric batches reach the affect as
# values.
function final_parameters(p::AbstractMatrix{<:Number}, probs)
    final_p = Array(p)
    return map(eachindex(probs)) do i
        p0 = probs[i].p
        p0 isa AbstractVector{<:Number} && length(p0) == size(final_p, 1) || return p0
        column = @view final_p[:, i]
        return all(isequal.(p0, column)) ? p0 : restructure_parameters(p0, column)
    end
end

restructure_parameters(p::StaticArrays.StaticArray, column) =
    similar_type(p)(column)
restructure_parameters(p, column) = copyto!(similar(p), column)

function batched_Wfact_t(jac, isinplace::Bool, mass_diag, ipiv)
    return batched_Wfact_t(AnalyticBatchedJacobian{isinplace}(jac), mass_diag, ipiv)
end

function generate_problem(
        prob::SciMLBase.AbstractODEProblem,
        u0,
        p,
        jac_prototype,
        colorvec,
        ipiv = nothing;
        alg = nothing
    )
    _f = let f = prob.f.f, kernel = DiffEqBase.isinplace(prob) ? gpu_kernel : gpu_kernel_oop
        function (du, u, p, t)
            version = get_backend(u)
            wgs = workgroupsize(version, size(u, 2))
            return kernel(version)(
                f, du, u, p, t; ndrange = size(u, 2),
                workgroupsize = wgs
            )
        end
    end

    mass_matrix, mass_diag = batched_mass_matrix(prob.f.mass_matrix, u0)

    # Without the problem's `jac`, a stiff solve (signalled by `jac_prototype`) differentiates
    # the batched right-hand side instead.
    _Wfact!_t = if SciMLBase.has_jac(prob.f)
        batched_Wfact_t(prob.f.jac, DiffEqBase.isinplace(prob), mass_diag, ipiv)
    elseif jac_prototype !== nothing
        batched_Wfact_t(batched_jacobian(_f, u0, alg), mass_diag, ipiv)
    else
        nothing
    end

    if SciMLBase.has_tgrad(prob.f)
        _tgrad = let tgrad = prob.f.tgrad,
                kernel = DiffEqBase.isinplace(prob) ? gpu_kernel : gpu_kernel_oop

            function (J, u, p, t)
                version = get_backend(u)
                wgs = workgroupsize(version, size(u, 2))
                return kernel(version)(
                    tgrad, J, u, p, t;
                    ndrange = size(u, 2),
                    workgroupsize = wgs
                )
            end
        end
    else
        _tgrad = nothing
    end

    f_func = ODEFunction(
        _f; Wfact_t = _Wfact!_t,
        #colorvec,
        jac_prototype,
        sparsity = nothing,
        tgrad = _tgrad,
        mass_matrix
    )
    return prob = ODEProblem(
        f_func, u0, prob.tspan, p;
        prob.kwargs...
    )
end

# `alg` is accepted for the shared call sites: an SDE solve only uses the problem's own `jac`.
function generate_problem(
        prob::SDEProblem, u0, p, jac_prototype, colorvec, ipiv = nothing; alg = nothing
    )
    if prob.noise_rate_prototype !== nothing
        error("Incompatible problem detected. EnsembleGPUArray currently requires `prob.noise_rate_prototype === nothing`, i.e. only diagonal noise is currently supported. Track https://github.com/SciML/DiffEqGPU.jl/issues/331 for more information.")
    end
    if batched_mass_matrix(prob.f.mass_matrix, u0)[1] !== I
        throw(
            ArgumentError(
                "`EnsembleGPUArray` does not support SDE problems with a mass matrix other than the identity."
            )
        )
    end

    _f = let f = prob.f.f, kernel = DiffEqBase.isinplace(prob) ? gpu_kernel : gpu_kernel_oop
        function (du, u, p, t)
            version = get_backend(u)
            wgs = workgroupsize(version, size(u, 2))
            return kernel(version)(
                f, du, u, p, t;
                ndrange = size(u, 2),
                workgroupsize = wgs
            )
        end
    end

    _g = let f = prob.f.g, kernel = DiffEqBase.isinplace(prob) ? gpu_kernel : gpu_kernel_oop
        function (du, u, p, t)
            version = get_backend(u)
            wgs = workgroupsize(version, size(u, 2))
            return kernel(version)(
                f, du, u, p, t;
                ndrange = size(u, 2),
                workgroupsize = wgs
            )
        end
    end

    _Wfact!_t = if SciMLBase.has_jac(prob.f)
        batched_Wfact_t(prob.f.jac, DiffEqBase.isinplace(prob), nothing, ipiv)
    else
        nothing
    end

    if SciMLBase.has_tgrad(prob.f)
        _tgrad = let tgrad = prob.f.tgrad,
                kernel = DiffEqBase.isinplace(prob) ? gpu_kernel : gpu_kernel_oop

            function (J, u, p, t)
                version = get_backend(u)
                wgs = workgroupsize(version, size(u, 2))
                return kernel(version)(
                    tgrad, J, u, p, t;
                    ndrange = size(u, 2),
                    workgroupsize = wgs
                )
            end
        end
    else
        _tgrad = nothing
    end

    f_func = SDEFunction(
        _f, _g; Wfact_t = _Wfact!_t,
        #colorvec,
        jac_prototype,
        sparsity = nothing,
        tgrad = _tgrad
    )
    return prob = SDEProblem(
        f_func, _g, u0, prob.tspan, p;
        prob.kwargs...
    )
end

# OrdinaryDiffEq's default initialization of a problem with a mass matrix is `CheckInit`,
# which inspects the mass matrix entry by entry. On the batched `(N ntraj) × (N ntraj)`
# matrix that scalar-indexes device memory and costs `O((N ntraj)^2)`, so the batched solve
# skips it unless the user chose an initialization. The initial states must then already be
# consistent with the algebraic equations.
function batched_initializealg(prob, kwargs)
    if prob.f.mass_matrix === I || haskey(kwargs, :initializealg)
        return (;)
    end
    return (; initializealg = SciMLBase.NoInit())
end

# Singular mass matrices are only supported, and tested, with Rosenbrock methods, whose
# `W = J - M / γ` comes from the batched `Wfact_t`.
function supports_singular_mass_matrix(alg)
    return nameof(parentmodule(typeof(alg))) === :OrdinaryDiffEqRosenbrock
end

# A stiff method builds its iteration matrix from the batched `Wfact_t`, which takes the
# per-trajectory Jacobians from `f.jac` or, without it, from automatic differentiation of the
# right-hand side (ODE problems only).
needs_batched_jacobian(prob, alg) = SciMLBase.has_jac(prob.f) || hasproperty(alg, :linsolve)
needs_batched_jacobian(prob::SDEProblem, alg) = SciMLBase.has_jac(prob.f)

# Storage for the `len × len × ntraj` batched iteration matrix of a stiff solve, or `nothing`.
function batched_jac_prototype(prob, alg, ensemblealg, u0, ntraj)
    needs_batched_jacobian(prob, alg) || return nothing
    len = length(prob.u0)
    if ensemblealg isa EnsembleGPUArray
        jac_prototype = allocate(ensemblealg.backend, eltype(u0), (len, len, ntraj))
        fill!(jac_prototype, false)
        return jac_prototype
    else
        return zeros(eltype(u0), len, len, ntraj)
    end
end

"""
    check_array_algorithm(prob, alg)

Throw an `ArgumentError` when `EnsembleGPUArray` (or `EnsembleCPUArray`) cannot solve `prob`
with `alg`, before any trajectory is set up.
"""
check_array_algorithm(prob, alg) = nothing
function check_array_algorithm(prob::SciMLBase.AbstractODEProblem, alg)
    M = prob.f.mass_matrix
    if is_singular_mass_matrix(M, length(prob.u0)) && !supports_singular_mass_matrix(alg)
        throw(
            ArgumentError(
                "`EnsembleGPUArray` solves problems with a singular mass matrix (DAEs) only with Rosenbrock methods such as `Rosenbrock23` or `Rodas5P`, but the algorithm is `$(nameof(typeof(alg)))`."
            )
        )
    end
    hasproperty(alg, :linsolve) || return nothing
    if alg.linsolve !== nothing && !(alg.linsolve isa LinSolveGPUSplitFactorize)
        throw(
            ArgumentError(
                "`EnsembleGPUArray` solves the linear systems of a stiff method with its own batched factorization, so `linsolve = $(nameof(typeof(alg.linsolve)))()` is not supported. Leave `linsolve` unset."
            )
        )
    end
    return nothing
end
