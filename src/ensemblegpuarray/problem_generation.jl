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

# Callbacks write into the batched `nparam × ntraj` parameter matrix in place. Read it back
# once per batch so each returned solution can carry its trajectory's final parameters.
final_batch_parameters(p::AbstractMatrix{<:Number}) = Array(p)
final_batch_parameters(p) = nothing

# Only the matrix layout of `pack_ordinary_parameters` exposes mutable per-trajectory
# parameters to affects; scalar and non-numeric batches reach the affect as values.
final_trajectory_problem(prob, ::Nothing, i) = prob
function final_trajectory_problem(prob, final_p::AbstractMatrix, i)
    p = prob.p
    p isa AbstractVector{<:Number} && length(p) == size(final_p, 1) || return prob
    column = @view final_p[:, i]
    all(isequal.(p, column)) && return prob
    return remake(prob; p = restructure_parameters(p, column))
end

restructure_parameters(p::StaticArrays.StaticArray, column) =
    similar_type(p)(column)
restructure_parameters(p, column) = copyto!(similar(p), column)

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

# `Wfact_t(W, u, p, gamma, t)` for the batched problem: per trajectory,
# `W[:, :, i] = J_i - M / gamma`, factorized in place (pivoted when `ipiv` is given), as
# OrdinaryDiffEq expects.
function batched_Wfact_t(jac, isinplace, mass_diag, ipiv)
    jac_kernel = isinplace ? batched_jac_kernel : batched_jac_kernel_oop
    return function (W, u, p, gamma, t)
        version = get_backend(u)
        wgs = workgroupsize(version, size(u, 2))
        jac_kernel(version)(jac, W, u, p, t; ndrange = size(u, 2), workgroupsize = wgs)
        subtract_mass_kernel(version)(
            W, mass_diag, gamma; ndrange = size(u, 2), workgroupsize = wgs
        )
        return batched_lufact!(version, W, ipiv)
    end
end

function generate_problem(
        prob::SciMLBase.AbstractODEProblem,
        u0,
        p,
        jac_prototype,
        colorvec,
        ipiv = nothing
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

    _Wfact!_t = if SciMLBase.has_jac(prob.f)
        batched_Wfact_t(prob.f.jac, DiffEqBase.isinplace(prob), mass_diag, ipiv)
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

function generate_problem(prob::SDEProblem, u0, p, jac_prototype, colorvec, ipiv = nothing)
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

# A stiff method builds its iteration matrix from the batched `Wfact_t`, which needs the
# per-trajectory Jacobian `f.jac`.
has_batched_jacobian(f) = SciMLBase.has_jac(f)

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
    if !has_batched_jacobian(prob.f)
        throw(
            ArgumentError(
                "`EnsembleGPUArray` needs the Jacobian of the right-hand side to use the stiff method `$(nameof(typeof(alg)))`: pass it as `ODEFunction(f; jac)`, or build the problem with ModelingToolkit using `jac = true`."
            )
        )
    end
    return nothing
end
