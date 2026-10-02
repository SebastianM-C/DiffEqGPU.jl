struct ParamWrapper{P, T}
    params::P
    data::T
end

function Adapt.adapt_structure(to, ps::ParamWrapper{P, T}) where {P, T}
    return ParamWrapper(
        adapt(to, ps.params),
        adapt(to, ps.data)
    )
end

# Per-trajectory parameter argument for `EnsembleGPUArray` kernels.
#
# `pack_ordinary_parameters` distinguishes layouts: a 1-D `AbstractVector{<:Number}` is a
# batch of scalars and yields `p[i]::Number`; an `nparam × ntraj` matrix (including
# `ntraj = 1`) yields the column view `p[:, i]`. A bare `Number` covers a singleton
# scalar left as a scalar. Non-numeric containers use `p[i]`.
@inline ensemble_param(p::Number, ::Integer) = p
@inline function ensemble_param(p::AbstractArray{<:Number}, i::Integer)
    if ndims(p) == 1
        return @inbounds p[i]
    else
        return @view p[:, i]
    end
end
@inline ensemble_param(p::AbstractArray, i::Integer) = @inbounds p[i]

# The reparameterization is adapted from:https://github.com/rtqichen/torchdiffeq/issues/122#issuecomment-738978844
# Map normalized time t∈[0,1] to each trajectory's physical tspan via a separate
# local `t_phys`. Reassigning the kernel argument `t` leaks across CPU workgroup lanes.
@kernel function gpu_kernel(
        f, du, @Const(u),
        @Const(params::AbstractArray{ParamWrapper{P, T}}),
        @Const(t)
    ) where {P, T}
    i = @index(Global, Linear)
    @inbounds p = params[i].params
    @inbounds tspan = params[i].data
    # reparameterization t->(t_0, t_f) from t->(0, 1).
    t_phys = (tspan[2] - tspan[1]) * t + tspan[1]
    @views @inbounds f(du[:, i], u[:, i], p, t_phys)
    @inbounds for j in 1:size(du, 1)
        du[j, i] = du[j, i] * (tspan[2] - tspan[1])
    end
end

@kernel function gpu_kernel_oop(
        f, du, @Const(u),
        @Const(params::AbstractArray{ParamWrapper{P, T}}),
        @Const(t)
    ) where {P, T}
    i = @index(Global, Linear)
    @inbounds p = params[i].params
    @inbounds tspan = params[i].data
    # reparameterization
    t_phys = (tspan[2] - tspan[1]) * t + tspan[1]
    @views @inbounds x = f(u[:, i], p, t_phys)
    @inbounds for j in 1:size(du, 1)
        du[j, i] = x[j] * (tspan[2] - tspan[1])
    end
end

@kernel function gpu_kernel(f, du, @Const(u), @Const(p), @Const(t))
    i = @index(Global, Linear)
    @views @inbounds f(du[:, i], u[:, i], ensemble_param(p, i), t)
end

@kernel function gpu_kernel_oop(f, du, @Const(u), @Const(p), @Const(t))
    i = @index(Global, Linear)
    @views @inbounds x = f(u[:, i], ensemble_param(p, i), t)
    @inbounds for j in 1:size(du, 1)
        du[j, i] = x[j]
    end
end

@kernel function jac_kernel(
        f, J, @Const(u),
        @Const(params::AbstractArray{ParamWrapper{P, T}}),
        @Const(t)
    ) where {P, T}
    i = @index(Global, Linear) - 1
    section = (1 + (i * size(u, 1))):((i + 1) * size(u, 1))
    @inbounds p = params[i + 1].params
    @inbounds tspan = params[i + 1].data

    # reparameterization
    t_phys = (tspan[2] - tspan[1]) * t + tspan[1]

    @views @inbounds f(J[section, section], u[:, i + 1], p, t_phys)
    @inbounds for j in section, k in section

        J[k, j] = J[k, j] * (tspan[2] - tspan[1])
    end
end

@kernel function jac_kernel_oop(
        f, J, @Const(u),
        @Const(params::AbstractArray{ParamWrapper{P, T}}),
        @Const(t)
    ) where {P, T}
    i = @index(Global, Linear) - 1
    section = (1 + (i * size(u, 1))):((i + 1) * size(u, 1))

    @inbounds p = params[i + 1].params
    @inbounds tspan = params[i + 1].data

    # reparameterization
    t_phys = (tspan[2] - tspan[1]) * t + tspan[1]

    @views @inbounds x = f(u[:, i + 1], p, t_phys)

    @inbounds for j in section, k in section

        J[k, j] = x[k, j] * (tspan[2] - tspan[1])
    end
end

@kernel function jac_kernel(f, J, @Const(u), @Const(p), @Const(t))
    i = @index(Global, Linear) - 1
    section = (1 + (i * size(u, 1))):((i + 1) * size(u, 1))
    @views @inbounds f(J[section, section], u[:, i + 1], ensemble_param(p, i + 1), t)
end

@kernel function jac_kernel_oop(f, J, @Const(u), @Const(p), @Const(t))
    i = @index(Global, Linear) - 1
    section = (1 + (i * size(u, 1))):((i + 1) * size(u, 1))
    @views @inbounds x = f(u[:, i + 1], ensemble_param(p, i + 1), t)
    @inbounds for j in section, k in section

        J[k, j] = x[k, j]
    end
end

@kernel function discrete_condition_kernel(condition, cur, @Const(u), @Const(t), @Const(p))
    i = @index(Global, Linear)
    @views @inbounds cur[i] = condition(
        u[:, i], t, FakeIntegrator(u[:, i], t, ensemble_param(p, i))
    )
end

@kernel function discrete_affect!_kernel(affect!, cur, u, t, p)
    i = @index(Global, Linear)
    @views @inbounds cur[i] &&
        affect!(FakeIntegrator(u[:, i], t, ensemble_param(p, i)))
end

# Applies `affect!` to every trajectory, for callbacks that fire for all of them at once.
@kernel function all_affect!_kernel(affect!, u, @Const(t), p)
    i = @index(Global, Linear)
    @views @inbounds affect!(FakeIntegrator(u[:, i], t, ensemble_param(p, i)))
end

@kernel function continuous_condition_kernel(
        condition, out, @Const(u), @Const(t),
        @Const(p)
    )
    i = @index(Global, Linear)
    @views @inbounds out[i] = condition(
        u[:, i], t, FakeIntegrator(u[:, i], t, ensemble_param(p, i))
    )
end

@kernel function continuous_affect!_kernel(
        affect!, affect_neg!, simultaneous_events, u, t, p
    )
    i = @index(Global, Linear)
    @inbounds event_direction = simultaneous_events[i]
    if event_direction == Int8(1)
        @views @inbounds apply_affect!(affect!, FakeIntegrator(u[:, i], t, ensemble_param(p, i)))
    elseif event_direction == Int8(-1)
        @views @inbounds apply_affect!(affect_neg!, FakeIntegrator(u[:, i], t, ensemble_param(p, i)))
    end
end

# A `ContinuousCallback` direction without an affect (`affect! = nothing` or
# `affect_neg! = nothing`) is still detected by the batched `VectorContinuousCallback`, which
# has no per-direction affects, so it must be a no-op here. Dispatching on `Nothing` keeps the
# branch out of the compiled kernel.
@inline apply_affect!(affect!, integrator) = affect!(integrator)
@inline apply_affect!(::Nothing, integrator) = nothing

"""
    maxthreads(backend)

Return the maximum work-group size used by DiffEqGPU kernels on `backend`.

This is a developer interface for backend extensions. A backend method must return a
positive integer that is valid for the backend's kernel launch configuration.

# Arguments

  - `backend`: a KernelAbstractions backend supported by DiffEqGPU.

# Returns

The backend-specific maximum number of threads in a work group.

# Examples

```julia
maxthreads(CPU())
```
"""
maxthreads(::CPU) = 1024

"""
    maybe_prefer_blocks(backend)

Return the backend configuration used for DiffEqGPU kernel launches.

This is a developer interface for backend extensions. A backend method may return a
configuration with block-oriented execution enabled when that is required for efficient
or correct kernel execution.

# Arguments

  - `backend`: a KernelAbstractions backend supported by DiffEqGPU.

# Returns

The backend instance to pass to subsequent kernel allocation and launch operations.

# Examples

```julia
maybe_prefer_blocks(CPU()) isa CPU
```
"""
maybe_prefer_blocks(::CPU) = CPU()

function workgroupsize(backend, n)
    return min(maxthreads(backend), n)
end

# `Wfact_t` for the batched problem is assembled in two passes: one kernel writes each
# trajectory's Jacobian into its slice `W[:, :, i]`, and a second subtracts the mass matrix,
# giving OrdinaryDiffEq's `W = J - M / γ`. Keeping the passes apart lets the Jacobian come
# from another source without touching the mass-matrix handling.

# With per-trajectory time spans the solver integrates in normalized time `t ∈ [0, 1]`, and
# the right-hand side is scaled by `tf - t0`, so its Jacobian is scaled by the same factor.
@kernel function batched_jac_kernel(
        jac, W, @Const(u),
        @Const(params::AbstractArray{ParamWrapper{P, T}}),
        @Const(t)
    ) where {P, T}
    i = @index(Global, Linear)
    _W = @inbounds @view(W[:, :, i])
    @inbounds p = params[i].params
    @inbounds tspan = params[i].data
    t_phys = (tspan[2] - tspan[1]) * t + tspan[1]
    @views @inbounds jac(_W, u[:, i], p, t_phys)
    @inbounds for j in eachindex(_W)
        _W[j] = _W[j] * (tspan[2] - tspan[1])
    end
end

@kernel function batched_jac_kernel(jac, W, @Const(u), @Const(p), @Const(t))
    i = @index(Global, Linear)
    _W = @inbounds @view(W[:, :, i])
    @views @inbounds jac(_W, u[:, i], ensemble_param(p, i), t)
end

@kernel function batched_jac_kernel_oop(
        jac, W, @Const(u),
        @Const(params::AbstractArray{ParamWrapper{P, T}}),
        @Const(t)
    ) where {P, T}
    i = @index(Global, Linear)
    _W = @inbounds @view(W[:, :, i])
    @inbounds p = params[i].params
    @inbounds tspan = params[i].data
    t_phys = (tspan[2] - tspan[1]) * t + tspan[1]
    @views @inbounds x = jac(u[:, i], p, t_phys)
    @inbounds for j in eachindex(_W)
        _W[j] = x[j] * (tspan[2] - tspan[1])
    end
end

@kernel function batched_jac_kernel_oop(jac, W, @Const(u), @Const(p), @Const(t))
    i = @index(Global, Linear)
    _W = @inbounds @view(W[:, :, i])
    @views @inbounds x = jac(u[:, i], ensemble_param(p, i), t)
    @inbounds for j in eachindex(_W)
        _W[j] = x[j]
    end
end

# Forward-mode seeding for the batched Jacobian: every trajectory's states `j0, …, j0 + C - 1`
# get the unit partials `1, …, C`, all other states zero partials.
@kernel function seed_jacobian_duals_kernel(ud, @Const(u), @Const(j0))
    i = @index(Global, Linear)
    D = eltype(ud)
    @inbounds for k in 1:size(u, 1)
        ud[k, i] = D(u[k, i], ForwardDiff.Partials(_unit_partials(D, k - j0 + 1)))
    end
end

@inline function _unit_partials(::Type{ForwardDiff.Dual{Tag, V, C}}, m) where {Tag, V, C}
    return ntuple(c -> ifelse(c == m, one(V), zero(V)), Val(C))
end

# The partials of the batched right-hand side are columns `j0, …, j0 + C - 1` of each
# trajectory's Jacobian.
@kernel function scatter_jacobian_partials_kernel(W, @Const(dud), @Const(j0))
    i = @index(Global, Linear)
    N = size(dud, 1)
    @inbounds for k in 1:N
        partials = ForwardDiff.partials(dud[k, i])
        for m in 1:length(partials)
            j = j0 + m - 1
            j <= N && (W[k, j, i] = partials[m])
        end
    end
end

# `mass_diag` is the diagonal of the mass matrix shared by every trajectory, or `nothing` for
# the identity.
@inline _mass_diagonal(::Nothing, j, W) = one(eltype(W))
@inline _mass_diagonal(mass_diag, j, W) = @inbounds mass_diag[j]

@kernel function subtract_mass_kernel(W, @Const(mass_diag), @Const(gamma))
    i = @index(Global, Linear)
    invgamma = inv(gamma)
    @inbounds for j in 1:size(W, 1)
        W[j, j, i] = W[j, j, i] - _mass_diagonal(mass_diag, j, W) * invgamma
    end
end

@kernel function gpu_kernel_tgrad(
        f::AbstractArray{T}, du, @Const(u), @Const(p),
        @Const(t)
    ) where {T}
    i = @index(Global, Linear)
    @inbounds f = f[i].tgrad
    @views @inbounds f(du[:, i], u[:, i], ensemble_param(p, i), t)
end
@kernel function gpu_kernel_oop_tgrad(
        f::AbstractArray{T}, du, @Const(u), @Const(p),
        @Const(t)
    ) where {T}
    i = @index(Global, Linear)
    @inbounds f = f[i].tgrad
    @views @inbounds x = f(u[:, i], ensemble_param(p, i), t)
    @inbounds for j in 1:size(du, 1)
        du[j, i] = x[j]
    end
end

"""
    lufact!(backend, W, ipiv)
    lufact!(backend, W)

Factorize each square matrix in a batched matrix array in place.

This is a developer interface implemented by backend extensions. The factorization is
consumed by `LinSolveGPUSplitFactorize`; each slice `W[:, :, i]` must be a square matrix.

The three-argument form uses partial (row) pivoting and stores the row interchanges of
`W[:, :, i]` in `ipiv[:, i]` in the LAPACK `getrf` convention: row `k` was interchanged
with row `ipiv[k, i]`, in order `k = 1, 2, …`. A generic KernelAbstractions
implementation covers every backend; backend extensions may specialize it, for example
with a vendor batched LU.

The two-argument form factorizes without pivoting. It fails on matrices that need row
interchanges, such as the iteration matrices of DAEs with algebraic equations whose
Jacobian has a zero on the diagonal, and is kept for callers that pair it with a
`LinSolveGPUSplitFactorize` constructed without pivots.

# Arguments

  - `backend`: the execution backend.
  - `W`: a three-dimensional array whose first two dimensions contain one matrix per batch
    index.
  - `ipiv`: an `Int32` matrix of size `(size(W, 1), size(W, 3))` on the same backend as
    `W`, overwritten with the pivot indices.

# Returns

`nothing`; `W` (and `ipiv`) are mutated in place.

# Examples

```julia
W = reshape([0.0f0, 2.0f0, 1.0f0, 3.0f0], 2, 2, 1)
ipiv = zeros(Int32, 2, 1)
lufact!(CPU(), W, ipiv)
```
"""
function lufact!(backend, W, ipiv)
    nbatch = size(W, 3)
    nbatch == 0 && return nothing
    wgs = workgroupsize(backend, nbatch)
    lufact_kernel(backend)(W, ipiv; ndrange = nbatch, workgroupsize = wgs)
    return nothing
end

function lufact!(::CPU, W)
    len = size(W, 1)
    for i in 1:size(W, 3)
        _W = @inbounds @view(W[:, :, i])
        generic_lufact!(_W, len)
    end
    return nothing
end

@kernel function lufact_kernel(W, ipiv)
    i = @index(Global, Linear)
    _W = @inbounds @view(W[:, :, i])
    _ipiv = @inbounds @view(ipiv[:, i])
    generic_lufact!(_W, _ipiv, size(W, 1))
end

# Pivot storage for the batched factorization of `W`: one `Int32` column per matrix,
# allocated on the backend of `W` and initialized to "no interchange".
function lu_pivots(W::AbstractArray{<:Any, 3})
    ipiv = similar(W, Int32, (size(W, 1), size(W, 3)))
    ipiv .= Int32.(axes(ipiv, 1))
    return ipiv
end
lu_pivots(::Nothing) = nothing

struct FakeIntegrator{uType, tType, P}
    u::uType
    t::tType
    p::P
end

### GPU Factorization
"""
    LinSolveGPUSplitFactorize()
    LinSolveGPUSplitFactorize(len, nfacts)
    LinSolveGPUSplitFactorize(len, nfacts, ipiv)

A parameter-parallel `SciMLLinearSolveAlgorithm` for applying pre-factorized
per-trajectory linear systems on a KernelAbstractions backend.

The matrix handed to the linear solve must already hold the batched LU factors computed
by [`lufact!`](@ref). When the factorization was pivoted, `ipiv` must be the pivot array
that `lufact!(backend, W, ipiv)` filled, so the row interchanges are applied to the
right-hand side; `ipiv = nothing` means the factors were computed without pivoting.

# Fields

  - `len::Int`: the size of each factored linear system.
  - `nfacts::Int`: the number of factorizations stored in the batched factorization array.
  - `ipiv`: the `(len, nfacts)` pivot array of the factorization, or `nothing`.

# Arguments

  - `len::Int`: the size of each factored linear system.
  - `nfacts::Int`: the number of factorizations stored in the batched factorization array.
  - `ipiv`: the pivot array, or `nothing` (the default) for unpivoted factors.

Most users do not need to construct this directly; `EnsembleGPUArray` installs it, with
the pivots of its own factorization, for compatible stiff ensemble solves.

# Returns

A `LinSolveGPUSplitFactorize` selector configured for the supplied factorization layout.

# Examples

```julia
linsolve = LinSolveGPUSplitFactorize(3, 256)
```
"""
struct LinSolveGPUSplitFactorize{P} <: LinearSolve.SciMLLinearSolveAlgorithm
    len::Int
    nfacts::Int
    ipiv::P
end
LinSolveGPUSplitFactorize(len, nfacts) = LinSolveGPUSplitFactorize(len, nfacts, nothing)
LinSolveGPUSplitFactorize() = LinSolveGPUSplitFactorize(0, 0)

LinearSolve.needs_concrete_A(::LinSolveGPUSplitFactorize) = true

function LinearSolve.init_cacheval(
        linsol::LinSolveGPUSplitFactorize, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{Bool, LinearSolve.LinearVerbosity},
        assumptions::LinearSolve.OperatorAssumptions
    )
    if linsol.len <= 0
        throw(
            ArgumentError(
                "`LinSolveGPUSplitFactorize` needs the size of each factored system: construct it as `LinSolveGPUSplitFactorize(len, nfacts)`. `EnsembleGPUArray` installs a configured one itself, so a stiff solver there can keep its default `linsolve`."
            )
        )
    end
    return LinSolveGPUSplitFactorize(linsol.len, length(u) ÷ linsol.len, linsol.ipiv)
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::LinSolveGPUSplitFactorize,
        args...; kwargs...
    )
    p = cache.cacheval
    A = cache.A
    b = cache.b
    x = cache.u
    version = get_backend(b)
    copyto!(x, b)
    wgs = workgroupsize(version, p.nfacts)
    # Note that the matrix is already factorized, only ldiv is needed.
    ldiv!_kernel(version)(
        A, x, p.ipiv, p.len, p.nfacts;
        ndrange = p.nfacts,
        workgroupsize = wgs
    )
    return SciMLBase.build_linear_solution(alg, x, nothing, cache)
end

# Old stuff
function (p::LinSolveGPUSplitFactorize)(x, A, b, update_matrix = false; kwargs...)
    version = get_backend(b)
    copyto!(x, b)
    wgs = workgroupsize(version, p.nfacts)
    ldiv!_kernel(version)(
        A, x, p.ipiv, p.len, p.nfacts;
        ndrange = p.nfacts,
        workgroupsize = wgs
    )
    return nothing
end

function (p::LinSolveGPUSplitFactorize)(::Type{Val{:init}}, f, u0_prototype)
    return LinSolveGPUSplitFactorize(size(u0_prototype)..., p.ipiv)
end

@kernel function ldiv!_kernel(W, u, @Const(ipiv), @Const(len), @Const(nfacts))
    i = @index(Global, Linear)
    section = (1 + ((i - 1) * len)):(i * len)
    _W = @inbounds @view(W[:, :, i])
    _u = @inbounds @view u[section]
    apply_pivots!(_u, ipiv, i, len)
    naivesolve!(_W, _u, len)
end

# Apply the row interchanges of factorization `i` to its right-hand side, in the order
# `getrf` performed them.
@inline apply_pivots!(u, ::Nothing, i, len) = nothing
@inline function apply_pivots!(u, ipiv, i, len)
    @inbounds for k in 1:len
        p = ipiv[k, i]
        if p != k
            u[k], u[p] = u[p], u[k]
        end
    end
    return nothing
end

function generic_lufact!(A::AbstractMatrix{T}, minmn) where {T}
    m = n = minmn
    #@cuprintf "\n\nbefore lufact!\n"
    #__printjac(A, ii)
    #@cuprintf "\n"
    @inbounds for k in 1:minmn
        #@cuprintf "inner factorization loop\n"
        # Scale first column
        Akkinv = inv(A[k, k])
        for i in (k + 1):m
            #@cuprintf "L\n"
            A[i, k] *= Akkinv
        end
        # Update the rest
        for j in (k + 1):n, i in (k + 1):m
            #@cuprintf "U\n"
            A[i, j] -= A[i, k] * A[k, j]
        end
    end
    #@cuprintf "after lufact!"
    #__printjac(A, ii)
    #@cuprintf "\n\n\n"
    return nothing
end

# LU with partial pivoting, storing the interchanges in `ipiv` as `getrf` does. Like
# `getrf`, a zero pivot column is skipped rather than reported: the factors are then
# singular and the solve produces non-finite values the integrator rejects.
function generic_lufact!(A::AbstractMatrix{T}, ipiv::AbstractVector, n) where {T}
    @inbounds for k in 1:n
        # Find the largest entry in the remaining part of column k
        p = k
        amax = abs(A[k, k])
        for i in (k + 1):n
            a = abs(A[i, k])
            if a > amax
                amax = a
                p = i
            end
        end
        ipiv[k] = p
        if p != k
            for j in 1:n
                A[k, j], A[p, j] = A[p, j], A[k, j]
            end
        end
        Akk = A[k, k]
        if !iszero(Akk)
            # Scale first column
            Akkinv = inv(Akk)
            for i in (k + 1):n
                A[i, k] *= Akkinv
            end
        end
        # Update the rest
        for j in (k + 1):n, i in (k + 1):n
            A[i, j] -= A[i, k] * A[k, j]
        end
    end
    return nothing
end

struct MyL{T} # UnitLowerTriangular
    data::T
end
struct MyU{T} # UpperTriangular
    data::T
end

function naivesub!(A::MyU, b::AbstractVector, n)
    x = b
    @inbounds for j in n:-1:1
        xj = x[j] = A.data[j, j] \ b[j]
        for i in (j - 1):-1:1 # counterintuitively 1:j-1 performs slightly better
            b[i] -= A.data[i, j] * xj
        end
    end
    return nothing
end
function naivesub!(A::MyL, b::AbstractVector, n)
    x = b
    @inbounds for j in 1:n
        xj = x[j]
        for i in (j + 1):n
            b[i] -= A.data[i, j] * xj
        end
    end
    return nothing
end

function naivesolve!(A::AbstractMatrix, x::AbstractVector, n)
    naivesub!(MyL(A), x, n)
    naivesub!(MyU(A), x, n)
    return nothing
end
