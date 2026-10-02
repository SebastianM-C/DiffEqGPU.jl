using DiffEqGPU, LinearAlgebra, Test
using Adapt: adapt
using KernelAbstractions: KernelAbstractions
using OrdinaryDiffEqRosenbrock: Rodas5P, Rosenbrock23
using SciMLBase: EnsembleProblem, ODEFunction, ODEProblem, ReturnCode, remake, solve

include("utils.jl")

# Metal and oneAPI have no double precision.
const T = GROUP in ("Metal", "oneAPI") ? Float32 : Float64

# Batched matrices whose leading entries vanish, so that factorizing them without row
# interchanges divides by zero.
function zero_pivot_batch(len, nbatch)
    A = zeros(T, len, len, nbatch)
    for i in 1:nbatch
        M = randn(T, len, len) + T(len) * I
        M[1, 1] = 0
        M[2, 2] = 0
        M[1, 2] = M[2, 1] = 1 + T(i) / nbatch
        A[:, :, i] = M
    end
    return A
end

@testset "pivoted batched LU and solve" begin
    len, nbatch = 6, 5
    A = zero_pivot_batch(len, nbatch)
    b = randn(T, len * nbatch)

    W = adapt(backend, copy(A))
    ipiv = DiffEqGPU.lu_pivots(W)
    DiffEqGPU.lufact!(backend, W, ipiv)
    x = adapt(backend, zeros(T, len * nbatch))
    linsolve = LinSolveGPUSplitFactorize(len, nbatch, ipiv)
    linsolve(x, W, adapt(backend, b))

    x_host = Array(x)
    for i in 1:nbatch
        section = (1 + (i - 1) * len):(i * len)
        F = lu(A[:, :, i])
        @test Array(ipiv)[:, i] == F.ipiv
        @test x_host[section] ≈ A[:, :, i] \ b[section] rtol = 100 * eps(T)^(3 // 4)
    end

    # Without pivoting the zero leading entry poisons the factors.
    W0 = adapt(backend, copy(A))
    if backend isa KernelAbstractions.CPU || GROUP == "CUDA"
        DiffEqGPU.lufact!(backend, W0)
        x0 = adapt(backend, zeros(T, len * nbatch))
        LinSolveGPUSplitFactorize(len, nbatch)(x0, W0, adapt(backend, b))
        @test !all(isfinite, Array(x0))
    end
end

# The batched solve runs one workgroup per matrix on GPUs, with the rows of a column split
# over its threads: sizes below, at and above the workgroup width, and batches that are not
# multiples of it.
@testset "batched solve, len = $len, nbatch = $nbatch" for len in (2, 3, 31, 33, 124),
        nbatch in (1, 7, 65)
    A = zeros(T, len, len, nbatch)
    for i in 1:nbatch
        M = randn(T, len, len) + T(len) * I
        len >= 2 && (M[1, 1] = 0)
        A[:, :, i] = M
    end
    b = randn(T, len * nbatch)
    W = adapt(backend, copy(A))
    ipiv = DiffEqGPU.lu_pivots(W)
    DiffEqGPU.lufact!(backend, W, ipiv)
    x = adapt(backend, zeros(T, len * nbatch))
    LinSolveGPUSplitFactorize(len, nbatch, ipiv)(x, W, adapt(backend, b))
    x_host = Array(x)
    tol = 100 * len * eps(T)^(3 // 4)
    @test all(
        isapprox(x_host[(1 + (i - 1) * len):(i * len)], A[:, :, i] \ b[(1 + (i - 1) * len):(i * len)]; rtol = tol)
            for i in 1:nbatch
    )

    # Unpivoted factors of diagonally dominant matrices.
    D = copy(A)
    for i in 1:nbatch, k in 1:len
        D[k, k, i] = 2 * len
    end
    if backend isa KernelAbstractions.CPU || GROUP == "CUDA"
        W0 = adapt(backend, copy(D))
        DiffEqGPU.lufact!(backend, W0)
        x0 = adapt(backend, zeros(T, len * nbatch))
        LinSolveGPUSplitFactorize(len, nbatch)(x0, W0, adapt(backend, b))
        x0_host = Array(x0)
        @test all(
            isapprox(x0_host[(1 + (i - 1) * len):(i * len)], D[:, :, i] \ b[(1 + (i - 1) * len):(i * len)]; rtol = tol)
                for i in 1:nbatch
        )
    end
end

@testset "LinSolveGPUSplitFactorize() needs the system size" begin
    prob = ODEProblem(
        ODEFunction((du, u, p, t) -> (du .= -u); jac = (J, u, p, t) -> (J .= -I)),
        ones(T, 2), (zero(T), one(T))
    )
    @test_throws ArgumentError solve(prob, Rodas5P(linsolve = LinSolveGPUSplitFactorize()))
end

# u' = A u with a structural zero at A[1, 1] and eigenvalues -1 and -k: once the fast mode
# has decayed and dt * γ * k ≫ 1, the leading entry -1 / (dt γ) of W = J - I / (dt γ) is
# small next to the entry -k below it, so the factorization takes a row interchange. The
# batched solves must still match the per-trajectory CPU solves.
function coupled!(du, u, p, t)
    du[1] = u[2]
    du[2] = -p[1] * u[1] - (p[1] + 1) * u[2]
    du[3] = u[1] - u[3]
    return nothing
end
function coupled_jac!(J, u, p, t)
    fill!(J, 0)
    J[1, 2] = 1
    J[2, 1] = -p[1]
    J[2, 2] = -(p[1] + 1)
    J[3, 1] = 1
    J[3, 3] = -1
    return nothing
end
coupled_prob = ODEProblem(
    ODEFunction(coupled!; jac = coupled_jac!), T[1, 0, 0], (zero(T), T(5)), T[1.0e4]
)
pivot_ps = T[1.0e4, 2.0e4, 5.0e4, 1.0e5]
pivot_eprob = EnsembleProblem(
    coupled_prob; prob_func = (prob, ctx) -> remake(prob; p = T[pivot_ps[ctx.sim_id]]),
    safetycopy = false
)

@testset "stiff EnsembleGPUArray solves with pivoted iteration matrices ($(nameof(typeof(alg))))" for alg in (Rodas5P(), Rosenbrock23())
    tol = T == Float64 ? 1.0e-8 : 1.0f-5
    esol = solve(
        pivot_eprob, alg, EnsembleGPUArray(backend); trajectories = length(pivot_ps),
        abstol = tol, reltol = tol, saveat = T(0.5)
    )
    for (i, k) in enumerate(pivot_ps)
        ref = solve(remake(coupled_prob; p = T[k]), alg; abstol = tol, reltol = tol, saveat = T(0.5))
        @test esol.u[i].retcode == ReturnCode.Success
        @test maximum(maximum(abs, a .- b) for (a, b) in zip(esol.u[i].u, ref.u)) < 1000 * tol
    end
end
