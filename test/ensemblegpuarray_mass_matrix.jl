using DiffEqGPU, LinearAlgebra, Test
using Adapt: adapt
using OrdinaryDiffEq: Tsit5
using OrdinaryDiffEqRosenbrock: Rodas5P, Rosenbrock23
using OrdinaryDiffEqSDIRK: TRBDF2
import SciMLBase
using SciMLBase: EnsembleProblem, ODEFunction, ODEProblem, remake, solve
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

include("utils.jl")

const ks = [0.5, 1.0, 1.5, 2.0]

# Index-1 DAE: x' = -k x, 0 = y + sin(y) - x. With x(0) = 1 the consistent y(0) solves
# y + sin(y) = 1.
const y0 = 0.5109734293885691

function dae!(du, u, p, t)
    du[1] = -p[1] * u[1]
    du[2] = u[2] + sin(u[2]) - u[1]
    return nothing
end
function dae_jac!(J, u, p, t)
    J[1, 1] = -p[1]
    J[1, 2] = 0
    J[2, 1] = -1
    J[2, 2] = 1 + cos(u[2])
    return nothing
end
dae_residual(u) = u[2] + sin(u[2]) - u[1]

# Mass matrix with a non-unit differential entry: 2 x' = -k x, 0 = y + sin(y) - x, z' = x - z.
function dae3!(du, u, p, t)
    du[1] = -p[1] * u[1]
    du[2] = u[2] + sin(u[2]) - u[1]
    du[3] = u[1] - u[3]
    return nothing
end
function dae3_jac!(J, u, p, t)
    fill!(J, 0)
    J[1, 1] = -p[1]
    J[2, 1] = -1
    J[2, 2] = 1 + cos(u[2])
    J[3, 1] = 1
    J[3, 3] = -1
    return nothing
end

function max_error(esol, cpu_sols)
    return maximum(
        maximum(maximum(abs, a .- b) for (a, b) in zip(g.u, c.u))
            for (g, c) in zip(esol.u, cpu_sols)
    )
end

function compare_with_cpu(prob, prob_func, alg; kwargs...)
    eprob = EnsembleProblem(prob; prob_func, safetycopy = false)
    esol = solve(
        eprob, alg, EnsembleGPUArray(backend, 0.0); trajectories = length(ks), kwargs...
    )
    cpu = [solve(prob_func(prob, (; sim_id = i)), alg; kwargs...) for i in eachindex(ks)]
    return esol, cpu
end

@testset "Diagonal mass matrix DAE, $(nameof(typeof(alg)))" for (alg, tol) in (
        (Rodas5P(), 1.0e-7), (Rosenbrock23(), 1.0e-6),
    )
    M = Diagonal([1.0, 0.0])
    prob = ODEProblem(ODEFunction(dae!; jac = dae_jac!, mass_matrix = M), [1.0, y0], (0.0, 2.0), [1.0])
    prob_func = (pr, ctx) -> remake(pr; p = [ks[ctx.sim_id]])
    esol, cpu = compare_with_cpu(prob, prob_func, alg; saveat = 0.1, abstol = 1.0e-8, reltol = 1.0e-8)
    @test all(s -> SciMLBase.successful_retcode(s), esol.u)
    @test max_error(esol, cpu) < tol
    @test maximum(maximum(abs ∘ dae_residual, s.u) for s in esol.u) < 1.0e-6
    # Trajectories keep their own parameter: x(2) = exp(-2k).
    @test [s.u[end][1] for s in esol.u] ≈ exp.(-2 .* ks) rtol = 1.0e-5

    # The same problem with a mass matrix whose differential entry is not one.
    M3 = Diagonal([2.0, 0.0, 1.0])
    prob3 = ODEProblem(
        ODEFunction(dae3!; jac = dae3_jac!, mass_matrix = M3), [1.0, y0, 0.0], (0.0, 2.0), [1.0]
    )
    esol3, cpu3 = compare_with_cpu(prob3, prob_func, alg; saveat = 0.1, abstol = 1.0e-8, reltol = 1.0e-8)
    @test all(s -> SciMLBase.successful_retcode(s), esol3.u)
    @test max_error(esol3, cpu3) < tol
    @test [s.u[end][1] for s in esol3.u] ≈ exp.(-ks) rtol = 1.0e-5
end

@testset "Diagonal mass matrix DAE in Float32" begin
    M = Diagonal(Float32[1, 0])
    prob = ODEProblem(
        ODEFunction(dae!; jac = dae_jac!, mass_matrix = M), Float32[1, y0], (0.0f0, 2.0f0), Float32[1]
    )
    prob_func = (pr, ctx) -> remake(pr; p = Float32[ks[ctx.sim_id]])
    esol, _ = compare_with_cpu(prob, prob_func, Rodas5P(); saveat = 0.1f0, abstol = 1.0f-6, reltol = 1.0f-6)
    @test all(s -> SciMLBase.successful_retcode(s), esol.u)
    @test eltype(esol.u[1].u[end]) == Float32
    @test maximum(maximum(abs ∘ dae_residual, s.u) for s in esol.u) < 1.0f-5
    @test [s.u[end][1] for s in esol.u] ≈ exp.(-2 .* ks) rtol = 1.0e-4
end

@testset "Batched Wfact_t" begin
    # W[:, :, i] = s_i J_i - M / γ, factorized with partial pivoting, where s_i = tf - t0 is
    # the time scaling of the per-trajectory time span path (1 otherwise).
    N, ntraj, γ, tn = 3, 2, 0.1, 0.25
    u = [1.0 0.5; y0 0.3; 0.2 0.7]
    mass_diag = [2.0, 0.0, 1.0]
    pk = [1.5, 2.5]
    tspans = [(0.0, 2.0), (1.0, 4.0)]
    function expected(i, scale)
        J = zeros(N, N)
        dae3_jac!(J, u[:, i], (pk[i],), 0.0)
        return lu(scale * J - Diagonal(mass_diag) / γ)
    end
    for (p, scales) in (
            ([pk[1], pk[2]], (1.0, 1.0)),
            (
                [DiffEqGPU.ParamWrapper((pk[i],), tspans[i]) for i in 1:ntraj],
                Tuple(ts[2] - ts[1] for ts in tspans),
            ),
        )
        W = adapt(backend, zeros(N, N, ntraj))
        ipiv = DiffEqGPU.lu_pivots(W)
        Wt = DiffEqGPU.batched_Wfact_t(dae3_jac!, true, adapt(backend, mass_diag), ipiv)
        pb = p isa Vector{Float64} ? adapt(backend, reshape(p, 1, :)) : adapt(backend, p)
        Wt(W, adapt(backend, u), pb, γ, tn)
        Wh, ipivh = Array(W), Array(ipiv)
        for i in 1:ntraj
            F = expected(i, scales[i])
            @test Wh[:, :, i] ≈ F.factors
            @test ipivh[:, i] == F.ipiv
        end
    end
end

@testset "DAE whose iteration matrix needs pivoting" begin
    # The first algebraic equation does not involve its own variable, so the diagonal of the
    # iteration matrix J - M / γ has a structural zero in that row and an unpivoted
    # factorization breaks down at the first step.
    function zp!(du, u, p, t)
        du[1] = -p[1] * u[1]
        du[2] = u[3] - u[1]
        du[3] = u[2] - 2 * u[3]
        return nothing
    end
    function zp_jac!(J, u, p, t)
        fill!(J, 0)
        J[1, 1] = -p[1]
        J[2, 1] = -1
        J[2, 3] = 1
        J[3, 2] = 1
        J[3, 3] = -2
        return nothing
    end
    f = ODEFunction(zp!; jac = zp_jac!, mass_matrix = Diagonal([1.0, 0.0, 0.0]))
    prob = ODEProblem(f, [1.0, 2.0, 1.0], (0.0, 2.0), [1.0])
    prob_func = (prob, ctx) -> remake(prob; p = [ks[ctx.sim_id]])
    for (alg, err) in ((Rodas5P(), 1.0e-6), (Rosenbrock23(), 1.0e-5))
        esol, cpu = compare_with_cpu(
            prob, prob_func, alg; abstol = 1.0e-8, reltol = 1.0e-8, saveat = 0.5
        )
        @test all(s -> SciMLBase.successful_retcode(s), esol.u)
        @test max_error(esol, cpu) < err
    end
end

@testset "Per-trajectory time spans with a mass matrix" begin
    # Per-trajectory time spans go through the time-normalized path, which needs isbits `p`.
    M = Diagonal([1.0, 0.0])
    prob = ODEProblem(ODEFunction(dae!; jac = dae_jac!, mass_matrix = M), [1.0, y0], (0.0, 1.0), (1.0,))
    prob_func = (pr, ctx) -> remake(pr; p = (ks[ctx.sim_id],), tspan = (0.0, 1.0 + ctx.sim_id))
    esol, cpu = compare_with_cpu(prob, prob_func, Rodas5P(); abstol = 1.0e-8, reltol = 1.0e-8, save_everystep = false)
    @test all(s -> SciMLBase.successful_retcode(s), esol.u)
    @test maximum(maximum(abs, g.u[end] .- c.u[end]) for (g, c) in zip(esol.u, cpu)) < 1.0e-6
end

@testset "ModelingToolkit DAE" begin
    @variables x(t) = 1.0 y(t)
    @parameters k = 1.0
    @named sys = System([D(x) ~ -k * x, 0 ~ y + sin(y) - x], t; guesses = [y => 0.5])
    # `split = false` gives a plain parameter vector, which `EnsembleGPUArray` batches.
    csys = mtkcompile(sys; split = false)
    prob = ODEProblem(csys, [], (0.0, 2.0); jac = true)
    @test prob.f.mass_matrix isa Diagonal
    # `EnsembleGPUArray` does not initialize the trajectories: start from a consistent state.
    u0 = solve(prob, Rodas5P(); abstol = 1.0e-10, reltol = 1.0e-10).u[1]
    setter = ModelingToolkit.SymbolicIndexingInterface.setsym_oop(prob, [k])
    prob_func = (pr, ctx) -> remake(pr; u0, p = setter(pr, [ks[ctx.sim_id]])[2])
    ix = ModelingToolkit.SymbolicIndexingInterface.variable_index(prob, x)
    iy = ModelingToolkit.SymbolicIndexingInterface.variable_index(prob, y)
    for (alg, tol) in ((Rodas5P(), 1.0e-7), (Rosenbrock23(), 1.0e-6))
        esol, cpu = compare_with_cpu(prob, prob_func, alg; saveat = 0.1, abstol = 1.0e-8, reltol = 1.0e-8)
        @test all(s -> SciMLBase.successful_retcode(s), esol.u)
        @test max_error(esol, cpu) < tol
        @test maximum(maximum(u -> abs(u[iy] + sin(u[iy]) - u[ix]), s.u) for s in esol.u) < 1.0e-6
    end
end

@testset "Unsupported problems and algorithms" begin
    prob_func = (pr, ctx) -> remake(pr; p = [ks[ctx.sim_id]])
    function ensemble_solve(f, alg; u0 = [1.0, y0])
        prob = ODEProblem(f, u0, (0.0, 1.0), [1.0])
        return solve(
            EnsembleProblem(prob; prob_func), alg, EnsembleGPUArray(backend, 0.0);
            trajectories = 2
        )
    end
    singular = ODEFunction(dae!; jac = dae_jac!, mass_matrix = Diagonal([1.0, 0.0]))
    @test_throws "singular mass matrix" ensemble_solve(singular, Tsit5())
    @test_throws "singular mass matrix" ensemble_solve(singular, TRBDF2())
    @test_throws "diagonal mass matrices only" ensemble_solve(
        ODEFunction(dae!; jac = dae_jac!, mass_matrix = [1.0 0.5; 0.0 0.0]), Rodas5P()
    )
    @test_throws "linsolve" ensemble_solve(
        singular, Rodas5P(linsolve = DiffEqGPU.LinearSolve.LUFactorization())
    )
    # The batched factorization is accepted when given explicitly.
    sol = ensemble_solve(singular, Rodas5P(linsolve = LinSolveGPUSplitFactorize(2, 2)))
    @test all(s -> SciMLBase.successful_retcode(s), sol.u)
end
