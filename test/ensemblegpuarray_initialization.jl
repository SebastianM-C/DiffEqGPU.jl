using DiffEqGPU, ModelingToolkit, OrdinaryDiffEq, Test
using ModelingToolkit: t_nounits as t, D_nounits as D
using SciMLBase: SciMLBase, EnsembleProblem, ODEProblem, ReturnCode, remake, solve

include("utils.jl")

const SII = ModelingToolkit.SymbolicIndexingInterface

# `x(0)` comes from a nonlinear initialization equation and `b` is solved for during
# initialization, so the initialized state and parameters both depend on `a`. For `a < -1/4`
# the initialization equation `x^2 + x ~ a` has no real solution.
@variables x(t) y(t)
@parameters a b = missing [guess = 1.0]
@named sys = System(
    [D(x) ~ -a * x + b, D(y) ~ -y], t;
    initialization_eqs = [x^2 + x ~ a, b ~ 2 * x + y]
)

function lazy_prob_func(prob, as)
    setter = SII.setsym_oop(prob, [a])
    return function (pr, ctx)
        u0, p = setter(pr, [as[ctx.sim_id]])
        return remake(pr; u0, p, lazy_initialization = true)
    end
end

function ensemble_solve(prob, as, ensemblealg; kwargs...)
    eprob = EnsembleProblem(prob; prob_func = lazy_prob_func(prob, as), safetycopy = false)
    return solve(
        eprob, Tsit5(), ensemblealg; trajectories = length(as), saveat = 0.25, kwargs...
    )
end

function cpu_solves(prob, as; kwargs...)
    pf = lazy_prob_func(prob, as)
    return [
        solve(pf(prob, (; sim_id = i)), Tsit5(); saveat = 0.25, kwargs...)
            for i in eachindex(as)
    ]
end

function test_matches_cpu(gpu, cpu; atol)
    @test SciMLBase.successful_retcode(gpu)
    @test gpu.u[1] ≈ cpu.u[1] atol = atol
    @test gpu.ps[b] ≈ cpu.ps[b] atol = atol
    @test gpu.ps[a] == cpu.ps[a]
    @test length(gpu.u) == length(cpu.u)
    @test maximum(maximum(abs, g - c) for (g, c) in zip(gpu.u, cpu.u)) < 10 * atol
    return
end

# A flat parameter vector, so the batch runs on any backend. The initialization function
# of such a problem is wrapped by AutoSpecialize, which the host initialization unwraps.
csys = mtkcompile(sys; split = false)
prob = ODEProblem(csys, [y => 1.0, a => 1.0], (0.0, 1.0); guesses = [x => 0.5])
tol = (; abstol = 1.0e-9, reltol = 1.0e-9)

@testset "Initialized trajectories match their CPU solves" begin
    as = [0.5, 1.0, 2.0, 3.0]
    cpu = cpu_solves(prob, as; tol...)
    gpu = ensemble_solve(prob, as, EnsembleGPUArray(backend, 0.0); tol...)
    for i in eachindex(as)
        test_matches_cpu(gpu.u[i], cpu[i]; atol = 1.0e-7)
        # The initialization equation holds at t0, not the guess `x = 0.5`.
        x0 = gpu.u[i][x, 1]
        @test x0^2 + x0 ≈ as[i] atol = 1.0e-8
    end
end

@testset "A failed initialization fails only its trajectory" begin
    as = [0.5, 1.0, -1.0, 3.0]
    cpu = cpu_solves(prob, as; tol...)
    @test cpu[3].retcode == ReturnCode.InitialFailure
    gpu = ensemble_solve(prob, as, EnsembleGPUArray(backend, 0.0); tol...)
    @test gpu.u[3].retcode == ReturnCode.InitialFailure
    @test length(gpu.u[3].t) == 1
    for i in (1, 2, 4)
        test_matches_cpu(gpu.u[i], cpu[i]; atol = 1.0e-7)
    end

    allfail = ensemble_solve(prob, [-1.0, -2.0], EnsembleGPUArray(backend, 0.0); tol...)
    @test all(sol -> sol.retcode == ReturnCode.InitialFailure, allfail.u)
end

@testset "Threaded host preparation" begin
    as = [0.5, 1.0, -1.0, 3.0, 2.0, 0.25]
    serial = ensemble_solve(prob, as, EnsembleGPUArray(backend, 0.0); tol...)
    threaded = ensemble_solve(
        prob, as, EnsembleGPUArray(backend, 0.0; threaded_host = true); tol...
    )
    @test EnsembleGPUArray(backend; threaded_host = true).threaded_host
    @test !EnsembleGPUArray(backend).threaded_host
    for i in eachindex(as)
        @test threaded.u[i].retcode == serial.u[i].retcode
        @test threaded.u[i].u == serial.u[i].u
        @test threaded.u[i].prob.p == serial.u[i].prob.p
    end
end

@testset "initializealg" begin
    as = [0.5, 1.0]
    # Without initialization the trajectories start from the guess.
    noinit = ensemble_solve(
        prob, as, EnsembleGPUArray(backend, 0.0); initializealg = SciMLBase.NoInit(), tol...
    )
    @test all(sol -> sol.u[1][SII.variable_index(prob, x)] == 0.5, noinit.u)

    override = ensemble_solve(
        prob, as, EnsembleGPUArray(backend, 0.0);
        initializealg = SciMLBase.OverrideInit(; abstol = 1.0e-12, reltol = 1.0e-12), tol...
    )
    for (i, sol) in enumerate(override.u)
        x0 = sol[x, 1]
        @test x0^2 + x0 ≈ as[i] atol = 1.0e-11
    end

    @test_throws ArgumentError ensemble_solve(
        prob, as, EnsembleGPUArray(backend, 0.0);
        initializealg = DiffEqGPU.BrownFullBasicInit()
    )
end

@testset "Float32 initialization" begin
    prob32 = ODEProblem(
        csys, [y => 1.0f0, a => 1.0f0], (0.0f0, 1.0f0); guesses = [x => 0.5f0]
    )
    @test eltype(prob32.u0) == Float32
    as = Float32[0.5, 1.0, 2.0]
    gpu = ensemble_solve(
        prob32, as, EnsembleGPUArray(backend, 0.0); abstol = 1.0f-6, reltol = 1.0f-6
    )
    for (i, sol) in enumerate(gpu.u)
        @test SciMLBase.successful_retcode(sol)
        @test eltype(sol.u[1]) == Float32
        x0 = sol[x, 1]
        @test x0^2 + x0 ≈ as[i] atol = 1.0f-5
    end
end

if backend isa DiffEqGPU.CPU
    # `MTKParameters` only reach the kernels of the CPU backend.
    @testset "Split parameters" begin
        sprob = ODEProblem(
            mtkcompile(sys), [y => 1.0, a => 1.0], (0.0, 1.0); guesses = [x => 0.5]
        )
        as = [0.5, 1.0, 2.0]
        cpu = cpu_solves(sprob, as; tol...)
        gpu = ensemble_solve(sprob, as, EnsembleGPUArray(backend, 0.0); tol...)
        for i in eachindex(as)
            test_matches_cpu(gpu.u[i], cpu[i]; atol = 1.0e-7)
        end
    end
end
