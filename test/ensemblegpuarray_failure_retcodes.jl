using DiffEqGPU, Test
using OrdinaryDiffEq: Tsit5, Rosenbrock23
import SciMLBase
using SciMLBase: EnsembleProblem, ODEFunction, ODEProblem, ReturnCode, remake, solve

include("utils.jl")

# u' = u², u(0) = c blows up at t = 1 / c: with these initial values only the last trajectory
# blows up before the end of the time span, and the batch stops there.
blowup(du, u, p, t) = (du[1] = u[1]^2; nothing)
blowup_jac(J, u, p, t) = (J[1, 1] = 2 * u[1]; nothing)
const prob = ODEProblem(ODEFunction(blowup; jac = blowup_jac), [1.0], (0.0, 2.0))
exact(c, t) = 1 / (1 / c - t)

function ensemble(cs, alg; kwargs...)
    eprob = EnsembleProblem(
        prob; prob_func = (prob, ctx) -> remake(prob; u0 = [cs[ctx.sim_id]]), safetycopy = false
    )
    return solve(
        eprob, alg, EnsembleGPUArray(backend, 0.0); trajectories = length(cs),
        abstol = 1.0e-8, reltol = 1.0e-8, kwargs...
    ).u
end

@testset "Only the trajectory that fails gets the failure ($(nameof(typeof(alg))))" for alg in (
        Tsit5(), Rosenbrock23(),
    )
    cs = [0.2, 0.25, 0.3, 1.0]
    sols = ensemble(cs, alg)
    culprit = sols[end]
    @test !SciMLBase.successful_retcode(culprit)
    @test culprit.retcode != ReturnCode.Failure
    for s in sols[1:(end - 1)]
        @test s.retcode == ReturnCode.Failure
        # Stopped early, with a solution that is valid up to the stop.
        @test s.t[end] < 2
        @test s.t[end] == culprit.t[end]
    end
    for (c, s) in zip(cs[1:(end - 1)], sols[1:(end - 1)])
        @test all(isapprox(u[1], exact(c, t); rtol = 1.0e-5) for (u, t) in zip(s.u, s.t))
    end

    # The culprit is found wherever it is in the batch.
    sols = ensemble([0.2, 1.0, 0.25], alg)
    @test [s.retcode == ReturnCode.Failure for s in sols] == [true, false, true]
end

@testset "Unchanged without a failure or a culprit" begin
    sols = ensemble([0.2, 0.25, 0.3], Tsit5())
    @test all(s -> s.retcode == ReturnCode.Success, sols)
    # A user `internalnorm` replaces the per-trajectory norm that finds the culprit, so every
    # trajectory reports the batch's failure.
    sols = ensemble(
        [0.2, 0.25, 1.0], Tsit5(); internalnorm = (u, t) -> sqrt(sum(abs2, u) / length(u))
    )
    @test allequal(s.retcode for s in sols)
    @test !SciMLBase.successful_retcode(sols[1])
end
