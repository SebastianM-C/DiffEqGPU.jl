using DiffEqGPU, Test
using OrdinaryDiffEq: Tsit5, Rosenbrock23
using DiffEqCallbacks: PeriodicCallback, PresetTimeCallback
import SciMLBase
using SciMLBase: EnsembleProblem, ODEFunction, ODEProblem, remake, solve

include("utils.jl")

const tol = 1.0e-9
state_tolerance(::Tsit5) = 1.0e-6
state_tolerance(::Rosenbrock23) = 1.0e-4

decay(du, u, p, t) = (du[1] = -p[1] * u[1]; nothing)
decay_jac(J, u, p, t) = (J[1, 1] = -p[1]; nothing)
const ks = [0.5, 1.0, 1.5, 2.0]
decay_prob_func = (prob, ctx) -> remake(prob; p = [ks[ctx.sim_id]])
const decay_prob = ODEProblem(
    ODEFunction(decay; jac = decay_jac), [1.0], (0.0, 2.0), [1.0]
)

# Each trajectory solved on its own with the same callback.
function serial_solutions(prob, prob_func, n, alg; kwargs...)
    return map(1:n) do i
        solve(prob_func(deepcopy(prob), (; sim_id = i)), alg; abstol = tol, reltol = tol, kwargs...)
    end
end

max_state_error(sols, refs) = maximum(
    maximum(maximum(abs, a .- b) for (a, b) in zip(s.u, r.u)) for (s, r) in zip(sols, refs)
)

kick!(integrator) = (integrator.u[1] += 0.1; nothing)

@testset "PeriodicCallback ($(nameof(typeof(alg))), $(name))" for alg in (Tsit5(), Rosenbrock23()),
        (name, kwargs) in (
            ("default", (;)),
            ("phase", (; phase = 0.1)),
            ("initial and final affect", (; initial_affect = true, final_affect = true)),
        )
    cb() = PeriodicCallback(kick!, 0.25; save_positions = (false, false), kwargs...)
    prob = remake(decay_prob; callback = cb())
    eprob = EnsembleProblem(prob; prob_func = decay_prob_func, safetycopy = false)
    sol = solve(
        eprob, alg, EnsembleGPUArray(backend, 0.0); trajectories = length(ks),
        abstol = tol, reltol = tol, saveat = 0.05
    )
    refs = serial_solutions(decay_prob, decay_prob_func, length(ks), alg; callback = cb(), saveat = 0.05)
    @test all(s -> s.retcode == SciMLBase.ReturnCode.Success, sol.u)
    @test max_state_error(sol.u, refs) < state_tolerance(alg)
    # Without the kicks the solution decays monotonically.
    @test any(s -> any(>(0.05), diff(first.(s.u))), sol.u)
end

@testset "PeriodicCallback writing parameters" begin
    # A periodic "gear" that doubles the decay rate at every stop until it reaches a cap,
    # depending on each trajectory's own state.
    function shift!(integrator)
        if integrator.u[1] > 0.3 && integrator.p[1] < 4
            integrator.p[1] *= 2
        end
        return nothing
    end
    cb() = PeriodicCallback(shift!, 0.2; save_positions = (false, false))
    prob = remake(decay_prob; callback = cb())
    eprob = EnsembleProblem(prob; prob_func = decay_prob_func, safetycopy = false)
    sol = solve(
        eprob, Tsit5(), EnsembleGPUArray(backend, 0.0); trajectories = length(ks),
        abstol = tol, reltol = tol, saveat = 0.05
    )
    refs = serial_solutions(decay_prob, decay_prob_func, length(ks), Tsit5(); callback = cb(), saveat = 0.05)
    @test max_state_error(sol.u, refs) < state_tolerance(Tsit5())
    @test [s.prob.p[1] for s in sol.u] == [r.prob.p[1] for r in refs]
    @test any(s -> s.prob.p[1] != ks[1], sol.u)
end

@testset "The stops do not depend on the number of trajectories" begin
    cb = PeriodicCallback(kick!, 0.25; save_positions = (false, false))
    prob = remake(decay_prob; callback = cb)
    naccept = map((4, 64)) do n
        eprob = EnsembleProblem(
            prob; prob_func = (prob, ctx) -> remake(prob; p = [1.0]), safetycopy = false
        )
        sol = solve(
            eprob, Tsit5(), EnsembleGPUArray(backend, 0.0); trajectories = n,
            abstol = tol, reltol = tol, save_everystep = false
        )
        sol.u[1].stats.naccept
    end
    @test naccept[1] == naccept[2]
end

@testset "Unsupported periodic and preset-time callbacks" begin
    eprob(cb) = EnsembleProblem(
        remake(decay_prob; callback = cb); prob_func = decay_prob_func, safetycopy = false
    )
    custom = PeriodicCallback(kick!, 0.25; initialize = (c, u, t, integrator) -> nothing)
    @test_throws ArgumentError solve(
        eprob(custom), Tsit5(), EnsembleGPUArray(backend, 0.0); trajectories = length(ks)
    )
    preset = PresetTimeCallback([0.5, 1.0], kick!)
    @test_throws ArgumentError solve(
        eprob(preset), Tsit5(), EnsembleGPUArray(backend, 0.0); trajectories = length(ks)
    )
end
