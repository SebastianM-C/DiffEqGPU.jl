using DiffEqGPU, Test
using OrdinaryDiffEq: Tsit5, Rosenbrock23
using SciMLBase: CallbackSet, ContinuousCallback, DiscreteCallback, EnsembleProblem,
    ODEFunction, ODEProblem, remake, solve

include("utils.jl")

const tol = 1.0e-9
const algs = (Tsit5(), Rosenbrock23())
# The batch shares one step size, set by the hardest trajectory, so a trajectory does not
# get exactly its own steps. The low-order Rosenbrock23 shows that most.
state_tolerance(::Tsit5) = 1.0e-6
state_tolerance(::Rosenbrock23) = 1.0e-4

# Reference: each trajectory solved on its own with the same callback, so that its
# parameters are mutated exactly as in the ensemble.
function serial_solutions(prob, prob_func, n, alg; kwargs...)
    return map(1:n) do i
        trajectory = prob_func(deepcopy(prob), (; sim_id = i))
        solve(trajectory, alg; abstol = tol, reltol = tol, kwargs...)
    end
end

max_state_error(sols, refs) = maximum(
    maximum(maximum(abs, a .- b) for (a, b) in zip(s.u, r.u)) for (s, r) in zip(sols, refs)
)

# u' = -k u, u(0) = 1: crosses 0.5 once (downward) for every k > 0.
decay(du, u, p, t) = (du[1] = -p[1] * u[1]; nothing)
decay_jac(J, u, p, t) = (J[1, 1] = -p[1]; nothing)
decay_condition(u, t, integrator) = u[1] - 0.5
bump!(integrator) = (integrator.u[1] += 0.3; nothing)
ks = [0.5, 1.0, 1.5, 2.0]
decay_prob_func = (prob, ctx) -> remake(prob; p = [ks[ctx.sim_id]])
# Stiff methods on `EnsembleGPUArray` need the Jacobian.
decay_prob = ODEProblem(ODEFunction(decay; jac = decay_jac), [1.0], (0.0, 3.0), [1.0])
bump = ContinuousCallback(decay_condition, bump!; save_positions = (false, false))

@testset "`callback` keyword is applied ($(nameof(typeof(alg))))" for alg in algs
    eprob = EnsembleProblem(decay_prob; prob_func = decay_prob_func, safetycopy = false)
    sol = solve(
        eprob, alg, EnsembleGPUArray(backend, 0.0); trajectories = length(ks),
        abstol = tol, reltol = tol, saveat = 0.1, callback = bump
    )
    refs = serial_solutions(decay_prob, decay_prob_func, length(ks), alg; saveat = 0.1, callback = bump)
    # Without the bump the solution decays monotonically.
    @test all(s -> any(diff(first.(s.u)) .> 0.1), sol.u)
    @test max_state_error(sol.u, refs) < state_tolerance(alg)
end

@testset "`merge_callbacks = false` replaces the problem callback" begin
    # Fires at a different level than `bump`, so the merged callbacks never coincide.
    sink!(integrator) = (integrator.u[1] -= 0.1; nothing)
    sink = ContinuousCallback(
        (u, t, integrator) -> u[1] - 0.3, sink!; save_positions = (false, false)
    )
    prob = remake(decay_prob; callback = sink)
    eprob = EnsembleProblem(prob; prob_func = decay_prob_func, safetycopy = false)
    replaced = solve(
        eprob, Tsit5(), EnsembleGPUArray(backend, 0.0); trajectories = length(ks),
        abstol = tol, reltol = tol, saveat = 0.1, callback = bump, merge_callbacks = false
    )
    refs = serial_solutions(decay_prob, decay_prob_func, length(ks), Tsit5(); saveat = 0.1, callback = bump)
    @test max_state_error(replaced.u, refs) < 1.0e-6

    merged = solve(
        eprob, Tsit5(), EnsembleGPUArray(backend, 0.0); trajectories = length(ks),
        abstol = tol, reltol = tol, saveat = 0.1, callback = bump
    )
    refs = serial_solutions(
        prob, decay_prob_func, length(ks), Tsit5(); saveat = 0.1, callback = bump
    )
    @test max_state_error(merged.u, refs) < 1.0e-6
end

# u' = gear * amp * cos(t), u(0) = 0. Trajectories with amp > 0.5 cross 0.5 upward (the
# affect shifts the "gear" p[1] from 1 to 2) and later downward (no affect); amp = 0.4 never
# crosses and must keep its parameters.
geared(du, u, p, t) = (du[1] = p[1] * p[2] * cos(t); nothing)
geared_jac(J, u, p, t) = (J[1, 1] = 0; nothing)
shift_condition(u, t, integrator) = u[1] - 0.5
shift_up!(integrator) = (integrator.p[1] += 1; nothing)
amps = [1.0, 0.8, 0.6, 0.4]
geared_prob_func = (prob, ctx) -> remake(prob; p = [1.0, amps[ctx.sim_id]])

@testset "one-sided callback, parameter write ($(nameof(typeof(alg))))" for alg in algs
    shift = ContinuousCallback(
        shift_condition, shift_up!, nothing; save_positions = (false, false)
    )
    prob = ODEProblem(
        ODEFunction(geared; jac = geared_jac), [0.0], (0.0, 3.0), [1.0, 1.0]; callback = shift
    )
    eprob = EnsembleProblem(prob; prob_func = geared_prob_func, safetycopy = false)
    sol = solve(
        eprob, alg, EnsembleGPUArray(backend, 0.0); trajectories = length(amps),
        abstol = tol, reltol = tol, saveat = 0.1
    )
    refs = serial_solutions(prob, geared_prob_func, length(amps), alg; saveat = 0.1)
    @test max_state_error(sol.u, refs) < state_tolerance(alg)
    @test [s.prob.p for s in sol.u] == [[2.0, 1.0], [2.0, 0.8], [2.0, 0.6], [1.0, 0.4]]
    @test [s.prob.p for s in sol.u] == [r.prob.p for r in refs]
end

@testset "unsupported callback hooks are refused" begin
    hooked = ContinuousCallback(
        decay_condition, bump!;
        initialize = (c, u, t, integrator) -> nothing, save_positions = (false, false)
    )
    eprob = EnsembleProblem(decay_prob; prob_func = decay_prob_func, safetycopy = false)
    @test_throws ArgumentError solve(
        eprob, Tsit5(), EnsembleGPUArray(backend, 0.0); trajectories = length(ks),
        callback = hooked
    )
    finalized = DiscreteCallback(
        (u, t, integrator) -> false, integrator -> nothing;
        finalize = (c, u, t, integrator) -> nothing
    )
    @test_throws ArgumentError solve(
        eprob, Tsit5(), EnsembleGPUArray(backend, 0.0); trajectories = length(ks),
        callback = finalized
    )
end

@testset "callbacks with per-trajectory time spans are refused" begin
    prob = ODEProblem(decay, [1.0], (0.0, 3.0), (1.0,))
    eprob = EnsembleProblem(
        prob; prob_func = (prob, ctx) -> remake(prob; tspan = (0.0, 2.0 + ctx.sim_id)),
        safetycopy = false
    )
    @test_throws ArgumentError solve(
        eprob, Tsit5(), EnsembleGPUArray(backend, 0.0); trajectories = 4, callback = bump
    )
    # Without callbacks, differing time spans are still supported.
    sol = solve(
        eprob, Tsit5(), EnsembleGPUArray(backend, 0.0); trajectories = 4,
        abstol = tol, reltol = tol, save_everystep = false
    )
    @test [s.u[end][1] for s in sol.u] ≈ [exp(-(2.0 + i)) for i in 1:4] rtol = 1.0e-6
end
