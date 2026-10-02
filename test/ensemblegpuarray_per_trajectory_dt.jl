using DiffEqGPU, LinearAlgebra, Test
using OrdinaryDiffEqRosenbrock: Rodas5P, Rosenbrock23
using DiffEqCallbacks: PeriodicCallback
import SciMLBase
using SciMLBase: EnsembleProblem, ODEFunction, ODEProblem, ContinuousCallback, remake, solve
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
const SII = ModelingToolkit.SymbolicIndexingInterface

include("utils.jl")

# `EnsembleGPUArray(...; per_trajectory_dt = true)` integrates every trajectory with its own
# steps, following OrdinaryDiffEq's Rodas5P and its default controller, so each trajectory of a
# batch is compared with an OrdinaryDiffEq solve of that trajectory alone.

const AutoFiniteDiff = DiffEqGPU.ADTypes.AutoFiniteDiff
const alg = Rodas5P(autodiff = AutoFiniteDiff())
lanes(; kwargs...) = EnsembleGPUArray(backend, 0.0; per_trajectory_dt = true, kwargs...)

max_state_error(a, b) = maximum(maximum(abs, x .- y) for (x, y) in zip(a.u, b.u))

# Robertson's stiff kinetics, as an ODE and as a mass-matrix DAE (index 1); the trajectories
# scale the rate constants.
function rober_ode!(du, u, p, t)
    y1, y2, y3 = u
    k1, k2, k3 = p
    du[1] = -k1 * y1 + k3 * y2 * y3
    du[2] = k1 * y1 - k3 * y2 * y3 - k2 * y2^2
    du[3] = k2 * y2^2
    return nothing
end
function rober_dae!(du, u, p, t)
    y1, y2, y3 = u
    k1, k2, k3 = p
    du[1] = -k1 * y1 + k3 * y2 * y3
    du[2] = k1 * y1 - k3 * y2 * y3 - k2 * y2^2
    du[3] = y1 + y2 + y3 - 1
    return nothing
end
const rober_p = [0.04, 3.0e7, 1.0e4]
const rober_func = (prob, ctx) -> remake(prob; p = rober_p .* (1 + 0.5 * (ctx.sim_id - 1) / 7))
const rober_problems = (
    ODE = ODEProblem(rober_ode!, [1.0, 0.0, 0.0], (0.0, 100.0), rober_p),
    DAE = ODEProblem(
        ODEFunction(rober_dae!; mass_matrix = Diagonal([1.0, 1.0, 0.0])),
        [1.0, 0.0, 0.0], (0.0, 100.0), rober_p
    ),
)

@testset "Robertson $name: steps of each trajectory's own solve" for (name, prob) in pairs(rober_problems)
    kwargs = (; abstol = 1.0e-8, reltol = 1.0e-6, saveat = 10.0)
    eprob = EnsembleProblem(prob; prob_func = rober_func, safetycopy = false)
    sol = solve(eprob, alg, lanes(); trajectories = 8, kwargs...)
    for i in 1:8
        ref = solve(rober_func(prob, (; sim_id = i)), alg; kwargs...)
        s = sol.u[i]
        @test s.retcode == SciMLBase.ReturnCode.Success
        @test s.t == ref.t
        @test max_state_error(s, ref) < 1.0e-7
        # The controller is OrdinaryDiffEq's; the step sequences agree up to rounding, which
        # can flip a single accept decision.
        @test abs(s.stats.naccept - ref.stats.naccept) <= 1
        @test abs(s.stats.nreject - ref.stats.nreject) <= 1
    end
    # Each trajectory took its own steps.
    @test length(unique(s.stats.naccept for s in sol.u)) > 1
end

@testset "Robertson $name: fixed steps reproduce Rodas5P" for (name, prob) in pairs(rober_problems)
    # Every step passes this tolerance and `dtmax` keeps the steps at the initial `dt`: this
    # checks the stages, the iteration matrix, the Jacobian and the dense output on their own.
    prob = remake(prob; tspan = (0.0, 0.05))
    eprob = EnsembleProblem(prob; prob_func = rober_func, safetycopy = false)
    sol = solve(
        eprob, alg, lanes(); trajectories = 4, dt = 1.0e-4, dtmax = 1.0e-4, saveat = 0.00333,
        abstol = 1.0, reltol = 1.0
    )
    for i in 1:4
        ref = solve(rober_func(prob, (; sim_id = i)), alg; adaptive = false, dt = 1.0e-4, saveat = 0.00333)
        @test sol.u[i].t ≈ ref.t
        @test max_state_error(sol.u[i], ref) < 1.0e-12
    end
end

# Two-state decay with a periodic kick or a periodic parameter write.
decay!(du, u, p, t) = (du[1] = -p[1] * u[1]; du[2] = u[1] - p[2] * u[2]; nothing)
const decay_ks = [0.5, 1.0, 1.5, 2.0]
const decay_func = (prob, ctx) -> remake(prob; p = [decay_ks[ctx.sim_id], 0.3])
kick!(integrator) = (integrator.u[1] += 0.1; nothing)
function shift!(integrator)
    if integrator.u[1] > 0.3 && integrator.p[1] < 4
        integrator.p[1] *= 2
    end
    return nothing
end

@testset "PeriodicCallback $(name)" for (name, cb) in (
        ("kick", () -> PeriodicCallback(kick!, 0.25; save_positions = (false, false))),
        ("kick with phase", () -> PeriodicCallback(kick!, 0.25; phase = 0.1, save_positions = (false, false))),
        (
            "kick with initial and final affect",
            () -> PeriodicCallback(
                kick!, 0.25; initial_affect = true, final_affect = true, save_positions = (false, false)
            ),
        ),
        ("parameter write", () -> PeriodicCallback(shift!, 0.1; save_positions = (false, false))),
    )
    prob = ODEProblem(decay!, [1.0, 0.0], (0.0, 2.0), [1.0, 0.3]; callback = cb())
    kwargs = (; abstol = 1.0e-6, reltol = 1.0e-6, saveat = 0.05)
    eprob = EnsembleProblem(prob; prob_func = decay_func, safetycopy = false)
    sol = solve(eprob, alg, lanes(); trajectories = length(decay_ks), kwargs...)
    for i in eachindex(decay_ks)
        ref = solve(decay_func(remake(prob; callback = cb()), (; sim_id = i)), alg; kwargs...)
        @test sol.u[i].retcode == SciMLBase.ReturnCode.Success
        @test sol.u[i].t ≈ ref.t
        # Saves at an affect's time hold the value before the affect, as in OrdinaryDiffEq.
        @test max_state_error(sol.u[i], ref) < 1.0e-7
        @test sol.u[i].prob.p ≈ ref.prob.p
    end
end

@testset "A failing trajectory stops alone" begin
    prob = ODEProblem(decay!, [1.0, 0.0], (0.0, 2.0), [1.0, 0.3])
    ps = [[1.0, 0.3], [NaN, 0.3], [2.0, 0.3]]
    eprob = EnsembleProblem(prob; prob_func = (prob, ctx) -> remake(prob; p = ps[ctx.sim_id]), safetycopy = false)
    sol = solve(eprob, alg, lanes(); trajectories = 3, abstol = 1.0e-6, reltol = 1.0e-6, saveat = 0.5)
    @test sol.u[2].retcode == SciMLBase.ReturnCode.Unstable
    for i in (1, 3)
        ref = solve(remake(prob; p = ps[i]), alg; abstol = 1.0e-6, reltol = 1.0e-6, saveat = 0.5)
        @test sol.u[i].retcode == SciMLBase.ReturnCode.Success
        @test max_state_error(sol.u[i], ref) < 1.0e-7
    end
end

@testset "restore_stop_dt" begin
    # Opt-in: after a step shortened onto a stop, continue from the unshortened proposal.
    # Fewer steps when several steps fit between the stops, at the accuracy of the tolerance.
    cb() = PeriodicCallback(kick!, 0.1; save_positions = (false, false))
    prob = ODEProblem(decay!, [1.0, 0.0], (0.0, 2.0), [1.0, 0.3]; callback = cb())
    eprob = EnsembleProblem(prob; prob_func = decay_func, safetycopy = false)
    kwargs = (; abstol = 1.0e-7, reltol = 1.0e-7, saveat = 0.1)
    default = solve(eprob, alg, lanes(); trajectories = length(decay_ks), kwargs...)
    restored = solve(eprob, alg, lanes(); trajectories = length(decay_ks), restore_stop_dt = true, kwargs...)
    steps(sol) = sum(s.stats.naccept + s.stats.nreject for s in sol.u)
    @test steps(restored) < 0.8 * steps(default)
    for i in eachindex(decay_ks)
        ref = solve(
            decay_func(remake(prob; callback = cb()), (; sim_id = i)), alg;
            abstol = 1.0e-11, reltol = 1.0e-11, saveat = 0.1
        )
        @test restored.u[i].retcode == SciMLBase.ReturnCode.Success
        @test max_state_error(restored.u[i], ref) < 1.0e-6
    end
end

@testset "ModelingToolkit periodic event" begin
    @variables x(t) = 0.0 v(t) = 0.0
    @parameters k = 1.0
    @discretes g(t) = 1.0
    shift_up = ModelingToolkit.ImperativeAffect(modified = (; g), observed = (; v)) do m, o, ctx, integrator
        return (; g = (o.v > 0.25 && m.g < 3) ? m.g + 1 : m.g)
    end
    @named gearbox = System(
        [D(x) ~ v, D(v) ~ k - g * v], t;
        discrete_events = [
            ModelingToolkit.SymbolicDiscreteCallback(
                0.1, shift_up; discrete_parameters = [g], save_positions = (false, false)
            ),
        ]
    )
    sys = mtkcompile(gearbox)
    prob = ODEProblem(sys, [], (0.0, 3.0); affect_transform = DiffEqGPU.gpu_affect_transform, save_discretes = false)
    ks = [0.3, 0.6, 0.9, 2.0]
    set = SII.setsym_oop(prob, [k])
    prob_func = function (prob, ctx)
        u0, p = set(prob, [ks[ctx.sim_id]])
        return remake(prob; u0, p, lazy_initialization = true)
    end
    kwargs = (; abstol = 1.0e-6, reltol = 1.0e-6, saveat = 0.1)
    sol = solve(EnsembleProblem(prob; prob_func, safetycopy = false), alg, lanes(); trajectories = length(ks), kwargs...)
    cpu = [solve(prob_func(prob, (; sim_id = i)), alg; kwargs...) for i in eachindex(ks)]
    @test all(s -> s.retcode == SciMLBase.ReturnCode.Success, sol.u)
    @test maximum(max_state_error(s, c) for (s, c) in zip(sol.u, cpu)) < 1.0e-7
    @test [s.prob.ps[g] for s in sol.u] == [c.prob.ps[g] for c in cpu]
    @test length(unique(s.prob.ps[g] for s in sol.u)) > 1
end

@testset "Unsupported settings throw, at setup and in the solve" begin
    prob = ODEProblem(decay!, [1.0, 0.0], (0.0, 1.0), [1.0, 0.3])
    eprob = EnsembleProblem(prob; prob_func = decay_func, safetycopy = false)
    good = (; saveat = 0.1)
    @test DiffEqGPU.check_per_trajectory_dt(prob, alg, lanes(); good...) === nothing
    @test DiffEqGPU.check_per_trajectory_dt(prob, alg, lanes(); save_everystep = false) === nothing
    cases = (
        (Rosenbrock23(autodiff = AutoFiniteDiff()), good),
        (Rodas5P(), good),
        (alg, (;)),
        (alg, (; good..., save_idxs = [1])),
        (alg, (; good..., abstol = [1.0e-6, 1.0e-6])),
        (alg, (; good..., adaptive = false, dt = 0.01)),
        (alg, (; good..., callback = ContinuousCallback((u, t, i) -> u[1] - 0.5, i -> nothing))),
        (alg, (; good..., callback = PeriodicCallback(kick!, 0.25))),
    )
    for (a, kwargs) in cases
        @test_throws ArgumentError DiffEqGPU.check_per_trajectory_dt(prob, a, lanes(); kwargs...)
        @test_throws ArgumentError solve(eprob, a, lanes(); trajectories = 2, kwargs...)
    end
    # Options stored in the problem are checked too.
    @test_throws ArgumentError DiffEqGPU.check_per_trajectory_dt(remake(prob; save_idxs = [1]), alg, lanes(); good...)
    @test_throws ArgumentError DiffEqGPU.check_per_trajectory_dt(prob, alg, EnsembleGPUArray(backend, 0.0); good...)
end
