using DiffEqGPU, Test
using OrdinaryDiffEq: Tsit5, Rosenbrock23, Rodas5P
import SciMLBase
using SciMLBase: EnsembleProblem, remake, solve
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
const SII = ModelingToolkit.SymbolicIndexingInterface

include("utils.jl")

# ModelingToolkit events on `EnsembleGPUArray`: periodic events whose `ImperativeAffect`
# writes a discrete parameter (a "gear" that shifts at most once per tick), built with
# `gpu_affect_transform` and swept over a parameter with lazy remakes.

const transform = (; affect_transform = DiffEqGPU.gpu_affect_transform, save_discretes = false)
const tol = 1.0e-9

function sweep(prob, sym, values)
    set = SII.setsym_oop(prob, [sym])
    return function (prob, ctx)
        u0, p = set(prob, [values[ctx.sim_id]])
        return remake(prob; u0, p, lazy_initialization = true)
    end
end

function compare_with_cpu(prob, prob_func, n, alg; kwargs...)
    eprob = EnsembleProblem(prob; prob_func, safetycopy = false)
    esol = solve(eprob, alg, EnsembleGPUArray(backend, 0.0); trajectories = n, kwargs...)
    cpu = [solve(prob_func(prob, (; sim_id = i)), alg; kwargs...) for i in 1:n]
    return esol, cpu
end

max_state_error(esol, cpu) = maximum(
    maximum(maximum(abs, a .- b) for (a, b) in zip(g.u, c.u)) for (g, c) in zip(esol.u, cpu)
)

@variables x(t) = 0.0 v(t) = 0.0
@parameters k = 1.0
@discretes g(t) = 1.0

const shift_up = ModelingToolkit.ImperativeAffect(
    modified = (; g), observed = (; v)
) do m, o, ctx, integrator
    return (; g = (o.v > 0.25 && m.g < 3) ? m.g + 1 : m.g)
end

const gearbox = mtkcompile(
    System(
        [D(x) ~ v, D(v) ~ k - g * v], t;
        discrete_events = [
            ModelingToolkit.SymbolicDiscreteCallback(0.1, shift_up; discrete_parameters = [g]),
        ],
        name = :gearbox
    )
)
const ks = [0.3, 0.6, 0.9, 2.0]

@testset "Periodic ImperativeAffect, $(nameof(typeof(alg)))" for (alg, err) in (
        (Tsit5(), 1.0e-8), (Rosenbrock23(), 1.0e-5),
    )
    prob = ODEProblem(gearbox, [], (0.0, 3.0); transform...)
    prob_func = sweep(prob, k, ks)
    kwargs = (; abstol = tol, reltol = tol, saveat = 0.1)
    esol, cpu = compare_with_cpu(prob, prob_func, length(ks), alg; kwargs...)
    @test all(s -> SciMLBase.successful_retcode(s), esol.u)
    @test max_state_error(esol, cpu) < err
    gears = [s.prob.ps[g] for s in esol.u]
    @test gears == [c.prob.ps[g] for c in cpu]
    # The sweep reaches different gears, so the per-trajectory writes are exercised.
    @test length(unique(gears)) > 1
end

@testset "The transformed problem solves on the CPU as the original" begin
    plain = ODEProblem(gearbox, [], (0.0, 3.0))
    lowered = ODEProblem(gearbox, [], (0.0, 3.0); transform...)
    a = solve(plain, Tsit5(); abstol = tol, reltol = tol, saveat = 0.1)
    b = solve(lowered, Tsit5(); abstol = tol, reltol = tol, saveat = 0.1)
    @test a.u == b.u
end

# The index-1 DAE of the initialization tests, whose decay rate the gear scales.
@variables y(t)
@parameters c = 1.0
const dae_shift = ModelingToolkit.ImperativeAffect(
    modified = (; g), observed = (; y)
) do m, o, ctx, integrator
    return (; g = (o.y > 0.3 && m.g < 3) ? m.g + 1 : m.g)
end
function dae_system(; reinitializealg = SciMLBase.NoInit(), affect = dae_shift)
    event = ModelingToolkit.SymbolicDiscreteCallback(
        0.1, affect; discrete_parameters = [g], reinitializealg
    )
    return mtkcompile(
        System(
            [D(x) ~ -c * g * x, 0 ~ y + sin(y) - x], t;
            discrete_events = [event], guesses = [y => 0.5], name = :dae
        )
    )
end

@testset "Periodic ImperativeAffect on a DAE, $(nameof(typeof(alg)))" for (alg, err) in (
        (Rodas5P(), 1.0e-6), (Rosenbrock23(), 1.0e-5),
    )
    prob = ODEProblem(dae_system(), [x => 1.0], (0.0, 2.0); transform...)
    prob_func = sweep(prob, c, [0.1, 0.4, 1.0, 2.0])
    kwargs = (; abstol = 1.0e-8, reltol = 1.0e-8, saveat = 0.1)
    esol, cpu = compare_with_cpu(prob, prob_func, 4, alg; kwargs...)
    @test all(s -> SciMLBase.successful_retcode(s), esol.u)
    @test max_state_error(esol, cpu) < err
    @test [s.prob.ps[g] for s in esol.u] == [r.prob.ps[g] for r in cpu]
end

@testset "Unsupported events" begin
    lowered(sys, u0 = []) = ODEProblem(sys, u0, (0.0, 1.0); transform...)
    # Affects given as equations.
    equations = mtkcompile(
        System(
            [D(x) ~ v, D(v) ~ k - g * v], t;
            discrete_events = [
                ModelingToolkit.SymbolicDiscreteCallback(
                    0.1, [g ~ Pre(g) + 1]; discrete_parameters = [g]
                ),
            ],
            name = :equations
        )
    )
    @test_throws ArgumentError lowered(equations)
    # A write to a parameter that is not discrete.
    write_k = ModelingToolkit.ImperativeAffect(modified = (; k)) do m, o, ctx, integrator
        return (; k = 2 * m.k)
    end
    shared = mtkcompile(
        System(
            [D(x) ~ v, D(v) ~ k - g * v], t;
            discrete_events = [ModelingToolkit.SymbolicDiscreteCallback(0.1, write_k)],
            name = :shared
        )
    )
    @test_throws ArgumentError lowered(shared)
    # Reinitializing a DAE after the event would run on the whole batch.
    @test_throws ArgumentError lowered(
        dae_system(; reinitializealg = SciMLBase.CheckInit()), [x => 1.0]
    )
    # A custom `initialize`.
    with_initialize = mtkcompile(
        System(
            [D(x) ~ v, D(v) ~ k - g * v], t;
            discrete_events = [
                ModelingToolkit.SymbolicDiscreteCallback(
                    0.1, shift_up; discrete_parameters = [g], initialize = shift_up
                ),
            ],
            name = :with_initialize
        )
    )
    @test_throws ArgumentError lowered(with_initialize)
    # Affects built without the transform are refused by the solve.
    for prob in (
            ODEProblem(gearbox, [], (0.0, 1.0)),
            # An equation affect on a DAE unknown, which ModelingToolkit solves for.
            ODEProblem(
                mtkcompile(
                    System(
                        [D(x) ~ -x, 0 ~ y + sin(y) - x], t;
                        discrete_events = [
                            ModelingToolkit.SymbolicDiscreteCallback(
                                0.1, [x ~ Pre(x) + 0.1]
                            ),
                        ],
                        guesses = [y => 0.5], name = :implicit
                    )
                ), [x => 1.0], (0.0, 1.0)
            ),
        )
        eprob = EnsembleProblem(prob; safetycopy = false)
        @test_throws ArgumentError solve(
            eprob, Rosenbrock23(), EnsembleGPUArray(backend, 0.0); trajectories = 2
        )
    end
end
