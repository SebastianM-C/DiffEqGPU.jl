using DiffEqGPU, LinearAlgebra, Test
using OrdinaryDiffEqRosenbrock: Rodas5P, Rosenbrock23
import SciMLBase
using SciMLBase: EnsembleProblem, remake, solve
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
import SymbolicIndexingInterface as SII

include("utils.jl")

# An index-1 DAE whose algebraic variable has only a guess, swept over a parameter with lazy
# remakes, as a parameter sweep builds its trajectories. Each trajectory has to be
# initialized before the batched solve, and the batched solve has to respect the singular
# mass matrix; without either, the algebraic variable drifts off the constraint while the
# solve still reports success.
@variables x(t) = 1.0 y(t)
@parameters k = 1.0
@named sys = System([D(x) ~ -k * x, 0 ~ y + sin(y) - x], t; guesses = [y => 0.5])
const csys = mtkcompile(sys)
const prob = ODEProblem(csys, [], (0.0, 2.0); jac = true)
const ix = SII.variable_index(prob, x)
const iy = SII.variable_index(prob, y)
residual(u) = u[iy] + sin(u[iy]) - u[ix]

const ks = [0.5, 1.0, 1.5, 2.0]
const set_k = SII.setsym_oop(prob, [k])
prob_func = function (prob, ctx)
    u0, p = set_k(prob, [ks[ctx.sim_id]])
    return remake(prob; u0, p, lazy_initialization = true)
end

@testset "Lazily remade DAE, $(nameof(typeof(alg)))" for (alg, err) in (
        (Rodas5P(), 1.0e-6), (Rosenbrock23(), 1.0e-5),
    )
    kwargs = (; abstol = 1.0e-8, reltol = 1.0e-8, saveat = 0.1)
    eprob = EnsembleProblem(prob; prob_func, safetycopy = false)
    esol = solve(eprob, alg, EnsembleGPUArray(backend, 0.0); trajectories = length(ks), kwargs...)
    cpu = [solve(prob_func(prob, (; sim_id = i)), alg; kwargs...) for i in eachindex(ks)]
    @test all(s -> SciMLBase.successful_retcode(s), esol.u)
    for (g, c) in zip(esol.u, cpu)
        # The initial state is the initialized one, not the guess.
        @test abs(residual(g.u[1])) < 1.0e-8
        @test maximum(abs ∘ residual, g.u) < 1.0e-6
        @test maximum(maximum(abs, a .- b) for (a, b) in zip(g.u, c.u)) < err
    end
end
