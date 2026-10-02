using DiffEqGPU, ModelingToolkit, SciMLBase, Test
using ModelingToolkit: t_nounits as t, D_nounits as D
using ModelingToolkit: MTKParameters
using OrdinaryDiffEq: Tsit5, Rosenbrock23
const SS = ModelingToolkit.SciMLStructures

include("utils.jl")

# `EnsembleGPUArray` batches split `MTKParameters`: the tunable and discrete portions per
# trajectory, everything else shared by all trajectories.

const TOL = (; abstol = 1.0e-8, reltol = 1.0e-8)

# A copy of `p` with its own buffers, so a callback writing the discretes of one solve does
# not leak into another.
fresh(p::MTKParameters) = SS.replace(SS.Discrete(), p, copy(SS.canonicalize(SS.Discrete(), p)[1]))

function max_error(esol, cpu)
    return maximum(eachindex(cpu)) do i
        maximum(maximum(abs, a .- b) for (a, b) in zip(esol.u[i].u, cpu[i].u))
    end
end

@variables x(t) = 1.0 y(t) = 0.0
@parameters a = 1.0 b = 2.0 c = 0.5 flag::Bool = true n::Int = 2
@discretes g(t) = 1.0
# The symbolic event only makes `g` a discrete; it never fires, and the problem's callback is
# replaced below.
never = ModelingToolkit.SymbolicContinuousCallback(
    [x ~ -10.0], [g ~ Pre(g) + 1]; discrete_parameters = [g]
)
@named sys = System(
    [
        D(x) ~ -a * x + ifelse(flag, b, 0.0) * 0.1 * y * g - 0.01 * n,
        D(y) ~ -c * y + 0.2 * x * g,
    ], t; continuous_events = [never]
)
sys = mtkcompile(sys)
const ix = findfirst(isequal(x), unknowns(sys))
final_g(sol) = sol.prob.p.discrete[1][1]

# The discrete steps up once `x` falls below 0.5; the callback writes it through the
# trajectory's parameters, on the CPU and on the device alike.
bump = ContinuousCallback(
    (u, t, integ) -> u[ix] - 0.5,
    integ -> (integ.p.discrete[1][1] += 1; nothing);
    save_positions = (false, false)
)

function sweep(prob, as)
    remade(i) = (pr = remake(prob; p = [a => as[i], b => oftype(as[i], 1.5 + 0.25i)]); remake(pr; p = fresh(pr.p)))
    return remade, EnsembleProblem(prob; prob_func = (prob, ctx) -> remade(ctx.sim_id), safetycopy = false)
end

const as = [0.8, 1.0, 1.3, 1.6]
prob = remake(
    ODEProblem{true, SciMLBase.FullSpecialize}(sys, [], (0.0, 3.0); jac = true);
    callback = bump
)
@test prob.p isa MTKParameters
@test !isempty(prob.p.discrete) && !isempty(prob.p.constant)
remade, eprob = sweep(prob, as)

@testset "Tunables, Bool/Int constants and a callback-written discrete with $(nameof(typeof(alg)))" for alg in (
        Tsit5(), Rosenbrock23(),
    )
    cpu = [solve(remade(i), alg; saveat = 0.1, TOL...) for i in eachindex(as)]
    # Every trajectory crosses x = 0.5 once
    @test all(s -> final_g(s) == 2.0, cpu)
    esol = solve(
        eprob, alg, EnsembleGPUArray(backend, 0.0); trajectories = length(as),
        saveat = 0.1, TOL...
    )
    @test all(SciMLBase.successful_retcode, esol.u)
    @test max_error(esol, cpu) < 1.0e-6
    # The returned solutions carry the discrete the callback wrote, and their own tunables
    @test all(s -> final_g(s) == 2.0, esol.u)
    @test [s.ps[a] for s in esol.u] == as
end

@testset "Float32" begin
    prob32 = remake(
        ODEProblem{true, SciMLBase.FullSpecialize}(
            sys, [x => 1.0f0, y => 0.0f0, a => 1.0f0, b => 2.0f0, c => 0.5f0, g => 1.0f0],
            (0.0f0, 3.0f0); jac = true
        );
        callback = bump
    )
    @test eltype(prob32.u0) == Float32 == eltype(prob32.p.tunable)
    remade32, eprob32 = sweep(prob32, Float32.(as))
    cpu = [solve(remade(i), Tsit5(); saveat = 0.1, TOL...) for i in eachindex(as)]
    esol = solve(
        eprob32, Tsit5(), EnsembleGPUArray(backend, 0.0); trajectories = length(as),
        saveat = 0.1f0, abstol = 1.0f-6, reltol = 1.0f-6
    )
    @test eltype(esol.u[1].u[1]) == Float32
    @test max_error(esol, cpu) < 1.0e-4
    @test all(s -> final_g(s) == 2.0f0, esol.u)

    # Float64 parameters are batched in the precision of a Float32 state
    ext = Base.get_extension(DiffEqGPU, :ModelingToolkitBaseExt)
    batched = DiffEqGPU.pack_parameters([remade(1), remade(2)], Float32)
    @test batched isa ext.BatchedMTKParameters
    @test eltype(batched.tunable) == Float32 && eltype(batched.discrete[1]) == Float32
    @test batched.shared[1] == map(c -> c isa BitVector ? Vector{Bool}(c) : c, prob.p.constant)
end

@testset "Trajectories differing outside the tunables are refused" begin
    sub = ModelingToolkit.subset_tunables(sys, [sys.a])
    # Without the symbolic event, whose compiled condition needs a full integrator
    subprob = remake(
        ODEProblem{true, SciMLBase.FullSpecialize}(sub, [], (0.0, 1.0)); callback = nothing
    )
    @test SS.canonicalize(SS.Tunable(), subprob.p)[1] == [1.0]
    bad = EnsembleProblem(
        subprob; safetycopy = false,
        prob_func = (prob, ctx) -> remake(prob; p = [sub.b => Float64(ctx.sim_id)])
    )
    @test_throws ArgumentError solve(
        bad, Tsit5(), EnsembleGPUArray(backend, 0.0); trajectories = 2
    )
    # Varying only the remaining tunable is fine
    good = EnsembleProblem(
        subprob; safetycopy = false,
        prob_func = (prob, ctx) -> remake(prob; p = [sub.a => Float64(ctx.sim_id)])
    )
    esol = solve(
        good, Tsit5(), EnsembleGPUArray(backend, 0.0); trajectories = 2, saveat = 0.1, TOL...
    )
    cpu = [solve(good.prob_func(subprob, (; sim_id = i)), Tsit5(); saveat = 0.1, TOL...) for i in 1:2]
    @test max_error(esol, cpu) < 1.0e-6
end

# Constant array parameters of different lengths form a vector-of-vectors buffer; build such
# `MTKParameters` directly, with a `BitVector` constant beside it.
@testset "Ragged and BitVector constant buffers" begin
    lookup(v, i) = v[clamp(round(Int, i), 1, length(v))]
    function f!(du, u, p, t)
        tables, flags = p.constant
        k = p.tunable[1]
        du[1] = -k * u[1] + lookup(tables[1], 3) + (flags[2] ? lookup(tables[2], 2) : 0.0)
        return nothing
    end
    mkp(k) = MTKParameters(
        [k], Float64[], (), ([[1.0, 2.0, 3.0], [0.5, 0.25]], BitVector([false, true])), (), ()
    )
    rprob = ODEProblem(f!, [1.0], (0.0, 1.0), mkp(1.0))
    ks = [0.5, 1.0, 2.0]
    reprob = EnsembleProblem(
        rprob; safetycopy = false, prob_func = (prob, ctx) -> remake(prob; p = mkp(ks[ctx.sim_id]))
    )
    esol = solve(reprob, Tsit5(), EnsembleGPUArray(backend, 0.0); trajectories = 3, TOL...)
    # u' = -k u + 3.25 with u(0) = 1
    exact(k) = 3.25 / k + (1 - 3.25 / k) * exp(-k)
    @test all(i -> isapprox(esol.u[i].u[end][1], exact(ks[i]); rtol = 1.0e-6), eachindex(ks))

    # A buffer that cannot be uploaded is refused
    nonnumeric = MTKParameters([1.0], Float64[], (), (), ([sin],), ())
    nprob = EnsembleProblem(
        remake(rprob; p = nonnumeric); safetycopy = false, prob_func = (prob, ctx) -> prob
    )
    @test_throws ArgumentError solve(
        nprob, Tsit5(), EnsembleGPUArray(backend, 0.0); trajectories = 2
    )
end
