using DiffEqGPU, OrdinaryDiffEq, StaticArrays, Test

include("../utils.jl")

# x' = 1 from x = 0, so every crossing time is known exactly and the final value says which
# affects ran. Each `bump` fires once, on the upcrossing of its own level.
rhs(u, p, t) = SVector(1.0f0)
prob = ODEProblem{false}(rhs, SVector(0.0f0), (0.0f0, 2.0f0))

function bump(level, by)
    return ContinuousCallback(
        (u, t, integrator) -> u[1] - level,
        integrator -> (integrator.u = integrator.u .+ by);
        affect_neg! = nothing, save_positions = (false, false)
    )
end

function gpu_final(alg, callbacks)
    sol = solve(
        EnsembleProblem(prob; safetycopy = false), alg, EnsembleGPUKernel(backend, 0.0);
        trajectories = 2, callback = CallbackSet(callbacks...), merge_callbacks = true,
        save_everystep = false, dt = 0.01f0
    )
    return [s.u[end][1] for s in sol.u]
end

cpu_final(callbacks) = solve(prob, Tsit5(); callback = CallbackSet(callbacks...)).u[end][1]

# The affect applied must be the one of the callback that fired, not the first one in the set:
# three small bumps that all fire give 2 + 0.001 + 0.01 + 0.1 in every ordering. Applying a
# wrong affect is off by percents; the tolerance only absorbs Float32 Vern7's dense output,
# through which the state at each event is read (https://github.com/SciML/DiffEqGPU.jl/issues/554).
bumps = (bump(0.25f0, 0.001f0), bump(0.75f0, 0.01f0), bump(1.25f0, 0.1f0))
orderings = ((1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1))

@testset "Multiple continuous callbacks ($(nameof(typeof(alg))))" for
    alg in (GPUTsit5(), GPUVern7())
    @testset "ordering $order" for order in orderings
        @test all(x -> isapprox(x, 2.111f0; rtol = 1.0f-4), gpu_final(alg, bumps[collect(order)]))
    end

    # The first jump carries x past the second level, so only the first callback fires.
    jump_a, jump_b = bump(0.5f0, 10.0f0), bump(1.5f0, 1000.0f0)
    for callbacks in ((jump_a, jump_b), (jump_b, jump_a))
        @test all(x -> isapprox(x, cpu_final(callbacks); rtol = 1.0f-4), gpu_final(alg, callbacks))
    end
end

@testset "VectorContinuousCallback is refused" begin
    vcc = VectorContinuousCallback(
        (out, u, t, integrator) -> (out[1] = u[1] - 0.5f0),
        (integrator, idx) -> (integrator.u = integrator.u .+ 1.0f0), 1;
        save_positions = (false, false)
    )
    err = try
        solve(
            EnsembleProblem(prob; safetycopy = false), GPUTsit5(),
            EnsembleGPUKernel(backend, 0.0); trajectories = 2, callback = vcc,
            merge_callbacks = true, save_everystep = false, dt = 0.01f0
        )
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("VectorContinuousCallback", sprint(showerror, err))
end
