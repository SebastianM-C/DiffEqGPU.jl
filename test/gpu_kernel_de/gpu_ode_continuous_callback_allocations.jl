using DiffEqGPU, StaticArrays, SciMLBase, Test

include("../utils.jl")

# A continuous callback used to make the kernel heap-allocate the integrator once per
# trajectory: it escaped through calls that were not inlined. A large ensemble then ran out
# of the device heap (CUDA's default limit is 8 MiB; 100_000 trajectories needed ~43 MB).
rhs(u, p, t) = SVector(1.0f0)
prob = ODEProblem{false}(rhs, SVector(0.0f0), (0.0f0, 2.0f0))

struct CrossHalf end
@inline (::CrossHalf)(u, t, integrator) = u[1] - 0.5f0
struct AddTen end
@inline (::AddTen)(integrator) = (integrator.u = integrator.u .+ 10.0f0; nothing)

callback = CallbackSet(
    ContinuousCallback(
        CrossHalf(), AddTen(); affect_neg! = nothing, save_positions = (false, false)
    )
)

solve_ensemble(alg, trajectories) = solve(
    EnsembleProblem(prob; safetycopy = false), alg, EnsembleGPUKernel(backend, 0.0);
    trajectories, callback, merge_callbacks = true, save_everystep = false, dt = 0.01f0
)

# An explicit and a stiff solver: both go through the same callback handling, but the stiff
# integrators carry more state.
algs = (GPUTsit5(), GPURodas5P())

@testset "A continuous callback does not exhaust the device heap ($(nameof(typeof(alg))))" for
    alg in algs
    sol = solve_ensemble(alg, 100_000)
    @test length(sol.u) == 100_000
    @test all(s -> isapprox(s.u[end][1], 12.0f0; rtol = 1.0f-5), sol.u)
end

if GROUP == "CUDA"
    @testset "A continuous callback does not heap-allocate in the kernel ($(nameof(typeof(alg))))" for
        alg in algs
        io = IOBuffer()
        CUDA.@device_code_llvm io = io dump_module = true solve_ensemble(alg, 4)
        kernels = filter(
            f -> occursin("gpu_ode_asolve_kernel", first(split(f, '\n'))),
            split(String(take!(io)), "\ndefine ")
        )
        @test !isempty(kernels)
        for kernel in kernels
            @test !any(
                l -> occursin("call", l) && occursin("gpu_gc_pool_alloc", l),
                split(kernel, '\n')
            )
        end
    end
end
