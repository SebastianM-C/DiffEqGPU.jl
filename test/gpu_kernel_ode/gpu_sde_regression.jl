using DiffEqGPU, OrdinaryDiffEq, StaticArrays, LinearAlgebra, CUDA, Statistics
using Random

Random.seed!(100)

@info "Convergence Test"

# dX_t = u*dt + udW_t
f(u, p, t) = u
g(u, p, t) = u
u0 = @SVector [0.5f0]

tspan = (0.0f0, 1.0f0)
prob = SDEProblem(f, g, u0, tspan)

monteprob = EnsembleProblem(prob)

dt = Float32(1 // 2^(8))
sol = solve(monteprob, GPUEM(), EnsembleGPUKernel(), dt = dt, trajectories = 1000,
            adaptive = false)

sol_array = Array(sol)

us = reshape(mean(sol_array, dims = 3), size(sol_array, 2))

us_exact = 0.5f0 * exp.(sol[1].t)

@test norm(us - us_exact, Inf) < 1e-1

@info "Diagonal Noise"

function lorenz(u, p, t)
    du1 = 10.0(u[2] - u[1])
    du2 = u[1] * (28.0 - u[3]) - u[2]
    du3 = u[1] * u[2] - (8 / 3) * u[3]
    return SVector{3}(du1, du2, du3)
end

function σ_lorenz(u, p, t)
    return SVector{3}(3.0f0, 3.0f0, 3.0f0)
end

u0 = @SVector [1.0f0, 0.0f0, 0.0f0]
tspan = (0.0f0, 10.0f0)
prob = SDEProblem(lorenz, σ_lorenz, u0, tspan)
monteprob = EnsembleProblem(prob)
dt = Float32(1 // 2^(8))

sol = solve(monteprob, GPUEM(), EnsembleGPUKernel(0.0), dt = dt, trajectories = 10,
            adaptive = false)

@test sol.converged == true

sol = solve(monteprob, GPUEM(), EnsembleGPUKernel(0.0), dt = dt, trajectories = 10,
            adaptive = false, save_everystep = false)

@test sol.converged == true
@test length(sol[1].u) == 2

@info "Non-Diagonal Noise"

function f(u, p, t)
    return 1.01 .* u
end

function g(u, p, t)
    du1_1 = 0.3u[1]
    du1_2 = 0.6u[1]
    du1_3 = 0.9u[1]
    du1_4 = 0.12u[1]
    du2_1 = 1.2u[2]
    du2_2 = 0.2u[2]
    du2_3 = 0.3u[2]
    du2_4 = 1.8u[2]
    return SMatrix{2, 4}(du1_1, du1_2, du1_3, du1_4, du2_1, du2_2, du2_3, du2_4)
end

u0 = @SVector ones(Float32, 2)
noise_rate_prototype = @SMatrix zeros(Float32, 2, 4)
prob = SDEProblem(f, g, u0, (0.0f0, 1.0f0), noise_rate_prototype = noise_rate_prototype)
monteprob = EnsembleProblem(prob)

sol = solve(monteprob, GPUEM(), EnsembleGPUKernel(0.0), dt = dt, trajectories = 10,
            adaptive = false)

@test sol.converged == true

sol = solve(monteprob, GPUEM(), EnsembleGPUKernel(0.0), dt = dt, trajectories = 10,
            adaptive = false, save_everystep = false)

@test sol.converged == true
@test length(sol[1].u) == 2
