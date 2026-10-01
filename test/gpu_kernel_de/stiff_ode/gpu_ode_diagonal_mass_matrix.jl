using DiffEqGPU, LinearAlgebra, SciMLBase, StaticArrays, Test

include("../../utils.jl")

# A semi-explicit DAE in mass-matrix form with a known solution: `nd` differential states
# u' = -p u + y and `nd` algebraic ones 0 = u / 2 - y, so u(t) = u(0) exp((1/2 - p) t).
const nd = 3
const N = 2nd

function rhs(x, p, t)
    return SVector{N}(
        ntuple(Val(N)) do i
            i <= nd ? -p[1] * x[i] + x[nd + i] : x[i - nd] / 2 - x[i]
        end
    )
end

u0_diff = SVector{nd, Float32}(ntuple(i -> 1.0f0 + i / nd, Val(nd)))
u0 = vcat(u0_diff, u0_diff / 2)
p = SVector(2.0f0)
tspan = (0.0f0, 1.0f0)
exact = u0_diff * exp((0.5f0 - p[1]) * tspan[2])

mass_diag = vcat(ones(Float32, nd), zeros(Float32, nd))
mass_matrices = (
    dense = Matrix(Diagonal(mass_diag)),
    diagonal = Diagonal(mass_diag),
    static_diagonal = Diagonal(SVector{N}(mass_diag)),
)

problem(mm) = ODEProblem{false}(ODEFunction{false}(rhs; mass_matrix = mm), u0, tspan, p)

function final_states(alg, mm)
    sol = solve(
        EnsembleProblem(problem(mm); safetycopy = false), alg,
        EnsembleGPUKernel(backend, 0.0); trajectories = 2, adaptive = true,
        dt = 0.01f0, abstol = 1.0f-7, reltol = 1.0f-7, save_everystep = false
    )
    return [s.u[end] for s in sol.u]
end

@testset "A diagonal mass matrix stays diagonal" begin
    for mm in (mass_matrices.diagonal, mass_matrices.static_diagonal)
        f = DiffEqGPU.make_prob_compatible(problem(mm)).f
        @test f.mass_matrix isa Diagonal{Float32, <:SVector{N}}
        @test f.mass_matrix == Diagonal(mass_diag)
    end
end

# Not GPUKvaerno3/GPUKvaerno5: with a singular mass matrix they do not finish this problem,
# and with a non-identity invertible one their results are wrong, independently of how the
# mass matrix is stored. Their `W` goes through the same `add_mass_matrix`.
@testset "Diagonal mass matrix ($(nameof(typeof(alg))))" for alg in (
        GPURosenbrock23(), GPURodas4(), GPURodas5P(),
    )
    dense = final_states(alg, mass_matrices.dense)
    @test all(u -> isapprox(u[1:nd], exact; rtol = 1.0f-4), dense)
    # Only the diagonal of W changes, in the same arithmetic as the dense update.
    @test final_states(alg, mass_matrices.diagonal) == dense
    @test final_states(alg, mass_matrices.static_diagonal) == dense
end
