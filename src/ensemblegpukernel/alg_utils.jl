# `A + α * M`, the matrix the stiff solvers factorize. For a diagonal mass matrix only the
# diagonal of `A` changes, so it is updated in one pass over `A`: the generic
# `StaticMatrix + Diagonal` reads all `N²` entries of the `Diagonal`, which for large `N`
# compiles far slower than the dense case (minutes at `N = 124`).
@inline add_mass_matrix(A, M, α) = A + α * M
@inline function add_mass_matrix(
        A::StaticArrays.SMatrix{N, N, T}, M::LinearAlgebra.Diagonal{<:Any, <:StaticArrays.SVector{N}},
        α
    ) where {N, T}
    d = M.diag
    return StaticArrays.SMatrix{N, N, T}(
        ntuple(Val(N * N)) do k
            i = (k - 1) % N + 1
            i == (k - 1) ÷ N + 1 ? A[k] + α * d[i] : A[k]
        end
    )
end

function alg_order(alg::Union{GPUODEAlgorithm, GPUSDEAlgorithm})
    error("Order is not defined for this algorithm")
end

alg_order(alg::GPUTsit5) = 5
alg_order(alg::GPUTsit5IController) = 5
alg_order(alg::GPUVern7) = 7
alg_order(alg::GPUVern9) = 9
alg_order(alg::GPURosenbrock23) = 2
alg_order(alg::GPURodas4) = 4
alg_order(alg::GPURodas5P) = 5
alg_order(alg::GPUKvaerno3) = 3
alg_order(alg::GPUKvaerno5) = 5

alg_order(alg::GPUEM) = 1
alg_order(alg::GPUSIEA) = 2

function finite_diff_jac(f, jac_prototype, x)
    dx = sqrt(eps(RecursiveArrayTools.recursive_bottom_eltype(x)))
    jac = MMatrix{size(x, 1), size(x, 1), eltype(x)}(1I)
    for i in eachindex(x)
        x_dx = convert(MArray, x)
        x_dx[i] = x_dx[i] + dx
        x_dx = convert(SArray, x_dx)
        jac[:, i] .= (f(x_dx) - f(x)) / dx
    end
    return convert(SMatrix, jac)
end

function alg_autodiff(alg::GPUODEAlgorithm)
    error("This algorithm does not have an autodifferentiation option defined.")
end

alg_autodiff(::GPUODEImplicitAlgorithm{AD}) where {AD} = AD
