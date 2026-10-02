module CUDAExt
using CUDA: CUDA, CUDABackend
import DiffEqGPU

function DiffEqGPU.EnsembleGPUArray(cpu_offload::Float64; kwargs...)
    return DiffEqGPU.EnsembleGPUArray(CUDABackend(), cpu_offload; kwargs...)
end
DiffEqGPU.maxthreads(::CUDABackend) = 256
DiffEqGPU.maybe_prefer_blocks(::CUDABackend) = CUDABackend(; prefer_blocks = true)

function DiffEqGPU.lufact!(::CUDABackend, W)
    CUDA.CUBLAS.getrf_strided_batched!(W, false)
    return nothing
end

function DiffEqGPU.lufact!(::CUDABackend, W, ipiv)
    CUDA.CUBLAS.getrf_strided_batched!(W, ipiv)
    return nothing
end

end
