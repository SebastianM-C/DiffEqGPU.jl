module JLArraysExt
using JLArrays: JLBackend
import DiffEqGPU

DiffEqGPU.maxthreads(::JLBackend) = 256
DiffEqGPU.maybe_prefer_blocks(::JLBackend) = JLBackend()
# JLArrays runs kernels with KernelAbstractions' CPU execution, which does not support
# barriers inside loops.
DiffEqGPU.cooperative_ldiv(::JLBackend) = false

end
