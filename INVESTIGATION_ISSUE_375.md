# Investigation Report: Issue #375 - EnsembleGPUKernel fails with MTK generated problems

## Issue Summary

When combining ModelingToolkit (MTK) code generation with GPU ensemble solving via `EnsembleGPUKernel`, the system throws an inline allocation error:

```
CuArray only supports element types that are allocated inline.
```

## Root Cause Analysis

### The Problem

The failure occurs at `src/solve.jl:280` in the `batch_solve_up_kernel` function:

```julia
probs = adapt(dev, adapt.((dev,), probs))
```

This line attempts to transfer an array of `ImmutableODEProblem` objects to GPU memory. The issue is that MTK-generated problems contain complex nested type structures that are not compatible with GPU memory requirements.

### Why MTK Problems Fail

CUDA (and other GPU backends) require that array element types be "allocated inline" - meaning they must be `isbits` types with known, fixed sizes at compile time. MTK-generated ODE problems contain:

1. **RuntimeGeneratedFunctions** - Functions generated at runtime
2. **GeneratedFunctionWrapper** - Wrappers around generated code
3. **ComposedFunction** - Multiple layers of function composition
4. **Closures** - Functions that capture variables from their environment

These types cannot be transferred to GPU memory because:
- They contain pointers and references
- They have dynamic/unknown sizes
- They may contain heap-allocated data
- They are not `isbits` types

### Code Flow

1. User creates an `ODEProblem` from MTK `System` with code generation
2. `EnsembleProblem` wraps it with parameter variations via `prob_func`
3. `solve()` calls `batch_solve()` → `batch_solve_up_kernel()`
4. Problems are converted to `ImmutableODEProblem` via `make_prob_compatible()`
5. **FAILS HERE**: `adapt(dev, ...)` tries to transfer to GPU
6. CUDA rejects the complex nested type structure

## Current Architecture

### How EnsembleGPUKernel Works

```
src/solve.jl:batch_solve_up_kernel()
├── Convert problems to ImmutableODEProblem
├── adapt(dev, adapt.((dev,), probs))  ← FAILS HERE
├── vectorized_solve() or vectorized_asolve()
└── ode_solve_kernel() / ode_asolve_kernel()
    └── GPU kernels process each problem in parallel
```

### Current Adapt Strategy

The only custom `Adapt.adapt_structure` rule is for `ParamWrapper` in `src/ensemblegpuarray/kernels.jl:10`:

```julia
function Adapt.adapt_structure(to, ps::ParamWrapper{P, T}) where {P, T}
    ParamWrapper(adapt(to, ps.params), adapt(to, ps.data))
end
```

There are no custom adapt rules for `ImmutableODEProblem` or MTK-generated function types.

## Potential Solutions

### Option 1: Extract Function and Adapt Components Separately (Recommended)

Instead of adapting the entire problem, extract the function and only adapt the data:

```julia
# In batch_solve_up_kernel(), before GPU transfer:
function adapt_prob_for_gpu(dev, prob::ImmutableODEProblem)
    # Keep function on host, only transfer data
    gpu_u0 = adapt(dev, prob.u0)
    gpu_p = adapt(dev, prob.p)
    gpu_tspan = adapt(dev, prob.tspan)
    # Return a GPU-compatible structure
    return (f = prob.f, u0 = gpu_u0, p = gpu_p, tspan = gpu_tspan, kwargs = prob.kwargs)
end
```

**Pros:**
- Minimal changes to existing code
- Functions stay on CPU, only data goes to GPU
- Works with any function type

**Cons:**
- Requires kernel modification to accept this structure
- May have performance implications

### Option 2: Add Adapt Rules for ImmutableODEProblem

Define custom adaptation logic:

```julia
function Adapt.adapt_structure(to, prob::ImmutableODEProblem{iip, spec, F, U, T, P, K}) where {iip, spec, F, U, T, P, K}
    # Only adapt the data fields, not the function
    ImmutableODEProblem{iip, spec, F, typeof(adapt(to, prob.u0)), T, typeof(adapt(to, prob.p)), K}(
        prob.f,  # Keep function as-is
        adapt(to, prob.u0),
        adapt(to, prob.tspan),
        adapt(to, prob.p),
        prob.kwargs
    )
end
```

**Pros:**
- Leverages existing Adapt.jl infrastructure
- Minimal changes to calling code

**Cons:**
- Functions still need to be GPU-compatible
- Won't solve MTK's complex function issue directly

### Option 3: Function Serialization/Reconstruction

Serialize MTK functions to a GPU-compatible format:

1. Extract the mathematical operations from MTK generated code
2. Convert to a simple callable structure
3. Reconstruct on GPU side

**Pros:**
- Most robust long-term solution
- Could enable full GPU compilation

**Cons:**
- Complex implementation
- Requires deep MTK integration
- May not be feasible for all function types

### Option 4: Workaround - Use EnsembleGPUArray Instead

For MTK problems, fallback to `EnsembleGPUArray` which doesn't serialize the entire problem:

```julia
# Detect MTK problems and use different strategy
if is_mtk_generated(prob)
    # Use EnsembleGPUArray approach
    sol = solve(monteprob, Tsit5(), EnsembleGPUArray(backend), trajectories = n)
else
    # Use EnsembleGPUKernel
    sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(backend), trajectories = n)
end
```

**Pros:**
- Quick workaround
- No deep changes needed

**Cons:**
- Performance may be worse
- Doesn't solve the underlying issue

## Recommended Implementation Plan

### Phase 1: Immediate Fix (Workaround)

1. Add detection for MTK-generated problems
2. Document that MTK problems should use `EnsembleGPUArray`
3. Provide clear error message when incompatible

### Phase 2: Proper Solution

1. Implement Option 1 or 2 above
2. Modify kernel code to handle separated function/data
3. Add comprehensive tests with MTK problems
4. Update documentation

### Phase 3: Upstream Coordination

1. Work with MTK team on GPU-compatible code generation
2. Explore function serialization options
3. Consider specialized MTK-GPU integration package

## Testing Strategy

### Test Cases Needed

1. **Basic MTK problem** - Simple ODE from MTK
2. **MTK with code generation** - Both `split=true` and `split=false`
3. **Parameter variations** - Ensure `prob_func` works correctly
4. **Different systems** - Test various MTK system types
5. **Comparison** - Verify results match CPU execution

### Minimal Test Example

```julia
using DiffEqGPU, OrdinaryDiffEq, ModelingToolkit, StaticArrays, CUDA

@parameters σ ρ β
@variables x(t) y(t) z(t)

eqs = [D(D(x)) ~ σ * (y - x),
       D(y) ~ x * (ρ - z) - y,
       D(z) ~ x * y - β * z]

@mtkcompile sys = System(eqs, t) split=false

u0 = SA[D(x) => 2f0, x => 1f0, y => 0f0, z => 0f0]
p = SA[σ => 28f0, ρ => 10f0, β => 8f0 / 3f0]

prob = ODEProblem{false, SciMLBase.FullSpecialize}(sys, [u0; p], (0f0, 100f0))

# This should work after fix
prob_func(prob, i, repeat) = remake(prob, p = rand(Float32, 3))
monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)
sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend()), trajectories = 3)
```

## Additional Considerations

### Performance Implications

- Keeping functions on CPU may require kernel to make host calls
- This could negate GPU performance benefits
- Need benchmarking to quantify impact

### Compatibility

- Solution must work across all GPU backends (CUDA, AMDGPU, Metal, oneAPI)
- Adapt.jl is backend-agnostic, so should be compatible

### Documentation Needs

- Clear guidance on MTK + GPU ensemble usage
- Examples showing working patterns
- Performance expectations

## Files Modified (for implementation)

- `src/solve.jl` - Modify `batch_solve_up_kernel()`
- `src/utils.jl` - Add GPU adaptation helpers
- `src/ensemblegpukernel/kernels.jl` - Update kernel signatures if needed
- `test/gpu_kernel_de/` - Add MTK compatibility tests
- `docs/` - Add MTK examples and documentation

## Related Issues/PRs

- This is a known limitation of GPU kernel approaches
- Similar issues may exist in other SciML GPU packages
- Coordinate with MTK team on solutions

## Conclusion

The issue stems from a fundamental incompatibility between MTK's runtime code generation and GPU memory requirements. A proper fix requires either:

1. Modifying how problems are transferred to GPU (separate function from data)
2. Changing MTK's code generation to produce GPU-compatible functions
3. Using a different ensemble algorithm for MTK problems

The recommended approach is to implement Option 1 as a short-term fix while working with the MTK team on a long-term solution.
