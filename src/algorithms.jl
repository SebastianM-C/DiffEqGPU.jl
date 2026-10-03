"""
    EnsembleCPUArray()

An `EnsembleArrayAlgorithm` which utilizes the CPU kernels to parallelize each ODE solve
with their separate ODE integrator on each kernel. This method is meant to be a debugging
counterpart to `EnsembleGPUArray`, having the same behavior and using the same
KernelAbstractions.jl process to build the combined ODE, but without the restrictions of
`f` being a GPU-compatible kernel function.

It is unlikely that this method is useful beyond library development and debugging, as
almost any case should be faster with `EnsembleThreads` or `EnsembleDistributed`.

# Returns

An `EnsembleCPUArray` algorithm selector for the CPU-backed array ensemble path.

# Examples

```julia
ensemblealg = EnsembleCPUArray()
solve(ensemble_prob, Tsit5(), ensemblealg; trajectories = 100)
```
"""
struct EnsembleCPUArray <: EnsembleArrayAlgorithm end

"""
    EnsembleGPUArray(backend, cpu_offload = 0.2; threaded_host = false, per_trajectory_dt = false,
        pivot_threshold = 1.0e-7)

An `EnsembleArrayAlgorithm` that uses one kernel per trajectory while storing the
trajectories in a batched array. This is the appropriate choice when the ODE solver or
right-hand side cannot be compiled into one fused `EnsembleGPUKernel` solve.

# Fields

  - `backend`: the `KernelAbstractions` backend used for the batched computation.
  - `cpu_offload`: the fraction of trajectories solved on the CPU. The default is `0.2`.
  - `threaded_host`: whether the host-side preparation of each batch runs on several threads.
  - `per_trajectory_dt`: whether every trajectory takes its own adaptive steps.
  - `pivot_threshold`: the pivot check of the per-trajectory sparse LU.

# Arguments

  - `backend`: a `KernelAbstractions` backend, such as `CUDA.CUDABackend()` or
    `KernelAbstractions.CPU()`.
  - `cpu_offload`: the fraction of trajectories to offload to CPU execution. The
    two-argument constructor stores this value as a `Float64`; the one-argument constructor
    defaults to `0.2`.

# Keyword Arguments

  - `threaded_host`: when `true`, the `prob_func` calls and the host-side initialization of
    the trajectories of a batch run in `Threads.@threads`, one task per trajectory. This
    requires `prob_func` to be thread-safe: it must not mutate shared state, and with
    `safetycopy = false` it must return a problem that does not share mutable state with
    the other trajectories (`remake` gives a new problem). Defaults to `false`, which
    prepares the trajectories in order on the calling thread.
  - `per_trajectory_dt`: when `true`, every trajectory keeps its own time, step size and
    step acceptance, instead of the whole batch advancing with one shared step; see
    "Per-trajectory steps" below. Supports `Rodas5P` only. Defaults to `false`.
  - `pivot_threshold`: with `per_trajectory_dt` and a sparse `jac_prototype`, a trajectory
    whose fixed-order factorization keeps less than this fraction of a pivot
    (`min_k |U[k, k]| / |W[k, k]|`, see `DiffEqGPU.lane_pivot_status`) switches to a pivoted
    dense LU. The dense LU costs a step about the same whether one trajectory or many use it,
    so a model whose fixed order stays accurate at smaller ratios can lower the threshold;
    `0` switches only on a NaN. Defaults to `1.0e-7`.

# Returns

An `EnsembleGPUArray` algorithm selector.

# Throws

The constructor itself does not validate backend support. Unsupported right-hand sides,
callbacks, or solver combinations throw when the ensemble is solved.

# Limitations

`EnsembleGPUArray` requires being able to generate a kernel for `f` using
KernelAbstractions.jl and solving the resulting ODE defined over `CuArray` input types.
This introduces the following limitations on its usage:

  - Not all standard Julia `f` functions are allowed. Only Julia `f` functions which are
    capable of being compiled into a GPU kernel are allowed. This notably means that
    certain features of Julia can cause issues inside of kernel, like:

      + Allocating memory (building arrays)
      + Linear algebra (anything that calls BLAS)
      + Broadcast

  - Not all ODE solvers are allowed, only those from OrdinaryDiffEq.jl. The tested feature
    set from OrdinaryDiffEq.jl includes:

      + Explicit Runge-Kutta methods
      + Implicit Runge-Kutta methods
      + Rosenbrock methods
      + DiscreteCallbacks and ContinuousCallbacks
  - Stiff methods use the Jacobian `f.jac` when the problem provides one. Otherwise the
    per-trajectory Jacobians come from forward-mode automatic differentiation of the
    batched right-hand side, which must then accept `ForwardDiff.Dual` states. This takes
    `cld(N, c)` evaluations of the right-hand side per Jacobian, for `N` states and a
    chunk size `c` of at most 8, and preallocates `2 N ntraj (c + 1)` numbers. The time
    derivative uses `f.tgrad` when given, else OrdinaryDiffEq's own differentiation of the
    batched right-hand side. Differentiating through the ensemble solve (the `rrule`)
    still requires `f.jac`. The linear systems are solved with a batched factorization,
    so a `linsolve` other than [`DiffEqGPU.LinSolveGPUSplitFactorize`](@ref) throws an
    `ArgumentError`.
  - Mass matrices must be diagonal: a `UniformScaling`, a `Diagonal`, or a matrix whose
    off-diagonal entries are zero. Other mass matrices throw an `ArgumentError`. All
    trajectories use the mass matrix of the first one.
  - A singular mass matrix (a DAE with algebraic variables) is supported with Rosenbrock
    methods, such as `Rosenbrock23` and `Rodas5P`, and throws an `ArgumentError` with other
    algorithms. Each trajectory is initialized on the host (see Initialization below); a
    problem without initialization data must have a `u0` that satisfies the algebraic
    equations, which `initializealg = CheckInit()` verifies per trajectory.
  - The per-trajectory iteration matrices are factorized with partial pivoting, so DAEs whose
    algebraic equations have a zero on the diagonal of their Jacobian are supported.
  - To use multiple GPUs over clusters, one must manually set up one process per GPU. See
    the multi-GPU tutorial for more details.

# Failures

All trajectories share one integrator, so the solve of a batch stops when any trajectory
fails, for example when its step size collapses. The trajectory that made the shared step
fail (the one with the largest error, or with a non-finite state) is returned with the
batch's return code, such as `ReturnCode.Unstable`; the other trajectories are returned with
`ReturnCode.Failure`, with solutions that are valid up to the time the batch stopped. With a
user `internalnorm`, which replaces the per-trajectory norm that identifies the failing
trajectory, every trajectory is returned with the batch's return code.

# Callbacks

Callbacks can be given on the problem or as the `callback` keyword of `solve`. As in
`solve` for a single problem, the keyword is merged with the problem's callback unless
`merge_callbacks = false`, in which case it replaces it.

Conditions and affects run once per trajectory inside a kernel, on a stand-in integrator
that holds the trajectory's `u`, `t` and `p`. An affect may modify `u` and, when the
parameters are an array per trajectory, `p`; each returned solution's `prob.p` holds the
trajectory's parameters at the end of the solve.

All trajectories share one integrator, so:

  - an event in any trajectory shortens the step of every trajectory;
  - `save_positions` saves the state of every trajectory at every event;
  - a `ContinuousCallback` direction without an affect (`affect! = nothing` or
    `affect_neg! = nothing`) is still located, and the affect is skipped;
  - callbacks with a custom `initialize` or `finalize`, and callbacks combined with
    trajectories that have different time spans, throw an `ArgumentError`.

A `PeriodicCallback` from DiffEqCallbacks.jl is supported (load DiffEqCallbacks): its stops
are shared by all trajectories, so the number of stops does not grow with the number of
trajectories, and its affect runs per trajectory at every stop. Its `phase`,
`initial_affect` and `final_affect` are honored; a custom `initialize` or `finalize` throws
an `ArgumentError`.

# Initialization

Each trajectory is initialized on the host, before the batch is stacked, as the CPU solve
of that trajectory would be initialized:

  - A problem with initialization data (such as a ModelingToolkit problem, including one
    made with `remake(prob; ..., lazy_initialization = true)`) is initialized with
    `OverrideInit`. The returned solution is built from the initialized problem, so its `u0`
    and parameters, including parameters solved for during initialization, are the
    initialized ones.
  - The `initializealg` keyword of `solve` selects the algorithm: `OverrideInit` (whose
    `nlsolve`, `abstol` and `reltol` are used), `CheckInit`, which checks the algebraic
    equations of a mass-matrix problem, or `NoInit`. Other algorithms throw an
    `ArgumentError`. The nonlinear solver defaults to `SimpleTrustRegion` (or
    `SimpleGaussNewton` for a least-squares initialization problem), and the tolerances to
    the `abstol` and `reltol` of the solve.
  - A trajectory whose initialization fails is left out of the batch and returned with
    `ReturnCode.InitialFailure`; its solution holds only the initial time and state. The
    other trajectories are solved normally.

The batched problem itself is solved without initialization.

# ModelingToolkit parameters

Problems whose parameters are ModelingToolkit `MTKParameters` are batched by portion. The
tunable and discrete portions are stored per trajectory; the constant portion is uploaded
once and shared by all trajectories, so it must be the same in every trajectory (an
`ArgumentError` names the portion otherwise; make the parameters a sweep varies tunable,
for example with `ModelingToolkit.subset_tunables`). The nonnumeric and caches portions must
be empty. Floating-point buffers take the floating-point type of the state. The initials
portion is not uploaded: only initialization reads it. Each returned solution carries the
discrete values its trajectory ended with in `sol.prob.p`, so callbacks that write discretes
are visible after the solve; the timeseries of discretes is not saved.

# ModelingToolkit events

The events of a ModelingToolkit system run on `EnsembleGPUArray` when the problem is built
with [`DiffEqGPU.gpu_affect_transform`](@ref):

```julia
prob = ODEProblem(sys, u0, tspan; affect_transform = DiffEqGPU.gpu_affect_transform,
    save_discretes = false)
```

The transform keeps ModelingToolkit's callbacks and their timing, and makes each affect run
once per trajectory inside a kernel. Supported affects are `ImperativeAffect`s that write
unknowns and discrete parameters; periodic discrete events (a `SymbolicDiscreteCallback`
with a period) suit the batch best, since their stops are shared by all trajectories. Each
returned solution's `prob.ps` holds the discrete values its trajectory ended with;
`save_discretes = false` turns off saving their timeseries, which would read the batched
parameters. Affects given as equations, writes to other parameters, custom `initialize` or
`finalize` affects, and, for a system with algebraic equations, events whose
`reinitializealg` is not `NoInit()` throw an `ArgumentError`, as does solving a problem
whose affects were built without the transform.

!!! warn

    Callbacks with `terminate!` do not work well with `EnsembleGPUArray` because the entire
    integration halts when any trajectory halts. Use with caution.

# Step-size control

All trajectories of a batch advance with one shared step. The error estimate is measured
per trajectory (the RMS over its own components) and the largest of these decides
acceptance and the next step, so every trajectory meets `abstol`/`reltol` as it would in a
solve of its own. The shared step is therefore the one the hardest trajectory needs: a batch
mixing easy and hard trajectories takes as many steps as the hard ones. Pass
`internalnorm = DiffEqGPU.ComponentNorm(keep)` to measure each trajectory's error over the
components `keep` only (see [`DiffEqGPU.ComponentNorm`](@ref)); any other `internalnorm`
replaces this norm and receives the batched state array.

# Per-trajectory steps

With `per_trajectory_dt = true`, DiffEqGPU integrates the batch with its own `Rodas5P`
stepper instead of OrdinaryDiffEq's integrator. Every trajectory has its own time, step
size, step-size controller and step acceptance; the batch still shares the kernel launches,
and a step attempt is computed for all unfinished trajectories at once. Each trajectory
then takes the steps a solve of its own would take, with OrdinaryDiffEq's `Rodas5P`
defaults (the PI controller, the initial step size), so a batch of easy and hard
trajectories no longer advances at the pace of the hardest one. Because the shared step is
set by the hardest trajectory, it also makes the other trajectories more accurate than
their tolerances ask for; per-trajectory steps give each trajectory the accuracy of its own
solve, so compare the two at matched achieved error, not at equal tolerance.

A trajectory that fails (`ReturnCode.Unstable`, `DtLessThanMin`, `MaxIters`) stops alone; the
others continue. Supported: `Rodas5P` with `autodiff = AutoFiniteDiff()` (forward or central
differences), diagonal mass matrices including singular ones, scalar `abstol` and `reltol`,
`saveat` (or `save_everystep = false`), `save_start`, `save_end`, `dt`, `dtmax`, `tstops`,
`maxiters`, `internalnorm = DiffEqGPU.ComponentNorm(keep)`, a sparse `jac_prototype` (see
below), and DiffEqCallbacks' `PeriodicCallback`s, including those of ModelingToolkit's
periodic events, with `save_positions = (false, false)`. Other callbacks and options throw an
`ArgumentError`. Saved values between two steps are interpolated with `Rodas5P`'s dense
output; at a time where a periodic affect fires, the value before the affect is saved, as
OrdinaryDiffEq does with `save_positions = (false, false)`.

The step-size control is OrdinaryDiffEq's, including how it continues after a step was
shortened to land on a stop (a `tstops` entry or a periodic callback's time): the next step
grows from the shortened step, as in the current OrdinaryDiffEq release.

With a `SparseMatrixCSC` `jac_prototype` (the pattern of the Jacobian of one trajectory), the
stepper stores each trajectory's Jacobian as the values of that pattern, computes it with one
right-hand side evaluation per color of a column coloring instead of one per state, and
factorizes the iteration matrices with a sparse LU. The pivot order of that LU is chosen
once, from the iteration matrices of the first step of all trajectories (a maximum-product
row matching and a minimum-degree ordering), and used without pivoting for every
factorization of the solve; this works when the large entries stay where they are, as in
models whose algebraic equations keep their structure, and makes the factorizations and
solves a fraction of dense LU's. The pattern must contain every entry the right-hand side can
make nonzero: with the coloring, a missing entry also corrupts other stored entries. Each
trajectory's factorization and solves run on a group of threads. A trajectory whose fixed pivot
order loses a pivot to cancellation during the elimination (below `pivot_threshold`)
switches to a pivoted dense LU, the factorization a solve of that trajectory on its own uses,
for the rest of its solve; it stops with `ReturnCode.InternalLinearSolveFailed` only if that is
singular too.

# Examples

```julia
using DiffEqGPU, CUDA, OrdinaryDiffEq
function lorenz(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
    return
end

u0 = Float32[1.0; 0.0; 0.0]
tspan = (0.0f0, 100.0f0)
p = [10.0f0, 28.0f0, 8 / 3.0f0]
prob = ODEProblem(lorenz, u0, tspan, p)
prob_func = (prob, ctx) -> remake(prob, p = rand(Float32, 3) .* p)
monteprob = EnsembleProblem(prob; prob_func, safetycopy = false)
@time sol = solve(
    monteprob, Tsit5(), EnsembleGPUArray(CUDADevice()),
    trajectories = 10_000, saveat = 1.0f0
)
```
"""
struct EnsembleGPUArray{Backend} <: EnsembleArrayAlgorithm
    backend::Backend
    cpu_offload::Float64
    threaded_host::Bool
    per_trajectory_dt::Bool
    pivot_threshold::Float64
end

function EnsembleGPUArray(
        backend, cpu_offload; threaded_host::Bool = false, per_trajectory_dt::Bool = false,
        pivot_threshold::Real = LANE_PIVOT_THRESHOLD
    )
    0 <= pivot_threshold < Inf || throw(
        ArgumentError("`pivot_threshold` must be finite and nonnegative; got $pivot_threshold.")
    )
    return EnsembleGPUArray(
        backend, Float64(cpu_offload), threaded_host, per_trajectory_dt, Float64(pivot_threshold)
    )
end

"""
    EnsembleGPUKernel(backend, cpu_offload = 0.0)

A massively parallel ensemble algorithm that generates one GPU kernel for the complete
fixed-step ODE or SDE solve. It minimizes kernel-launch overhead, at the cost of stricter
requirements on the problem and solver.

# Fields

  - `dev`: the `KernelAbstractions` backend used to launch the fused kernel.
  - `cpu_offload`: the fraction of trajectories solved on the CPU. The default is `0.0`.

# Arguments

  - `backend`: a `KernelAbstractions` backend, such as `CUDA.CUDABackend()` or
    `KernelAbstractions.CPU()`.
  - `cpu_offload`: the fraction of trajectories to offload to CPU execution. The
    two-argument constructor stores this value as a `Float64`; the one-argument constructor
    defaults to `0.0`.

# Returns

An `EnsembleGPUKernel` algorithm selector.

# Throws

The constructor itself does not validate kernel compatibility. Unsupported state containers,
right-hand sides, callbacks, or solver combinations throw when the ensemble is solved.

# Limitations

  - Not all standard Julia `f` functions are allowed. Only Julia `f` functions which are
    capable of being compiled into a GPU kernel are allowed. This notably means that
    certain features of Julia can cause issues inside a kernel, like:

    + Allocating memory (building arrays)
    + Linear algebra (anything that calls BLAS)
    + Broadcast

  - Only out-of-place `f` definitions are allowed. Coupled with the requirement of not
    allowing for memory allocations, this means that the ODE must be defined with
    `StaticArray` initial conditions.
  - Only specific ODE solvers are allowed. This includes:

    + GPUTsit5
    + GPUVern7
    + GPUVern9
  - To use multiple GPUs over clusters, one must manually set up one process per GPU. See
    the multi-GPU tutorial for more details.

# Examples

```julia
using DiffEqGPU, CUDA, OrdinaryDiffEq, StaticArrays

function lorenz(u, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]
    du1 = σ * (u[2] - u[1])
    du2 = u[1] * (ρ - u[3]) - u[2]
    du3 = u[1] * u[2] - β * u[3]
    return SVector{3}(du1, du2, du3)
end

u0 = @SVector [1.0f0; 0.0f0; 0.0f0]
tspan = (0.0f0, 10.0f0)
p = @SVector [10.0f0, 28.0f0, 8 / 3.0f0]
prob = ODEProblem{false}(lorenz, u0, tspan, p)
prob_func = (prob, ctx) -> remake(prob, p = (@SVector rand(Float32, 3)) .* p)
monteprob = EnsembleProblem(prob; prob_func, safetycopy = false)

@time sol = solve(
    monteprob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend()), trajectories = 10_000,
    adaptive = false, dt = 0.1f0
)
```
"""
struct EnsembleGPUKernel{Dev} <: EnsembleKernelAlgorithm
    dev::Dev
    cpu_offload::Float64
end

cpu_alg = Dict(
    GPUTsit5 => (GPUSimpleTsit5(), GPUSimpleATsit5()),
    GPUVern7 => (GPUSimpleVern7(), GPUSimpleAVern7()),
    GPUVern9 => (GPUSimpleVern9(), GPUSimpleAVern9())
)

# Work around the fact that Zygote cannot handle the task system
# Work around the fact that Zygote isderiving fails with constants?
function EnsembleGPUArray(dev; kwargs...)
    return EnsembleGPUArray(dev, 0.2; kwargs...)
end

function EnsembleGPUKernel(dev)
    return EnsembleGPUKernel(dev, 0.0)
end

function ChainRulesCore.rrule(::Type{<:EnsembleGPUArray})
    return EnsembleGPUArray(0.0), _ -> NoTangent()
end

ZygoteRules.@adjoint function EnsembleGPUArray(dev)
    EnsembleGPUArray(dev, 0.0), _ -> nothing
end
