module ModelingToolkitBaseExt

using ModelingToolkitBase: ModelingToolkitBase, BlockedArray, MTKParameters, System,
    unknowns
using Adapt: adapt
import DiffEqGPU
import Adapt
using StaticArraysCore: SVector
import SciMLBase

const SciMLStructures = ModelingToolkitBase.SciMLStructures
const BlockArrays = parentmodule(BlockedArray)

const MTKPARAMETERS_PORTIONS = (
    :tunable, :initials, :discrete, :constant, :nonnumeric, :caches,
)

## Batched `MTKParameters` for `EnsembleGPUArray`

# A vector of vectors of different lengths, stored as one flat buffer so it can live on the
# device: element `i` is `data[(offsets[i] + 1):offsets[i + 1]]`, returned as a view.
struct RaggedVector{V, D, O} <: AbstractVector{V}
    data::D
    offsets::O
end
# The element type is computed rather than taken from a `view`, because the constructor also
# runs inside kernels (KernelAbstractions rebuilds `@Const` arguments there), where a view's
# bounds check cannot be compiled.
function RaggedVector(data::D, offsets::O) where {D, O}
    V = SubArray{eltype(D), 1, D, Tuple{UnitRange{Int}}, true}
    return RaggedVector{V, D, O}(data, offsets)
end
function RaggedVector(::Type{T}, vs::AbstractVector{<:AbstractVector}) where {T}
    offsets = zeros(Int, length(vs) + 1)
    for (i, v) in enumerate(vs)
        offsets[i + 1] = offsets[i] + length(v)
    end
    data = Vector{T}(undef, offsets[end])
    for (i, v) in enumerate(vs)
        copyto!(data, offsets[i] + 1, v, 1, length(v))
    end
    return RaggedVector(data, offsets)
end
Base.size(r::RaggedVector) = (length(r.offsets) - 1,)
Base.@propagate_inbounds function Base.getindex(r::RaggedVector, i::Int)
    return view(r.data, (r.offsets[i] + 1):r.offsets[i + 1])
end
function Adapt.adapt_structure(to, r::RaggedVector)
    return RaggedVector(Adapt.adapt(to, r.data), Adapt.adapt(to, r.offsets))
end

# The `MTKParameters` of a batch of trajectories. The tunable and discrete portions are
# stored per trajectory, as one column each; everything else is a single copy shared by all
# trajectories. Indexing with a trajectory number gives that trajectory's `MTKParameters`,
# whose tunable and discrete buffers are views into its columns, so a callback writing a
# discrete writes the batched storage.
struct BatchedMTKParameters{T, D, A, I, S} <: AbstractVector{Any}
    # ntunable × ntraj
    tunable::T
    # One ndiscrete × ntraj matrix per discrete partition (clock)
    discrete::D
    # The block axes of each discrete partition, or `nothing` for an unblocked one
    discrete_axes::A
    # An empty buffer standing in for the initials, which only initialization reads
    initials::I
    # (constant, nonnumeric, caches)
    shared::S
end
Base.size(b::BatchedMTKParameters) = (size(b.tunable, 2),)
Base.@propagate_inbounds function Base.getindex(b::BatchedMTKParameters, i::Int)
    discrete = map((d, ax) -> discrete_column(d, ax, i), b.discrete, b.discrete_axes)
    constant, nonnumeric, caches = b.shared
    return MTKParameters(
        view(b.tunable, :, i), b.initials, discrete, constant, nonnumeric, caches
    )
end
Base.@propagate_inbounds discrete_column(d, ::Nothing, i) = view(d, :, i)
Base.@propagate_inbounds discrete_column(d, axes, i) = BlockedArray(view(d, :, i), axes)

# The block axes are static, so they need no adapting.
function Adapt.adapt_structure(to, b::BatchedMTKParameters)
    return BatchedMTKParameters(
        Adapt.adapt(to, b.tunable), Adapt.adapt(to, b.discrete), b.discrete_axes,
        Adapt.adapt(to, b.initials), Adapt.adapt(to, b.shared)
    )
end

# Floating-point buffers take the floating-point type of the batched state; integer and
# `Bool` buffers keep theirs.
convert_float(::Type{T}, x::AbstractArray{<:AbstractFloat}) where {T} = T.(x)
convert_float(::Type, x::AbstractArray) = Array(x)

# One column per trajectory. `stack` keeps a single trajectory, and an empty portion, a matrix.
batch_columns(::Type{T}, cols) where {T} = stack(convert_float(T, Array(c)) for c in cols)

static_axis(ax) = isbits(ax) ? ax : BlockArrays.BlockedOneTo(SVector{length(ax.lasts)}(ax.lasts))
static_axes(x::BlockedArray) = map(static_axis, axes(x))
static_axes(x) = nothing

function shared_buffer(::Type{T}, portion, x::AbstractVector{<:Number}) where {T}
    return convert_float(T, x isa BitArray ? Vector{Bool}(x) : x)
end
function shared_buffer(::Type{T}, portion, x::AbstractVector{<:AbstractVector{<:Number}}) where {T}
    S = isempty(x) || eltype(eltype(x)) <: AbstractFloat ? T : eltype(eltype(x))
    return RaggedVector(S, x)
end
function shared_buffer(::Type, portion, x)
    throw(
        ArgumentError(
            "EnsembleGPUArray cannot upload a `$(typeof(x))` buffer from the $portion portion of `MTKParameters` to the device. Only vectors of numbers and vectors of numeric vectors are supported there."
        )
    )
end

# A portion holding no buffers, or only empty ones, has nothing for the device.
function empty_portion(portion, x::Tuple)
    all(isempty, x) && return ()
    throw(
        ArgumentError(
            "EnsembleGPUArray does not support `MTKParameters` with a non-empty $portion portion: it holds $(join(map(b -> string(typeof(b)), filter(!isempty, collect(x))), ", ")). Keep such values out of the parameters, for example by making them constants of a numeric type."
        )
    )
end

# Everything except the tunable and discrete portions is uploaded once, from the first
# trajectory, so it has to be the same in every trajectory. The initials are exempt: they
# can differ after per-trajectory initialization, and only initialization reads them.
function check_shared_portions(ps)
    p1 = first(ps)
    for portion in (:constant, :nonnumeric, :caches)
        ref = getproperty(p1, portion)
        i = findfirst(ps) do p
            x = getproperty(p, portion)
            !(x === ref || isequal(x, ref))
        end
        i === nothing && continue
        throw(
            ArgumentError(
                "EnsembleGPUArray batches only the tunable and discrete portions of `MTKParameters`, and shares the others between all trajectories, but the $portion portion of trajectory $i differs from that of the first trajectory. Make the parameters that vary between trajectories tunable (for example with `ModelingToolkit.subset_tunables`), or solve trajectories with different $portion values in separate ensembles."
            )
        )
    end
    return nothing
end

function DiffEqGPU.pack_parameters(p1::MTKParameters, probs, ::Type{T}) where {T}
    ps = map(prob -> prob.p, probs)
    check_shared_portions(ps)
    tunable = batch_columns(T, map(p -> p.tunable, ps))
    discrete = ntuple(k -> batch_columns(T, map(p -> p.discrete[k], ps)), length(p1.discrete))
    discrete_axes = map(static_axes, p1.discrete)
    constant = map(x -> shared_buffer(T, :constant, x), p1.constant)
    nonnumeric = empty_portion(:nonnumeric, p1.nonnumeric)
    caches = empty_portion(:caches, p1.caches)
    return BatchedMTKParameters(
        tunable, discrete, discrete_axes, T[], (constant, nonnumeric, caches)
    )
end

# The discrete portion is the one callbacks change. Read it back once per batch and give each
# trajectory whose discretes changed new parameters holding the final values.
function DiffEqGPU.final_parameters(b::BatchedMTKParameters, probs)
    isempty(b.discrete) && return map(prob -> prob.p, probs)
    discrete = map(Array, b.discrete)
    return map(eachindex(probs)) do i
        p = probs[i].p
        initial = collect(SciMLStructures.canonicalize(SciMLStructures.Discrete(), p)[1])
        final = reduce(vcat, map(d -> d[:, i], discrete))
        # Compare in the batch's precision, so the rounding of the upload is no change
        final == oftype(final, initial) && return p
        return SciMLStructures.replace(
            SciMLStructures.Discrete(), p, convert(Vector{eltype(initial)}, final)
        )
    end
end

function DiffEqGPU.make_parameter_compatible(p::MTKParameters)
    compatible = MTKParameters(
        DiffEqGPU.make_static_storage(p.tunable),
        DiffEqGPU.make_static_storage(p.initials),
        # Callback-updated discretes live in a `BlockedArray` (one block per clock
        # partition). `make_static_storage` does not know that type, so let its own Adapt
        # rule rebuild it around static data.
        adapt(DiffEqGPU.StaticAdaptor(), p.discrete),
        DiffEqGPU.make_static_storage(p.constant),
        DiffEqGPU.make_static_storage(p.nonnumeric),
        DiffEqGPU.make_static_storage(p.caches)
    )
    # Converting the storage to `SArray` cannot rescue content that is not isbits in the
    # first place — a nonnumeric buffer holding a type or a closure over an array, say.
    # Such a `p` cannot be a field of the isbits problem uploaded to the device, so say so
    # here rather than failing somewhere inside the kernel.
    isbits(compatible) && return compatible
    offenders = filter(MTKPARAMETERS_PORTIONS) do portion
        !isbits(getproperty(compatible, portion))
    end
    throw(
        ArgumentError(
            "These `MTKParameters` cannot be used by EnsembleGPUKernel: the $(join(offenders, ", ", " and ")) $(length(offenders) == 1 ? "portion holds" : "portions hold") values that are not isbits, so the problem cannot be uploaded to the device. Either keep such values out of the parameter set, or give their type an isbits stand-in by adding a `DiffEqGPU.make_static_storage` method for it."
        )
    )
end

function DiffEqGPU.lower_initialization_problem(prob::SciMLBase.SCCNonlinearProblem)
    sys = prob.f.sys
    sys isa System || throw(
        ArgumentError(
            "Only ModelingToolkit-generated SCC nonlinear initialization problems can be lowered for EnsembleGPUKernel."
        )
    )
    any(p -> nameof(typeof(p)) === :HomotopyProblem, prob.probs) && throw(
        ArgumentError(
            "SCC nonlinear initialization problems containing homotopy blocks are not supported by EnsembleGPUKernel. Recompile the system with `mtkcompile(sys; homotopy = false)` to replace every `homotopy(actual, simplified)` node by `actual`, which builds an equivalent initialization without homotopy blocks."
        )
    )

    block_states = map(prob.probs) do block_prob
        block_u0 = SciMLBase.state_values(block_prob)
        block_u0 !== nothing && return block_u0
        # A linear block is solved by one exact Newton step, which lands on the solution
        # from any seed, so a zero seed stands in for the missing state.
        block_prob isa SciMLBase.LinearProblem || throw(
            ArgumentError(
                "Every nonlinear SCC initialization block must have an initial state for EnsembleGPUKernel."
            )
        )
        zero(block_prob.b)
    end
    u0 = reduce(vcat, block_states)
    length(u0) == length(unknowns(sys)) || throw(
        ArgumentError("SCC initialization block sizes do not match the full state size.")
    )
    f = SciMLBase.NonlinearFunction{false, SciMLBase.FullSpecialize}(
        sys; u0, p = prob.p, check_compatibility = false
    )
    nonlinear_prob = SciMLBase.NonlinearProblem{false}(f, u0, prob.p)

    offset = 0
    blocks = map(prob.probs, block_states) do block_prob, block_u0
        n = length(block_u0)
        block = DiffEqGPU.ImmutableSCCBlock{
            offset + 1, n, block_prob isa SciMLBase.LinearProblem,
        }()
        offset += n
        block
    end
    return DiffEqGPU.ImmutableSCCNonlinearProblem(nonlinear_prob, Tuple(blocks))
end

# ModelingToolkit emits the initialization maps as isbits `RuntimeGeneratedFunction`s
# under `SciMLBase.FullSpecialize` (ModelingToolkit.jl#5043). Those rebuild `u0` and `p`
# in the buffer types of whatever value provider they are handed, so against the static
# storage `make_prob_compatible` installs they produce static results and run unchanged
# inside the kernel. Nothing is lowered here.
device_compatible_map(::Nothing) = true
device_compatible_map(map) = isbitstype(typeof(map))

# The maps read `state_values`/`parameter_values` off whatever they are handed. Inside the
# kernel that is the nonlinear solution; here it is the lowered initialization problem, and
# the SCC wrapper is not itself a value provider.
map_value_provider(prob) = prob
map_value_provider(prob::DiffEqGPU.ImmutableSCCNonlinearProblem) = prob.problem

function DiffEqGPU.make_initialization_maps_compatible(
        prob, initprob, umap, pmap, ::MTKParameters
    )
    device_compatible_map(umap) && device_compatible_map(pmap) || throw(
        ArgumentError(
            "EnsembleGPUKernel requires ModelingToolkit's device-compatible initialization maps, which are emitted only under `SciMLBase.FullSpecialize`. Rebuild the problem as `ODEProblem{iip, SciMLBase.FullSpecialize}(sys, ...)`; ModelingToolkit ignores a `specialize` keyword argument."
        )
    )
    # The maps are evaluated inside the kernel, which cannot allocate, so the `u0` they
    # build has to be static too. ModelingToolkit fixes that container at
    # code-generation time from `u0_constructor`, so it cannot be repaired here.
    umap === nothing || isbits(umap(map_value_provider(initprob))) || throw(
        ArgumentError(
            "ModelingToolkit's state initialization map builds a `$(typeof(umap(map_value_provider(initprob))))`, which a kernel can neither allocate nor hold. It has to build a static, immutable `u0` instead, and that is decided when the problem is constructed. Build the problem out-of-place and with static storage: `ODEProblem{false, SciMLBase.FullSpecialize}(sys, ...; u0_constructor = static_constructor, p_constructor = static_constructor)` where `static_constructor(values) = SVector{length(values)}(values)`. Out-of-place is required as well as static: an `MVector` is a mutable struct and so is not isbits, while an in-place problem cannot write into an immutable `SVector`."
        )
    )
    return umap, pmap
end


## Events: running ModelingToolkit affects per trajectory

const SII = ModelingToolkitBase.SymbolicIndexingInterface

# The per-trajectory stand-in for the integrator that `EnsembleGPUArray` kernels call affects
# with: its `u` is the trajectory's column of the batched state and its `p` the trajectory's
# parameters (a column of a `BatchedMTKParameters`).
SII.state_values(integrator::DiffEqGPU.FakeIntegrator) = integrator.u
SII.parameter_values(integrator::DiffEqGPU.FakeIntegrator) = integrator.p
SII.current_time(integrator::DiffEqGPU.FakeIntegrator) = integrator.t
# A compiled `ImperativeAffect` ends with `reset_jumps && reset_aggregated_jumps!(integ)`;
# `gpu_affect_transform` refuses affects that reset jumps, but the call is still compiled for
# the stand-in, and the generic method reads fields it does not have.
ModelingToolkitBase.JumpProcesses.reset_aggregated_jumps!(
    ::DiffEqGPU.FakeIntegrator, uprev = nothing; kwargs...
) = nothing

const LOWERING_HINT = "Build the problem with `affect_transform = DiffEqGPU.gpu_affect_transform, save_discretes = false` so that its events run on `EnsembleGPUArray`."

function DiffEqGPU.gpu_affect_transform(
        affect, event::ModelingToolkitBase.AbstractCallback, sys; role
    )
    if role === :initialize || role === :finalize
        affect === nothing || affect === SciMLBase.INITIALIZE_DEFAULT ||
            affect === SciMLBase.FINALIZE_DEFAULT ||
            refuse_event("an event with a custom `$role`, which would run once on the batched integrator instead of once per trajectory")
        return affect
    end
    affect === ModelingToolkitBase.EMPTY_AFFECT && return affect
    affect isa ModelingToolkitBase.FunctionalAffect ||
        refuse_event("an affect of type `$(nameof(typeof(affect)))`; only affects given as an `ImperativeAffect` run per trajectory on the device, so write an affect given as equations as an `ImperativeAffect`")
    parts = ModelingToolkitBase.functional_affect_parts(affect)
    parts.reset_jumps && refuse_event("an affect that resets jump aggregators")
    isbits(parts.user_affect) ||
        refuse_event("an `ImperativeAffect` whose function captures non-isbits data")
    isbits(parts.ctx) || refuse_event("an `ImperativeAffect` with a non-isbits context")
    foreach(pairs(parts.setters)) do (name, setter)
        check_setter(name, setter)
    end
    reinit = event.reinitializealg
    if !(reinit isa SciMLBase.NoInit) && ModelingToolkitBase.has_alg_equations(sys)
        refuse_event("an event of a system with algebraic equations whose `reinitializealg` is $(nameof(typeof(reinit)))(); reinitializing would run on the whole batch, so use `reinitializealg = SciMLBase.NoInit()`")
    end
    return DiffEqGPU.GPUArrayAffect(ModelingToolkitBase.without_parameter_hooks(affect))
end

function refuse_event(what)
    throw(ArgumentError("`EnsembleGPUArray` does not support $what."))
end

# An affect may write the trajectory's unknowns, which are its column of the batched state,
# and its discrete parameters, which are stored per trajectory. The other parameters are
# shared by all trajectories or fixed for the solve.
check_setter(name, ::SII.SetStateIndex) = nothing
function check_setter(name, setter::SII.SetParameterIndex)
    index = setter.idx
    index isa ModelingToolkitBase.ParameterIndex &&
        index.portion isa SciMLStructures.Discrete && return nothing
    portion = index isa ModelingToolkitBase.ParameterIndex ? nameof(typeof(index.portion)) :
        nameof(typeof(index))
    return refuse_event("an affect writing `$name`, a $portion parameter; affects may write unknowns and discrete parameters only")
end
function check_setter(name, setter)
    return refuse_event("an affect writing `$name` through a `$(nameof(typeof(setter)))`; affects may write unknowns and discrete parameters only")
end

# Affects compiled without `gpu_affect_transform` would call parameter hooks, or solve
# equations, on the per-trajectory stand-in for the integrator, which supports neither.
function DiffEqGPU.check_device_affect(
        affect::Union{ModelingToolkitBase.FunctionalAffect, ModelingToolkitBase.ImplicitAffect},
        ensemblealg
    )
    return throw(
        ArgumentError(
            "This ModelingToolkit affect cannot run on $(nameof(typeof(ensemblealg))) as it is. $LOWERING_HINT"
        )
    )
end

end
