# `initialize` and `finalize` would run once on the batched integrator, whose `u` and `p`
# hold every trajectory, rather than once per trajectory. Refuse them instead of running
# them with the wrong meaning.
function check_callback_hooks(callback, ensemblealg)
    if callback.initialize !== SciMLBase.INITIALIZE_DEFAULT ||
            callback.finalize !== SciMLBase.FINALIZE_DEFAULT
        throw(
            ArgumentError(
                "$(nameof(typeof(ensemblealg))) does not support callbacks with a custom `initialize` or `finalize`: they would run once on the batched integrator instead of once per trajectory."
            )
        )
    end
    return nothing
end

"""
    DiffEqGPU.gpu_affect_transform(affect, event, sys; role)

The `affect_transform` that lets the events of a ModelingToolkit system run on
`EnsembleGPUArray`. Pass it, together with `save_discretes = false`, when building the
problem:

```julia
prob = ODEProblem(sys, u0, tspan; affect_transform = DiffEqGPU.gpu_affect_transform,
    save_discretes = false)
```

ModelingToolkit calls it on every compiled affect of the system's events. It returns each
affect of an `ImperativeAffect` as a [`DiffEqGPU.GPUArrayAffect`](@ref) that runs once per
trajectory inside a kernel, and leaves the callbacks' timing, which ModelingToolkit builds,
unchanged. The transformed problem also solves on the CPU, with the same results.

It throws an `ArgumentError` for an affect that cannot run per trajectory on the device:
an affect given as equations (an `AffectSystem`), one that writes anything but unknowns and
discrete parameters, one with a non-isbits function or context, one that resets jump
aggregators, a custom `initialize` or `finalize`, and, for a system with algebraic
equations, an event whose `reinitializealg` is not `NoInit()`.

Requires ModelingToolkit (its `ModelingToolkitBase`) to be loaded.
"""
function gpu_affect_transform(affect, event, sys; role)
    throw(
        ArgumentError(
            "`DiffEqGPU.gpu_affect_transform` transforms the events of a ModelingToolkit system; load ModelingToolkit."
        )
    )
end

"""
    DiffEqGPU.GPUArrayAffect(affect)

An affect that [`DiffEqGPU.gpu_affect_transform`](@ref) has checked to run once per
trajectory inside an `EnsembleGPUArray` kernel. Calling it calls `affect` with the
integrator, or with the per-trajectory stand-in for it inside a kernel.
"""
struct GPUArrayAffect{F}
    affect::F
end

(a::GPUArrayAffect)(integrator) = a.affect(integrator)

# Throws if `affect` cannot run per trajectory inside an `EnsembleGPUArray` kernel, telling
# how to make it run there; extensions add methods for the affects their packages compile.
check_device_affect(affect, ensemblealg) = nothing

# A discrete callback whose condition and scheduling read only the time and the tstops of
# the integrator, such as DiffEqCallbacks' `PeriodicCallback`, can keep that machinery on the
# batched integrator, because all trajectories share one time span: it fires for every
# trajectory at once, and only its user affect must run per trajectory, as
# `batched_affect` does. Extensions add methods returning the rebuilt callback; `nothing`
# means `callback` is not of such a kind.
batched_time_callback(callback, ensemblealg) = nothing

# `affect!(integrator)` applied to every trajectory of the batched integrator.
function batched_affect(affect!)
    return function (integrator)
        version = get_backend(integrator.u)
        wgs = workgroupsize(version, size(integrator.u, 2))
        all_affect!_kernel(version)(
            affect!, integrator.u, integrator.t, integrator.p;
            ndrange = size(integrator.u, 2),
            workgroupsize = wgs
        )
        return nothing
    end
end

function generate_callback(callback::ContinuousCallback, I, ensemblealg)
    if ensemblealg isa EnsembleGPUKernel
        return callback
    end
    check_callback_hooks(callback, ensemblealg)
    check_device_affect(callback.affect!, ensemblealg)
    check_device_affect(callback.affect_neg!, ensemblealg)
    _condition = callback.condition
    _affect! = callback.affect!
    _affect_neg! = callback.affect_neg!

    condition = function (out, u, t, integrator)
        version = get_backend(u)
        wgs = workgroupsize(version, size(u, 2))
        continuous_condition_kernel(version)(
            _condition, out, u, t, integrator.p;
            ndrange = size(u, 2),
            workgroupsize = wgs
        )
        return nothing
    end

    affect! = function (integrator, simultaneous_events::AbstractVector)
        version = get_backend(integrator.u)
        wgs = workgroupsize(version, size(integrator.u, 2))
        # DiffEqBase passes a `@view` of its host mask buffer. GPU backends only have a
        # memcpy path for `Array` sources, so materialize the view to avoid scalar indexing.
        host_events = convert(Vector{eltype(simultaneous_events)}, simultaneous_events)
        simultaneous_events_device = similar(
            integrator.u, eltype(host_events), length(host_events)
        )
        copyto!(simultaneous_events_device, host_events)
        return continuous_affect!_kernel(version)(
            _affect!, _affect_neg!, simultaneous_events_device, integrator.u,
            integrator.t, integrator.p;
            ndrange = size(integrator.u, 2),
            workgroupsize = wgs
        )
    end

    # `idxs` refers to the components of one trajectory, so it is not forwarded: the batched
    # condition reads the whole state.
    return VectorContinuousCallback(
        condition, affect!, I;
        save_positions = callback.save_positions,
        rootfind = callback.rootfind,
        interp_points = callback.interp_points,
        dtrelax = callback.dtrelax,
        abstol = callback.abstol,
        reltol = callback.reltol,
        repeat_nudge = callback.repeat_nudge,
        initializealg = callback.initializealg
    )
end

function generate_callback(callback::CallbackSet, I, ensemblealg)
    return CallbackSet(
        map(
            cb -> generate_callback(cb, I, ensemblealg),
            (
                callback.continuous_callbacks...,
                callback.discrete_callbacks...,
            )
        )...
    )
end

generate_callback(::Tuple{}, I, ensemblealg) = nothing
generate_callback(::Nothing, I, ensemblealg) = nothing

# Without this method a `VectorContinuousCallback` falls through to the method below that
# expects a problem, and fails with an unrelated `FieldError`.
function generate_callback(::VectorContinuousCallback, I, ensemblealg)
    throw(
        ArgumentError(
            "`VectorContinuousCallback` is not supported by $(nameof(typeof(ensemblealg))). Pass its conditions as separate `ContinuousCallback`s in a `CallbackSet` instead."
        )
    )
end

# The problem's callback and the `callback` keyword, with the semantics of
# `DiffEqBase.merge_problem_kwargs`: a `callback` keyword is merged with the problem's
# callback when `merge_callbacks = true` (the default) and replaces it otherwise.
function ensemble_callbacks(prob; kwargs...)
    prob_cb = get(prob.kwargs, :callback, nothing)
    kwarg_cb = get(kwargs, :callback, nothing)
    if haskey(kwargs, :callback) && !get(kwargs, :merge_callbacks, true)
        prob_cb = nothing
    end
    return prob_cb, kwarg_cb
end

isempty_callback(cb) = cb === nothing || isempty(cb)

function has_ensemble_callbacks(prob; kwargs...)
    prob_cb, kwarg_cb = ensemble_callbacks(prob; kwargs...)
    return !isempty_callback(prob_cb) || !isempty_callback(kwarg_cb)
end

function generate_callback(prob, I, ensemblealg; kwargs...)
    prob_cb, kwarg_cb = ensemble_callbacks(prob; kwargs...)
    if isempty_callback(prob_cb) && isempty_callback(kwarg_cb)
        return nothing
    else
        return CallbackSet(
            generate_callback(prob_cb, I, ensemblealg),
            generate_callback(kwarg_cb, I, ensemblealg)
        )
    end
end
