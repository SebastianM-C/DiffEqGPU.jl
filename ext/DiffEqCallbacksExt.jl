module DiffEqCallbacksExt

using DiffEqGPU: DiffEqGPU
using DiffEqCallbacks: DiffEqCallbacks, PeriodicCallback, PeriodicCallbackAffect
using SciMLBase: SciMLBase, DiscreteCallback

# A `PeriodicCallback` schedules its stops with `add_tstop!` and fires when the integrator's
# time reaches the next one, so on the batched integrator it fires for every trajectory at
# the same times. It is rebuilt around the per-trajectory affect: its initialization may
# apply the affect itself (`initial_affect = true`), so the original closures, which hold
# the affect, cannot be reused.
function DiffEqGPU.batched_time_callback(callback::DiscreteCallback, ensemblealg)
    periodic = callback.affect!
    periodic isa PeriodicCallbackAffect || return nothing
    DiffEqGPU.check_device_affect(periodic.affect!, ensemblealg)
    init, condition = callback.initialize, callback.condition
    if !(
            all(n -> hasfield(typeof(init), n), (:phase, :initial_affect, :initialize)) &&
                hasfield(typeof(condition), :final_affect)
        )
        throw(
            ArgumentError(
                "$(nameof(typeof(ensemblealg))) cannot read the settings of this `PeriodicCallback`; its DiffEqCallbacks version is not supported."
            )
        )
    end
    user_initialize = getfield(init, :initialize)
    # Without an `initialize` keyword, `PeriodicCallback` uses its own default.
    default_initialize = parentmodule(typeof(user_initialize)) === DiffEqCallbacks
    if !(default_initialize || user_initialize === SciMLBase.INITIALIZE_DEFAULT) ||
            callback.finalize !== SciMLBase.FINALIZE_DEFAULT
        throw(
            ArgumentError(
                "$(nameof(typeof(ensemblealg))) does not support a `PeriodicCallback` with a custom `initialize` or `finalize`: they would run once on the batched integrator instead of once per trajectory."
            )
        )
    end
    initialize = default_initialize ? (;) : (; initialize = user_initialize)
    return PeriodicCallback(
        DiffEqGPU.batched_affect(periodic.affect!), periodic.Δt;
        phase = getfield(init, :phase),
        initial_affect = getfield(init, :initial_affect),
        final_affect = getfield(condition, :final_affect),
        save_positions = callback.save_positions,
        initializealg = callback.initializealg,
        initialize...
    )
end

end
