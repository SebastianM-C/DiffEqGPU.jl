@inline function step!(integ::GPUVern7I{false, S, T}, ts, us) where {T, S}
    @unpack dt = integ
    t = integ.t
    p = integ.p
    @unpack c2, c3, c4, c5, c6, c7, c8, a021, a031, a032, a041, a043, a051, a053, a054,
    a061, a063, a064, a065, a071, a073, a074, a075, a076, a081, a083, a084,
    a085, a086, a087, a091, a093, a094, a095, a096, a097, a098, a101, a103,
    a104, a105, a106, a107, b1, b4, b5, b6, b7, b8, b9 = integ.tab

    tmp = integ.tmp
    f = integ.f
    integ.uprev = integ.u
    uprev = integ.u

    integ.tprev = t
    saved_in_cb = false
    adv_integ = true
    ## Check if tstops are within the range of time-series
    if integ.tstops !== nothing && integ.tstops_idx <= length(integ.tstops) &&
       (integ.tstops[integ.tstops_idx] - integ.t - integ.dt - 100 * eps(integ.t) < 0)
        integ.t = integ.tstops[integ.tstops_idx]
        integ.tstops_idx += 1
    else
        ##Advance the integrator
        integ.t += dt
    end

    if integ.u_modified
        k1 = f(uprev, p, t)
        integ.u_modified = false
    else
        @inbounds k1 = integ.k10
    end

    k1 = f(uprev, p, t)
    a = dt * a021
    k2 = f(uprev + a * k1, p, t + c2 * dt)
    k3 = f(uprev + dt * (a031 * k1 + a032 * k2), p, t + c3 * dt)
    k4 = f(uprev + dt * (a041 * k1 + a043 * k3), p, t + c4 * dt)
    k5 = f(uprev + dt * (a051 * k1 + a053 * k3 + a054 * k4), p, t + c5 * dt)
    k6 = f(uprev + dt * (a061 * k1 + a063 * k3 + a064 * k4 + a065 * k5), p, t + c6 * dt)
    k7 = f(uprev + dt * (a071 * k1 + a073 * k3 + a074 * k4 + a075 * k5 + a076 * k6), p,
           t + c7 * dt)
    k8 = f(uprev +
           dt * (a081 * k1 + a083 * k3 + a084 * k4 + a085 * k5 + a086 * k6 + a087 * k7), p,
           t + c8 * dt)
    g9 = uprev +
         dt *
         (a091 * k1 + a093 * k3 + a094 * k4 + a095 * k5 + a096 * k6 + a097 * k7 + a098 * k8)
    g10 = uprev +
          dt * (a101 * k1 + a103 * k3 + a104 * k4 + a105 * k5 + a106 * k6 + a107 * k7)
    k9 = f(g9, p, t + dt)
    k10 = f(g10, p, t + dt)

    integ.u = uprev +
              dt * (b1 * k1 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7 + b8 * k8 + b9 * k9)

    @inbounds begin # Necessary for interpolation
        integ.k1 = k1
        integ.k2 = k2
        integ.k3 = k3
        integ.k4 = k4
        integ.k5 = k5
        integ.k6 = k6
        integ.k7 = k7
        integ.k8 = k8
        integ.k9 = k9
        integ.k10 = k10
    end

    if integ.cb !== nothing
        _, saved_in_cb = apply_discrete_callback!(integ, ts, us,
                                                  integ.cb.discrete_callbacks...)
    else
        saved_in_cb = false
    end

    return saved_in_cb
end

function vern7_kernel(probs, _us, _ts, dt, callback, tstops, nsteps,
                      saveat, ::Val{save_everystep}) where {save_everystep}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(probs) && return

    # get the actual problem for this thread
    prob = @inbounds probs[i]

    # get the input/output arrays for this thread
    ts = @inbounds view(_ts, :, i)
    us = @inbounds view(_us, :, i)

    integ = gpuvern7_init(prob.f, false, prob.u0, prob.tspan[1], dt, prob.p, tstops,
                          callback, save_everystep)

    u0 = prob.u0
    tspan = prob.tspan

    cur_t = 0
    if saveat !== nothing
        cur_t = 1
        if prob.tspan[1] == ts[1]
            cur_t += 1
            @inbounds us[1] = u0
        end
    else
        @inbounds ts[integ.step_idx] = prob.tspan[1]
        @inbounds us[integ.step_idx] = prob.u0
    end

    integ.step_idx += 1
    # FSAL
    while integ.t < tspan[2]
        saved_in_cb = step!(integ, ts, us)
        if saveat === nothing && save_everystep && !saved_in_cb
            @inbounds us[integ.step_idx] = integ.u
            @inbounds ts[integ.step_idx] = integ.t
            integ.step_idx += 1
        elseif saveat !== nothing
            # while cur_t <= length(saveat) && saveat[cur_t] <= integ.t
            #     savet = saveat[cur_t]
            #     θ = (savet - integ.tprev) / integ.dt
            #     b1θ, b2θ, b3θ, b4θ, b5θ, b6θ, b7θ = SimpleDiffEq.bθs(integ.rs, θ)
            #     @inbounds us[cur_t] = integ.uprev +
            #                           integ.dt *
            #                           (b1θ * integ.k1 + b2θ * integ.k2 + b3θ * integ.k3 +
            #                            b4θ * integ.k4 + b5θ * integ.k5 + b6θ * integ.k6 +
            #                            b7θ * integ.k7)
            #     @inbounds ts[cur_t] = savet
            #     cur_t += 1
            # end
        end
    end

    if saveat === nothing && !save_everystep
        @inbounds us[2] = integ.u
        @inbounds ts[2] = integ.t
    end

    return nothing
end

#############################Adaptive Version#####################################

@inline function step!(integ::GPUAVern7I{false, S, T}, ts, us) where {S, T}
    beta1, beta2, qmax, qmin, gamma, qoldinit, _ = build_adaptive_tsit5_controller_cache(eltype(integ.u))
    dt = integ.dtnew
    t = integ.t
    p = integ.p
    tf = integ.tf

    @unpack c2, c3, c4, c5, c6, c7, c8, a021, a031, a032, a041, a043, a051, a053, a054,
    a061, a063, a064, a065, a071, a073, a074, a075, a076, a081, a083, a084,
    a085, a086, a087, a091, a093, a094, a095, a096, a097, a098, a101, a103,
    a104, a105, a106, a107, b1, b4, b5, b6, b7, b8, b9, btilde1, btilde4,
    btilde5, btilde6, btilde7, btilde8, btilde9, btilde10, extra, interp = integ.tab

    tmp = integ.tmp
    f = integ.f
    integ.uprev = integ.u
    uprev = integ.u

    qold = integ.qold
    abstol = integ.abstol
    reltol = integ.reltol

    if integ.u_modified
        k1 = f(uprev, p, t)
        integ.u_modified = false
    else
        @inbounds k1 = integ.k10
    end

    EEst = Inf

    while EEst > 1.0
        dt < 1e-14 && error("dt<dtmin")

        k1 = f(uprev, p, t)
        a = dt * a021
        k2 = f(uprev + a * k1, p, t + c2 * dt)
        k3 = f(uprev + dt * (a031 * k1 + a032 * k2), p, t + c3 * dt)
        k4 = f(uprev + dt * (a041 * k1 + a043 * k3), p, t + c4 * dt)
        k5 = f(uprev + dt * (a051 * k1 + a053 * k3 + a054 * k4), p, t + c5 * dt)
        k6 = f(uprev + dt * (a061 * k1 + a063 * k3 + a064 * k4 + a065 * k5), p, t + c6 * dt)
        k7 = f(uprev + dt * (a071 * k1 + a073 * k3 + a074 * k4 + a075 * k5 + a076 * k6), p,
               t + c7 * dt)
        k8 = f(uprev +
               dt * (a081 * k1 + a083 * k3 + a084 * k4 + a085 * k5 + a086 * k6 + a087 * k7),
               p,
               t + c8 * dt)
        g9 = uprev +
             dt *
             (a091 * k1 + a093 * k3 + a094 * k4 + a095 * k5 + a096 * k6 + a097 * k7 +
              a098 * k8)
        g10 = uprev +
              dt * (a101 * k1 + a103 * k3 + a104 * k4 + a105 * k5 + a106 * k6 + a107 * k7)
        k9 = f(g9, p, t + dt)
        k10 = f(g10, p, t + dt)

        u = uprev +
            dt * (b1 * k1 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7 + b8 * k8 + b9 * k9)

        tmp = dt *
              (btilde1 * k1 + btilde4 * k4 + btilde5 * k5 + btilde6 * k6 + btilde7 * k7 +
               btilde8 * k8 + btilde9 * k9 + btilde10 * k10)

        tmp = tmp ./ (abstol .+ max.(abs.(uprev), abs.(u)) * reltol)
        EEst = DiffEqBase.ODE_DEFAULT_NORM(tmp, t)

        if iszero(EEst)
            q = inv(qmax)
        else
            q11 = EEst^beta1
            q = q11 / (qold^beta2)
        end

        if EEst > 1
            dt = dt / min(inv(qmin), q11 / gamma)
        else # EEst <= 1
            q = max(inv(qmax), min(inv(qmin), q / gamma))
            qold = max(EEst, qoldinit)
            dtnew = dt / q #dtnew
            dtnew = min(abs(dtnew), abs(tf - t - dt))

            @inbounds begin # Necessary for interpolation
                integ.k1 = k1
                integ.k2 = k2
                integ.k3 = k3
                integ.k4 = k4
                integ.k5 = k5
                integ.k6 = k6
                integ.k7 = k7
                integ.k8 = k8
                integ.k9 = k9
                integ.k10 = k10
            end

            integ.dt = dt
            integ.dtnew = dtnew
            integ.qold = qold
            integ.tprev = t
            integ.u = u

            if (tf - t - dt) < 1e-14
                integ.t = tf
            else
                if integ.tstops !== nothing && integ.tstops_idx <= length(integ.tstops) &&
                   integ.tstops[integ.tstops_idx] - integ.t - integ.dt -
                   100 * eps(integ.t) < 0
                    integ.t = integ.tstops[integ.tstops_idx]
                    integ.tstops_idx += 1
                else
                    ##Advance the integrator
                    integ.t += dt
                end
            end
        end
    end
    if integ.cb !== nothing
        _, saved_in_cb = DiffEqBase.apply_discrete_callback!(integ,
                                                             integ.cb.discrete_callbacks...)
    else
        saved_in_cb = false
    end
    return nothing
end

function avern7_kernel(probs, _us, _ts, dt, callback, tstops, abstol, reltol,
                       saveat, ::Val{save_everystep}) where {save_everystep}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(probs) && return

    # get the actual problem for this thread
    prob = @inbounds probs[i]
    # get the input/output arrays for this thread
    ts = @inbounds view(_ts, :, i)
    us = @inbounds view(_us, :, i)
    # TODO: optimize contiguous view to return a CuDeviceArray

    u0 = prob.u0
    tspan = prob.tspan
    f = prob.f
    p = prob.p

    t = tspan[1]
    tf = prob.tspan[2]

    cur_t = 0
    if saveat !== nothing
        cur_t = 1
        if tspan[1] == ts[1]
            cur_t += 1
            @inbounds us[1] = u0
        end
    else
        @inbounds ts[1] = tspan[1]
        @inbounds us[1] = u0
    end

    integ = gpuavern7_init(prob.f, false, prob.u0, prob.tspan[1], prob.tspan[2], dt, prob.p,
                           abstol, reltol, DiffEqBase.ODE_DEFAULT_NORM, tstops, callback)

    while integ.t < tspan[2]
        step!(integ, ts, us)
        if saveat === nothing && save_everystep
            error("Do not use saveat == nothing & save_everystep = true in adaptive version")
        elseif saveat !== nothing
            # while cur_t <= length(saveat) && saveat[cur_t] <= integ.t
            #     savet = saveat[cur_t]
            #     θ = (savet - integ.tprev) / integ.dt
            #     b1θ, b2θ, b3θ, b4θ, b5θ, b6θ, b7θ = SimpleDiffEq.bθs(integ.rs, θ)
            #     @inbounds us[cur_t] = integ.uprev +
            #                           integ.dt *
            #                           (b1θ * integ.k1 + b2θ * integ.k2 + b3θ * integ.k3 +
            #                            b4θ * integ.k4 + b5θ * integ.k5 + b6θ * integ.k6 +
            #                            b7θ * integ.k7)
            #     @inbounds ts[cur_t] = savet
            #     cur_t += 1
            # end
        end
    end

    if saveat === nothing && !save_everystep
        @inbounds us[2] = integ.u
        @inbounds ts[2] = integ.t
    end

    return nothing
end
