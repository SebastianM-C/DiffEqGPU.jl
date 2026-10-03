# Per-trajectory adaptive steps for `EnsembleGPUArray(...; per_trajectory_dt = true)`.
#
# The batch is integrated by DiffEqGPU's own Rodas5P stepper instead of OrdinaryDiffEq's
# integrator. Every trajectory ("lane") has its own time, step size, controller state and
# step acceptance; a step attempt is computed for all unfinished lanes in the same kernel
# launches. The state of lane `i` is column `i` of the `N × B` arrays, and its scalars are
# entry `i` of length-`B` vectors. The stepper follows OrdinaryDiffEq's `Rodas5P` with its
# default `PIController`, so a lane takes the steps a solve of its own would take.

_per_trajectory_dt(::EnsembleArrayAlgorithm) = false
_per_trajectory_dt(ensemblealg::EnsembleGPUArray) = ensemblealg.per_trajectory_dt

"""
    LanePeriodic(affect!, Δt, phase, initial_affect, final_affect, save_positions)

A DiffEqCallbacks `PeriodicCallback` as the per-trajectory stepper runs it: the stops
`t0 + phase + k Δt` before the end of the time span are shared by all trajectories, and
`affect!` runs per trajectory, in a kernel, when that trajectory steps onto a stop.
"""
struct LanePeriodic{A, T}
    affect!::A
    Δt::T
    phase::T
    initial_affect::Bool
    final_affect::Bool
    save_positions::NTuple{2, Bool}
end
function LanePeriodic(affect!, Δt, phase, initial_affect, final_affect, save_positions)
    Δt, phase = promote(Δt, phase)
    return LanePeriodic{typeof(affect!), typeof(Δt)}(
        affect!, Δt, phase, initial_affect, final_affect, save_positions
    )
end

# Lane status. Every lane starts `LANE_ACTIVE`; the others are final.
const LANE_ACTIVE = Int8(0)
const LANE_SUCCESS = Int8(1)
const LANE_UNSTABLE = Int8(2)
const LANE_DTMIN = Int8(3)
const LANE_MAXITERS = Int8(4)
const LANE_PIVOT = Int8(5)

function lane_retcode(status)
    status == LANE_SUCCESS && return ReturnCode.Success
    status == LANE_UNSTABLE && return ReturnCode.Unstable
    status == LANE_DTMIN && return ReturnCode.DtLessThanMin
    status == LANE_MAXITERS && return ReturnCode.MaxIters
    status == LANE_PIVOT && return ReturnCode.InternalLinearSolveFailed
    return ReturnCode.Failure
end

# The Rodas5P tableau in the form OrdinaryDiffEq's `RosenbrockCache` uses: stage `s` is
# evaluated at `uprev + Σ A[s, j] k_j` and time `t + c[s] dt`, its right-hand side adds
# `dt d[s] ∂f/∂t + M Σ C[s, j] k_j / dt`, the step is `uprev + Σ b_j k_j`, the error
# estimate is `k_8`, and the dense output uses the rows of `H`.
struct LaneRodasTableau{T}
    A::SMatrix{8, 8, T, 64}
    C::SMatrix{8, 8, T, 64}
    b::SVector{8, T}
    c::SVector{8, T}
    d::SVector{8, T}
    H::SMatrix{3, 8, T, 24}
    gamma::T
end

function LaneRodasTableau(::Type{T}) where {T}
    g = Rodas5PTableau(T, T)
    z = zero(T)
    o = one(T)
    a6 = (g.a61, g.a62, g.a63, g.a64, g.a65)
    A = SMatrix{8, 8, T}(
        permutedims(
            T[
                z z z z z z z z
                g.a21 z z z z z z z
                g.a31 g.a32 z z z z z z
                g.a41 g.a42 g.a43 z z z z z
                g.a51 g.a52 g.a53 g.a54 z z z z
                a6... z z z
                a6... o z z
                a6... o o z
            ]
        )'
    )
    C = SMatrix{8, 8, T}(
        permutedims(
            T[
                z z z z z z z z
                g.C21 z z z z z z z
                g.C31 g.C32 z z z z z z
                g.C41 g.C42 g.C43 z z z z z
                g.C51 g.C52 g.C53 g.C54 z z z z
                g.C61 g.C62 g.C63 g.C64 g.C65 z z z
                g.C71 g.C72 g.C73 g.C74 g.C75 g.C76 z z
                g.C81 g.C82 g.C83 g.C84 g.C85 g.C86 g.C87 z
            ]
        )'
    )
    b = SVector{8, T}(a6..., o, o, o)
    c = SVector{8, T}(z, g.c2, g.c3, g.c4, g.c5, o, o, o)
    d = SVector{8, T}(g.d1, g.d2, g.d3, g.d4, g.d5, z, z, z)
    H = SMatrix{3, 8, T}(
        permutedims(
            T[
                g.h21 g.h22 g.h23 g.h24 g.h25 g.h26 g.h27 g.h28
                g.h31 g.h32 g.h33 g.h34 g.h35 g.h36 g.h37 g.h38
                g.h41 g.h42 g.h43 g.h44 g.h45 g.h46 g.h47 g.h48
            ]
        )'
    )
    return LaneRodasTableau{T}(A, C, b, c, d, H, T(g.γ))
end

"""
    LaneControllerOptions

The settings of OrdinaryDiffEq's default `PIController` for `Rodas5P` (order 5): the gains
`beta1 = 7 / 50` and `beta2 = 2 / 25`, the step-ratio bounds `qmin = 1 / 5` and `qmax = 10`
(`qmax_first_step = 10000` before the first accepted step), the safety factor
`gamma = 9 / 10`, the deadband `qsteady_min = 1`, `qsteady_max = 6 / 5` (that of an
adaptive implicit algorithm, which a Rosenbrock method is) and `qoldinit = 1e-4`.
"""
struct LaneControllerOptions{T}
    beta1::T
    beta2::T
    qmin::T
    qmax::T
    qmax_first_step::T
    gamma::T
    qsteady_min::T
    qsteady_max::T
    qoldinit::T
end

function LaneControllerOptions(::Type{T}; order = 5) where {T}
    return LaneControllerOptions{T}(
        T(7 // (10order)), T(2 // (5order)), T(1 // 5), T(10), T(10000), T(9 // 10),
        T(1), T(6 // 5), T(1 // 10^4)
    )
end

"""
    lane_pi_controller(EEst, errold, dt, dtpropose, first_step, opts)

One decision of OrdinaryDiffEq's `PIController` for one trajectory, after a step attempt
of size `dt` whose error estimate is `EEst` (finite; `EEst <= 1` means the step is within
tolerance). Returns `(accept, dtnew, errold_new)`.

  - `errold`: the error estimate of the last accepted step (`opts.qoldinit` before the first).
  - `dtpropose`: the step an accepted attempt continues from. OrdinaryDiffEq restores the
    proposal from before shortening a step to land on a stop, but it shortens twice per
    step (in `apply_step!` and again in `loopheader!`), so only the first attempt and an
    attempt after a rejection keep the unshortened proposal; after an accepted step it is
    the shortened step. `lane_controller_kernel` passes it accordingly.
  - `first_step`: no step accepted yet; the step may then grow by up to
    `opts.qmax_first_step` instead of `opts.qmax`.
  - `dtnew`: the next step to attempt (before `dtmax` and the next stop are applied).
  - `errold_new`: `errold` for the next decision.
"""
@inline function lane_pi_controller(
        EEst, errold, dt, dtpropose, first_step, opts::LaneControllerOptions
    )
    (; qmin, qmax, gamma, beta1, beta2, qmax_first_step, qoldinit, qsteady_min, qsteady_max) = opts
    qmax = first_step ? qmax_first_step : qmax
    # q is the shrink factor dt_old / dt_new: the PI term, safety factor `gamma`, and bounds.
    q11 = zero(EEst)
    if iszero(EEst)
        q = inv(qmax)
    else
        # `fastpower`, as OrdinaryDiffEq: an approximate power through Float32, so that the
        # step sizes, and with them the steps, match a solve of the trajectory on its own.
        q11 = fastpower(EEst, beta1)
        q = q11 / fastpower(errold, beta2)
        @fastmath q = clamp(q / gamma, inv(qmax), inv(qmin))
    end
    accept = EEst <= one(EEst)
    if accept
        # Within the deadband the step is kept. The proposal continues from the step the
        # controller asked for, not from one shortened to land on a stop.
        if qsteady_min <= q <= qsteady_max
            q = one(q)
        end
        return true, dtpropose / q, max(EEst, qoldinit)
    else
        # Shrink the failed step by the proportional term only, at most by 1 / qmin; the
        # previous error stays the one of the last accepted step.
        return false, dt / min(inv(qmin), q11 / gamma), errold
    end
end

# All lanes' state on the device. `N` states per lane, `B` lanes.
struct LaneStepper{T, Tt, M, S, V, I8, I32, BV, TB, CO, F, P, CB}
    u::M            # N × B current (trial) state
    uprev::M        # N × B state at the start of the step
    du::M           # N × B right-hand side of the current stage
    fsal::M         # N × B f(uprev, t)
    dT::M           # N × B ∂f/∂t at (uprev, t)
    tmp::M          # N × B linear-solve right-hand side and solution
    K::S            # N × B × 8 stage increments
    J::Any          # N × N × B Jacobians (kept across rejected steps); B × nnz with a sparse pattern
    W::Any          # N × N × B iteration matrices, LU-factored in place; B × nslot with a sparse pattern
    ipiv::Any
    mass_diag::Any  # length-N diagonal of the mass matrix, or `nothing`
    up::M           # finite-difference scratch, (ncols B) × N
    fp::M
    fm::M
    ncols::Int      # columns (colors with a sparse pattern) per finite-difference launch
    central::Bool
    t::V            # B lane times
    dt::V           # B step sizes of the next attempt
    dtprop::V       # B proposals before shortening to a stop
    tprev::V        # B times at the start of the last accepted step
    hprev::V        # B sizes of the last accepted step
    EEst::Any       # B error estimates (state type)
    errold::Any
    status::I8      # B lane status
    fresh::BV       # B: f, J and ∂f/∂t must be computed at (uprev, t)
    clamped::BV     # B: the next attempt was shortened to land on the next stop
    naccept::I32
    nreject::I32
    stop_idx::I32   # B index of the next entry of `stops`
    landed::I32     # B index of the stop the last accepted step landed on, or 0
    save_idx::I32   # B index of the next entry of `savet`
    save_hi::I32    # B last save point strictly inside the last accepted step
    save_at::I32    # B save point equal to the end of the last accepted step, or 0
    stops::Any      # sorted stop times, ending with the final time
    stop_mask::Any  # UInt32 per stop: bit k set if periodic callback k fires there
    savet::Any      # save times
    saves::S        # N × nsave × B saved states
    tab::TB
    ctrl::CO
    abstol::T
    reltol::T
    dtmin::Tt
    dtmax::Tt
    maxiters::Int32
    f::F
    p::P
    callbacks::CB
    sparsity::Any   # `LaneSparsity` of a sparse `jac_prototype`, or `nothing` (dense)
    lu::Any         # `LaneSparseLU` once the pivot order is chosen, or `nothing`
    lu_scratch::Any # B × N scratch of the sparse factorization and solves
    pivot_min::Any  # B smallest pivot ratio of the sparse factorizations so far
    dense_lane::Any # B: the lane's W is factorized densely (`LaneDenseFallback`), or `nothing`
    dense_fallback::Any
    norm_weights::Any # N weights of a `ComponentNorm` (1 kept, 0 not), or `nothing`
    nkeep::Int      # the number of components in the error norm
    backend::Any
end

# ---------------------------------------------------------------------------------------
# Kernels. Lane `i` is skipped unless `status[i] == LANE_ACTIVE` (and, where noted, unless
# `fresh[i]`).

@kernel function lane_rhs_kernel(
        f, iip, du, @Const(u), @Const(p), @Const(t), @Const(dt), @Const(cs), @Const(status),
        @Const(fresh), @Const(only_fresh)
    )
    i = @index(Global, Linear)
    @inbounds if status[i] == LANE_ACTIVE && (!only_fresh || fresh[i])
        @views trajectory_rhs!(f, iip, du[:, i], u[:, i], p, i, t[i] + cs * dt[i])
    end
end

# The time derivative at (uprev, t) by a forward difference, as FiniteDiff takes it for
# `AutoFiniteDiff`: the step `sqrt(eps) max(|t|, 1)`. Uses `du` as scratch.
@kernel function lane_tgrad_kernel(
        f, iip, dT, du, @Const(uprev), @Const(fsal), @Const(p), @Const(t), @Const(status),
        @Const(fresh), @Const(rel)
    )
    i = @index(Global, Linear)
    @inbounds if status[i] == LANE_ACTIVE && fresh[i]
        ti = t[i]
        th = ti + rel * max(abs(ti), one(ti))
        h = th - ti
        @views trajectory_rhs!(f, iip, du[:, i], uprev[:, i], p, i, th)
        for k in 1:size(dT, 1)
            dT[k, i] = (du[k, i] - fsal[k, i]) / h
        end
    end
end

# `fd_jacobian_kernel` for the lanes that need a new Jacobian, at their own times.
@kernel function lane_fd_jacobian_kernel(
        f, iip, J, up, fp, fm, @Const(u), @Const(f0), @Const(p), @Const(t), @Const(jlo),
        @Const(ncols), @Const(rel), @Const(central), @Const(status), @Const(fresh)
    )
    g = @index(Global, Linear)
    i = (g - 1) ÷ ncols + 1
    j = jlo + (g - 1) % ncols
    N = size(u, 1)
    @inbounds if status[i] == LANE_ACTIVE && fresh[i]
        ti = t[i]
        uc = view(up, g, :)
        fpc = view(fp, g, :)
        for k in 1:N
            uc[k] = u[k, i]
        end
        uj = u[j, i]
        uc[j] = uj + rel * max(one(uj), abs(uj))
        h = uc[j] - uj
        trajectory_rhs!(f, iip, fpc, uc, p, i, ti)
        if central
            uc[j] = uj - h
            fmc = view(fm, g, :)
            trajectory_rhs!(f, iip, fmc, uc, p, i, ti)
            for k in 1:N
                J[k, j, i] = (fpc[k] - fmc[k]) / (2 * h)
            end
        else
            for k in 1:N
                J[k, j, i] = (fpc[k] - f0[k, i]) / h
            end
        end
    end
end

# W = J - M / (dt γ), as OrdinaryDiffEq's `Wfact_t`; factorized afterwards.
@kernel function lane_w_kernel(W, @Const(J), @Const(mass_diag), @Const(dt), @Const(gamma), @Const(status))
    r, c, i = @index(Global, NTuple)
    @inbounds if status[i] == LANE_ACTIVE
        w = J[r, c, i]
        if r == c
            w -= _mass_diagonal(mass_diag, r, W) / (dt[i] * gamma)
        end
        W[r, c, i] = w
    end
end

# First stage right-hand side: f(uprev, t) + dt d₁ ∂f/∂t.
@kernel function lane_stage1_kernel(tmp, @Const(fsal), @Const(dT), @Const(dt), @Const(d1), @Const(status))
    k, i = @index(Global, NTuple)
    @inbounds if status[i] == LANE_ACTIVE
        tmp[k, i] = fsal[k, i] + dt[i] * d1 * dT[k, i]
    end
end

# Store k_{s-1} = -(solution of the last solve) and form the stage state
# u = uprev + Σ_{j<s} a[j] k_j, with `a` row `s` of the tableau's A. The row is passed as an
# `SVector` and the loop has a constant bound so that it unrolls into registers: indexing the
# tableau's matrices with the runtime `s` puts them in local memory, which made this kernel
# about 40× slower.
@kernel function lane_stage_state_kernel(u, K, @Const(tmp), @Const(uprev), @Const(s), @Const(a), @Const(status))
    k, i = @index(Global, NTuple)
    @inbounds if status[i] == LANE_ACTIVE
        K[k, i, s - 1] = -tmp[k, i]
        acc = uprev[k, i]
        for j in 1:7
            if j < s
                acc += a[j] * K[k, i, j]
            end
        end
        u[k, i] = acc
    end
end

# Stage `s` right-hand side: du + dt d_s ∂f/∂t + M Σ_{j<s} c[j] k_j / dt, with `c` row `s`
# of the tableau's C and `d_s = d[s]` (see `lane_stage_state_kernel` for why a row).
@kernel function lane_stage_rhs_kernel(
        tmp, @Const(du), @Const(dT), @Const(K), @Const(mass_diag), @Const(dt), @Const(s),
        @Const(c), @Const(d_s), @Const(status)
    )
    k, i = @index(Global, NTuple)
    @inbounds if status[i] == LANE_ACTIVE
        h = dt[i]
        acc = zero(eltype(tmp))
        for j in 1:7
            if j < s
                acc += c[j] * K[k, i, j]
            end
        end
        tmp[k, i] = du[k, i] + h * d_s * dT[k, i] +
            _mass_diagonal(mass_diag, k, tmp) * acc / h
    end
end

# Store k_8 and form the step u = uprev + Σ b_j k_j.
@kernel function lane_final_kernel(u, K, @Const(tmp), @Const(uprev), @Const(tab), @Const(status))
    k, i = @index(Global, NTuple)
    @inbounds if status[i] == LANE_ACTIVE
        K[k, i, 8] = -tmp[k, i]
        acc = uprev[k, i]
        for j in 1:8
            acc += tab.b[j] * K[k, i, j]
        end
        u[k, i] = acc
    end
end

# The weight of component k in the error norm: 1, or that of a `ComponentNorm` (1 kept, 0 not).
@inline _norm_weight(::Nothing, k, ::Type{T}) where {T} = one(T)
@inline _norm_weight(weights, k, ::Type{T}) where {T} = @inbounds T(weights[k])

# EEst = RMS over the lane's components of k_8 / (abstol + max(|uprev|, |u|) reltol), the
# default `internalnorm` and `calculate_residuals` of OrdinaryDiffEq; with `weights`, over the
# `nkeep` components a `ComponentNorm` keeps.
@kernel function lane_error_kernel(
        EEst, @Const(K), @Const(u), @Const(uprev), @Const(abstol), @Const(reltol), @Const(status),
        @Const(weights), @Const(nkeep)
    )
    i = @index(Global, Linear)
    @inbounds if status[i] == LANE_ACTIVE
        N = size(u, 1)
        T = eltype(EEst)
        acc = zero(T)
        for k in 1:N
            sk = abstol + max(abs(uprev[k, i]), abs(u[k, i])) * reltol
            e = K[k, i, 8] / sk
            acc += _norm_weight(weights, k, T) * e * e
        end
        EEst[i] = sqrt(acc / nkeep)
    end
end

# The next attempt from `t` with proposal `dtp`, as `modify_dt_for_tstops!`: a step that
# reaches the next stop to within rounding lands on it (its time is set to the stop).
@inline function _next_dt(t, dtp, stops, idx)
    @inbounds stop = stops[idx]
    distance = stop - t
    tol = 100 * eps(max(abs(t), abs(stop)))
    return dtp + tol < distance ? (dtp, false) : (min(dtp, distance), true)
end

# One controller decision per lane after a step attempt. Accepted steps advance `t` and
# record what the save and affect kernels need; every lane gets the next attempt's `dt`.
@kernel function lane_controller_kernel(
        t, dt, dtprop, tprev, hprev, errold, status, fresh, clamped, naccept, nreject,
        stop_idx, landed, save_idx, save_hi, save_at, @Const(EEst), @Const(stops), @Const(savet),
        @Const(ctrl), @Const(dtmin), @Const(dtmax), @Const(maxiters)
    )
    i = @index(Global, Linear)
    @inbounds if status[i] == LANE_ACTIVE
        e = EEst[i]
        h = dt[i]
        landed[i] = Int32(0)
        save_at[i] = Int32(0)
        save_hi[i] = save_idx[i] - Int32(1)
        if !isfinite(e)
            status[i] = LANE_UNSTABLE
            fresh[i] = false
        else
            first_step = naccept[i] == 0
            accept, dtnew,
                errnew = lane_pi_controller(e, errold[i], h, dtprop[i], first_step, ctrl)
            errold[i] = errnew
            ti = t[i]
            if accept
                naccept[i] += Int32(1)
                k = stop_idx[i]
                on_stop = clamped[i]
                tnew = on_stop ? stops[k] : ti + h
                tprev[i] = ti
                hprev[i] = h
                t[i] = tnew
                fresh[i] = true
                if on_stop
                    landed[i] = k
                    k += Int32(1)
                    stop_idx[i] = k
                end
                # Save points inside (tprev, tnew) are interpolated; one at tnew is saved
                # after the affects.
                s = save_idx[i]
                while s <= length(savet) && savet[s] < tnew
                    s += Int32(1)
                end
                save_hi[i] = s - Int32(1)
                if s <= length(savet) && savet[s] == tnew
                    save_at[i] = s
                    s += Int32(1)
                end
                save_idx[i] = s
                if k > length(stops)
                    status[i] = LANE_SUCCESS
                end
                # As `calc_dt_propose!`: the interval the clock represents from the new time.
                dtnew = (tnew + dtnew) - tnew
            else
                nreject[i] += Int32(1)
                fresh[i] = false
            end
            if status[i] == LANE_ACTIVE
                dtnew = min(dtnew, dtmax)
                if dtnew <= dtmin
                    status[i] = LANE_DTMIN
                elseif naccept[i] + nreject[i] >= maxiters
                    status[i] = LANE_MAXITERS
                else
                    dt[i], clamped[i] = _next_dt(t[i], dtnew, stops, stop_idx[i])
                    # See `lane_pi_controller`: after an accepted step OrdinaryDiffEq keeps
                    # the shortened step as its proposal.
                    dtprop[i] = accept ? dt[i] : dtnew
                end
            end
        end
    else
        # A lane that finished in an earlier attempt has nothing left to save or apply.
        landed[i] = Int32(0)
        save_at[i] = Int32(0)
        save_hi[i] = Int32(0)
        fresh[i] = false
    end
end

# Rodas5P's dense output between uprev (θ = 0) and u (θ = 1):
# (1 - θ) uprev + θ (u + (1 - θ) (k₁ + θ (k₂ + θ k₃))), with k_d = Σ H[d, s] K_s.
@kernel function lane_save_interior_kernel(
        saves, @Const(u), @Const(uprev), @Const(K), @Const(savet), @Const(tprev),
        @Const(hprev), @Const(save_hi), @Const(save_idx), @Const(save_at), @Const(fresh),
        @Const(tab)
    )
    k, i = @index(Global, NTuple)
    @inbounds begin
        hi = save_hi[i]
        # The save points of the last accepted step start after the ones of the steps before.
        lo = hi + Int32(1)
        while lo > 1 && savet[lo - 1] > tprev[i]
            lo -= Int32(1)
        end
        if fresh[i] && lo <= hi
            kd1 = zero(eltype(u))
            kd2 = zero(eltype(u))
            kd3 = zero(eltype(u))
            for s in 1:8
                ks = K[k, i, s]
                kd1 += tab.H[1, s] * ks
                kd2 += tab.H[2, s] * ks
                kd3 += tab.H[3, s] * ks
            end
            y0 = uprev[k, i]
            y1 = u[k, i]
            for s in lo:hi
                θ = (savet[s] - tprev[i]) / hprev[i]
                θ1 = 1 - θ
                saves[k, s, i] = θ1 * y0 + θ * (y1 + θ1 * (kd1 + θ * (kd2 + θ * kd3)))
            end
        end
    end
end

@kernel function lane_save_exact_kernel(saves, @Const(u), @Const(save_at))
    k, i = @index(Global, NTuple)
    @inbounds begin
        s = save_at[i]
        if s > 0
            saves[k, s, i] = u[k, i]
        end
    end
end

# `affect!` for the lanes whose last accepted step landed on a stop where callback `cb`
# fires (or, with `all_lanes`, for every lane at its current time).
@kernel function lane_affect_kernel(
        affect!, u, @Const(t), p, @Const(landed), @Const(stop_mask), @Const(cb), @Const(all_lanes)
    )
    i = @index(Global, Linear)
    @inbounds begin
        k = landed[i]
        fires = all_lanes || (k > 0 && (stop_mask[k] >> (cb - 1)) & one(UInt32) == one(UInt32))
        if fires
            @views affect!(FakeIntegrator(u[:, i], t[i], ensemble_param(p, i)))
        end
    end
end

# Accepted lanes continue from u.
@kernel function lane_accept_kernel(uprev, @Const(u), @Const(landed_or_fresh))
    k, i = @index(Global, NTuple)
    @inbounds if landed_or_fresh[i]
        uprev[k, i] = u[k, i]
    end
end

# ---------------------------------------------------------------------------------------
# Initial step size, as OrdinaryDiffEq's `ode_determine_initdt` for an in-place problem.

@kernel function lane_initdt1_kernel(
        dt, u1, @Const(u0), @Const(f0), @Const(mass_diag), @Const(abstol), @Const(reltol),
        @Const(dtmax), @Const(smalldt), @Const(weights), @Const(nkeep)
    )
    i = @index(Global, Linear)
    @inbounds begin
        N = size(u0, 1)
        T = eltype(u0)
        d0 = zero(T)
        d1 = zero(T)
        for k in 1:N
            sk = abstol + abs(u0[k, i]) * reltol
            w = _norm_weight(weights, k, T)
            d0 += w * (u0[k, i] / sk)^2
            d1 += w * (f0[k, i] / _mass_diagonal(mass_diag, k, u0) / sk)^2
        end
        d0 = sqrt(d0 / nkeep)
        d1 = sqrt(d1 / nkeep)
        dt0 = (d0 < 1.0e-5 || d1 < 1.0e-5) ? smalldt : (d0 / d1) / 100
        dt0 = min(dt0, dtmax)
        dt[i] = dt0
        for k in 1:N
            u1[k, i] = u0[k, i] + dt0 * f0[k, i] / _mass_diagonal(mass_diag, k, u0)
        end
    end
end

@kernel function lane_initdt2_kernel(
        dt, @Const(u0), @Const(f0), @Const(f1), @Const(mass_diag), @Const(abstol),
        @Const(reltol), @Const(dtmax), @Const(dtmin), @Const(order), @Const(weights), @Const(nkeep)
    )
    i = @index(Global, Linear)
    @inbounds begin
        N = size(u0, 1)
        T = eltype(u0)
        dt0 = dt[i]
        d2 = zero(T)
        d1 = zero(T)
        same = true
        for k in 1:N
            sk = abstol + abs(u0[k, i]) * reltol
            m = _mass_diagonal(mass_diag, k, u0)
            w = _norm_weight(weights, k, T)
            d2 += w * ((f1[k, i] - f0[k, i]) / m / sk)^2
            d1 += w * (f0[k, i] / m / sk)^2
            same &= f1[k, i] == f0[k, i]
        end
        d2 = sqrt(d2 / nkeep) / dt0
        d1 = sqrt(d1 / nkeep)
        if same
            dt[i] = max(dtmin, 100dt0)
        else
            m12 = max(d1, d2)
            dt1 = m12 <= 1.0e-15 ? max(oftype(dt0, 1.0e-6), dt0 / 1000) :
                oftype(dt0, 10^(-(2 + log10(m12)) / order))
            dt[i] = max(dtmin, min(100dt0, dt1, dtmax))
        end
    end
end

# ---------------------------------------------------------------------------------------
# Host side.

function _lane_options(prob, kwargs)
    opts = merge(NamedTuple(prob.kwargs), NamedTuple(kwargs))
    known = (
        :abstol, :reltol, :saveat, :save_start, :save_end, :save_everystep, :dt, :dtmax,
        :tstops, :maxiters, :callback, :merge_callbacks, :initializealg, :verbose,
        :unstable_check, :dense, :internalnorm,
        :save_discretes,
    )
    unknown = filter(k -> !(k in known), keys(opts))
    isempty(unknown) || throw(
        ArgumentError(
            "`EnsembleGPUArray(...; per_trajectory_dt = true)` does not support the solve options $(join(map(k -> "`$k`", unknown), ", "))."
        )
    )
    if get(opts, :dense, false) === true
        throw(ArgumentError("`per_trajectory_dt = true` does not support `dense = true`; pass `saveat`."))
    end
    # ModelingToolkit stores its problem keyword `save_discretes` as a solve option. The
    # per-trajectory stepper saves the states only.
    if get(opts, :save_discretes, false) !== false
        throw(
            ArgumentError(
                "`per_trajectory_dt = true` does not save discrete values: pass `save_discretes = false`."
            )
        )
    end
    return opts
end

function _lane_scalar_tolerance(x, name)
    x isa Number || throw(
        ArgumentError(
            "`per_trajectory_dt = true` supports a scalar `$name` only; got a `$(typeof(x))`."
        )
    )
    return x
end

function _lane_save_times(saveat, save_start, save_end, t0, tf)
    ts = if saveat isa Number
        collect(typeof(t0), (t0 + saveat):saveat:tf)
    else
        sort!(typeof(t0)[s for s in saveat if t0 < s <= tf])
    end
    save_start && pushfirst!(ts, t0)
    if save_end && (isempty(ts) || last(ts) != tf)
        push!(ts, tf)
    end
    return ts
end

function _lane_callbacks(prob, ensemblealg; kwargs...)
    prob_cb, kwarg_cb = ensemble_callbacks(prob; kwargs...)
    cbs = Any[]
    for cb in (prob_cb, kwarg_cb)
        isempty_callback(cb) && continue
        set = cb isa CallbackSet ? cb : CallbackSet(cb)
        isempty(set.continuous_callbacks) || throw(
            ArgumentError(
                "`per_trajectory_dt = true` does not support continuous callbacks."
            )
        )
        for d in set.discrete_callbacks
            lane = batched_time_callback(d, ensemblealg)
            lane isa LanePeriodic || throw(
                ArgumentError(
                    "`per_trajectory_dt = true` supports DiffEqCallbacks' `PeriodicCallback` only (load DiffEqCallbacks); got another `DiscreteCallback`."
                )
            )
            lane.save_positions == (false, false) || throw(
                ArgumentError(
                    "`per_trajectory_dt = true` supports periodic callbacks with `save_positions = (false, false)` only; for a ModelingToolkit event pass `save_positions = (false, false)` to it."
                )
            )
            push!(cbs, lane)
        end
    end
    length(cbs) <= 32 || throw(ArgumentError("`per_trajectory_dt = true` supports up to 32 periodic callbacks."))
    return Tuple(cbs)
end

# Sorted stop times (user `tstops`, periodic stops, the final time) and, per stop, the
# bitmask of the periodic callbacks that fire there.
function _lane_stops(callbacks, tstops, t0, tf, ::Type{Tt}) where {Tt}
    stops = Dict{Tt, UInt32}()
    for s in tstops
        t0 < s < tf && (stops[Tt(s)] = get(stops, Tt(s), UInt32(0)))
    end
    stops[Tt(tf)] = UInt32(0)
    for (k, cb) in enumerate(callbacks)
        bit = UInt32(1) << (k - 1)
        tstart = convert(typeof(cb.Δt), t0 + cb.phase)
        index = iszero(cb.phase) ? 1 : 0
        while true
            s = Tt(tstart + index * cb.Δt)
            s < tf || break
            if s > t0
                stops[s] = get(stops, s, UInt32(0)) | bit
            end
            index += 1
        end
        cb.final_affect && (stops[Tt(tf)] |= bit)
    end
    times = sort!(collect(keys(stops)))
    return times, UInt32[stops[s] for s in times]
end

function _lane_jacobian_mode(alg)
    ad = hasproperty(alg, :autodiff) ? alg.autodiff : nothing
    ad isa ADTypes.AutoSparse && (ad = ADTypes.dense_ad(ad))
    if ad isa ADTypes.AutoFiniteDiff
        fdtype = ad.fdjtype
        fdtype isa Union{Val{:forward}, Val{:central}} && return fdtype isa Val{:central}
    end
    throw(
        ArgumentError(
            "`per_trajectory_dt = true` computes the Jacobian with finite differences: pass `Rodas5P(autodiff = AutoFiniteDiff())` (forward or central differences)."
        )
    )
end

function _check_lane_algorithm(alg)
    if !(nameof(typeof(alg)) === :Rodas5P && nameof(parentmodule(typeof(alg))) === :OrdinaryDiffEqRosenbrock)
        throw(
            ArgumentError(
                "`EnsembleGPUArray(...; per_trajectory_dt = true)` supports `Rodas5P` only; got `$(nameof(typeof(alg)))`."
            )
        )
    end
    return nothing
end

# Run the per-trajectory stepper on the batched initial states `u0` (N × B, on the device)
# and parameters `p` of the trajectories `probs`. Returns the saved times, the saved states
# (N × nsave × B, on the host), the number of saves per lane, the lane statuses and the step
# counts.
"""
    DiffEqGPU.check_per_trajectory_dt(prob, alg, ensemblealg; adaptive = true, kwargs...)

Throw the `ArgumentError` that solving the trajectories of `prob` with `alg` on
`ensemblealg` (an `EnsembleGPUArray` with `per_trajectory_dt = true`) and the solve keywords
`kwargs` would throw for an unsupported setting, without solving anything; return `nothing`
otherwise. It checks the algorithm (`Rodas5P` with a forward or central finite-difference
Jacobian), the mass matrix (diagonal), the solve options (including those stored in
`prob.kwargs`), scalar tolerances, the saving options, the callbacks (DiffEqCallbacks'
`PeriodicCallback`s with `save_positions = (false, false)`, at most 32) and the time span
(forward). Callers that build the problems of an ensemble can call it once on the base
problem, before preparing the trajectories. A check that needs all trajectories, such as
their time spans being equal, still happens in the solve.
"""
function check_per_trajectory_dt(prob, alg, ensemblealg; kwargs...)
    _lane_settings(prob, alg, ensemblealg; kwargs...)
    return nothing
end

# Every check of the per-trajectory path, and the settings they resolve.
function _lane_settings(prob, alg, ensemblealg; adaptive = true, kwargs...)
    _per_trajectory_dt(ensemblealg) || throw(
        ArgumentError(
            "the per-trajectory stepper needs `EnsembleGPUArray(...; per_trajectory_dt = true)`; got `$(nameof(typeof(ensemblealg)))` without it."
        )
    )
    adaptive || throw(
        ArgumentError("`per_trajectory_dt = true` needs an adaptive solve (`adaptive = true`).")
    )
    _check_lane_algorithm(alg)
    central = _lane_jacobian_mode(alg)
    N = length(prob.u0)
    mass = _mass_matrix_diagonal(prob.f.mass_matrix, N)
    pattern = nothing
    if _lane_sparse(prob.f.jac_prototype)
        pattern = _lane_sparsity_pattern(prob.f.jac_prototype, N)
        _lane_structurally_nonsingular(_lane_w_pattern(pattern, mass)) || throw(
            ArgumentError(
                "`per_trajectory_dt = true`: the `jac_prototype` pattern (with the diagonal where the mass matrix is nonzero) is structurally singular, so every iteration matrix would be singular."
            )
        )
    end
    opts = _lane_options(prob, kwargs)
    t0, tf = prob.tspan
    tf > t0 || throw(ArgumentError("`per_trajectory_dt = true` needs a forward time span."))
    abstol = _lane_scalar_tolerance(get(opts, :abstol, 1 // 10^6), :abstol)
    reltol = _lane_scalar_tolerance(get(opts, :reltol, 1 // 10^3), :reltol)
    saveat = get(opts, :saveat, ())
    save_everystep = get(opts, :save_everystep, isempty(saveat))
    if save_everystep && isempty(saveat)
        throw(
            ArgumentError(
                "`per_trajectory_dt = true` saves at given times: pass `saveat`, or `save_everystep = false` to save the start and end only."
            )
        )
    end
    save_start = get(opts, :save_start, saveat isa Number || isempty(saveat) || t0 in saveat)
    save_end = get(opts, :save_end, true)
    callbacks = _lane_callbacks(prob, ensemblealg; kwargs...)
    norm = get(opts, :internalnorm, nothing)
    if norm !== nothing
        norm isa ComponentNorm || throw(
            ArgumentError(
                "`per_trajectory_dt = true` supports `internalnorm = DiffEqGPU.ComponentNorm(keep)` only; got a `$(typeof(norm))`."
            )
        )
        _check_component_norm(norm, N)
    end
    return (;
        opts, central, pattern, abstol, reltol, saveat, save_start, save_end, callbacks,
        dtmax = get(opts, :dtmax, tf - t0), maxiters = get(opts, :maxiters, 100_000),
        tstops = get(opts, :tstops, ()),
        norm,
    )
end

function lane_solve(probs, alg, ensemblealg, u0, p; kwargs...)
    prob = probs[1]
    set = _lane_settings(prob, alg, ensemblealg; kwargs...)
    backend = ensemblealg.backend
    opts = set.opts
    central = set.central
    T = eltype(u0)
    t0, tf = prob.tspan
    Tt = promote_type(typeof(t0), typeof(tf))
    t0 = Tt(t0)
    tf = Tt(tf)
    N, B = size(u0)
    iip = Val(DiffEqBase.isinplace(prob))
    f = prob.f.f
    abstol = T(set.abstol)
    reltol = T(set.reltol)
    savet = Tt.(_lane_save_times(set.saveat, set.save_start, set.save_end, t0, tf))
    dtmax = Tt(set.dtmax)
    maxiters = Int32(set.maxiters)
    dtmin = eps(max(abs(t0), abs(tf)))
    callbacks = set.callbacks
    stops, stop_mask = _lane_stops(callbacks, set.tstops, t0, tf, Tt)

    _, mass_diag = batched_mass_matrix(prob.f.mass_matrix, u0)
    isdae = mass_diag !== nothing && any(iszero, Array(mass_diag))

    dev(x) = adapt(backend, x)
    zN() = fill!(similar(u0, T, N, B), zero(T))
    norm_weights, nkeep = if set.norm === nothing
        nothing, N
    else
        w = zeros(T, N)
        w[set.norm.keep] .= 1
        copyto!(similar(u0, T, N), w), length(set.norm.keep)
    end
    pattern = set.pattern
    sparsity = pattern === nothing ? nothing : dev(_lane_sparsity(pattern))
    ncolumns = pattern === nothing ? N : length(sparsity.color_ptr) - 1
    ncols = let by_threads = cld(FD_TARGET_THREADS, B),
            by_memory = FD_SCRATCH_BYTES ÷ ((central ? 3 : 2) * N * B * sizeof(T))

        clamp(min(by_threads, by_memory), 1, ncolumns)
    end
    if pattern === nothing
        J = fill!(similar(u0, T, N, N, B), zero(T))
        W = similar(J)
        ipiv = lu_pivots(W)
    else
        # W and its operation list are set up after the first Jacobians (`_lane_sparse_setup`).
        J = _lane_values(u0, T, B, nnz(pattern))
        W = similar(u0, T, B, 0)
        ipiv = nothing
    end
    lv(x, ::Type{X}) where {X} = fill!(similar(u0, X, B), x)
    saves = fill!(similar(u0, T, N, max(length(savet), 1), B), zero(T))

    st = LaneStepper(
        copy(u0), copy(u0), zN(), zN(), zN(), zN(),
        fill!(similar(u0, T, N, B, 8), zero(T)), J, W, ipiv, mass_diag,
        similar(u0, T, ncols * B, N), similar(u0, T, ncols * B, N),
        similar(u0, T, central ? ncols * B : 0, N), ncols, central,
        lv(t0, Tt), lv(zero(Tt), Tt), lv(zero(Tt), Tt), lv(t0, Tt), lv(zero(Tt), Tt),
        lv(zero(T), T), lv(T(1 // 10^4), T), lv(LANE_ACTIVE, Int8), lv(true, Bool),
        lv(false, Bool),
        lv(Int32(0), Int32), lv(Int32(0), Int32), lv(Int32(1), Int32), lv(Int32(0), Int32),
        lv(Int32(1), Int32), lv(Int32(0), Int32), lv(Int32(0), Int32),
        dev(stops), dev(stop_mask), dev(savet), saves,
        LaneRodasTableau(T), LaneControllerOptions(T), abstol, reltol, dtmin, dtmax,
        maxiters, f, p, callbacks, sparsity, nothing,
        pattern === nothing ? nothing : _lane_values(u0, T, B, N),
        pattern === nothing ? nothing : lv(one(T), T),
        pattern === nothing ? nothing : lv(false, Bool),
        pattern === nothing ? nothing : LaneDenseFallback(),
        norm_weights, nkeep,
        backend
    )

    # The start: save, initial affects, then the initial step size from the state after them.
    if !isempty(savet) && first(savet) == t0
        view(st.saves, :, 1, :) .= st.u
        fill!(st.save_idx, Int32(2))
    end
    for (k, cb) in enumerate(callbacks)
        cb.initial_affect || continue
        lane_affect_kernel(backend)(
            cb.affect!, st.uprev, st.t, st.p, st.landed, st.stop_mask, Int32(k), true;
            ndrange = B, workgroupsize = workgroupsize(backend, B)
        )
    end
    copyto!(st.u, st.uprev)
    _lane_initial_dt!(st, opts, iip, isdae, t0)

    wgs = workgroupsize(backend, B)
    if pattern !== nothing
        # The pivot order comes from the first iteration matrices of all lanes; the first
        # step attempt then uses the Jacobians computed for it.
        _lane_jacobian!(st, iip, wgs)
        st = _lane_sparse_setup(st, pattern, _mass_matrix_diagonal(prob.f.mass_matrix, N))
        fill!(st.fresh, false)
    end
    while count(==(LANE_ACTIVE), st.status) > 0
        lane_step!(st, iip, wgs)
    end

    status = Array(st.status)
    if st.dense_lane !== nothing
        ndense = count(Array(st.dense_lane))
        ndense > 0 &&
            @debug "$ndense of $B trajectories switched to the dense LU: their static-pivot factorization failed the pivot check"
    end
    nsaved = Array(st.save_idx) .- 1
    return (;
        savet, saves = Array(st.saves), nsaved, status,
        naccept = Array(st.naccept), nreject = Array(st.nreject), p = st.p,
        pivot_min = st.pivot_min === nothing ? nothing : Array(st.pivot_min),
        dense_lane = st.dense_lane === nothing ? nothing : Array(st.dense_lane),
    )
end

function _lane_initial_dt!(st, opts, iip, isdae, t0)
    backend = st.backend
    B = length(st.t)
    Tt = eltype(st.t)
    dtmin = nextfloat(max(st.dtmin, eps(t0)))
    smalldt = max(dtmin, Tt(1 // 10^6))
    if haskey(opts, :dt)
        fill!(st.dt, Tt(opts[:dt]))
    elseif isdae
        fill!(st.dt, smalldt)
    else
        wgs = workgroupsize(backend, B)
        zt = fill!(similar(st.t), zero(Tt))
        lane_rhs_kernel(backend)(
            st.f, iip, st.fsal, st.uprev, st.p, st.t, zt, zero(Tt), st.status, st.fresh,
            false; ndrange = B, workgroupsize = wgs
        )
        lane_initdt1_kernel(backend)(
            st.dt, st.u, st.uprev, st.fsal, st.mass_diag, st.abstol, st.reltol, st.dtmax,
            smalldt, st.norm_weights, st.nkeep; ndrange = B, workgroupsize = wgs
        )
        lane_rhs_kernel(backend)(
            st.f, iip, st.du, st.u, st.p, st.t, st.dt, one(Tt), st.status, st.fresh, false;
            ndrange = B, workgroupsize = wgs
        )
        lane_initdt2_kernel(backend)(
            st.dt, st.uprev, st.fsal, st.du, st.mass_diag, st.abstol, st.reltol, st.dtmax,
            dtmin, 5, st.norm_weights, st.nkeep; ndrange = B, workgroupsize = wgs
        )
        copyto!(st.u, st.uprev)
    end
    copyto!(st.dtprop, st.dt)
    # The first attempt is shortened to the first stop like every other.
    dts = Array(st.dt)
    stops = Array(st.stops)
    ts = Array(st.t)
    next = [_next_dt(ts[i], dts[i], stops, 1) for i in eachindex(dts)]
    copyto!(st.dt, first.(next))
    copyto!(st.clamped, Bool[last(x) for x in next])
    return nothing
end

# f, ∂f/∂t and J at (uprev, t) for the lanes that moved (`fresh`); rejected lanes keep theirs.
function _lane_jacobian!(st, iip, wgs)
    backend = st.backend
    N, B = size(st.u)
    Tt = eltype(st.t)
    T = eltype(st.u)
    lane_rhs_kernel(backend)(
        st.f, iip, st.fsal, st.uprev, st.p, st.t, st.dt, zero(Tt), st.status, st.fresh, true;
        ndrange = B, workgroupsize = wgs
    )
    lane_tgrad_kernel(backend)(
        st.f, iip, st.dT, st.du, st.uprev, st.fsal, st.p, st.t, st.status, st.fresh,
        sqrt(eps(T)); ndrange = B, workgroupsize = wgs
    )
    rel = st.central ? cbrt(eps(T)) : sqrt(eps(T))
    if st.sparsity === nothing
        for jlo in 1:st.ncols:N
            ncols = min(st.ncols, N - jlo + 1)
            n = ncols * B
            lane_fd_jacobian_kernel(backend)(
                st.f, iip, st.J, st.up, st.fp, st.fm, st.uprev, st.fsal, st.p, st.t, jlo,
                ncols, rel, st.central, st.status, st.fresh;
                ndrange = n, workgroupsize = workgroupsize(backend, n)
            )
        end
    else
        ncolors = length(st.sparsity.color_ptr) - 1
        for clo in 1:st.ncols:ncolors
            nc = min(st.ncols, ncolors - clo + 1)
            n = nc * B
            lane_fd_jacobian_sparse_kernel(backend)(
                st.f, iip, st.J, st.up, st.fp, st.fm, st.uprev, st.fsal, st.p, st.t, clo,
                nc, rel, st.central, st.status, st.fresh, st.sparsity;
                ndrange = n, workgroupsize = workgroupsize(backend, n)
            )
        end
    end
    return nothing
end

# W = J - M / (dt γ), factorized, for every active lane.
function _lane_factorize!(st, wgs)
    backend = st.backend
    N, B = size(st.u)
    if st.lu === nothing
        lane_w_kernel(backend)(
            st.W, st.J, st.mass_diag, st.dt, st.tab.gamma, st.status;
            ndrange = (N, N, B), workgroupsize = (min(N, 16), min(N, 16), 1)
        )
        batched_lufact!(backend, st.W, st.ipiv)
    else
        S, L, Bp = _lane_coop_shape(LANE_SPARSE_FACTOR_THREADS, B)
        lane_sparse_factor_coop_kernel(backend)(
            st.W, st.lu_scratch, st.pivot_min, st.dense_lane, st.status, st.J, st.mass_diag,
            st.dt, st.tab.gamma, st.lu, B; ndrange = (S, Bp), workgroupsize = (S, L)
        )
        # Lanes whose static-pivot factorization tripped the pivot check, now or earlier.
        _lane_dense_factorize!(st)
    end
    return nothing
end

# tmp = W \ tmp for every active lane.
function _lane_ldiv!(st, wgs)
    N, B = size(st.u)
    if st.lu === nothing
        batched_ldiv!(st.backend, st.W, st.tmp, st.ipiv, N, B)
    else
        S, L, Bp = _lane_coop_shape(_lane_sparse_solve_threads(B), B)
        lane_sparse_solve_coop_kernel(st.backend)(
            st.tmp, st.lu_scratch, st.W, st.status, st.dense_lane, st.lu, B;
            ndrange = (S, Bp), workgroupsize = (S, L)
        )
        _lane_dense_ldiv!(st, st.tmp)
    end
    return nothing
end

# One step attempt of every active lane.
function lane_step!(st, iip, wgs)
    backend = st.backend
    N, B = size(st.u)
    tab = st.tab
    rhs!(du, u, cs, only_fresh) = lane_rhs_kernel(backend)(
        st.f, iip, du, u, st.p, st.t, st.dt, cs, st.status, st.fresh, only_fresh;
        ndrange = B, workgroupsize = wgs
    )
    nb = (N, B)
    nbwgs = (min(N, 32), max(1, min(B, 256 ÷ min(N, 32))))

    _lane_jacobian!(st, iip, wgs)
    _lane_factorize!(st, wgs)

    lane_stage1_kernel(backend)(
        st.tmp, st.fsal, st.dT, st.dt, tab.d[1], st.status; ndrange = nb, workgroupsize = nbwgs
    )
    _lane_ldiv!(st, wgs)
    for s in 2:8
        lane_stage_state_kernel(backend)(
            st.u, st.K, st.tmp, st.uprev, s, tab.A[s, :], st.status;
            ndrange = nb, workgroupsize = nbwgs
        )
        rhs!(st.du, st.u, tab.c[s], false)
        lane_stage_rhs_kernel(backend)(
            st.tmp, st.du, st.dT, st.K, st.mass_diag, st.dt, s, tab.C[s, :], tab.d[s], st.status;
            ndrange = nb, workgroupsize = nbwgs
        )
        _lane_ldiv!(st, wgs)
    end
    lane_final_kernel(backend)(
        st.u, st.K, st.tmp, st.uprev, tab, st.status; ndrange = nb, workgroupsize = nbwgs
    )
    lane_error_kernel(backend)(
        st.EEst, st.K, st.u, st.uprev, st.abstol, st.reltol, st.status, st.norm_weights,
        st.nkeep; ndrange = B, workgroupsize = wgs
    )

    lane_controller_kernel(backend)(
        st.t, st.dt, st.dtprop, st.tprev, st.hprev, st.errold, st.status, st.fresh,
        st.clamped, st.naccept, st.nreject, st.stop_idx, st.landed, st.save_idx, st.save_hi,
        st.save_at, st.EEst, st.stops, st.savet, st.ctrl, st.dtmin, st.dtmax, st.maxiters;
        ndrange = B, workgroupsize = wgs
    )

    # Accepted lanes (`fresh`): saves inside and at the end of the step, then the affects at
    # its end (DiffEqBase's `apply_discrete_callback!` saves before the affect), then
    # continue from u.
    if !isempty(st.savet)
        lane_save_interior_kernel(backend)(
            st.saves, st.u, st.uprev, st.K, st.savet, st.tprev, st.hprev, st.save_hi,
            st.save_idx, st.save_at, st.fresh, tab; ndrange = nb, workgroupsize = nbwgs
        )
    end
    if !isempty(st.savet)
        lane_save_exact_kernel(backend)(st.saves, st.u, st.save_at; ndrange = nb, workgroupsize = nbwgs)
    end
    for (k, cb) in enumerate(st.callbacks)
        lane_affect_kernel(backend)(
            cb.affect!, st.u, st.t, st.p, st.landed, st.stop_mask, Int32(k), false;
            ndrange = B, workgroupsize = wgs
        )
    end
    lane_accept_kernel(backend)(st.uprev, st.u, st.fresh; ndrange = nb, workgroupsize = nbwgs)
    return nothing
end

# The trajectories `probs` of a batch solved with per-trajectory steps, as solutions.
function _batch_solve_lanes(
        ensembleprob, alg, ensemblealg, I, probs, u0;
        sim_seeds, rng_func, master_rng, kwargs...
    )
    if !all(prob -> isequal(prob.tspan, probs[1].tspan), probs)
        throw(
            ArgumentError(
                "`EnsembleGPUArray(...; per_trajectory_dt = true)` needs all trajectories to have the same `tspan`."
            )
        )
    end
    backend = ensemblealg.backend
    p = adapt(backend, pack_parameters(probs, float(eltype(u0))))
    res = lane_solve(probs, alg, ensemblealg, adapt(backend, u0), p; kwargs...)
    ps = final_parameters(res.p, probs)
    return map(eachindex(probs)) do i
        n = res.nsaved[i]
        us = [res.saves[:, s, i] for s in 1:n]
        stats = SciMLBase.DEStats(0)
        stats.naccept = res.naccept[i]
        stats.nreject = res.nreject[i]
        ensembleprob.output_func(
            SciMLBase.build_solution(
                with_parameters(probs[i], ps[i]), alg, res.savet[1:n], us;
                stats, retcode = lane_retcode(res.status[i])
            ),
            _make_ensemble_context(I[i], sim_seeds, rng_func, master_rng)
        )[1]
    end
end
