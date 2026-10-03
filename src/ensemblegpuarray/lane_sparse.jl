# Sparse Jacobians and sparse LU for the per-trajectory stepper (`lanes.jl`).
#
# With a sparse `jac_prototype`, the stepper stores each lane's Jacobian as the values of
# that pattern, computes it by finite differences over a column coloring of the pattern
# (one right-hand side evaluation per color instead of per column), and factorizes
# W = J - M / (dt γ) with a sparse LU whose pivot order is fixed for the whole solve: a
# row permutation from a maximum-product matching of |W| (MC64's job 5) and a symmetric
# minimum-degree ordering, both computed once, from the lanes' first iteration matrices.
# Every lane then runs the same list of operations without pivoting. Values are stored
# lane-fastest (`B × nslot`), so a warp of lanes reads each slot coalesced.
#
# The pattern must contain every entry the right-hand side can make nonzero: with the
# coloring, a missing entry (i, j) corrupts the stored entry of another column of j's
# color in row i as well, not only the missing one.

# The Jacobian pattern (CSC, `nnz` entries) and its column coloring: the columns of a
# color share no row, so one perturbation per color recovers all of them.
struct LaneSparsity{I}
    colptr::I       # N + 1
    rowval::I       # nnz
    color_ptr::I    # ncolors + 1
    color_cols::I   # N, the columns of color c are color_cols[color_ptr[c]:color_ptr[c + 1] - 1]
end
function Adapt.adapt_structure(to, s::LaneSparsity)
    return LaneSparsity(adapt(to, s.colptr), adapt(to, s.rowval), adapt(to, s.color_ptr), adapt(to, s.color_cols))
end

# The static-pivot LU of W[rowo, colo] as an operation list over `nslot` value slots.
struct LaneSparseLU{I}
    nslot::Int
    jslot::I    # slot of each Jacobian entry
    mslot::I    # slot of W[k, k] for each state k (0 if W[k, k] is not stored)
    rowo::I     # row k of the permuted matrix is row rowo[k] of W
    colo::I     # column k of the permuted matrix is column colo[k] of W
    piv::I      # slot of U[k, k]
    divptr::I   # the multipliers of pivot k: div_t[divptr[k]:divptr[k + 1] - 1]
    div_t::I
    fmaptr::I   # the updates of pivot k: V[t] -= V[l] V[u] for (t, l, u) in fmaptr[k]:fmaptr[k + 1] - 1
    fma_t::I
    fma_l::I
    fma_u::I
    Lptr::I     # strict lower factor (unit diagonal) by columns: (row, slot) pairs
    Lrow::I
    Lslot::I
    Uptr::I     # strict upper factor by columns: (row, slot) pairs
    Urow::I
    Uslot::I
end
function Adapt.adapt_structure(to, lu::LaneSparseLU)
    return LaneSparseLU((adapt(to, getfield(lu, f)) for f in fieldnames(LaneSparseLU))...)
end

"""
    lane_pivot_status(σ)

The status of a lane after its iteration matrix W has been factorized with the fixed pivot
order, from `σ = min_k |U[k, k]| / |W[k, k]|`: how much the pivots shrank during the
elimination (`σ = 1` when no pivot lost magnitude; scale-invariant, unlike a ratio to the
column maximum or the size of the multipliers). Returns `LANE_ACTIVE` to continue with the
factors, or a final status (`LANE_PIVOT`) to stop the lane. A NaN pivot makes `σ` NaN
(compare with `!(σ >= threshold)` to treat it as failing); an exactly zero or non-finite
pivot also makes the step's error estimate non-finite, which the controller turns into
`LANE_UNSTABLE`. `σ` measures the cancellation during the elimination only: a pivot that is
already small relative to its column in the current W (the order was chosen at the first
step's `dt`) keeps `σ = 1`.
"""
@inline function lane_pivot_status(σ)
    return !(σ >= 1e-8) ? LANE_PIVOT : LANE_ACTIVE
end

_lane_sparse(prototype) = prototype isa SparseMatrixCSC

# Threads per lane of the sparse factorization and solves. A thread per lane runs the whole
# operation list sequentially, which only keeps a GPU busy with many lanes; below that, the
# lane's updates are spread over a group of threads (measured crossover on an RTX 4080 with a
# 124-state model: about 8k lanes).
const LANE_SPARSE_COOP_MAX_LANES = 8192
const LANE_SPARSE_COOP_THREADS = 32
_lane_sparse_threads(B) = B >= LANE_SPARSE_COOP_MAX_LANES ? 1 : LANE_SPARSE_COOP_THREADS

# Workgroups of `S` threads for each of `L` lanes (128 threads), and the padded lane count.
function _lane_coop_shape(S, B)
    L = max(1, 128 ÷ S)
    return S, L, cld(B, L) * L
end

# B × n per-lane values, indexed `[lane, entry]`: stored lane-fastest for the per-lane kernels,
# entry-fastest (a lane's values contiguous) for the cooperative ones.
function _lane_values(like, ::Type{T}, B, n, threads) where {T}
    threads == 1 && return fill!(similar(like, T, B, n), zero(T))
    return PermutedDimsArray(fill!(similar(like, T, n, B), zero(T)), (2, 1))
end
_lane_entry_max(V) = vec(maximum(abs, V; dims = 1))
_lane_entry_max(V::PermutedDimsArray) = vec(maximum(abs, parent(V); dims = 2))

function _lane_sparsity_pattern(prototype, N)
    size(prototype) == (N, N) || throw(
        ArgumentError(
            "`per_trajectory_dt = true` needs an N × N `jac_prototype` (N = $N); got $(size(prototype, 1)) × $(size(prototype, 2))."
        )
    )
    # Every stored entry, explicit zeros included, is part of the pattern.
    rows, cols, _ = findnz(prototype)
    return sparse(rows, cols, fill(true, length(rows)), N, N)
end

# The pattern of W: that of J and the diagonal entries where the mass matrix is nonzero.
function _lane_w_pattern(PJ::SparseMatrixCSC{Bool}, mass)
    N = size(PJ, 1)
    P = Matrix(PJ)
    for k in 1:N
        (mass === nothing || !iszero(mass[k])) && (P[k, k] = true)
    end
    return P
end

# Greedy coloring of the columns, in order: the smallest color no column sharing a row
# with column j has taken.
function _lane_column_coloring(PJ::SparseMatrixCSC{Bool})
    N = size(PJ, 2)
    rows = rowvals(PJ)
    cols_of_row = [Int[] for _ in 1:N]
    for j in 1:N, e in nzrange(PJ, j)
        push!(cols_of_row[rows[e]], j)
    end
    color = zeros(Int, N)
    seen = zeros(Int, N + 1)   # seen[c] == j: color c is taken by a neighbour of j
    for j in 1:N
        for e in nzrange(PJ, j), j2 in cols_of_row[rows[e]]
            j2 < j && (seen[color[j2]] = j)
        end
        c = 1
        while seen[c] == j
            c += 1
        end
        color[j] = c
    end
    ncolors = maximum(color; init = 0)
    order = sortperm(color)                       # stable: columns ascending within a color
    ptr = cumsum([1; [count(==(c), color) for c in 1:ncolors]])
    return ptr, order
end

function _lane_sparsity(PJ::SparseMatrixCSC{Bool})
    ptr, cols = _lane_column_coloring(PJ)
    i32(x) = Int32.(x)
    colptr = cumsum([1; [length(nzrange(PJ, j)) for j in 1:size(PJ, 2)]])
    return LaneSparsity(i32(colptr), i32(rowvals(PJ)), i32(ptr), i32(cols))
end

# Whether the pattern admits a zero-free diagonal after a row permutation (a perfect
# bipartite matching, by augmenting paths); without one every W is singular.
function _lane_structurally_nonsingular(P::AbstractMatrix{Bool})
    N = size(P, 1)
    rows_of = [findall(view(P, :, j)) for j in 1:N]
    match = zeros(Int, N)   # match[i]: column matched to row i
    function augment(j, seen)
        for i in rows_of[j]
            seen[i] && continue
            seen[i] = true
            if match[i] == 0 || augment(match[i], seen)
                match[i] = j
                return true
            end
        end
        return false
    end
    return all(j -> augment(j, falses(N)), 1:N)
end

# Maximum-product matching (MC64 job 5): the row permutation that maximizes the product of
# |A| on the diagonal, as a minimum-cost assignment with cost log(max_i |A[i, j]|) -
# log|A[i, j]| (Hungarian algorithm, O(N³)). Pattern entries that are zero in `A` cost
# more than any matching without them, so they are used only where nothing else matches. Returns
# `rowof` with A[rowof, :] having the matched entries on the diagonal. The costs are summed
# along augmenting paths, so they are kept in at least double precision: rounding them
# would decide between near-equal matchings arbitrarily.
function _lane_maxprod_matching(A::AbstractMatrix{<:Real}, P::AbstractMatrix{Bool})
    N = size(A, 1)
    Tc = promote_type(Float64, float(eltype(A)))
    a = abs.(Tc.(A))
    logs = [a[i, j] > 0 ? log(a[i, j]) : Tc(-Inf) for i in 1:N, j in 1:N]
    span = zero(Tc)
    for j in 1:N
        c = [logs[i, j] for i in 1:N if P[i, j] && isfinite(logs[i, j])]
        isempty(c) || (span = max(span, maximum(c) - minimum(c)))
    end
    # Above the total cost of any matching on nonzero entries (N entries, each at most `span`),
    # so a matching uses a zero entry only where no zero-free one exists.
    zero_cost = N * span + 1
    C = fill(Tc(Inf), N, N)
    for j in 1:N
        cmax = maximum((logs[i, j] for i in 1:N if P[i, j]); init = Tc(-Inf))
        for i in 1:N
            P[i, j] || continue
            C[i, j] = isfinite(logs[i, j]) ? cmax - logs[i, j] : (isfinite(cmax) ? zero_cost : zero(Tc))
        end
    end
    # Hungarian algorithm on rows (potentials u, v; p[j] = row assigned to column j)
    u = zeros(Tc, N + 1)
    v = zeros(Tc, N + 1)
    p = zeros(Int, N + 1)
    way = zeros(Int, N + 1)
    for i in 1:N
        p[1] = i
        j0 = 1
        minv = fill(Tc(Inf), N + 1)
        used = falses(N + 1)
        while true
            used[j0] = true
            i0 = p[j0]
            delta = Tc(Inf)
            j1 = 0
            for j in 2:(N + 1)
                used[j] && continue
                cur = C[i0, j - 1] - u[i0 + 1] - v[j]
                if cur < minv[j]
                    minv[j] = cur
                    way[j] = j0
                end
                if minv[j] < delta
                    delta = minv[j]
                    j1 = j
                end
            end
            isfinite(delta) || throw(ArgumentError("the `jac_prototype` pattern is structurally singular"))
            for j in 1:(N + 1)
                if used[j]
                    u[p[j] + 1] += delta
                    v[j] -= delta
                else
                    minv[j] -= delta
                end
            end
            j0 = j1
            p[j0] == 0 && break
        end
        while true
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            j0 == 1 && break
        end
    end
    return [p[j + 1] for j in 1:N]
end

# Minimum-degree ordering of the symmetric pattern S (exact degrees on the elimination
# graph, ties to the lowest index): the symmetric permutation that keeps the fill of an
# LU without pivoting small. O(N² + fill), fine for the few hundred states of a model.
function _lane_min_degree(S::AbstractMatrix{Bool})
    N = size(S, 1)
    adj = [BitSet(j for j in 1:N if j != i && (S[i, j] || S[j, i])) for i in 1:N]
    alive = trues(N)
    order = Int[]
    for _ in 1:N
        best = 0
        bestdeg = typemax(Int)
        for v in 1:N
            alive[v] || continue
            d = length(adj[v])
            if d < bestdeg
                bestdeg = d
                best = v
            end
        end
        push!(order, best)
        alive[best] = false
        nbrs = adj[best]
        for a in nbrs
            delete!(adj[a], best)
            union!(adj[a], nbrs)
            delete!(adj[a], a)
        end
        adj[best] = BitSet()
    end
    return order
end

# The operation list of the LU without pivoting of W[rowo, colo], for the W pattern P.
function _lane_sparse_lu(P::AbstractMatrix{Bool}, PJ::SparseMatrixCSC{Bool}, rowo, colo)
    N = size(P, 1)
    F = P[rowo, colo]
    for k in 1:N, i in (k + 1):N
        F[i, k] || continue
        for j in (k + 1):N
            F[k, j] && (F[i, j] = true)
        end
    end
    slot = zeros(Int, N, N)
    s = 0
    for j in 1:N, i in 1:N
        F[i, j] && (slot[i, j] = (s += 1))
    end
    pos_row = invperm(rowo)
    pos_col = invperm(colo)
    rows = rowvals(PJ)
    jslot = [slot[pos_row[rows[e]], pos_col[j]] for j in 1:N for e in nzrange(PJ, j)]
    mslot = [P[k, k] ? slot[pos_row[k], pos_col[k]] : 0 for k in 1:N]
    piv = [slot[k, k] for k in 1:N]
    div_t = Int[]
    divptr = [1]
    fma_t = Int[]
    fma_l = Int[]
    fma_u = Int[]
    fmaptr = [1]
    for k in 1:N
        Ls = [i for i in (k + 1):N if F[i, k]]
        Us = [j for j in (k + 1):N if F[k, j]]
        append!(div_t, slot[i, k] for i in Ls)
        push!(divptr, length(div_t) + 1)
        for j in Us, i in Ls
            push!(fma_t, slot[i, j])
            push!(fma_l, slot[i, k])
            push!(fma_u, slot[k, j])
        end
        push!(fmaptr, length(fma_t) + 1)
    end
    Lptr = [1]
    Lrow = Int[]
    Lslot = Int[]
    Uptr = [1]
    Urow = Int[]
    Uslot = Int[]
    for j in 1:N
        for i in (j + 1):N
            F[i, j] && (push!(Lrow, i); push!(Lslot, slot[i, j]))
        end
        push!(Lptr, length(Lrow) + 1)
        for i in 1:(j - 1)
            F[i, j] && (push!(Urow, i); push!(Uslot, slot[i, j]))
        end
        push!(Uptr, length(Urow) + 1)
    end
    i32(x) = Int32.(x)
    return LaneSparseLU(
        s, i32(jslot), i32(mslot), i32(rowo), i32(colo), i32(piv), i32(divptr), i32(div_t),
        i32(fmaptr), i32(fma_t), i32(fma_l), i32(fma_u), i32(Lptr), i32(Lrow), i32(Lslot),
        i32(Uptr), i32(Urow), i32(Uslot)
    )
end

# The pivot order from the envelope `Wabs` (N × N, max over the lanes of |W|) of the first
# iteration matrices: the matching puts large entries on the diagonal, the ordering keeps
# the fill small, and the matched diagonal moves with the symmetric permutation.
function _lane_sparse_order(Wabs, P)
    rowof = _lane_maxprod_matching(Wabs, P)
    B = P[rowof, :]
    q = _lane_min_degree(B .| B')
    return rowof[q], q
end

# An upper bound of max over the lanes of |W| = |J - M / (dt γ)| from the lanes' Jacobian
# values `Jv` (B × nnz, on the device) and step sizes: max |J| per entry, plus
# M[k, k] / (min dt γ) on the diagonal.
function _lane_w_envelope(Jv, PJ, P, mass, dt, gamma)
    N = size(P, 1)
    jmax = Array(_lane_entry_max(Jv))
    T = eltype(jmax)
    hmin = T(minimum(dt) * gamma)
    W = zeros(T, N, N)
    rows = rowvals(PJ)
    for j in 1:N, e in nzrange(PJ, j)
        W[rows[e], j] = jmax[e]
    end
    for k in 1:N
        P[k, k] || continue
        m = mass === nothing ? one(T) : T(mass[k])
        W[k, k] += abs(m) / hmin
    end
    return W
end

# ---------------------------------------------------------------------------------------
# Kernels

# `lane_fd_jacobian_kernel` over the colors of the pattern: thread `g` perturbs the columns
# of color `clo + (g - 1) % ncolors` of lane `(g - 1) ÷ ncolors + 1` together and stores the
# entries of those columns. A row depends on at most one column of a color, so this equals
# one perturbation per column.
@kernel function lane_fd_jacobian_sparse_kernel(
        f, iip, Jv, up, fp, fm, @Const(u), @Const(f0), @Const(p), @Const(t), @Const(clo),
        @Const(ncolors), @Const(rel), @Const(central), @Const(status), @Const(fresh), @Const(sp)
    )
    g = @index(Global, Linear)
    i = (g - 1) ÷ ncolors + 1
    c = clo + (g - 1) % ncolors
    N = size(u, 1)
    @inbounds if status[i] == LANE_ACTIVE && fresh[i]
        ti = t[i]
        cols = sp.color_ptr[c]:(sp.color_ptr[c + 1] - 1)
        uc = view(up, g, :)
        fpc = view(fp, g, :)
        for k in 1:N
            uc[k] = u[k, i]
        end
        for q in cols
            j = sp.color_cols[q]
            uj = u[j, i]
            uc[j] = uj + rel * max(one(uj), abs(uj))
        end
        trajectory_rhs!(f, iip, fpc, uc, p, i, ti)
        if central
            for q in cols
                j = sp.color_cols[q]
                uj = u[j, i]
                uc[j] = uj - ((uj + rel * max(one(uj), abs(uj))) - uj)
            end
            fmc = view(fm, g, :)
            trajectory_rhs!(f, iip, fmc, uc, p, i, ti)
            for q in cols
                j = sp.color_cols[q]
                uj = u[j, i]
                h = (uj + rel * max(one(uj), abs(uj))) - uj
                for e in sp.colptr[j]:(sp.colptr[j + 1] - 1)
                    r = sp.rowval[e]
                    Jv[i, e] = (fpc[r] - fmc[r]) / (2 * h)
                end
            end
        else
            for q in cols
                j = sp.color_cols[q]
                uj = u[j, i]
                h = (uj + rel * max(one(uj), abs(uj))) - uj
                for e in sp.colptr[j]:(sp.colptr[j + 1] - 1)
                    r = sp.rowval[e]
                    Jv[i, e] = (fpc[r] - f0[r, i]) / h
                end
            end
        end
    end
end

# W = J - M / (dt γ) gathered into the slots, then factorized in place by the operation
# list. `d0` keeps the diagonal of the permuted W before elimination, for the pivot
# check; `pivot_min` keeps, per lane, the smallest |U[k, k]| / |W[k, k]| seen in the solve.
@kernel function lane_sparse_factor_kernel(
        Wv, d0, pivot_min, status, @Const(Jv), @Const(mass_diag), @Const(dt), @Const(gamma), @Const(lu)
    )
    i = @index(Global, Linear)
    @inbounds if status[i] == LANE_ACTIVE
        T = eltype(Wv)
        N = length(lu.piv)
        for s in 1:lu.nslot
            Wv[i, s] = zero(T)
        end
        for e in 1:length(lu.jslot)
            Wv[i, lu.jslot[e]] = Jv[i, e]
        end
        invh = inv(dt[i] * gamma)
        for k in 1:N
            s = lu.mslot[k]
            if s > 0
                Wv[i, s] -= _mass_diagonal(mass_diag, k, Wv) * invh
            end
        end
        for k in 1:N
            d0[i, k] = Wv[i, lu.piv[k]]
        end
        σ = one(T)
        for k in 1:N
            pk = Wv[i, lu.piv[k]]
            a = abs(d0[i, k])
            σ = min(σ, iszero(a) ? (iszero(pk) ? zero(T) : one(T)) : abs(pk) / a)
            ip = inv(pk)
            for q in lu.divptr[k]:(lu.divptr[k + 1] - 1)
                Wv[i, lu.div_t[q]] *= ip
            end
            for q in lu.fmaptr[k]:(lu.fmaptr[k + 1] - 1)
                ts = lu.fma_t[q]
                Wv[i, ts] = muladd(-Wv[i, lu.fma_l[q]], Wv[i, lu.fma_u[q]], Wv[i, ts])
            end
        end
        pivot_min[i] = min(pivot_min[i], σ)
        status[i] = lane_pivot_status(σ)
    end
end

# x = W \ x for every active lane, with the factors of `lane_sparse_factor_kernel`; `y` is
# B × N scratch. Forward and back substitution by columns (as the cooperative kernel).
@kernel function lane_sparse_solve_kernel(x, y, @Const(Wv), @Const(status), @Const(lu))
    i = @index(Global, Linear)
    @inbounds if status[i] == LANE_ACTIVE
        N = length(lu.piv)
        for r in 1:N
            y[i, r] = x[lu.rowo[r], i]
        end
        for k in 1:N
            yk = y[i, k]
            for q in lu.Lptr[k]:(lu.Lptr[k + 1] - 1)
                r = lu.Lrow[q]
                y[i, r] = muladd(-Wv[i, lu.Lslot[q]], yk, y[i, r])
            end
        end
        for k in N:-1:1
            yk = y[i, k] / Wv[i, lu.piv[k]]
            y[i, k] = yk
            for q in lu.Uptr[k]:(lu.Uptr[k + 1] - 1)
                r = lu.Urow[q]
                y[i, r] = muladd(-Wv[i, lu.Uslot[q]], yk, y[i, r])
            end
        end
        for r in 1:N
            x[lu.colo[r], i] = y[i, r]
        end
    end
end

# The same factorization and solve with `S` threads per lane (the first dimension of the
# workgroup) for batches too small to keep the device busy with a thread per lane: the
# updates of a pivot are independent, so they are spread over the lane's threads, with one
# barrier per pivot. Pivot k's updates use l = W[i, k] / U[k, k] as they go, and column k - 1
# is scaled in the same round (no update of pivot k reads or writes it). The values are
# indexed `[lane, slot]` like the per-lane kernels, but stored slot-major
# (`PermutedDimsArray`), so a lane's slots are contiguous. `B` is the number of lanes (the
# launch is padded to whole workgroups).
# The lane of a cooperative thread exists (the launch is padded) and is active; evaluated in
# every segment between barriers, as a value kept across `@synchronize` would have to be
# `@private` on the CPU backend.
@inline _coop_active(i, B, status) = i <= B && @inbounds(status[i] == LANE_ACTIVE)

@kernel function lane_sparse_factor_coop_kernel(
        Wv, d0, pivot_min, status, @Const(Jv), @Const(mass_diag), @Const(dt), @Const(gamma),
        @Const(lu), @Const(B)
    )
    s, i = @index(Global, NTuple)
    @uniform S = @groupsize()[1]
    @uniform N = length(lu.piv)
    @uniform T = eltype(Wv)
    @inbounds if _coop_active(i, B, status)
        for q in s:S:(lu.nslot)
            Wv[i, q] = zero(T)
        end
    end
    @synchronize
    @inbounds if _coop_active(i, B, status)
        for e in s:S:length(lu.jslot)
            Wv[i, lu.jslot[e]] = Jv[i, e]
        end
    end
    @synchronize
    @inbounds if _coop_active(i, B, status)
        invh = inv(dt[i] * gamma)
        for k in s:S:N
            q = lu.mslot[k]
            if q > 0
                Wv[i, q] -= _mass_diagonal(mass_diag, k, Wv) * invh
            end
        end
    end
    @synchronize
    @inbounds if _coop_active(i, B, status)
        for k in s:S:N
            d0[i, k] = Wv[i, lu.piv[k]]
        end
    end
    for k in 1:N
        @inbounds if _coop_active(i, B, status)
            ip = inv(Wv[i, lu.piv[k]])
            for q in (lu.fmaptr[k] + s - 1):S:(lu.fmaptr[k + 1] - 1)
                t = lu.fma_t[q]
                Wv[i, t] = muladd(-(Wv[i, lu.fma_l[q]] * ip), Wv[i, lu.fma_u[q]], Wv[i, t])
            end
            if k > 1
                ipp = inv(Wv[i, lu.piv[k - 1]])
                for q in (lu.divptr[k - 1] + s - 1):S:(lu.divptr[k] - 1)
                    Wv[i, lu.div_t[q]] *= ipp
                end
            end
        end
        @synchronize
    end
    @inbounds if _coop_active(i, B, status)
        ip = inv(Wv[i, lu.piv[N]])
        for q in (lu.divptr[N] + s - 1):S:(lu.divptr[N + 1] - 1)
            Wv[i, lu.div_t[q]] *= ip
        end
        if s == 1
            σ = one(T)
            for k in 1:N
                pk = Wv[i, lu.piv[k]]
                a = abs(d0[i, k])
                σ = min(σ, iszero(a) ? (iszero(pk) ? zero(T) : one(T)) : abs(pk) / a)
            end
            pivot_min[i] = min(pivot_min[i], σ)
            status[i] = lane_pivot_status(σ)
        end
    end
end

@kernel function lane_sparse_solve_coop_kernel(x, y, @Const(Wv), @Const(status), @Const(lu), @Const(B))
    s, i = @index(Global, NTuple)
    @uniform S = @groupsize()[1]
    @uniform N = length(lu.piv)
    @inbounds if _coop_active(i, B, status)
        for r in s:S:N
            y[i, r] = x[lu.rowo[r], i]
        end
    end
    @synchronize
    for k in 1:N
        @inbounds if _coop_active(i, B, status)
            yk = y[i, k]
            for q in (lu.Lptr[k] + s - 1):S:(lu.Lptr[k + 1] - 1)
                r = lu.Lrow[q]
                y[i, r] = muladd(-Wv[i, lu.Lslot[q]], yk, y[i, r])
            end
        end
        @synchronize
    end
    # y[k] keeps the value before the division by the pivot (every thread divides it itself).
    for k in N:-1:1
        @inbounds if _coop_active(i, B, status)
            yk = y[i, k] / Wv[i, lu.piv[k]]
            for q in (lu.Uptr[k] + s - 1):S:(lu.Uptr[k + 1] - 1)
                r = lu.Urow[q]
                y[i, r] = muladd(-Wv[i, lu.Uslot[q]], yk, y[i, r])
            end
        end
        @synchronize
    end
    @inbounds if _coop_active(i, B, status)
        for r in s:S:N
            x[lu.colo[r], i] = y[i, r] / Wv[i, lu.piv[r]]
        end
    end
end

# Choose the pivot order from the lanes' current Jacobians `st.J` and step sizes, and
# allocate the slots: the stepper with its `LaneSparseLU`.
function _lane_sparse_setup(st, PJ, mass)
    P = _lane_w_pattern(PJ, mass)
    Wabs = _lane_w_envelope(st.J, PJ, P, mass, st.dt, st.tab.gamma)
    rowo, colo = _lane_sparse_order(Wabs, P)
    lu = _lane_sparse_lu(P, PJ, rowo, colo)
    T = eltype(st.J)
    W = _lane_values(st.u, T, length(st.t), lu.nslot, st.lu_threads)
    st = @set st.lu = adapt(st.backend, lu)
    return @set st.W = W
end
