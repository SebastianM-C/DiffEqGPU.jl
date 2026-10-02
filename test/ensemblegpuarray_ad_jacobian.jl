using DiffEqGPU, ForwardDiff, LinearAlgebra, StaticArrays, Test
using Adapt: adapt
using OrdinaryDiffEq: OrdinaryDiffEq
using OrdinaryDiffEqRosenbrock: Rodas5P, Rosenbrock23
import SciMLBase
using SciMLBase: EnsembleProblem, ODEFunction, ODEProblem, remake, solve
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

include("utils.jl")

# Stiff ensembles of problems without a Jacobian: `EnsembleGPUArray` differentiates the batched
# right-hand side instead.

const mus = [10.0, 30.0, 100.0, 300.0]

function vdp!(du, u, p, t)
    du[1] = u[2]
    du[2] = p[1] * ((1 - u[1]^2) * u[2] - u[1])
    return nothing
end
function vdp_jac!(J, u, p, t)
    J[1, 1] = 0
    J[1, 2] = 1
    J[2, 1] = p[1] * (-2 * u[1] * u[2] - 1)
    J[2, 2] = p[1] * (1 - u[1]^2)
    return nothing
end
vdp(u, p, t) = SVector(u[2], p[1] * ((1 - u[1]^2) * u[2] - u[1]))

# A tabulated coefficient looked up with `searchsortedlast` and guarded by `isnan`, as in
# registered interpolation functions of symbolic models.
const XS = SVector(0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
const YS = SVector(1.0, 2.0, 0.5, 0.25, 3.0, 1.0)
function interp1(xs, ys, x)
    isnan(x) && return zero(x)
    i = clamp(searchsortedlast(xs, x), 1, length(xs) - 1)
    w = (x - xs[i]) / (xs[i + 1] - xs[i])
    return (1 - w) * ys[i] + w * ys[i + 1]
end
function tabulated!(du, u, p, t)
    du[1] = -p[1] * interp1(XS, YS, abs(u[2])) * u[1]
    du[2] = u[1] - u[2]^3
    return nothing
end

max_error(a, b) = maximum(
    maximum(maximum(abs, x .- y) for (x, y) in zip(s.u, r.u)) for (s, r) in zip(a, b)
)

function ensemble(prob, prob_func, alg; kwargs...)
    return solve(
        EnsembleProblem(prob; prob_func, safetycopy = false), alg,
        EnsembleGPUArray(backend, 0.0); trajectories = length(mus), kwargs...
    ).u
end

@testset "Batched forward-mode Jacobian" begin
    N, ntraj = 2, 5
    u = rand(N, ntraj) .+ 0.5
    p = reshape(collect(10.0:10.0:50.0), 1, :)
    for f! in (vdp!, tabulated!)
        batched_f = (du, u, p, t) -> DiffEqGPU.gpu_kernel(backend)(
            f!, du, u, p, t; ndrange = size(u, 2), workgroupsize = size(u, 2)
        )
        ud = adapt(backend, u)
        J = DiffEqGPU.ADBatchedJacobian(batched_f, ud; chunksize = 1)
        W = adapt(backend, zeros(N, N, ntraj))
        J(W, ud, adapt(backend, p), 0.0)
        Wh = Array(W)
        for i in 1:ntraj
            Jref = ForwardDiff.jacobian(
                (du, x) -> f!(du, x, p[:, i], 0.0), zeros(N), u[:, i]
            )
            @test Wh[:, :, i] ≈ Jref
        end
    end
end

function l96!(du, u, p, t)
    N = length(u)
    for k in 1:N
        du[k] = (u[mod1(k + 1, N)] - u[mod1(k - 2, N)]) * u[mod1(k - 1, N)] - u[k] + p[1]
    end
    return nothing
end

function batched_rhs(f, iip)
    kernel = iip ? DiffEqGPU.gpu_kernel : DiffEqGPU.gpu_kernel_oop
    return (du, u, p, t) -> kernel(backend)(
        f, du, u, p, t; ndrange = size(u, 2), workgroupsize = size(u, 2)
    )
end

# `params(i)` are trajectory `i`'s parameters as the reference sees them, and `scale(i)` the
# factor the batched right-hand side applies to its derivative.
function check_fd_jacobian(f, iip, N, u, p, params, scale; central, kwargs...)
    ntraj = size(u, 2)
    rtol = central ? 1.0e-8 : 1.0e-5
    ud = adapt(backend, u)
    J = DiffEqGPU.FDBatchedJacobian(f, Val(iip), batched_rhs(f, iip), ud; central, kwargs...)
    W = adapt(backend, zeros(N, N, ntraj))
    J(W, ud, adapt(backend, p), 0.0)
    Wh = Array(W)
    for i in 1:ntraj
        Jref = if iip
            ForwardDiff.jacobian((du, x) -> f(du, x, params(i), 0.0), zeros(N), u[:, i])
        else
            ForwardDiff.jacobian(x -> f(x, params(i), 0.0), u[:, i])
        end
        @test isapprox(Wh[:, :, i], scale(i) * Jref; rtol)
    end
    # The state is left as it was.
    @test Array(ud) == u
    return J
end

@testset "Batched finite-difference Jacobian" begin
    N, ntraj = 2, 5
    u = rand(N, ntraj) .+ 0.5
    p = reshape(collect(10.0:10.0:50.0), 1, :)
    for central in (false, true)
        for f! in (vdp!, tabulated!)
            check_fd_jacobian(f!, true, N, u, p, i -> p[:, i], i -> 1; central)
        end
        # Out-of-place right-hand side.
        check_fd_jacobian(vdp, false, N, u, p, i -> p[:, i], i -> 1; central)
        # Per-trajectory time spans: the right-hand side is scaled by the span's length.
        tspans = [(0.0, 1.0 + i) for i in 1:ntraj]
        pw = [DiffEqGPU.ParamWrapper((p[1, i],), tspans[i]) for i in 1:ntraj]
        check_fd_jacobian(
            vdp!, true, N, u, pw, i -> (p[1, i],), i -> tspans[i][2] - tspans[i][1]; central
        )
    end
end

@testset "Finite-difference Jacobian in several launches" begin
    N, ntraj = 5, 3
    u = rand(N, ntraj) .+ 0.5
    p = reshape([8.0, 9.0, 10.0], 1, :)
    for central in (false, true)
        # The memory budget allows one column per launch, the thread target two, and by
        # default all of them fit in one launch.
        for (kwargs, ncols) in (
                ((; scratch_bytes = 1), 1), ((; target_threads = 2 * ntraj), 2), ((;), N),
            )
            J = check_fd_jacobian(l96!, true, N, u, p, i -> p[:, i], i -> 1; central, kwargs...)
            @test J.ncols == ncols
        end
    end
end

@testset "The Jacobian follows the algorithm's `autodiff`" begin
    ADTypes = DiffEqGPU.ADTypes
    u0 = adapt(backend, zeros(3, 4))
    f = (du, u, p, t) -> nothing
    jac(ad) = DiffEqGPU.batched_jacobian(f, u0, (; autodiff = ad))
    chunk(J) = ForwardDiff.npartials(eltype(J.ud))
    @test chunk(jac(ADTypes.AutoForwardDiff())) == ForwardDiff.pickchunksize(3, 8)
    @test chunk(jac(ADTypes.AutoForwardDiff(; chunksize = 1))) == 1
    @test chunk(jac(ADTypes.AutoSparse(ADTypes.AutoForwardDiff(; chunksize = 2)))) == 2
    @test !jac(ADTypes.AutoFiniteDiff()).central
    @test jac(ADTypes.AutoFiniteDiff(; fdtype = Val(:central))).central
    @test_throws ArgumentError jac(ADTypes.AutoFiniteDiff(; fdtype = Val(:complex)))
    @test_throws ArgumentError jac(ADTypes.AutoEnzyme())
end

@testset "Stiff ODE without a Jacobian, $(nameof(typeof(alg))) with $name" for alg in (
        Rodas5P, Rosenbrock23,
    ), (name, ad, err) in (
        ("chunk 1", DiffEqGPU.ADTypes.AutoForwardDiff(; chunksize = 1), 1.0e-7),
        ("forward differences", DiffEqGPU.ADTypes.AutoFiniteDiff(), 1.0e-4),
        ("central differences", DiffEqGPU.ADTypes.AutoFiniteDiff(; fdtype = Val(:central)), 1.0e-5),
    )
    prob_func = (pr, ctx) -> remake(pr; p = [mus[ctx.sim_id]])
    kwargs = (; abstol = 1.0e-8, reltol = 1.0e-8, saveat = 0.1)
    analytic = ensemble(
        ODEProblem(ODEFunction(vdp!; jac = vdp_jac!), [2.0, 0.0], (0.0, 2.0), [10.0]),
        prob_func, alg(); kwargs...
    )
    sol = ensemble(
        ODEProblem(vdp!, [2.0, 0.0], (0.0, 2.0), [10.0]), prob_func, alg(; autodiff = ad);
        kwargs...
    )
    @test all(s -> SciMLBase.successful_retcode(s), sol)
    @test max_error(sol, analytic) < err
end

@testset "Stiff ODE without a Jacobian, $(nameof(typeof(alg)))" for alg in (
        Rodas5P(), Rosenbrock23(),
    )
    prob_func = (pr, ctx) -> remake(pr; p = [mus[ctx.sim_id]])
    kwargs = (; abstol = 1.0e-8, reltol = 1.0e-8, saveat = 0.1)
    analytic = ensemble(
        ODEProblem(ODEFunction(vdp!; jac = vdp_jac!), [2.0, 0.0], (0.0, 2.0), [10.0]),
        prob_func, alg; kwargs...
    )
    ad = ensemble(ODEProblem(vdp!, [2.0, 0.0], (0.0, 2.0), [10.0]), prob_func, alg; kwargs...)
    @test all(s -> SciMLBase.successful_retcode(s), ad)
    @test max_error(ad, analytic) < 1.0e-7

    # Out-of-place right-hand side.
    oop = ensemble(
        ODEProblem{false}(vdp, SVector(2.0, 0.0), (0.0, 2.0), SVector(10.0)),
        (pr, ctx) -> remake(pr; p = SVector(mus[ctx.sim_id])), alg; kwargs...
    )
    @test max_error(oop, analytic) < 1.0e-7

    # Float32. The default initial step makes this problem fail in Float32 with an analytic
    # Jacobian too, so give one.
    prob_func32 = (pr, ctx) -> remake(pr; p = Float32[mus[ctx.sim_id]])
    kwargs32 = (; abstol = 1.0f-5, reltol = 1.0f-5, saveat = 0.1f0, dt = 1.0f-4)
    analytic32 = ensemble(
        ODEProblem(ODEFunction(vdp!; jac = vdp_jac!), Float32[2, 0], (0.0f0, 2.0f0), Float32[10]),
        prob_func32, alg; kwargs32...
    )
    ad32 = ensemble(
        ODEProblem(vdp!, Float32[2, 0], (0.0f0, 2.0f0), Float32[10]), prob_func32, alg;
        kwargs32...
    )
    @test all(s -> SciMLBase.successful_retcode(s), ad32)
    @test eltype(ad32[1].u[end]) == Float32
    # Rounding differences between the two Jacobians change the Float32 step sequence, so
    # compare their errors against the Float64 solution rather than each other.
    @test max_error(ad32, analytic) < 2 * max_error(analytic32, analytic) + 1.0e-3
end

@testset "Tabulated right-hand side without a Jacobian" begin
    prob = ODEProblem(tabulated!, [1.0, 0.2], (0.0, 3.0), [1.0])
    prob_func = (pr, ctx) -> remake(pr; p = [mus[ctx.sim_id] / 10])
    sol = ensemble(prob, prob_func, Rodas5P(); abstol = 1.0e-8, reltol = 1.0e-8, saveat = 0.1)
    cpu = [
        solve(prob_func(prob, (; sim_id = i)), Rodas5P(); abstol = 1.0e-8, reltol = 1.0e-8, saveat = 0.1)
            for i in eachindex(mus)
    ]
    @test all(s -> SciMLBase.successful_retcode(s), sol)
    @test max_error(sol, cpu) < 1.0e-6
end

# Index-1 DAE: 2 x' = -k x, 0 = y + sin(y) - x, z' = x - z, with a consistent start.
function dae3!(du, u, p, t)
    du[1] = -p[1] * u[1]
    du[2] = u[2] + sin(u[2]) - u[1]
    du[3] = u[1] - u[3]
    return nothing
end
const y0 = 0.5109734293885691

@testset "Mass-matrix DAE without a Jacobian, $(nameof(typeof(alg)))" for (alg, tol) in (
        (Rodas5P(), 1.0e-7), (Rosenbrock23(), 1.0e-6),
    )
    prob = ODEProblem(
        ODEFunction(dae3!; mass_matrix = Diagonal([2.0, 0.0, 1.0])),
        [1.0, y0, 0.0], (0.0, 2.0), [1.0]
    )
    prob_func = (pr, ctx) -> remake(pr; p = [mus[ctx.sim_id] / 100])
    kwargs = (; abstol = 1.0e-8, reltol = 1.0e-8, saveat = 0.1)
    sol = ensemble(prob, prob_func, alg; kwargs...)
    cpu = [solve(prob_func(prob, (; sim_id = i)), alg; kwargs...) for i in eachindex(mus)]
    @test all(s -> SciMLBase.successful_retcode(s), sol)
    @test max_error(sol, cpu) < tol
    @test maximum(maximum(u -> abs(u[2] + sin(u[2]) - u[1]), s.u) for s in sol) < 1.0e-6
end

@testset "ModelingToolkit DAE without a Jacobian" begin
    @variables x(t) = 1.0 y(t)
    @parameters k = 1.0
    @named sys = System([D(x) ~ -k * x, 0 ~ y + sin(y) - x], t; guesses = [y => 0.5])
    # `split = false` gives a plain parameter vector, which `EnsembleGPUArray` batches.
    csys = mtkcompile(sys; split = false)
    prob = ODEProblem(csys, [], (0.0, 2.0))
    @test !SciMLBase.has_jac(prob.f)
    # `EnsembleGPUArray` does not initialize the trajectories: start from a consistent state.
    u0 = solve(prob, Rodas5P(); abstol = 1.0e-10, reltol = 1.0e-10).u[1]
    setter = ModelingToolkit.SymbolicIndexingInterface.setsym_oop(prob, [k])
    prob_func = (pr, ctx) -> remake(pr; u0, p = setter(pr, [mus[ctx.sim_id] / 100])[2])
    ix = ModelingToolkit.SymbolicIndexingInterface.variable_index(prob, x)
    iy = ModelingToolkit.SymbolicIndexingInterface.variable_index(prob, y)
    kwargs = (; abstol = 1.0e-8, reltol = 1.0e-8, saveat = 0.1)
    sol = ensemble(prob, prob_func, Rodas5P(); kwargs...)
    cpu = [solve(prob_func(prob, (; sim_id = i)), Rodas5P(); kwargs...) for i in eachindex(mus)]
    @test all(s -> SciMLBase.successful_retcode(s), sol)
    @test max_error(sol, cpu) < 1.0e-7
    @test maximum(maximum(u -> abs(u[iy] + sin(u[iy]) - u[ix]), s.u) for s in sol) < 1.0e-6
end
