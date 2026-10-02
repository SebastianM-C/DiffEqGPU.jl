# Host-side initialization for `EnsembleGPUArray`.
#
# Each trajectory's problem is initialized on the host before the batch is stacked, as the
# CPU solve would do at `init`. The batched problem carries no initialization data, so the
# batched solve itself runs without initialization.

const NonlinearSolveBase = SimpleNonlinearSolve.NonlinearSolveBase

_threaded_host(::EnsembleArrayAlgorithm) = false
_threaded_host(ensemblealg::EnsembleGPUArray) = ensemblealg.threaded_host

# The initialization algorithm a trajectory is initialized with on the host, from the
# `initializealg` keyword of the ensemble solve.
_host_initializealg(::Nothing) = DiffEqBase.DefaultInit()
_host_initializealg(alg::Union{DiffEqBase.DefaultInit, SciMLBase.OverrideInit, SciMLBase.NoInit, CheckInit}) = alg
function _host_initializealg(alg)
    throw(
        ArgumentError(
            "`initializealg = $(nameof(typeof(alg)))()` is not supported by `EnsembleGPUArray`, which initializes each trajectory on the host. Use `OverrideInit` (the default for problems with initialization data, such as ModelingToolkit problems), `CheckInit`, or `NoInit` with consistent initial conditions."
        )
    )
end

# Whether the batched solve has to be told not to initialize: the host already did, or the
# user chose an algorithm, which the host consumed.
function _host_initialized(probs, initializealg)
    return initializealg !== nothing ||
        any(prob -> SciMLBase.has_initialization_data(prob.f), probs)
end

_tolerance(tol::Number, ::Type{T}) where {T} = convert(T, tol)
_tolerance(tol::AbstractArray, ::Type{T}) where {T} = convert(T, minimum(tol))

# OrdinaryDiffEq initializes with the integrator's tolerances, so the defaults are the ODE
# solve's defaults.
function _initialization_tolerances(prob, alg, abstol, reltol)
    T = real(eltype(prob.u0))
    T <: AbstractFloat || (T = Float64)
    _abstol = alg isa SciMLBase.OverrideInit && alg.abstol !== nothing ? alg.abstol :
        something(abstol, 1.0e-6)
    _reltol = alg isa SciMLBase.OverrideInit && alg.reltol !== nothing ? alg.reltol :
        something(reltol, 1.0e-3)
    return _tolerance(_abstol, T), _tolerance(_reltol, T)
end

_default_initialization_nlsolve(::SciMLBase.NonlinearLeastSquaresProblem) = SimpleGaussNewton()
_default_initialization_nlsolve(_) = SimpleTrustRegion()

# Under `AutoSpecialize`/`AutoDespecialize` the initialization function is wrapped in
# function wrappers whose signatures only cover the duals and parameter wrappers that
# NonlinearSolve.jl uses, so the SimpleNonlinearSolve solvers cannot call it. Solve with
# the raw function instead.
function _raw_function(f)
    @static if isdefined(NonlinearSolveBase, :get_raw_f)
        f = NonlinearSolveBase.get_raw_f(f)
    end
    return SciMLBase.unwrapped_f(f)
end

_unwrap_initialization_problem(initprob) = initprob
function _unwrap_initialization_problem(
        initprob::Union{SciMLBase.NonlinearProblem, SciMLBase.NonlinearLeastSquaresProblem}
    )
    raw = _raw_function(initprob.f.f)
    raw === initprob.f.f && return initprob
    return remake(initprob; f = SciMLBase.unwrapped_f(initprob.f, raw))
end

function _unwrapped_initialization_function(f)
    initdata = f.initialization_data
    initprob = _unwrap_initialization_problem(initdata.initializeprob)
    initprob === initdata.initializeprob && return f
    return @set f.initialization_data.initializeprob = initprob
end

# The residual of the algebraic equations of a mass-matrix problem, checked like `CheckInit`.
function _check_initialization(prob, abstol)
    M = prob.f.mass_matrix
    M isa LinearAlgebra.UniformScaling && return true
    algebraic = [all(iszero, row) for row in eachrow(M)]
    any(algebraic) || return true
    du = if SciMLBase.isinplace(prob)
        du = similar(prob.u0)
        prob.f(du, prob.u0, prob.p, prob.tspan[1])
        du
    else
        prob.f(prob.u0, prob.p, prob.tspan[1])
    end
    return LinearAlgebra.norm(vec(du)[algebraic]) <= abstol
end

"""
    _initialize_trajectory(prob, initializealg, abstol, reltol)

Initialize one trajectory's problem on the host. Returns the problem with the initialized
`u0` and `p`, and whether initialization succeeded.
"""
function _initialize_trajectory(prob, initializealg, abstol, reltol)
    alg = _host_initializealg(initializealg)
    alg isa SciMLBase.NoInit && return prob, true
    has_initdata = SciMLBase.has_initialization_data(prob.f)
    if alg isa CheckInit || (alg isa DiffEqBase.DefaultInit && !has_initdata)
        _abstol, _ = _initialization_tolerances(prob, alg, abstol, reltol)
        return prob, _check_initialization(prob, _abstol)
    end
    has_initdata || return prob, true

    override = alg isa SciMLBase.OverrideInit ? alg : SciMLBase.OverrideInit()
    _abstol, _reltol = _initialization_tolerances(prob, override, abstol, reltol)
    f = _unwrapped_initialization_function(prob.f)
    initprob = f.initialization_data.initializeprob
    nlsolve_alg = something(override.nlsolve, _default_initialization_nlsolve(initprob))
    u0, p, success = try
        SciMLBase.get_initial_values(
            prob, prob, f, override, Val(SciMLBase.isinplace(prob));
            nlsolve_alg, abstol = _abstol, reltol = _reltol
        )
    catch err
        # The SimpleNonlinearSolve solvers throw on a singular Jacobian instead of
        # returning a failed retcode.
        err isa Union{LinearAlgebra.SingularException, LinearAlgebra.LAPACKException} ||
            rethrow()
        nothing, nothing, false
    end
    success && all(isfinite, u0) || return prob, false
    return remake(prob; u0, p, lazy_initialization = true), true
end

# Build every trajectory's problem with `prob_func` and initialize it, on several threads
# when the ensemble algorithm asks for it.
function _prepare_trajectories(
        ensembleprob, ensemblealg, I, sim_seeds, rng_func, master_rng;
        initializealg = nothing, abstol = nothing, reltol = nothing, kwargs...
    )
    # Contexts draw from the master RNG, so they are made in order on this thread.
    ctxs = [_make_ensemble_context(i, sim_seeds, rng_func, master_rng) for i in I]
    prepare = function (ctx)
        prob = ensembleprob.safetycopy ? deepcopy(ensembleprob.prob) : ensembleprob.prob
        return _initialize_trajectory(
            ensembleprob.prob_func(prob, ctx), initializealg, abstol, reltol
        )
    end
    results = if _threaded_host(ensemblealg) && Threads.nthreads() > 1 && length(I) > 1
        out = Vector{Any}(undef, length(ctxs))
        Threads.@threads for k in eachindex(ctxs)
            out[k] = prepare(ctxs[k])
        end
        identity.(out)
    else
        map(prepare, ctxs)
    end
    return map(first, results), map(last, results)
end

# The solution returned for a trajectory whose initialization failed, typed like the
# solutions of the batch so the ensemble result keeps a concrete element type.
function _initial_failure_solution(prob, alg)
    u0 = reshape(Array(prob.u0), :, 1)
    return SciMLBase.build_solution(
        prob, alg, [prob.tspan[1]], [@view(u0[:, 1])];
        stats = SciMLBase.DEStats(0), retcode = ReturnCode.InitialFailure
    )
end
