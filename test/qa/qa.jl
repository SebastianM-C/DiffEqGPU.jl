using SciMLTesting, DiffEqGPU
using JET
using Test

# ExplicitImports only sees an extension once its trigger package is loaded, so every
# trigger is loaded here to bring all nine `ext/` modules into the checked module set.
# None of them needs a device to load: they only add `maxthreads`/`maybe_prefer_blocks`
# (and `lufact!`/`EnsembleGPUArray`) methods, so precompiling and importing them works on
# a plain CPU runner. Metal logs "only supported on Apple Silicon" at init off macOS but
# still loads, so it is not a device check.
using AMDGPU, CUDA, JLArrays, Metal, OpenCL, oneAPI
# The non-backend extensions: none of these needs a device either.
using DiffEqCallbacks, Enzyme, ModelingToolkitBase

# ExplicitImports silently skips an extension that fails to load, so assert the
# extension modules actually exist rather than trusting a green run_qa.
@testset "Extensions loaded" begin
    for ext in (
            :AMDGPUExt, :CUDAExt, :JLArraysExt, :MetalExt, :OpenCLExt, :oneAPIExt,
            :DiffEqCallbacksExt, :EnzymeExt, :ModelingToolkitBaseExt,
        )
        @test Base.get_extension(DiffEqGPU, ext) !== nothing
    end
end

const REEXPORTED_API = (
    :BrownFullBasicInit,
    :CheckInit,
    :EnsembleDistributed,
    :EnsembleProblem,
    :EnsembleSerial,
    :EnsembleSolution,
    :EnsembleThreads,
    :terminate!,
)

run_qa(
    DiffEqGPU;
    reexports_allow = REEXPORTED_API,
    ei_kwargs = (;
        # StaticVecOrMat is re-exported by StaticArrays but owned by StaticArraysCore.
        # It is a non-public type alias used only in method-signature dispatch for the
        # vendored GPU linear-solve kernels; importing it from its true owner would
        # still leave it non-public, so the via-owners exception is the natural place.
        all_explicit_imports_via_owners = (;
            # BlockedArray is re-exported by ModelingToolkitBase but owned by BlockArrays,
            # which DiffEqGPU does not depend on; the MTK extension imports it from its
            # trigger package.
            ignore = (:StaticVecOrMat, :BlockedArray),
        ),
        # `state_values` is owned by SymbolicIndexingInterface (not a DiffEqGPU dependency)
        # and reached through SciMLBase, which re-exports it.
        all_qualified_accesses_via_owners = (; ignore = (:state_values,)),
        # Non-public names accessed qualified from upstream packages. These are genuine
        # internal/extension APIs; they will drop out of this list as those packages
        # declare `public` (verified flagged against the registered releases on Julia
        # 1.12, where these checks run).
        all_qualified_accesses_are_public = (;
            ignore = (
                # SciMLBase callback/rootfind/ensemble internals (not yet `public`)
                :AbstractContinuousCallback, :AbstractDiscreteCallback,
                :DEFAULT_REDUCTION, :FINALIZE_DEFAULT, :INITIALIZE_DEFAULT,
                :LeftRootFind, :NoRootFind, :RootfindOpt, :build_linear_solution,
                :default_rng_func, :generate_sim_seeds, :solve_batch,
                :is_trivial_initialization, :specialization, :tighten_container_eltype,
                # ForwardDiff differentiation API (documented but not `public`)
                :Chunk, :Dual, :Partials, :construct_seeds, :derivative, :jacobian,
                :npartials, :partials, :pickchunksize, :value,
                # LinearSolve cache/algorithm extension interface (not `public`)
                :LinearCache, :SciMLLinearSolveAlgorithm, :init_cacheval,
                :needs_concrete_A,
                # SimpleDiffEq Tsit5 tableau-cache internals (not `public`)
                :_build_atsit5_caches, :_build_tsit5_caches, :bθs,
                # LinearAlgebra Hermitian/Symmetric union (stdlib, not `public`)
                :HermOrSym,
                # Core compiler inference used to size a Channel/Vector eltype; no
                # public cross-version replacement (Base.infer_return_type is 1.11+,
                # and the LTS floor is Julia 1.10).
                :Compiler, :return_type,
                # NonlinearSolveBase accessor for the function an AutoSpecialize wrapper
                # holds, used to solve initialization problems with SimpleNonlinearSolve.
                :get_raw_f,
                # CUDA batched LU used by DiffEqGPU.lufact!. CUDA's cuBLAS wrappers
                # are not `public`, and there is no public batched-getrf spelling.
                :getrf_strided_batched!,
                # Enzyme rule interface (EnzymeRules is not `public`), and the DiffEqGPU
                # internal transfer function the Enzyme extension writes rules for.
                :augmented_primal, :reverse, :_kernel_transfer,
                # DiffEqGPU's batched-callback hooks that the DiffEqCallbacks and
                # ModelingToolkitBase extensions add methods to or call.
                :batched_affect, :batched_time_callback, :check_device_affect,
                # DiffEqGPU's opt-out hook for the cooperative batched `ldiv!`, which the
                # JLArrays extension turns off.
                :cooperative_ldiv,
                # DiffEqGPU's ModelingToolkit hooks and internal problem types that the
                # ModelingToolkitBase extension implements or builds.
                :ImmutableSCCBlock, :ImmutableSCCNonlinearProblem, :final_parameters,
                :lower_initialization_problem, :make_initialization_maps_compatible,
                :make_parameter_compatible, :pack_parameters,
                # Reached through ModelingToolkitBase/SciMLBase by the MTK extension:
                # the SciMLStructures module and SymbolicIndexingInterface's
                # `state_values` (neither re-exporter marks them `public`).
                :SciMLStructures, :state_values,
                # Running ModelingToolkit event affects per trajectory: DiffEqGPU's
                # per-trajectory stand-in for the integrator, which the extension gives
                # SymbolicIndexingInterface methods; the SymbolicIndexingInterface and
                # JumpProcesses modules as reached through ModelingToolkitBase, and the
                # setter types the extension classifies (SetStateIndex,
                # SetParameterIndex); and the ModelingToolkitBase types that event
                # transforms receive and inspect. None is marked `public` upstream.
                :FakeIntegrator, :SymbolicIndexingInterface, :JumpProcesses,
                :SetStateIndex, :SetParameterIndex, :AbstractCallback, :ImplicitAffect,
                :ParameterIndex,
            ),
        ),
        # Non-public names imported explicitly from upstream packages. The
        # StaticArrays/StaticArraysCore internals back the vendored GPU LU/linsolve
        # kernels; `setindex` is the immutable Base helper used by the GPU LU pivot.
        all_explicit_imports_are_public = (;
            ignore = (
                :var"@_inline_meta", :LU, :StaticLUMatrix, :StaticVecOrMat,
                :StaticMatrix, :StaticVector, :similar_type, :setindex,
                # DiffEqCallbacks' PeriodicCallback affect type, by which the extension
                # recognizes a PeriodicCallback and reads its affect and period;
                # DiffEqCallbacks does not mark it `public`.
                :PeriodicCallbackAffect,
                # BlockArrays' BlockedArray as re-exported by ModelingToolkitBase, the
                # storage of split MTKParameters; the extension's only route to it.
                :BlockedArray,
            ),
        ),
    ),
)
