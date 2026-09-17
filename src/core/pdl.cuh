#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

// Programmatic dependent launch lowers to griddepcontrol.{launch_dependents,wait}, which first
// appear in sm_90. A target without that hardware keeps this source unchanged: the launch loses
// the programmatic-stream-serialization attribute (so full stream ordering applies again) and the
// device hooks compile to nothing. Only the producer/consumer overlap is lost, never correctness.
#if !defined(NINFER_ENABLE_PDL)
#  define NINFER_ENABLE_PDL 0
#endif

// The device pass must decide on its own architecture; a host pass would not have __CUDA_ARCH__.
#if defined(__CUDA_ARCH__)
#  define NINFER_PDL_DEVICE (__CUDA_ARCH__ >= 900)
#else
#  define NINFER_PDL_DEVICE (NINFER_ENABLE_PDL != 0)
#endif

namespace ninfer::pdl {

struct LaunchConfig {
    dim3 grid;
    dim3 block;
    std::size_t dynamic_smem_bytes = 0;
    cudaStream_t stream            = nullptr;
};

// Launches a consumer kernel as a programmatic dependent of the immediately preceding producer
// kernel in the same stream. Every consumer control path that reads producer output must first call
// wait_for_dependencies().
template <class... KernelArgs, class... CallArgs>
[[nodiscard]] inline cudaError_t
launch_dependent(const LaunchConfig& launch, void (*kernel)(KernelArgs...), CallArgs&&... args) {
    cudaLaunchConfig_t config{};
    config.gridDim          = launch.grid;
    config.blockDim         = launch.block;
    config.dynamicSmemBytes = launch.dynamic_smem_bytes;
    config.stream           = launch.stream;

#if NINFER_ENABLE_PDL
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    config.attrs    = &attribute;
    config.numAttrs = 1;
#endif

    return cudaLaunchKernelEx(&config, kernel, std::forward<CallArgs>(args)...);
}

// Every producer CTA must call this at least once or exit. This enables dependent scheduling but
// does not make producer writes visible to the consumer.
__device__ __forceinline__ void trigger_dependents() {
#if NINFER_PDL_DEVICE
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// Call on every consumer control path before its first access to producer-dependent data.
__device__ __forceinline__ void wait_for_dependencies() {
#if NINFER_PDL_DEVICE
    cudaGridDependencySynchronize();
#endif
}

} // namespace ninfer::pdl
