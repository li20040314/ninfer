#include "ops/linear_add/q4/q4_linear_add_dispatch.h"

#include "ops/linear/q4/q4_gemv_launch.cuh"
#include "ops/linear/q4/q4_ksplit_mma.cuh"
#include "ops/linear/q4/q4_mma_launch.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

struct ResidualEpilogue {
    __device__ __forceinline__ void operator()(__nv_bfloat16* destination, float value) const {
        *destination = __float2bfloat16_rn(__bfloat162float(*destination) + value);
    }
};

struct GemvResidualEpilogue {
    template <bool SplitOutput, int SplitRow>
    __device__ __forceinline__ void operator()(__nv_bfloat16* out, __nv_bfloat16*, int row,
                                               float value) const {
        static_assert(!SplitOutput);
        ResidualEpilogue{}(out + row, value);
    }
};

struct KSplitResidualEpilogue {
    __nv_bfloat16* residual;
    std::int32_t tokens;

    template <int Capacity>
    __device__ __forceinline__ void store(int row, int col, float4 value) const {
        if (col < tokens) {
            ResidualEpilogue{}(residual + static_cast<std::int64_t>(col) * 5120 + row, value.x);
            ResidualEpilogue{}(residual + static_cast<std::int64_t>(col) * 5120 + row + 8, value.z);
        }
        if (col + 1 < tokens) {
            ResidualEpilogue{}(residual + static_cast<std::int64_t>(col + 1) * 5120 + row, value.y);
            ResidualEpilogue{}(residual + static_cast<std::int64_t>(col + 1) * 5120 + row + 8,
                               value.w);
        }
    }
};

using GemvR1W8 =
    Q4RowSplitGemvSchedule<1, 8, 16, 1, Q4GemvActivationAccess::Direct,
                           Q4GemvLaneMapping::PackedByte2, Q4GemvDecodeMode::ScalarInteger,
                           Q4GemvCodeTransfer::SyncVector16, Q4GemvScaleAccess::Scalar16Shuffle,
                           Cache::ca, 6144 / 64, 1>;
// K=17408 (Qwen3.8-27B mlp/down): 272 groups per row split across 17 warps into one
// 16-group tile per warp; 136 scale pairs divide evenly across the warps.
using GemvR1W8K17408 =
    Q4RowSplitGemvSchedule<1, 17, 16, 1, Q4GemvActivationAccess::Direct,
                           Q4GemvLaneMapping::PackedByte2, Q4GemvDecodeMode::ScalarInteger,
                           Q4GemvCodeTransfer::SyncVector16, Q4GemvScaleAccess::Scalar16Shuffle,
                           Cache::ca, 17408 / 64, 1>;
using MmaR32C32  = Q4RowSplitMmaGemmSchedule<32, 32, 64, 16, 16, 3, 2, Q4FragmentPipeline::Serial,
                                             Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR32C64  = Q4RowSplitMmaGemmSchedule<32, 64, 64, 16, 32, 3, 2, Q4FragmentPipeline::Serial,
                                             Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR64C128 = Q4RowSplitMmaGemmSchedule<64, 128, 64, 64, 32, 2, 1, Q4FragmentPipeline::Serial,
                                             Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;

template <int InputRows, int Capacity>
void launch_ksplit(const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream) {
    using Geometry = Q4LinearGeometry<5120, InputRows>;
    auto* output   = static_cast<__nv_bfloat16*>(residual.data);
    q4_ksplit_mma_kernel<Geometry, (Capacity + 7) / 8 * 8, Capacity, KSplitResidualEpilogue,
                         Q4KSplitIdentityRows, true>
        <<<5120 / Q4KSplitMmaSchedule::kRowsPerCta, Q4KSplitMmaSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), output,
            KSplitResidualEpilogue{output, x.ne[1]}, {}, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

Q4LinearAddLaunch select_q4_linear_add_5120_6144(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_gemv<GemvR1W8, GemvResidualEpilogue>;
    if (tokens <= 4) return launch_ksplit<6144, 4>;
    if (tokens <= 8) return launch_ksplit<6144, 8>;
    if (tokens <= 16) return launch_ksplit<6144, 16>;
    if (tokens <= 24) return launch_ksplit<6144, 24>;
    if (tokens <= 32) return launch_ksplit<6144, 32>;
    if (tokens <= 96) return launch_q4_mma<MmaR32C32, ResidualEpilogue>;
    if (tokens <= 192) return launch_q4_mma<MmaR32C64, ResidualEpilogue>;
    return launch_q4_mma<MmaR64C128, ResidualEpilogue>;
}

Q4LinearAddLaunch select_q4_linear_add_5120_17408(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_gemv<GemvR1W8K17408, GemvResidualEpilogue>;
    if (tokens <= 4) return launch_ksplit<17408, 4>;
    if (tokens <= 8) return launch_ksplit<17408, 8>;
    if (tokens <= 16) return launch_ksplit<17408, 16>;
    if (tokens <= 24) return launch_ksplit<17408, 24>;
    if (tokens <= 32) return launch_ksplit<17408, 32>;
    if (tokens <= 96) return launch_q4_mma<MmaR32C32, ResidualEpilogue>;
    if (tokens <= 192) return launch_q4_mma<MmaR32C64, ResidualEpilogue>;
    return launch_q4_mma<MmaR64C128, ResidualEpilogue>;
}

} // namespace

Q4LinearAddLaunch select_q4_linear_add(std::int32_t rows, std::int32_t k, std::int32_t tokens) {
    if (tokens <= 0) {
        throw std::invalid_argument("q4 linear_add: unsupported shape or token extent");
    }
    if (rows == 5120 && k == 6144) { return select_q4_linear_add_5120_6144(tokens); }
    // Qwen3.8-27B mlp/down (hidden_size 5120, intermediate_size 17408).
    if (rows == 5120 && k == 17408) { return select_q4_linear_add_5120_17408(tokens); }
    throw std::invalid_argument("q4 linear_add: unsupported shape or token extent");
}

} // namespace ninfer::ops::detail
