#include "core/weight.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_geometry.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "core/device.h" // CUDA_CHECK
#include "ops/linear/q4/q4_ksplit_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ninfer::ops::detail {
namespace {

// The shape-specialised kernels bake the fused gate/up width and the contraction width into
// template arguments, so each registered geometry needs its own instantiation. Taking the numbers
// from the geometry table keeps that table the single source of truth: a profile cannot drift from
// the tuple the plan admits.
//
// The parameters are plain integers even though the geometry is a struct. nvcc's host stub
// generation cannot emit the stub for a `__global__` whose template arguments carry a class-type
// non-type template parameter, so the geometry stops at the profile boundary.
template <int GateUpRows_, int InputRows_>
struct Q4SwiGluProfile {
    static constexpr int kN            = GateUpRows_;
    static constexpr int kK            = InputRows_;
    static constexpr int kIntermediate = kN / 2;
    static constexpr int kGroupK       = 64;
    static constexpr int kGroups       = kK / kGroupK;
    static constexpr int kBytesPerGroup     = 32;
    static constexpr int kVecBytes          = 16;
    static constexpr int kGroupsPerWarpTile = 16;
    static constexpr int kVecsPerWarpTile   = kGroupsPerWarpTile * kBytesPerGroup / kVecBytes;
    static constexpr int kWarpsPerBlock     = 4;
    static constexpr int kBlockThreads      = kWarpsPerBlock * 32;
    static constexpr int kPairsPerBlock     = kWarpsPerBlock;
    static constexpr int kXVecs             = kK / 8; // x as uint4 (8 bf16 each)
    static constexpr int kTiles             = kGroups / kGroupsPerWarpTile;
    static_assert(kIntermediate % kPairsPerBlock == 0);
    static_assert(kBytesPerGroup == 2 * kVecBytes);
    static_assert(kGroups % kGroupsPerWarpTile == 0);
    static_assert(kVecsPerWarpTile == 32);
};

using SwiGlu27BProfile = Q4SwiGluProfile<kQ4SwiGlu27B.gate_up_rows, kQ4SwiGlu27B.k>;
using SwiGlu9BProfile  = Q4SwiGluProfile<kQ4SwiGlu9B.gate_up_rows, kQ4SwiGlu9B.k>;

template <class Profile>
struct Q4SwiGluSmallTGeometry {
    static constexpr int kInputRows    = Profile::kK;
    static constexpr int kGroupsPerRow = Profile::kK / Profile::kGroupK;
};

template <class Profile>
struct Q4SwiGluSmallTRows {
    static constexpr int kOutputRowsPerCta = 8;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + (local_row & 7) + (local_row >= 8 ? Profile::kIntermediate : 0);
    }
};

template <class Profile>
struct Q4SwiGluSmallTEpilogue {
    __nv_bfloat16* out;
    int columns;

    template <int ActiveCols>
    __device__ __forceinline__ void store(int row, int col0, float4 projected) const {
        if (col0 < columns) {
            out[static_cast<std::int64_t>(col0) * Profile::kIntermediate + row] =
                __float2bfloat16_rn(silu(projected.x) * projected.z);
        }
        if (col0 + 1 < columns) {
            out[static_cast<std::int64_t>(col0 + 1) * Profile::kIntermediate + row] =
                __float2bfloat16_rn(silu(projected.y) * projected.w);
        }
    }
};

template <class Profile, int ActiveCols>
void launch_small_t_active(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    constexpr int TileCols =
        ActiveCols <= 8 ? 8 : (ActiveCols <= 16 ? 16 : (ActiveCols <= 24 ? 24 : 32));
    constexpr int kBlocks = Profile::kIntermediate / Q4SwiGluSmallTRows<Profile>::kOutputRowsPerCta;
    const Q4SwiGluSmallTEpilogue<Profile> epilogue{static_cast<__nv_bfloat16*>(out.data), x.ne[1]};
    q4_ksplit_mma_kernel<Q4SwiGluSmallTGeometry<Profile>, TileCols, ActiveCols,
                         Q4SwiGluSmallTEpilogue<Profile>, Q4SwiGluSmallTRows<Profile>, true>
        <<<kBlocks, Q4KSplitMmaSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
            epilogue, Q4SwiGluSmallTRows<Profile>{}, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

// T=2..32 in four 8-column tiles. A switch keeps this a plain function: a constexpr table of
// pointers to the instantiations is a variable template, which the device host stub does not carry.
template <class Profile>
void launch_small_t_route(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    switch ((x.ne[1] - 1) / 8) {
    case 0:
        launch_small_t_active<Profile, 8>(x, w, out, stream);
        return;
    case 1:
        launch_small_t_active<Profile, 16>(x, w, out, stream);
        return;
    case 2:
        launch_small_t_active<Profile, 24>(x, w, out, stream);
        return;
    default:
        launch_small_t_active<Profile, 32>(x, w, out, stream);
        return;
    }
}

template <class Profile>
__device__ __forceinline__ void
q4_issue_pair_tile(uint4 (*__restrict__ s_code)[Profile::kVecsPerWarpTile],
                   uint4 (*__restrict__ s_scale)[2],
                   const std::uint8_t* __restrict__ gate_code_row,
                   const std::uint8_t* __restrict__ gate_scale_row,
                   const std::uint8_t* __restrict__ up_code_row,
                   const std::uint8_t* __restrict__ up_scale_row, int tile, int lane) {
    constexpr int kGroupsPerWarpTile = Profile::kGroupsPerWarpTile;
    constexpr int kBytesPerGroup     = Profile::kBytesPerGroup;
    const int g0                     = tile * kGroupsPerWarpTile;
    pipe_copy<16>(&s_code[0][lane],
                  reinterpret_cast<const uint4*>(gate_code_row + g0 * kBytesPerGroup) + lane);
    pipe_copy<16>(&s_code[1][lane],
                  reinterpret_cast<const uint4*>(up_code_row + g0 * kBytesPerGroup) + lane);
    if (lane < 2) {
        pipe_copy<16>(&s_scale[0][lane],
                      reinterpret_cast<const uint4*>(gate_scale_row + g0 * 2) + lane);
        pipe_copy<16>(&s_scale[1][lane],
                      reinterpret_cast<const uint4*>(up_scale_row + g0 * 2) + lane);
    }
    pipe_commit();
}

template <class Profile>
__global__ void q4_linear_swiglu_gemv_pair_kernel(const __nv_bfloat16* __restrict__ x,
                                                  const std::uint8_t* __restrict__ codes,
                                                  const std::uint8_t* __restrict__ scales,
                                                  __nv_bfloat16* __restrict__ out) {
    constexpr int kStages                 = 3;
    constexpr int kPrefetch               = kStages - 1;
    constexpr int kGroupsPerWarpTile      = Profile::kGroupsPerWarpTile;
    constexpr int kBytesPerGroup          = Profile::kBytesPerGroup;
    constexpr int kVecsPerWarpTile        = Profile::kVecsPerWarpTile;
    constexpr int kWarpsPerBlock          = Profile::kWarpsPerBlock;
    constexpr int kPairsPerBlock          = Profile::kPairsPerBlock;
    constexpr int kXVecs                  = Profile::kXVecs;
    constexpr int kTiles                  = Profile::kTiles;
    constexpr int kGroups                 = Profile::kGroups;
    constexpr int kIntermediate           = Profile::kIntermediate;
    __shared__ __align__(16) __nv_bfloat16 x_sh[Profile::kK];
    __shared__ uint4 code_tile[kWarpsPerBlock][kStages][2][kVecsPerWarpTile];
    __shared__ uint4 scale_tile[kWarpsPerBlock][kStages][2][2];

    auto* x_sh_v    = reinterpret_cast<uint4*>(x_sh);
    const auto* x_g = reinterpret_cast<const uint4*>(x);
    for (int i = static_cast<int>(threadIdx.x); i < kXVecs; i += static_cast<int>(blockDim.x)) {
        x_sh_v[i] = x_g[i];
    }
    __syncthreads();

    const int lane    = static_cast<int>(threadIdx.x) & 31;
    const int warp    = static_cast<int>(threadIdx.x) >> 5;
    const int out_row = static_cast<int>(blockIdx.x) * kPairsPerBlock + warp;

    const std::uint8_t* gate_code_row =
        codes + static_cast<std::int64_t>(out_row) * kGroups * kBytesPerGroup;
    const std::uint8_t* gate_scale_row = scales + static_cast<std::int64_t>(out_row) * kGroups * 2;
    const std::uint8_t* up_code_row =
        codes + static_cast<std::int64_t>(out_row + kIntermediate) * kGroups * kBytesPerGroup;
    const std::uint8_t* up_scale_row =
        scales + static_cast<std::int64_t>(out_row + kIntermediate) * kGroups * 2;
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x_sh);

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
        if (p < kTiles) {
            q4_issue_pair_tile<Profile>(code_tile[warp][p], scale_tile[warp][p], gate_code_row,
                                        gate_scale_row, up_code_row, up_scale_row, p, lane);
        } else {
            pipe_commit();
        }
    }

#pragma unroll 1
    for (int tile = 0; tile < kTiles; ++tile) {
        const int fetch = tile + kPrefetch;
        if (fetch < kTiles) {
            const int buf = fetch % kStages;
            q4_issue_pair_tile<Profile>(code_tile[warp][buf], scale_tile[warp][buf], gate_code_row,
                                        gate_scale_row, up_code_row, up_scale_row, fetch, lane);
        } else {
            pipe_commit();
        }
        pipe_wait<kPrefetch>();
        __syncwarp();

        const int buf           = tile % kStages;
        const auto* gate_codes  = reinterpret_cast<const std::uint8_t*>(code_tile[warp][buf][0]);
        const auto* up_codes    = reinterpret_cast<const std::uint8_t*>(code_tile[warp][buf][1]);
        const auto* gate_scales = reinterpret_cast<const std::uint16_t*>(scale_tile[warp][buf][0]);
        const auto* up_scales   = reinterpret_cast<const std::uint16_t*>(scale_tile[warp][buf][1]);
#pragma unroll
        for (int tile_group = 0; tile_group < kGroupsPerWarpTile; ++tile_group) {
            const float gate_scale =
                __half2float(__ushort_as_half(static_cast<std::uint16_t>(gate_scales[tile_group])));
            const float up_scale =
                __half2float(__ushort_as_half(static_cast<std::uint16_t>(up_scales[tile_group])));

            const int gate_packed =
                static_cast<int>(gate_codes[tile_group * kBytesPerGroup + lane]);
            const int gate_q0   = sign_extend<4>(gate_packed & 0x0f);
            const int gate_q1   = sign_extend<4>(gate_packed >> 4);
            const int up_packed = static_cast<int>(up_codes[tile_group * kBytesPerGroup + lane]);
            const int up_q0     = sign_extend<4>(up_packed & 0x0f);
            const int up_q1     = sign_extend<4>(up_packed >> 4);
            const int k0        = (tile * kGroupsPerWarpTile + tile_group) * Profile::kGroupK + lane * 2;
            const float2 xv     = __bfloat1622float2(x2[k0 >> 1]);
            gate_acc            = fmaf(static_cast<float>(gate_q0) * gate_scale, xv.x, gate_acc);
            gate_acc            = fmaf(static_cast<float>(gate_q1) * gate_scale, xv.y, gate_acc);
            up_acc              = fmaf(static_cast<float>(up_q0) * up_scale, xv.x, up_acc);
            up_acc              = fmaf(static_cast<float>(up_q1) * up_scale, xv.y, up_acc);
        }
        __syncwarp();
    }

    gate_acc = warp_reduce_sum(gate_acc);
    up_acc   = warp_reduce_sum(up_acc);
    if (lane == 0) { out[out_row] = __float2bfloat16(silu(gate_acc) * up_acc); }
}

// True when the weight names this registered geometry exactly, padding included.
template <class Profile>
bool matches_profile(const Weight& w) {
    return w.n == Profile::kN && w.k == Profile::kK && w.padded_shape[0] == Profile::kN &&
           w.padded_shape[1] == Profile::kK;
}

template <class Profile>
void launch_gemv_pair(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const int grid = Profile::kIntermediate / Profile::kPairsPerBlock;
    q4_linear_swiglu_gemv_pair_kernel<Profile><<<grid, Profile::kBlockThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q4_linear_swiglu_gemv_pair_launch(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream) {
    if (matches_profile<SwiGlu27BProfile>(w)) {
        launch_gemv_pair<SwiGlu27BProfile>(x, w, out, stream);
        return;
    }
    if (matches_profile<SwiGlu9BProfile>(w)) {
        launch_gemv_pair<SwiGlu9BProfile>(x, w, out, stream);
        return;
    }
    throw std::invalid_argument(
        "q4 linear_swiglu GEMV: the weight names no registered geometry "
        "([34816,5120] or [24576,4096])");
}

void q4_linear_swiglu_small_t_tiled_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream) {
    if (x.ne[1] < 2 || x.ne[1] > 32) {
        throw std::invalid_argument("Q4 LinearSwiGLU exact small-T requires T=2..32");
    }
    if (matches_profile<SwiGlu27BProfile>(w)) {
        launch_small_t_route<SwiGlu27BProfile>(x, w, out, stream);
        return;
    }
    if (matches_profile<SwiGlu9BProfile>(w)) {
        launch_small_t_route<SwiGlu9BProfile>(x, w, out, stream);
        return;
    }
    throw std::invalid_argument(
        "Q4 LinearSwiGLU small-T: the weight names no registered geometry "
        "([34816,5120] or [24576,4096])");
}

} // namespace ninfer::ops::detail
