#include "core/weight.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_geometry.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_rowsplit_gemm_simt.cuh"
#include "ops/linear/q4/q4_rowsplit_gemv.cuh"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"
#include "ops/linear/q5/q5_rowsplit_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using Q4AttnSimtR8C4Schedule = Q4RowSplitSimtGemmSchedule<8, 4, 16, 2, Cache::ca, 1>;
using Q4AttnSimtR8C8Schedule = Q4RowSplitSimtGemmSchedule<8, 8, 16, 2, Cache::ca, 1>;

// A full Q5 SIMT slab covers 1024 contraction elements, so the slab count is a function of the
// geometry and must not be inherited from the profile the kernel was first tuned for.
template <std::int32_t Hidden>
inline constexpr std::int32_t kSlabs = Hidden / 1024;

template <class Q4GemvSchedule, std::int32_t ParentRows, std::int32_t SplitRow,
          std::int32_t Hidden>
void launch_q4_gemv(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                    cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(div_up(ParentRows, Q4GemvSchedule::kRowsPerCta)), 1u,
                    1u);
    constexpr dim3 block(static_cast<unsigned>(Q4GemvSchedule::kThreads), 1u, 1u);
    q4_rowsplit_gemv_kernel<Q4GemvSchedule, true, SplitRow><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(key.data), ParentRows, Hidden);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, bool Full, std::int32_t ParentRows, std::int32_t SplitRow,
          std::int32_t Hidden>
void launch_q4_simt(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                    cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    const dim3 grid(static_cast<unsigned>(div_up(ParentRows, Schedule::kRowsPerCta)),
                    static_cast<unsigned>(div_up(cols, Schedule::kColsPerTile)), 1u);
    q4_rowsplit_gemm_simt_kernel<Schedule, Full, true, SplitRow>
        <<<grid, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(q.data),
            static_cast<__nv_bfloat16*>(key.data), q.ne[0], key.ne[0], ParentRows, Hidden, cols,
            weight.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, std::int32_t ParentRows, std::int32_t SplitRow, std::int32_t Hidden>
void launch_q4_simt_route(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                          cudaStream_t stream) {
    const bool full = (ParentRows % Schedule::kRowsPerCta) == 0 &&
                      ((Hidden / Q4RowSplitStorage::kGroupK) % Schedule::kGroupsPerStage) == 0 &&
                      (x.ne[1] % Schedule::kColsPerTile) == 0;
    if (full) {
        launch_q4_simt<Schedule, true, ParentRows, SplitRow, Hidden>(x, weight, q, key, stream);
    } else {
        launch_q4_simt<Schedule, false, ParentRows, SplitRow, Hidden>(x, weight, q, key, stream);
    }
}

template <class Q4GemvSchedule, std::int32_t ParentRows, std::int32_t SplitRow,
          std::int32_t Hidden>
void launch_q4(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
               cudaStream_t stream) {
    switch (x.ne[1]) {
    case 1:
        launch_q4_gemv<Q4GemvSchedule, ParentRows, SplitRow, Hidden>(x, weight, q, key, stream);
        return;
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
    case 9:
    case 10:
    case 11:
    case 12:
        launch_q4_simt_route<Q4AttnSimtR8C4Schedule, ParentRows, SplitRow, Hidden>(x, weight, q,
                                                                                 key, stream);
        return;
    case 8:
        launch_q4_simt_route<Q4AttnSimtR8C8Schedule, ParentRows, SplitRow, Hidden>(x, weight, q,
                                                                                 key, stream);
        return;
    default:
        throw std::invalid_argument("attention Q4 split-output requires T in [1,12]");
    }
}

template <std::int32_t ParentRows, std::int32_t SplitRow, std::int32_t Hidden>
void launch_q5_gemv(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                    cudaStream_t stream) {
    constexpr int kRowsPerBlock = 16;
    constexpr int kBlockThreads = kRowsPerBlock * 32;
    constexpr int kGrid         = ParentRows / kRowsPerBlock;
    static_assert(ParentRows % kRowsPerBlock == 0,
                  "the fused attention Q5 GEMV requires whole row blocks");
    q5_rowsplit_gemv_kernel<ParentRows, Hidden, kRowsPerBlock, 2, true, false, true, SplitRow>
        <<<kGrid, kBlockThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                              static_cast<const std::uint8_t*>(weight.qdata),
                                              static_cast<const std::uint8_t*>(weight.qhigh),
                                              static_cast<const std::uint8_t*>(weight.scales),
                                              static_cast<__nv_bfloat16*>(gate.data),
                                              static_cast<__nv_bfloat16*>(value.data));
    CUDA_CHECK(cudaGetLastError());
}

template <int Cols, std::int32_t ParentRows, std::int32_t SplitRow, std::int32_t Hidden>
void launch_q5_split4(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                      cudaStream_t stream) {
    constexpr int kThreads       = 4 * 32;
    constexpr std::int32_t kSlab = kSlabs<Hidden>;
    const dim3 grid(static_cast<unsigned>(ParentRows), 1u, 1u);
    q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Cols, kSlab, Hidden, true,
                                        SplitRow><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.qhigh),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(value.data), ParentRows, gate.ne[0], Hidden, Cols,
        weight.padded_shape[1], kSlab);
    CUDA_CHECK(cudaGetLastError());
}

template <std::int32_t ParentRows, std::int32_t SplitRow, std::int32_t Hidden>
void launch_q5_split4_exact(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                            cudaStream_t stream) {
    switch (x.ne[1]) {
    case 2:
        launch_q5_split4<2, ParentRows, SplitRow, Hidden>(x, weight, gate, value, stream);
        return;
    case 3:
        launch_q5_split4<3, ParentRows, SplitRow, Hidden>(x, weight, gate, value, stream);
        return;
    case 4:
        launch_q5_split4<4, ParentRows, SplitRow, Hidden>(x, weight, gate, value, stream);
        return;
    case 5:
        launch_q5_split4<5, ParentRows, SplitRow, Hidden>(x, weight, gate, value, stream);
        return;
    case 6:
        launch_q5_split4<6, ParentRows, SplitRow, Hidden>(x, weight, gate, value, stream);
        return;
    default:
        throw std::invalid_argument("attention Q5 split4 requires T in [2,6]");
    }
}

template <int ColsPerTile, std::int32_t ParentRows, std::int32_t SplitRow, std::int32_t Hidden>
void launch_q5_simt(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                    cudaStream_t stream) {
    constexpr int kRowsPerBlock  = 8;
    constexpr int kStages        = 2;
    constexpr int kThreads       = kRowsPerBlock * 32;
    constexpr std::int32_t kSlab = kSlabs<Hidden>;
    const std::int32_t cols      = x.ne[1];
    const dim3 grid(static_cast<unsigned>(div_up(ParentRows, kRowsPerBlock)),
                    static_cast<unsigned>(div_up(cols, ColsPerTile)), 1u);
    q5_rowsplit_gemm_simt_kernel<Q5RowSplitSimtSchedule, ColsPerTile, kRowsPerBlock, kStages, true,
                                 SplitRow><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.qhigh),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(value.data), ParentRows, gate.ne[0], Hidden, cols,
        weight.padded_shape[1], kSlab);
    CUDA_CHECK(cudaGetLastError());
}

template <std::int32_t ParentRows, std::int32_t SplitRow, std::int32_t Hidden>
void launch_q5(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
               cudaStream_t stream) {
    if (x.ne[1] == 1) {
        launch_q5_gemv<ParentRows, SplitRow, Hidden>(x, weight, gate, value, stream);
        return;
    }
    if (x.ne[1] <= 6) {
        launch_q5_split4_exact<ParentRows, SplitRow, Hidden>(x, weight, gate, value, stream);
        return;
    }
    if (x.ne[1] <= 12) {
        launch_q5_simt<4, ParentRows, SplitRow, Hidden>(x, weight, gate, value, stream);
        return;
    }
    throw std::invalid_argument("attention Q5 split-output requires T in [1,12]");
}

template <class Q4GemvSchedule, std::int32_t ParentRows, std::int32_t SplitRow,
          std::int32_t Hidden>
void launch_shape(const Tensor& x, const Weight& query_key_weight, const Weight& gate_value_weight,
                  Tensor& q, Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_q4<Q4GemvSchedule, ParentRows, SplitRow, Hidden>(x, query_key_weight, q, k, stream);
    launch_q5<ParentRows, SplitRow, Hidden>(x, gate_value_weight, gate, v, stream);
}

} // namespace

void q4_q5_attn_input_small_t_launch(const Tensor& x, const Weight& query_key_weight,
                                     const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream) {
    const std::int32_t fused  = query_key_weight.n;
    const std::int32_t split  = q.ne[0];
    const std::int32_t hidden = query_key_weight.padded_shape[1];

    // The 27B profile keeps its eight-warp static-ownership GEMV. The 9B profile must use the
    // runtime-ownership GEMV: a static per-row group count is only valid for its own contraction
    // width.
    if (q4_q5_attn_input_geometry_matches(kQ4Q5AttnInput27B, fused, split, hidden)) {
        launch_shape<Q4GemvR1Q8DirectSchedule, kQ4Q5AttnInput27B.fused_rows(),
                     kQ4Q5AttnInput27B.query_rows, kQ4Q5AttnInput27B.padded_k>(
            x, query_key_weight, gate_value_weight, q, gate, k, v, stream);
        return;
    }
    if (q4_q5_attn_input_geometry_matches(kQ4Q5AttnInput9B, fused, split, hidden)) {
        launch_shape<Q4GemvR4W1DirectSchedule, kQ4Q5AttnInput9B.fused_rows(),
                     kQ4Q5AttnInput9B.query_rows, kQ4Q5AttnInput9B.padded_k>(
            x, query_key_weight, gate_value_weight, q, gate, k, v, stream);
        return;
    }
    throw std::invalid_argument("Q4/Q5 attention input small-T: unsupported fusion geometry");
}

} // namespace ninfer::ops::detail
