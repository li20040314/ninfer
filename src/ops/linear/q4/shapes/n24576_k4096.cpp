#include "ops/linear/q4/q4_shapes.h"

namespace ninfer::ops::detail {

// Qwen3.5-9B: the fused group(mlp/gate, mlp/up) weight (2 x 12288 rows). Also reached through
// linear_swiglu's Materialized route, which re-enters the `linear` closed table for the same
// (n, k) pair. See n4096_k4096.cpp for why the generic Q4 GEMV variant is used instead of
// `launch_q4_gemv_r1_q8_direct`.
Q4Launch select_q4_n24576_k4096(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_gemv_r4_w1_direct;
    if (tokens <= 4) return launch_q4_simt_r8_c4;
    if (tokens <= 16) return launch_q4_simt_r8_c8;
    return launch_q4_mma_r64_c128;
}

} // namespace ninfer::ops::detail
