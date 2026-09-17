#include "ops/linear/q4/q4_shapes.h"

namespace ninfer::ops::detail {

// Qwen3.5-9B: attention/key (4 x 256). See n4096_k4096.cpp for why the generic Q4 GEMV variant
// is used instead of `launch_q4_gemv_r1_q8_direct`.
Q4Launch select_q4_n1024_k4096(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_gemv_r4_w1_direct;
    if (tokens <= 4) return launch_q4_simt_r8_c4;
    if (tokens <= 16) return launch_q4_simt_r8_c8;
    return launch_q4_mma_r64_c128;
}

} // namespace ninfer::ops::detail
