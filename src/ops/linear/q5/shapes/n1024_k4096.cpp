#include "ops/linear/q5/q5_shapes.h"

namespace ninfer::ops::detail {

// Qwen3.5-9B: attention/value (num_key_value_heads x head_dim = 4 x 256).
// See n4096_k4096.cpp for why the Q5 GEMV entry point cannot be used for this geometry.
Q5Launch select_q5_n1024_k4096(std::int32_t tokens) {
    if (tokens <= 4) return launch_q5_simt_r8_c4;
    if (tokens <= 16) return launch_q5_simt_r8_c8;
    return launch_q5_mma_r64_c128;
}

} // namespace ninfer::ops::detail
