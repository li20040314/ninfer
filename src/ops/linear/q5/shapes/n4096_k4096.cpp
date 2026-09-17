#include "ops/linear/q5/q5_shapes.h"

namespace ninfer::ops::detail {

// Qwen3.5-9B: attention/gate, attention/output, gdn/value, gdn/z and gdn/output all project to
// hidden_size = 4096. Only the runtime-shaped Q5 launchers are used: the Q5 GEMV entry point
// (`launch_q5_gemv_r16_s2_x`) is keyed on a compile-time k = 5120 and an exact n in {6144, 7168},
// so it cannot serve this geometry. Mirrors the all-generic ladder of q5/shapes/n1024_k5120.cpp.
Q5Launch select_q5_n4096_k4096(std::int32_t tokens) {
    if (tokens <= 4) return launch_q5_simt_r8_c4;
    if (tokens <= 16) return launch_q5_simt_r8_c8;
    return launch_q5_mma_r64_c128;
}

} // namespace ninfer::ops::detail
