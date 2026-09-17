#include "ops/linear/q5/q5_shapes.h"

namespace ninfer::ops::detail {

// Qwen3.5-9B: mlp/down (hidden_size 4096 over intermediate_size 12288). The 27B sibling of this
// projection is (5120, 17408), whose launcher ladder is not reusable because the Q5 GEMV entry
// point is keyed on a compile-time k = 5120 and an exact n in {6144, 7168}. Only the
// runtime-shaped Q5 launchers are usable here; see q5/shapes/n4096_k4096.cpp.
Q5Launch select_q5_n4096_k12288(std::int32_t tokens) {
    if (tokens <= 4) return launch_q5_simt_r8_c4;
    if (tokens <= 16) return launch_q5_simt_r8_c8;
    return launch_q5_mma_r64_c128;
}

} // namespace ninfer::ops::detail
