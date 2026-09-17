#include "ops/linear/q4/q4_shapes.h"

namespace ninfer::ops::detail {

// Qwen3.5-9B: attention/query (16 x 256) and gdn group(query, key) (2 x 2048).
// Only launchers that derive their shape from the runtime Weight are used here: the Q4 GEMV
// specialization selected by `launch_q4_gemv_r1_q8_direct` bakes in k = 5120
// (`kStaticGroupsPerRow = 80`), whereas `launch_q4_gemv_r4_w1_direct` uses the runtime path
// (`kStaticGroupsPerRow = 0`) and is already exercised at k = 2048 and k = 5120.
Q4Launch select_q4_n4096_k4096(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_gemv_r4_w1_direct;
    if (tokens <= 4) return launch_q4_simt_r8_c4;
    if (tokens <= 16) return launch_q4_simt_r8_c8;
    return launch_q4_mma_r64_c128;
}

} // namespace ninfer::ops::detail
