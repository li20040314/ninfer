#pragma once

#include "ops/linear/q4/q4_launch.h"

namespace ninfer::ops::detail {

[[nodiscard]] Q4Launch select_q4_n1024_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n4096_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n5120_k6144(std::int32_t tokens);
// Qwen3.8-27B mlp/down (hidden_size 5120, intermediate_size 17408).
[[nodiscard]] Q4Launch select_q4_n5120_k17408(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n6144_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n7168_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n34816_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n131072_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n131072_k2048(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n3456_k1152(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n4304_k1152(std::int32_t tokens);

// Qwen3.5-9B (hidden_size 4096).
[[nodiscard]] Q4Launch select_q4_n1024_k4096(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n2048_k4096(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n4096_k4096(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n5120_k4096(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n12288_k4096(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n24576_k4096(std::int32_t tokens);

} // namespace ninfer::ops::detail
