#pragma once

// The admitted Q4 LinearSwiGLU geometries, in one place.
//
// Each entry is an "exact problem": the tuple the plan admits and the two numbers the
// shape-specialised kernels bake into template arguments -- the fused gate/up width and the
// contraction width. Nothing here may be derived from another model.
//
// The fused width is twice the intermediate size and the contraction width is the model's hidden
// size, so the 27B entry (2*17408 over 5120) and the 9B entry (2*12288 over 4096) share no number.

#include <cstdint>

namespace ninfer::ops::detail {

struct Q4SwiGluGeometry {
    std::int32_t gate_up_rows;
    std::int32_t output_rows;
    std::int32_t k;
    std::int32_t padded_k;

    constexpr bool consistent() const noexcept {
        return output_rows * 2 == gate_up_rows && k == padded_k;
    }
};

// Qwen3.6/3.8 27B: intermediate 17408 over hidden 5120, the tuned target.
inline constexpr Q4SwiGluGeometry kQ4SwiGlu27B{34816, 17408, 5120, 5120};
// Qwen3.5 Small 9B: intermediate 12288 over hidden 4096, the Ada profile.
inline constexpr Q4SwiGluGeometry kQ4SwiGlu9B{24576, 12288, 4096, 4096};

inline constexpr Q4SwiGluGeometry kQ4SwiGluGeometries[]{kQ4SwiGlu27B, kQ4SwiGlu9B};

static_assert(kQ4SwiGlu27B.consistent() && kQ4SwiGlu9B.consistent(),
              "a registered Q4 LinearSwiGLU geometry must describe one closed problem");

// True when `problem` names exactly this geometry. Every field participates so a partially
// matching tuple cannot silently inherit an entry's kernels.
constexpr bool q4_swiglu_geometry_matches(const Q4SwiGluGeometry& geometry,
                                          const std::int32_t gate_up_rows,
                                          const std::int32_t output_rows, const std::int32_t k,
                                          const std::int32_t padded_k) noexcept {
    return geometry.gate_up_rows == gate_up_rows && geometry.output_rows == output_rows &&
           geometry.k == k && geometry.padded_k == padded_k;
}

} // namespace ninfer::ops::detail
