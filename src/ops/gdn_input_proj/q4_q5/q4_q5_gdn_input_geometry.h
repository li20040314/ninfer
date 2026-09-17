#pragma once

// The admitted Q4/Q5 GDN input-projection geometries, in one place.
//
// Each entry is an "exact problem": the tuple the plan admits and the numbers the shape-specialised
// kernels bake. Two profiles are registered, and they differ in the activation width, the
// contraction width, and the value/z head widths -- the query/key widths are shared.
//
// The value width is *not* a cross-model invariant. The 27B profile has 48 value heads (48*128 =
// 6144) while the 9B profile has 32 (32*128 = 4096), so an entry can only be told apart from the
// 35B-A3B profile -- which shares `value_rows == 4096` -- by its activation width. Nothing here may
// be derived from another model.

#include <cstdint>

namespace ninfer::ops::detail {

struct Q4Q5GdnInputGeometry {
    // The tuple `q4_q5_gdn_input_resolve_plan` admits.
    std::int32_t input_rows;
    std::int32_t qk_rows;
    std::int32_t value_z_rows;
    std::int32_t qkv_rows;
    std::int32_t z_rows;
    std::int32_t padded_k;

    // Derived shapes the kernels need explicitly: the convolution publishes query and key
    // separately, and the value/z parent's fold seam sits at `value_rows`.
    std::int32_t query_rows;
    std::int32_t key_rows;
    std::int32_t value_rows;

    constexpr std::int32_t value_z_seam() const noexcept { return value_rows; }
    constexpr std::int32_t channels() const noexcept { return qk_rows + value_rows; }
    // Where the value channels begin in the convolution's channel space, which the fused projection
    // epilogues index by. This is the fused q/k width, *not* the value width.
    constexpr std::int32_t conv_value_offset() const noexcept { return qk_rows; }

    constexpr bool consistent() const noexcept {
        return query_rows + key_rows == qk_rows && value_rows + z_rows == value_z_rows &&
               qk_rows + z_rows == qkv_rows && padded_k == input_rows;
    }
};

// Qwen3.6/3.8 27B: 16 key heads x 128 and 48 value heads x 128 over hidden 5120.
inline constexpr Q4Q5GdnInputGeometry kQ4Q5GdnInput27B{5120, 4096, 12288, 10240, 6144, 5120,
                                                       2048, 2048, 6144};
// Qwen3.5 Small 9B: the same 16 key heads x 128 and 32 value heads x 128 over hidden 4096.
inline constexpr Q4Q5GdnInputGeometry kQ4Q5GdnInput9B{4096, 4096, 8192, 8192, 4096, 4096,
                                                      2048, 2048, 4096};

inline constexpr Q4Q5GdnInputGeometry kQ4Q5GdnInputGeometries[]{kQ4Q5GdnInput27B, kQ4Q5GdnInput9B};

static_assert(kQ4Q5GdnInput27B.consistent() && kQ4Q5GdnInput9B.consistent(),
              "a registered GDN input geometry must describe one closed problem");

// True when `problem` is exactly this geometry. Every field participates so that a partially
// matching tuple cannot silently inherit an entry's kernels.
constexpr bool q4_q5_gdn_input_geometry_matches(const Q4Q5GdnInputGeometry& geometry,
                                                const std::int32_t input_rows,
                                                const std::int32_t qk_rows,
                                                const std::int32_t value_z_rows,
                                                const std::int32_t qkv_rows,
                                                const std::int32_t z_rows,
                                                const std::int32_t padded_k) noexcept {
    return geometry.input_rows == input_rows && geometry.qk_rows == qk_rows &&
           geometry.value_z_rows == value_z_rows && geometry.qkv_rows == qkv_rows &&
           geometry.z_rows == z_rows && geometry.padded_k == padded_k;
}

// Resolves the geometry a fused-operand problem names, or one with `input_rows == 0` when the tuple
// is not registered. `qk_rows` is the fused query+key width, `value_rows` the value width.
constexpr Q4Q5GdnInputGeometry
q4_q5_gdn_input_geometry(std::int32_t input_rows, std::int32_t qk_rows, std::int32_t value_rows,
                         std::int32_t z_rows) noexcept {
    for (const Q4Q5GdnInputGeometry& geometry : kQ4Q5GdnInputGeometries) {
        if (geometry.input_rows == input_rows && geometry.qk_rows == qk_rows &&
            geometry.value_rows == value_rows && geometry.z_rows == z_rows) {
            return geometry;
        }
    }
    return {0, 0, 0, 0, 0, 0, 0, 0, 0};
}

constexpr bool same_geometry(const Q4Q5GdnInputGeometry& lhs,
                             const Q4Q5GdnInputGeometry& rhs) noexcept {
    return q4_q5_gdn_input_geometry_matches(rhs, lhs.input_rows, lhs.qk_rows, lhs.value_z_rows,
                                            lhs.qkv_rows, lhs.z_rows, lhs.padded_k);
}

} // namespace ninfer::ops::detail
