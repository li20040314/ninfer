#pragma once

#include <cstdint>

namespace ninfer::ops::detail {

// Compile-time geometry of the fused attention input projection.
//
// The Op consumes two fused row-split weights: (query, key) packed into one parent and
// (gate, value) into another. Each fused parent therefore holds `query_rows + kv_rows` rows,
// and the implementations fold the query/gate half and the key/value half into a single launch
// through a compile-time split seam at `query_rows`.
//
// A geometry is admitted only when it appears in `kQ4Q5AttnInputGeometries`; every kernel
// instantiation, plan predicate, and operand check derives from that single table.
struct Q4Q5AttnInputGeometry {
    std::int32_t input_rows; // activation width
    std::int32_t query_rows; // query/gate rows, and the fold seam
    std::int32_t kv_rows;    // key/value rows
    std::int32_t padded_k;   // padded contraction width

    constexpr std::int32_t fused_rows() const noexcept { return query_rows + kv_rows; }
};

// Tuned profile: Qwen3.6/3.8 27B (hidden 5120), the RTX 5090 target.
inline constexpr Q4Q5AttnInputGeometry kQ4Q5AttnInput27B{5120, 6144, 1024, 5120};
// Qwen3.5 Small 9B (hidden 4096), the RTX 4060 profile.
inline constexpr Q4Q5AttnInputGeometry kQ4Q5AttnInput9B{4096, 4096, 1024, 4096};

inline constexpr Q4Q5AttnInputGeometry kQ4Q5AttnInputGeometries[]{kQ4Q5AttnInput27B,
                                                                 kQ4Q5AttnInput9B};

// The admitted geometry whose fused parent width and contraction width match, or a zeroed
// geometry when the fusion is not admitted.
constexpr Q4Q5AttnInputGeometry q4_q5_attn_input_geometry(std::int32_t fused_rows,
                                                          std::int32_t padded_k) noexcept {
    for (const Q4Q5AttnInputGeometry& geometry : kQ4Q5AttnInputGeometries) {
        if (geometry.fused_rows() == fused_rows && geometry.padded_k == padded_k) { return geometry; }
    }
    return {0, 0, 0, 0};
}

// True when the three fused-projection axes identify `geometry`.
constexpr bool q4_q5_attn_input_geometry_matches(const Q4Q5AttnInputGeometry& geometry,
                                                 std::int32_t fused_rows, std::int32_t split_row,
                                                 std::int32_t padded_k) noexcept {
    return fused_rows == geometry.fused_rows() && split_row == geometry.query_rows &&
           padded_k == geometry.padded_k;
}

} // namespace ninfer::ops::detail
