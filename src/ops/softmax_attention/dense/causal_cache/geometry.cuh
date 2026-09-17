#pragma once

#include "ops/softmax_attention/common/head_mapping.cuh"

namespace ninfer::ops {

template <int QHeadsValue, int KVHeadsValue, int SmallTSplitScaleValue>
struct CausalAttentionGeometry : AttentionHeadMapping<QHeadsValue, KVHeadsValue> {
    static_assert(SmallTSplitScaleValue > 0);

    static constexpr int SmallTSplitScale    = SmallTSplitScaleValue;
    static constexpr int SmallTMaximumSplits = 85 * SmallTSplitScale;
};

using CausalD256H24Kv4 = CausalAttentionGeometry<24, 4, 1>;
using CausalD256H16Kv2 = CausalAttentionGeometry<16, 2, 2>;
// Qwen3.5-9B: full-attention layers use 16 query heads over 4 KV heads at head_dim 256.
// The KV load per token matches the 24/4 profile, so the small-T split scale follows it.
using CausalD256H16Kv4 = CausalAttentionGeometry<16, 4, 1>;

} // namespace ninfer::ops
