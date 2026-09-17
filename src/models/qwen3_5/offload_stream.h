#pragma once

#include "artifact/materializer.h"
#include "core/arena.h"
#include "core/weight_view.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::models::qwen3_5 {

// P0 weight offload: weights beyond the device budget stay resident in host memory and are
// streamed into a frozen device slot region before their layer executes. The slot region is
// sized by the largest single-layer footprint and rotates: ensure_layer() rewrites slot
// contents for each streamed layer, so kernel-visible weight pointers never change but slots
// are shared between layers.
//
// Transfer path: the retained host objects are page-locked (cudaHostRegister, best effort) so
// cudaMemcpyAsync runs the direct DMA path instead of the WDDM staged-copy path, which would
// add a full host-side memcpy per transfer. Layer uploads are event-ordered rather than
// host-synchronized: the transfer stream waits for the previous layer's compute tail (single
// shared slot must not be rewritten while those kernels still read it), and the compute stream
// waits for the copy event before the layer's kernels launch. The CPU never blocks.
struct OffloadedObject {
    artifact::ObjectHandle object{};
    std::uint64_t slot_offset = 0;
    std::uint64_t bytes       = 0;
};

struct WeightOffloadSpec {
    bool enabled = false;
    // First text layer whose weights stream from host. Earlier layers, the embedding, the
    // output head, and the final norm stay device resident.
    std::uint32_t first_streamed_layer = 0;
    std::vector<OffloadedObject> objects;
    // Per text layer: indices into `objects` (empty for resident layers).
    std::vector<std::vector<std::size_t>> layer_objects;
    // Per WeightId: indices into `objects` parallel to the bound view parts. Empty entries
    // mark device-resident weights.
    std::vector<std::vector<std::size_t>> weight_objects;
    std::size_t resident_bytes = 0;
    // Device slot-region size: two rotating slots, each the largest single-layer streamed
    // footprint (layer L uploads into slot L%2 while layer L-1 computes from slot (L-1)%2),
    // not the total host-resident payload.
    std::size_t streamed_bytes = 0;
};

class WeightStreamScheduler {
public:
    WeightStreamScheduler(const WeightOffloadSpec& spec,
                          const artifact::MaterializedArtifact& backing,
                          cudaStream_t transfer_stream);
    ~WeightStreamScheduler();

    WeightStreamScheduler(const WeightStreamScheduler&)            = delete;
    WeightStreamScheduler& operator=(const WeightStreamScheduler&) = delete;

    // Event-ordered upload of every offloaded object of the requested text layer into its slot
    // region. Holds the transfer stream back only until the layer that *previously used this
    // layer's slot* (two layers ago, since the slots alternate) has finished reading it, so the
    // upload overlaps the immediately preceding layer's compute. Makes the compute stream wait on
    // the copy before any caller launches the layer's kernels. Returns without host-side
    // blocking; resident layers return immediately.
    void ensure_layer(int layer, cudaStream_t compute_stream) const;

    // Stable parent views for offloaded objects; the bound weight views borrow these.
    [[nodiscard]] const WeightParent& parent(std::size_t object_index) const {
        return parents_.at(object_index);
    }

    [[nodiscard]] bool enabled() const noexcept { return spec_.enabled; }

    [[nodiscard]] std::size_t slot_bytes() const noexcept { return slot_.bytes; }
    [[nodiscard]] std::size_t streamed_bytes() const noexcept { return spec_.streamed_bytes; }
    [[nodiscard]] std::size_t resident_bytes() const noexcept { return spec_.resident_bytes; }
    [[nodiscard]] std::uint32_t first_streamed_layer() const noexcept {
        return spec_.first_streamed_layer;
    }
    // Page-locking is best-effort (`cudaHostRegister` can refuse a large block and the copies then
    // silently take the staged path, which is both slower and harder to overlap), so the outcome is
    // reported rather than assumed.
    [[nodiscard]] std::size_t pinned_bytes() const noexcept { return pinned_bytes_; }
    [[nodiscard]] std::size_t unpinned_bytes() const noexcept { return unpinned_bytes_; }

private:
    // One contiguous device-side upload: the slot destinations are laid out per layer, but the
    // retained host objects are separate allocations, so each entry stays its own memcpy.
    struct LayerCopy {
        const std::byte* destination = nullptr;
        const std::byte* source      = nullptr;
        std::size_t bytes            = 0;
    };

    WeightOffloadSpec spec_;
    DeviceBuffer slot_;
    std::vector<WeightParent> parents_;
    // Stable host sources of the offloaded objects (kept for pinned-memory unregistration).
    std::vector<const std::byte*> host_sources_;
    std::vector<std::vector<LayerCopy>> layer_batches_;
    // Per offloaded object: whether the retained host bytes were page-locked successfully.
    std::vector<bool> host_pinned_;
    std::size_t pinned_bytes_   = 0;
    std::size_t unpinned_bytes_ = 0;
    // One ordering event per slot parity. The slot a layer uploads into was last read two layers
    // ago, so each event is armed at the *previous* call (when the compute stream's tail was the
    // end of the layer before it) and consumed at the next one. A single shared event would make
    // every upload wait on the immediately preceding layer, which serializes the whole pass.
    mutable cudaEvent_t compute_event_[2] = {nullptr, nullptr};
    mutable cudaEvent_t copy_event_    = nullptr; // Completion of the layer's uploads.
    cudaStream_t transfer_stream_      = nullptr;
};

} // namespace ninfer::models::qwen3_5
