#include "models/qwen3_5/offload_stream.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <stdexcept>

namespace ninfer::models::qwen3_5 {

WeightStreamScheduler::WeightStreamScheduler(const WeightOffloadSpec& spec,
                                             const artifact::MaterializedArtifact& backing,
                                             cudaStream_t transfer_stream)
    : spec_(spec), slot_(spec.enabled ? spec.streamed_bytes : 0),
      transfer_stream_(transfer_stream) {
    if (!spec_.enabled) { return; }
    if (spec_.streamed_bytes == 0) { throw std::invalid_argument("offload spec streams no bytes"); }

    if (cudaEventCreateWithFlags(&compute_event_[0], cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&compute_event_[1], cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&copy_event_, cudaEventDisableTiming) != cudaSuccess) {
        throw std::invalid_argument("offload scheduler failed to create ordering events");
    }

    parents_.reserve(spec_.objects.size());
    host_sources_.reserve(spec_.objects.size());
    host_pinned_.reserve(spec_.objects.size());
    layer_batches_.resize(spec_.layer_objects.size());
    for (const auto& entry : spec_.objects) {
        const auto& host = backing.host_parent(entry.object);
        const auto bytes = backing.host_bytes(entry.object).size();
        if (bytes != entry.bytes) {
            throw std::invalid_argument("offload slot size differs from retained host object");
        }
        WeightParent slot_parent;
        slot_parent.geometry             = host.geometry;
        slot_parent.weight_scale_divisor = host.weight_scale_divisor;
        slot_parent.data                 = static_cast<const std::byte*>(slot_.p) + entry.slot_offset;
        parents_.push_back(slot_parent);

        // Page-lock the retained host bytes so the per-layer uploads take the direct DMA path.
        // Registration is best-effort: on failure the upload falls back to the staged-copy path,
        // which costs a full extra host-side memcpy per transfer and overlaps poorly -- so the
        // outcome is counted rather than assumed.
        auto* mutable_host = const_cast<std::byte*>(host.data);
        const bool pinned =
            cudaHostRegister(mutable_host, bytes, cudaHostRegisterDefault) == cudaSuccess;
        host_pinned_.push_back(pinned);
        if (pinned) {
            pinned_bytes_ += bytes;
        } else {
            unpinned_bytes_ += bytes;
        }
        host_sources_.push_back(host.data);
    }

    // Pre-resolve every per-layer upload once: {slot destination, host source, bytes}.
    for (std::size_t layer = 0; layer < spec_.layer_objects.size(); ++layer) {
        layer_batches_[layer].reserve(spec_.layer_objects[layer].size());
        for (const auto object_index : spec_.layer_objects[layer]) {
            const auto& entry = spec_.objects.at(object_index);
            if (entry.bytes == 0) { continue; }
            layer_batches_[layer].push_back(
                LayerCopy{parents_.at(object_index).data, host_sources_.at(object_index),
                          static_cast<std::size_t>(entry.bytes)});
        }
    }
}

WeightStreamScheduler::~WeightStreamScheduler() {
    for (std::size_t index = 0; index < host_pinned_.size(); ++index) {
        if (host_pinned_[index] && index < host_sources_.size()) {
            // The scheduler is destroyed before the backing storage (Model member order), so the
            // registered pages still belong to live allocations here. Unregister failures cannot
            // be meaningfully handled at shutdown.
            cudaHostUnregister(const_cast<std::byte*>(host_sources_[index]));
        }
    }
    if (compute_event_[0] != nullptr) { cudaEventDestroy(compute_event_[0]); }
    if (compute_event_[1] != nullptr) { cudaEventDestroy(compute_event_[1]); }
    if (copy_event_ != nullptr) { cudaEventDestroy(copy_event_); }
}

void WeightStreamScheduler::ensure_layer(int layer, cudaStream_t compute_stream) const {
    if (!spec_.enabled) { return; }
    const auto index = static_cast<std::size_t>(layer);
    if (index >= layer_batches_.size()) { return; }
    const auto& batch = layer_batches_[index];
    if (batch.empty()) { return; }

    // The slot region is split in two and the layers alternate between the halves, so the upload of
    // layer L overwrites the half that layer L-2 read -- *not* the one layer L-1 is reading right
    // now. Holding the upload back until the immediately preceding layer finished (as a single
    // shared event does) makes the rotation a no-op and serializes every transfer behind the
    // previous layer's compute, which is the difference between sum(copy) and max(copy, compute).
    //
    // One event per half, armed one call ahead: at this call the compute stream's tail is the end of
    // layer index-1, which is exactly what the layer *after* next has to wait for before it reuses
    // this half -- so recording now serves the call two steps later, and the wait below consumes
    // what the previous call armed (the end of layer index-2).
    const std::size_t parity = (index - spec_.first_streamed_layer) % 2;

    if (index == spec_.first_streamed_layer) {
        // Pass boundary (a fresh decode step, or the next prefill chunk). Slot 0 was last read by
        // the *previous* pass, and the events armed during that pass are not ordered after it, so
        // re-arm here against the current tail. This is the only point left that has to serialize;
        // inside a pass every upload overlaps the preceding layer's compute.
        CUDA_CHECK(cudaEventRecord(compute_event_[parity], compute_stream));
    }
    CUDA_CHECK(cudaStreamWaitEvent(transfer_stream_, compute_event_[parity], 0));
    CUDA_CHECK(cudaEventRecord(compute_event_[parity ^ 1u], compute_stream));

    for (const auto& copy : batch) {
        CUDA_CHECK(cudaMemcpyAsync(const_cast<std::byte*>(copy.destination), copy.source,
                                   copy.bytes, cudaMemcpyHostToDevice, transfer_stream_));
    }
    // The layer's kernels consume the slot only after its uploads complete.
    CUDA_CHECK(cudaEventRecord(copy_event_, transfer_stream_));
    CUDA_CHECK(cudaStreamWaitEvent(compute_stream, copy_event_, 0));
}

} // namespace ninfer::models::qwen3_5
