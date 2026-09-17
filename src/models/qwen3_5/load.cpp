#include "models/qwen3_5/load.h"

#include "artifact/reader.h"
#include "models/qwen3_5/load/bindings.h"
#include "models/qwen3_5/offload_stream.h"

#include <algorithm>
#include <map>
#include <set>
#include <utility>

namespace ninfer::models::qwen3_5 {

struct LoadPlan::Impl {
    Config config;
    LoadOptions options;
    ModelWeights weights;
    std::vector<loading::PendingWeight> pending;
    artifact::MaterializationPlan materialization;
    WeightOffloadSpec offload;
    FrontendResources resources;
    InstanceInfo info;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

LoadPlan::~LoadPlan()                              = default;
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;

const Config& LoadPlan::config() const { return impl_->config; }

const ModelWeights& LoadPlan::weights() const { return impl_->weights; }

const FrontendResources& LoadPlan::resources() const { return impl_->resources; }

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    return impl_->materialization;
}

const artifact::ParameterReference& LoadPlan::parameter(WeightId id) const {
    return impl_->pending.at(id.index).reference;
}

std::span<const WeightUse> LoadPlan::uses(WeightId id) const {
    return impl_->pending.at(id.index).uses;
}

namespace {

constexpr std::uint64_t kOffloadSlotAlignment = 256;

template <class Function>
void for_each_block_weight(const BlockWeights& block, Function&& fn) {
    fn(block.input_norm);
    fn(block.post_attention_norm);
    if (const auto* attention = std::get_if<AttentionWeights>(&block.mixer)) {
        fn(attention->query);
        fn(attention->key);
        fn(attention->gate);
        fn(attention->value);
        fn(attention->query_norm);
        fn(attention->key_norm);
        fn(attention->output);
    } else {
        const auto& gdn = std::get<GdnWeights>(block.mixer);
        fn(gdn.query);
        fn(gdn.key);
        fn(gdn.value);
        fn(gdn.z);
        fn(gdn.a_projection);
        fn(gdn.b_projection);
        fn(gdn.a_log);
        fn(gdn.dt_bias);
        fn(gdn.convolution);
        fn(gdn.norm);
        fn(gdn.output);
    }
    if (const auto* dense = std::get_if<DenseWeights>(&block.ffn)) {
        fn(dense->gate);
        fn(dense->up);
        fn(dense->down);
    } else {
        const auto& moe = std::get<MoeWeights>(block.ffn);
        fn(moe.router);
        fn(moe.shared_score);
        for (const auto& expert : moe.experts) {
            fn(expert.gate);
            fn(expert.up);
            fn(expert.down);
        }
        fn(moe.shared.gate);
        fn(moe.shared.up);
        fn(moe.shared.down);
    }
}

std::uint64_t align_offset(std::uint64_t value, std::uint64_t alignment) {
    const auto remainder = value % alignment;
    return remainder == 0 ? value : value + (alignment - remainder);
}

// Decides the resident/streamed layer split from `options.weight_offload_ratio` (the streamed
// fraction of text-layer weight bytes), flips the affected parameter references to Host
// residency, rewrites the materialization plan accordingly (remaining device placements are
// re-offset because removing objects shifts every later allocation), and freezes the slot
// layout consumed by WeightStreamScheduler.
//
// `options.host_linear` changes what happens to the streamed suffix, not which layers it contains:
// instead of being copied into a rotating device slot each pass they are contracted in place on the
// CPU, so no slot region is allocated and the returned spec has `enabled == false` (no scheduler,
// no uploads). The split itself is shared, which is what lets the two switches compose.
void plan_weight_offload(const LoadOptions& options, std::vector<loading::PendingWeight>& pending,
                         const ModelWeights& weights,
                         artifact::MaterializationPlan& materialization,
                         WeightOffloadSpec& out_spec) {
    WeightOffloadSpec spec;
    const float ratio = options.weight_offload_ratio;
    if (ratio == 0.0F) {
        out_spec = std::move(spec);
        return;
    }
    if (!(ratio > 0.0F && ratio < 1.0F)) {
        throw artifact::ArtifactError("weight_offload_ratio must be within (0,1)");
    }

    std::map<std::size_t, const artifact::DevicePlacement*> placements;
    for (const auto& placement : materialization.device_objects) {
        placements.emplace(placement.object.index, &placement);
    }

    const auto weight_count = pending.size();
    const auto layer_count  = weights.text.layers.size();
    std::vector<std::vector<std::size_t>> layer_ids(layer_count);
    for (std::size_t layer = 0; layer < layer_count; ++layer) {
        for_each_block_weight(weights.text.layers[layer],
                              [&](WeightId id) { layer_ids[layer].push_back(id.index); });
    }

    std::vector<std::size_t> layer_bytes(layer_count, 0);
    std::vector<std::set<std::size_t>> layer_object_indices(layer_count);
    for (std::size_t layer = 0; layer < layer_count; ++layer) {
        for (const auto id : layer_ids[layer]) {
            for (const auto& part : pending[id].reference.binding.parts) {
                if (!layer_object_indices[layer].insert(part.object.index).second) { continue; }
                const auto found = placements.find(part.object.index);
                if (found == placements.end()) {
                    throw artifact::ArtifactError(pending[id].reference.name +
                                                  ": offload layer object lacks device placement");
                }
                layer_bytes[layer] += static_cast<std::size_t>(found->second->bytes);
            }
        }
    }

    std::size_t total_layer_bytes = 0;
    for (const auto bytes : layer_bytes) { total_layer_bytes += bytes; }

    // ---- Parameters the device does not need to hold at all -----------------------------------
    // Two reasons to host a weight, and they differ in how its bytes must be allocated. A device
    // kernel that dereferences the pointer directly (the embedding gather) needs page-locked
    // memory. A CPU contraction only needs the bytes addressable, so it takes the plain host copy
    // the reader already produced and skips the page-locking pass entirely.
    //
    // The embedding table is consulted as a row gather: a handful of rows per decode step out of
    // 248320. Resident, it pins its whole footprint forever; read straight out of page-locked host
    // memory it costs ~8 us per step (measured for one cold 5440-byte row, against 9.5 us for the
    // same row resident in device memory).
    //
    // The output head is the opposite case and is why it needs a contraction rather than a gather:
    // every step reads all of it. Hosting it only pays off because ops::linear routes a host
    // pointer to the CPU RowSplit kernel, which reads those bytes at DRAM bandwidth instead of
    // pushing them over PCIe; the device footprint it frees is handed to the resident layers.
    std::map<std::size_t, bool> host_objects; // object index -> page_locked
    std::size_t host_only_bytes = 0;
    const auto place_on_host = [&](WeightId id, bool page_locked) {
        auto& reference           = pending[id.index].reference;
        reference.residency       = artifact::Residency::Host;
        for (const auto& part : reference.binding.parts) {
            if (!host_objects.emplace(part.object.index, page_locked).second) { continue; }
            host_only_bytes += static_cast<std::size_t>(placements.at(part.object.index)->bytes);
        }
    };
    if (options.host_embedding) { place_on_host(weights.text.token_embedding, /*page_locked=*/true); }
    if (options.host_output_head) {
        place_on_host(weights.text.output_head, /*page_locked=*/false);
    }

    // `ratio` is the fraction of text-layer weight bytes that streams from host (offloaded to
    // CPU); the leading layers up to (1 - ratio) of the payload stay device resident. Freed
    // footprints are handed back to the resident layers rather than left idle.
    const auto target_resident = static_cast<std::size_t>(
        static_cast<double>(total_layer_bytes) * (1.0 - static_cast<double>(ratio))) +
                                 host_only_bytes;

    std::size_t accumulated    = 0;
    std::size_t first_streamed = layer_count;
    for (std::size_t layer = 0; layer < layer_count; ++layer) {
        if (accumulated >= target_resident) {
            first_streamed = layer;
            break;
        }
        accumulated += layer_bytes[layer];
    }
    if (first_streamed == layer_count) { first_streamed = 0; }

    std::vector<char> streamed_weight(weight_count, 0);
    for (std::size_t layer = first_streamed; layer < layer_count; ++layer) {
        for (const auto id : layer_ids[layer]) { streamed_weight[id] = 1; }
    }
    std::set<std::size_t> resident_objects;
    for (std::size_t id = 0; id < weight_count; ++id) {
        if (streamed_weight[id] != 0) { continue; }
        for (const auto& part : pending[id].reference.binding.parts) {
            resident_objects.insert(part.object.index);
        }
    }
    for (std::size_t layer = first_streamed; layer < layer_count; ++layer) {
        for (const auto object_index : layer_object_indices[layer]) {
            if (resident_objects.count(object_index) != 0) {
                throw artifact::ArtifactError(
                    "weight offload: layer " + std::to_string(layer) +
                    " shares weight objects with device-resident parameters");
            }
        }
    }

    // ---- CPU-computed suffix (`--host-linear`) -------------------------------------------------
    // The streamed layers keep the Host residency the split gave them and take the plain host copy
    // the reader already produced: no device slot region is allocated and no per-layer upload
    // happens at all. The consumers route a host-resident weight to the CPU RowSplit contraction,
    // which reads the same bytes at DRAM speed instead of pushing them over PCIe -- so this switch
    // removes the transfer the streaming path spent its time on rather than trying to hide it.
    //
    // Everything else is deliberately shared with the streaming path: the split, the residency
    // flip, the housekeeping that drops the streamed objects' device placements, and the
    // freed-byte accounting all run identically. `spec.enabled == false` is the only difference
    // that reaches the runtime, and it is what keeps the scheduler -- and its slot allocation --
    // out of the picture.
    const bool cpu_layers = options.host_linear;
    if (cpu_layers) {
        // Only the weights whose consumers have a CPU contraction route may leave the device:
        // the FFN gate/up pair (ops::linear_swiglu) and down, the mixer output projection
        // (ops::linear_add), and the fused input projections (ops::attn_input_proj /
        // ops::gdn_input_proj host routes for their paired Q4/Q5 parents). Everything else in a
        // streamed layer -- the gating projections and the per-block norms -- is read by device
        // kernels with no CPU counterpart, so hosting it would hand those kernels a host pointer
        // and fault. Those weights therefore stay device-resident (and keep their device
        // footprint); the ratio still picks which layers are affected, but a hosted suffix frees
        // nearly all of a streamed layer's bytes rather than half of them.
        //
        // The GDN projection weights are also consumed by the Verify-phase conv snapshot/record
        // variants, which have no host route -- unreachable here because host-linear refuses
        // speculative decoding up front.
        //
        // These tensors are page-locked at allocation, not left pageable: every token sweeps the
        // entire hosted suffix out of DRAM, and under real commit pressure (this host runs at
        // ~70 GB commit against 87 GB) Windows trims the arena's working set and decode spends
        // its time reading weights back from the pagefile -- measured at up to 200k hard faults
        // per second, 2x the whole contraction's time. Page-locking is a property of the
        // consumer: "the CPU reads it every token" needs residency just as much as a GPU kernel
        // reading it zero-copy does. ~12 GiB fits inside this host's ~13 GiB lock ceiling with
        // room for the embedding's own pinning; if the allocation cannot be locked the loader
        // fails loudly rather than silently degrading to a thrashing run.
        for (std::size_t layer = first_streamed; layer < layer_count; ++layer) {
            const auto& block = weights.text.layers[layer];
            if (const auto* dense = std::get_if<DenseWeights>(&block.ffn)) {
                place_on_host(dense->gate, /*page_locked=*/true);
                place_on_host(dense->up, /*page_locked=*/true);
                place_on_host(dense->down, /*page_locked=*/true);
            }
            if (const auto* attention = std::get_if<AttentionWeights>(&block.mixer)) {
                // The paired projection consumes all four rows-parents together, so they leave or
                // stay as a set.
                place_on_host(attention->query, /*page_locked=*/true);
                place_on_host(attention->key, /*page_locked=*/true);
                place_on_host(attention->gate, /*page_locked=*/true);
                place_on_host(attention->value, /*page_locked=*/true);
                place_on_host(attention->output, /*page_locked=*/true);
            } else if (const auto* gdn = std::get_if<GdnWeights>(&block.mixer)) {
                place_on_host(gdn->query, /*page_locked=*/true);
                place_on_host(gdn->key, /*page_locked=*/true);
                place_on_host(gdn->value, /*page_locked=*/true);
                place_on_host(gdn->z, /*page_locked=*/true);
                place_on_host(gdn->output, /*page_locked=*/true);
            }
        }
        spec.enabled              = false;
        spec.first_streamed_layer = static_cast<std::uint32_t>(first_streamed);
    } else {
        spec.enabled              = true;
        spec.first_streamed_layer = static_cast<std::uint32_t>(first_streamed);
        spec.weight_objects.assign(weight_count, {});
        spec.layer_objects.assign(layer_count, {});
        std::map<std::size_t, std::size_t> object_slot_index;
        for (std::size_t layer = first_streamed; layer < layer_count; ++layer) {
            auto& layer_slots = spec.layer_objects[layer];
            for (const auto id : layer_ids[layer]) {
                pending[id].reference.residency = artifact::Residency::Host;
                auto& part_slots                = spec.weight_objects[id];
                if (!part_slots.empty()) { continue; }
                part_slots.reserve(pending[id].reference.binding.parts.size());
                for (const auto& part : pending[id].reference.binding.parts) {
                    auto found = object_slot_index.find(part.object.index);
                    if (found == object_slot_index.end()) {
                        const auto& placement = *placements.at(part.object.index);
                        found =
                            object_slot_index.emplace(part.object.index, spec.objects.size()).first;
                        spec.objects.push_back(OffloadedObject{part.object, 0, placement.bytes});
                    }
                    part_slots.push_back(found->second);
                    if (std::find(layer_slots.begin(), layer_slots.end(), found->second) ==
                        layer_slots.end()) {
                        layer_slots.push_back(found->second);
                    }
                }
            }
        }

        // Slots rotate between two device regions: layer L uploads into slot L%2 while layer L-1's
        // kernels still read slot (L-1)%2, so the copy engine stays saturated without racing the
        // compute stream. The device allocation only needs 2x the largest single-layer footprint
        // instead of the full streamed payload; slot addresses stay fixed across steps, only the
        // slot *contents* are rewritten by ensure_layer() before each layer executes.
        std::uint64_t max_layer_slot = 0;
        for (std::size_t layer = first_streamed; layer < layer_count; ++layer) {
            std::uint64_t running = 0;
            for (const auto object_index : spec.layer_objects[layer]) {
                auto& entry = spec.objects[object_index];
                entry.slot_offset = align_offset(running, kOffloadSlotAlignment);
                running           = entry.slot_offset + entry.bytes;
            }
            max_layer_slot = std::max(max_layer_slot, running);
        }
        const std::uint64_t slot_stride = align_offset(max_layer_slot, kOffloadSlotAlignment);
        for (std::size_t layer = first_streamed; layer < layer_count; ++layer) {
            const auto parity_offset =
                static_cast<std::uint64_t>(layer - first_streamed) % 2 * slot_stride;
            for (const auto object_index : spec.layer_objects[layer]) {
                auto& entry = spec.objects[object_index];
                entry.slot_offset += parity_offset;
            }
        }
        // streamed_bytes is the slot-region size (two rotating single-layer footprints), not the
        // total streamed payload.
        spec.streamed_bytes = static_cast<std::size_t>(2 * slot_stride);
    }

    std::set<std::size_t> offloaded_objects;
    for (const auto& entry : spec.objects) { offloaded_objects.insert(entry.object.index); }
    std::vector<artifact::DevicePlacement> device_objects;
    device_objects.reserve(materialization.device_objects.size());
    std::uint64_t running_device = 0;
    for (auto placement : materialization.device_objects) {
        if (offloaded_objects.count(placement.object.index) != 0) {
            materialization.host_objects.push_back(
                artifact::HostPlacement{placement.object, {}});
            continue;
        }
        const auto hosted = host_objects.find(placement.object.index);
        if (hosted != host_objects.end()) {
            // `page_locked` decides the storage: a device kernel reading these bytes directly
            // cannot take ordinary pageable memory, while a CPU contraction has no such
            // requirement and would only pay for pinning it.
            materialization.host_objects.push_back(
                artifact::HostPlacement{placement.object, {}, hosted->second});
            continue;
        }
        placement.offset = align_offset(running_device, placement.alignment);
        running_device   = placement.offset + placement.bytes;
        device_objects.push_back(placement);
    }
    materialization.device_objects       = std::move(device_objects);
    materialization.device_capacity_bytes = running_device;
    spec.resident_bytes                        = static_cast<std::size_t>(running_device);

    out_spec = std::move(spec);
}

} // namespace

LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options) {
    auto out     = std::make_unique<LoadPlan::Impl>();
    out->options = options;
    out->config  = parse_config(reader.directory(), options);
    artifact::Binder binder(reader);
    out->resources = loading::bind_resources(binder, out->config);
    loading::Bindings bindings(binder);
    const auto& text  = out->config.text;
    out->weights.text = loading::bind_text(bindings, text, options);
    if (out->config.vision) {
        out->weights.vision = loading::bind_vision(bindings, *out->config.vision, text);
    }
    if (out->config.mtp) {
        out->weights.mtp = loading::bind_mtp(bindings, text, out->weights.text);
    }
    if (out->config.draft) {
        out->weights.draft =
            loading::bind_draft(bindings, *out->config.draft, text, out->weights.text,
                                std::string(options.speculative_component()));
    }
    if (options.proposal_enabled()) {
        const auto& proposal = reader.directory().component("text").proposal;
        if (!proposal) {
            throw artifact::ArtifactError("selected proposal head is absent from artifact");
        }
        out->weights.proposal = loading::bind_proposal(bindings, *proposal, text, options,
                                                       out->resources.public_token_count);
        if (out->config.draft && out->config.draft->dflash2) {
            const auto domain =
                proposal->indexed ? proposal->rows : out->resources.public_token_count;
            if (out->config.draft->dflash2->selector_top_k > domain) {
                throw artifact::ArtifactError(
                    "proposal domain is smaller than DFlash2 selector_top_k");
            }
        }
        if (out->weights.mtp) { out->weights.mtp->output_head = out->weights.proposal->head; }
        if (out->weights.draft) { out->weights.draft->output_head = out->weights.proposal->head; }
    }
    out->weights.text.output_head_use =
        bindings.use(out->weights.text.output_head, "text/final_hidden");
    if (out->weights.mtp) {
        out->weights.mtp->output_head_use =
            bindings.use(out->weights.mtp->output_head, "mtp/final_hidden");
    }
    if (out->weights.draft) {
        out->weights.draft->output_head_use =
            bindings.use(out->weights.draft->output_head,
                         std::string(options.speculative_component()) + "/final_hidden");
    }
    out->pending         = std::move(bindings.weights);
    out->materialization = std::move(binder).finish();
    plan_weight_offload(out->options, out->pending, out->weights, out->materialization,
                        out->offload);
    out->info.name       = reader.directory().metadata.value(
        "name", std::string(architecture_name(text.architecture)));
    out->info.metadata_json   = reader.directory().metadata.dump();
    out->info.provenance_json = reader.directory().provenance.dump();
    out->info.artifact_id     = reader.artifact_id();
    return LoadPlan(std::move(out));
}

std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                         const StartupObserver* observer) {
    if (!plan.impl_) { throw artifact::ArtifactError("load plan was already consumed"); }
    auto data    = std::move(plan.impl_);
    auto backing = artifact::materialize(*data->materialization.source,
                                         std::move(data->materialization), device, observer);
    auto bound   = loading::resolve_weights(std::move(data->pending), backing);
    std::unique_ptr<WeightStreamScheduler> scheduler;
    if (data->offload.enabled) {
        scheduler = std::make_unique<WeightStreamScheduler>(data->offload, backing,
                                                            device.transfer_stream);
        for (std::size_t id = 0; id < bound.size(); ++id) {
            const auto& object_slots = data->offload.weight_objects[id];
            if (object_slots.empty()) { continue; }
            auto& view = bound[id].view;
            if (object_slots.size() != view.parts.size()) {
                throw artifact::ArtifactError(bound[id].name +
                                              ": offload slot mapping differs from bound view");
            }
            for (std::size_t part = 0; part < view.parts.size(); ++part) {
                view.parts[part].parent = &scheduler->parent(object_slots[part]);
            }
        }
    }
    return std::unique_ptr<Model>(new Model(
        std::move(data->config), data->options, std::move(data->weights), std::move(bound),
        std::move(data->resources), std::move(data->info), std::move(backing),
        std::move(scheduler)));
}

std::unique_ptr<Model> load_model(const std::filesystem::path& path, LoadOptions options,
                                  DeviceContext& device, const StartupObserver* observer) {
    artifact::Reader reader(path);
    return materialize_model(plan_load(reader, options), device, observer);
}

} // namespace ninfer::models::qwen3_5
