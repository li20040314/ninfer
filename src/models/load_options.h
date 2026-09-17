#pragma once

#include "ninfer/types.h"

#include <string_view>

namespace ninfer::models {

struct LoadOptions {
    EnginePurpose purpose          = EnginePurpose::Generation;
    bool vision                    = false;
    SpeculativeBackend speculative = SpeculativeBackend::None;
    ProposalHead proposal_head     = ProposalHead::Full;
    float weight_offload_ratio     = 0.0F;
    // Keep the text embedding table in page-locked host memory, where the row gather reads it
    // directly, instead of pinning its whole footprint in device memory. Only meaningful with
    // weight_offload_ratio set: the freed bytes are handed to the streaming layers.
    bool host_embedding            = false;
    // Contract the output head on the CPU from host memory rather than keeping it device resident.
    // Like host_embedding it only takes effect with weight_offload_ratio set, because the point is
    // to hand the freed device bytes to the streaming layers.
    bool host_output_head          = false;
    // Contract the layers the ratio would otherwise stream on the CPU instead of uploading their
    // weights. Same dependency as host_output_head -- the point is the device bytes it frees and
    // the PCIe traffic it removes -- but it applies to the whole streamed suffix rather than one
    // parameter, so it is the switch that turns the offload path into a CPU-compute path.
    bool host_linear               = false;

    bool operator==(const LoadOptions&) const = default;

    [[nodiscard]] bool speculative_enabled() const noexcept {
        return speculative != SpeculativeBackend::None;
    }

    [[nodiscard]] bool mtp() const noexcept { return speculative == SpeculativeBackend::Mtp; }

    [[nodiscard]] bool dflash() const noexcept { return speculative == SpeculativeBackend::DFlash; }

    [[nodiscard]] bool dflash2() const noexcept {
        return speculative == SpeculativeBackend::DFlash2;
    }

    [[nodiscard]] bool masked_draft() const noexcept { return dflash() || dflash2(); }

    [[nodiscard]] bool proposal_enabled() const noexcept {
        return purpose == EnginePurpose::Generation && speculative != SpeculativeBackend::None &&
               proposal_head == ProposalHead::Optimized;
    }

    [[nodiscard]] std::string_view speculative_component() const noexcept {
        switch (speculative) {
        case SpeculativeBackend::None:
            return {};
        case SpeculativeBackend::Mtp:
            return "mtp";
        case SpeculativeBackend::DFlash:
            return "dflash";
        case SpeculativeBackend::DFlash2:
            return "dflash2";
        }
        return {};
    }
};

[[nodiscard]] constexpr bool is_masked_draft_backend(SpeculativeBackend backend) noexcept {
    return backend == SpeculativeBackend::DFlash || backend == SpeculativeBackend::DFlash2;
}

[[nodiscard]] inline LoadOptions load_options(const EngineOptions& options) noexcept {
    return {.purpose             = options.purpose,
            .vision              = options.enable_vision,
            .speculative         = options.speculative.backend,
            .proposal_head       = options.speculative.proposal_head,
            .weight_offload_ratio = options.weight_offload_ratio,
            .host_embedding       = options.host_embedding,
            .host_output_head     = options.host_output_head,
            .host_linear          = options.host_linear};
}

} // namespace ninfer::models
