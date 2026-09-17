#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_ksplit_launch.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {

#if NINFER_ENABLE_LARGE_STATIC_SMEM
namespace {
using Geometry = Q8N2048K4096;
using Access   = Q8KSplitScaleAccess;
using Stage    = Q8KSplitActivationStage;
using C4  = Q8KSplitSchedule<16, 8, 1, Access::Direct, Cache::cg, Cache::cg, Stage::RuntimeActive>;
using C8  = Q8KSplitSchedule<16, 8, 1, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C16 = Q8KSplitSchedule<16, 16, 1, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C24 = Q8KSplitSchedule<8, 24, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C32 = Q8KSplitSchedule<8, 32, 2, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C40 = Q8KSplitSchedule<8, 40, 2, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C48 = Q8KSplitSchedule<8, 48, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C56 = Q8KSplitSchedule<8, 56, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C64 = Q8KSplitSchedule<8, 64, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
} // namespace

Q8Launch select_q8_n2048_k4096(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_ksplit<Geometry, 4, C4>;
    if (tokens <= 8) return launch_q8_ksplit<Geometry, 8, C8>;
    if (tokens <= 16) return launch_q8_ksplit<Geometry, 16, C16>;
    if (tokens <= 24) return launch_q8_ksplit<Geometry, 24, C24>;
    if (tokens <= 32) return launch_q8_ksplit<Geometry, 32, C32>;
    if (tokens <= 40) return launch_q8_ksplit<Geometry, 40, C40>;
    if (tokens <= 48) return launch_q8_ksplit<Geometry, 48, C48>;
    if (tokens <= 56) return launch_q8_ksplit<Geometry, 56, C56>;
    if (tokens <= 64) return launch_q8_ksplit<Geometry, 64, C64>;
    if (tokens < 896) return launch_q8_mma_r32_c128;
    return launch_q8_mma_r64_c128;
}

#else
// The schedules registered for this geometry stage more than the 48 KiB of static __shared__
// this target allows, so the shape is not registered and the dispatcher reports it unsupported.
Q8Launch select_q8_n2048_k4096(std::int32_t) {
    throw std::invalid_argument(
        "q8 linear n2048 k4096: unavailable on this architecture (static shared memory limit)");
}
#endif

} // namespace ninfer::ops::detail
