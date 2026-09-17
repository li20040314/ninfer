#include "core/weight.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_geometry.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_rowsplit_gemm_simt.cuh"
#include "ops/linear/q4/q4_rowsplit_gemv.cuh"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"
#include "ops/linear/q5/q5_rowsplit_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// The shape-specialised kernels below bake the projection geometry into template arguments, so each
// registered geometry needs its own instantiation. Taking the numbers from the geometry table keeps
// that table the single source of truth: a profile cannot drift from the tuple the plan admits.
//
// The Q4 decode schedule is the one thing the table cannot express. `Q4GemvR1Q8DirectSchedule`
// bakes 5120/64 = 80 groups per row, so it only decodes a contraction width of 5120;
// `Q4GemvR4W1DirectSchedule` derives the group count from the runtime K and is the general choice.
template <Q4Q5GdnInputGeometry Geometry, class Q4GemvSchedule>
struct GdnProfile {
    static constexpr std::int32_t kQkRows     = Geometry.qk_rows;
    static constexpr std::int32_t kValueRows  = Geometry.value_rows;
    static constexpr std::int32_t kZRows      = Geometry.z_rows;
    static constexpr std::int32_t kValueZRows = Geometry.value_z_rows;
    static constexpr std::int32_t kHidden     = Geometry.input_rows;
    // One Q5 SIMT slab covers 1024 contraction elements, so the slab count is a function of the
    // geometry. It was baked as 5120/1024 == 5 while the 27B profile was the only one, and a fixed
    // 5 makes the Q5 side read past the activation and the weight row for any shorter contraction.
    static constexpr std::int32_t kSlabs = Geometry.input_rows / 1024;
    static_assert(Geometry.input_rows % 1024 == 0,
                  "the Q5 small-T kernels stage whole 1024-value slabs");
    using Q4Gemv = Q4GemvSchedule;
};

using Gdn27BProfile = GdnProfile<kQ4Q5GdnInput27B, Q4GemvR1Q8DirectSchedule>;
using Gdn9BProfile  = GdnProfile<kQ4Q5GdnInput9B, Q4GemvR4W1DirectSchedule>;

using Q4GdnSimtR8C4Schedule = Q4RowSplitSimtGemmSchedule<8, 4, 16, 2, Cache::ca, 1>;
using Q4GdnSimtR8C8Schedule = Q4RowSplitSimtGemmSchedule<8, 8, 16, 2, Cache::ca, 1>;

template <class Profile>
void launch_q4_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = typename Profile::Q4Gemv;
    const dim3 grid(static_cast<unsigned>(div_up(Profile::kQkRows, Schedule::kRowsPerCta)), 1u, 1u);
    constexpr dim3 block(static_cast<unsigned>(Schedule::kThreads), 1u, 1u);
    q4_rowsplit_gemv_kernel<Schedule><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
        nullptr, Profile::kQkRows, Profile::kHidden);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, bool Full, class Profile>
void launch_q4_simt(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t cols   = x.ne[1];
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(div_up(Profile::kQkRows, Schedule::kRowsPerCta)),
                    static_cast<unsigned>(div_up(cols, Schedule::kColsPerTile)), 1u);
    q4_rowsplit_gemm_simt_kernel<Schedule, Full><<<grid, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
        nullptr, out_ld, 0, Profile::kQkRows, Profile::kHidden, cols, weight.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, class Profile>
void launch_q4_simt_route(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const bool full =
        (Profile::kQkRows % Schedule::kRowsPerCta) == 0 &&
        ((Profile::kHidden / Q4RowSplitStorage::kGroupK) % Schedule::kGroupsPerStage) == 0 &&
        (x.ne[1] % Schedule::kColsPerTile) == 0;
    if (full) {
        launch_q4_simt<Schedule, true, Profile>(x, weight, out, stream);
    } else {
        launch_q4_simt<Schedule, false, Profile>(x, weight, out, stream);
    }
}

template <class Profile>
void launch_q4(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (x.ne[1] == 1) {
        launch_q4_gemv<Profile>(x, weight, out, stream);
        return;
    }
    if (x.ne[1] <= 4) {
        launch_q4_simt_route<Q4GdnSimtR8C4Schedule, Profile>(x, weight, out, stream);
        return;
    }
    if (x.ne[1] <= 15) {
        launch_q4_simt_route<Q4GdnSimtR8C8Schedule, Profile>(x, weight, out, stream);
        return;
    }
    throw std::invalid_argument("Q4/Q5 GDN independent launch requires T in [1,15]");
}

template <class Profile>
void launch_q5_gemv(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                    cudaStream_t stream) {
    constexpr int kRowsPerBlock = 16;
    constexpr int kThreads      = kRowsPerBlock * 32;
    q5_rowsplit_gemv_kernel<Profile::kValueZRows, Profile::kHidden, kRowsPerBlock, 2, true, false,
                            true, Profile::kValueRows>
        <<<Profile::kValueZRows / kRowsPerBlock, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data));
    CUDA_CHECK(cudaGetLastError());
}

template <int Cols, class Profile>
void launch_q5_split4(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                      cudaStream_t stream) {
    constexpr int kThreads    = 4 * 32;
    const std::int32_t out_ld = static_cast<std::int32_t>(value.nb[1] / sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(Profile::kValueZRows), 1u, 1u);
    q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Cols, Profile::kSlabs,
                                        Profile::kHidden, true, Profile::kValueRows>
        <<<grid, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                        static_cast<const std::uint8_t*>(weight.qdata),
                                        static_cast<const std::uint8_t*>(weight.qhigh),
                                        static_cast<const std::uint8_t*>(weight.scales),
                                        static_cast<__nv_bfloat16*>(value.data),
                                        static_cast<__nv_bfloat16*>(z.data), Profile::kValueZRows,
                                        out_ld, Profile::kHidden, Cols, weight.padded_shape[1],
                                        Profile::kSlabs);
    CUDA_CHECK(cudaGetLastError());
}

template <class Profile>
void launch_q5_split4_exact(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                            cudaStream_t stream) {
    switch (x.ne[1]) {
    case 2:
        launch_q5_split4<2, Profile>(x, weight, value, z, stream);
        return;
    case 3:
        launch_q5_split4<3, Profile>(x, weight, value, z, stream);
        return;
    case 4:
        launch_q5_split4<4, Profile>(x, weight, value, z, stream);
        return;
    case 5:
        launch_q5_split4<5, Profile>(x, weight, value, z, stream);
        return;
    case 6:
        launch_q5_split4<6, Profile>(x, weight, value, z, stream);
        return;
    default:
        throw std::invalid_argument("GDN Q5 split4 requires T in [2,6]");
    }
}

template <class Profile>
void launch_q5_simt_r8_c8(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                          cudaStream_t stream) {
    constexpr int kColsPerTile  = 8;
    constexpr int kRowsPerBlock = 8;
    constexpr int kStages       = 2;
    constexpr int kThreads      = kRowsPerBlock * 32;
    const std::int32_t cols     = x.ne[1];
    const std::int32_t out_ld   = static_cast<std::int32_t>(value.nb[1] / sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(div_up(Profile::kValueZRows, kRowsPerBlock)),
                    static_cast<unsigned>(div_up(cols, kColsPerTile)), 1u);
    q5_rowsplit_gemm_simt_kernel<Q5RowSplitSimtSchedule, kColsPerTile, kRowsPerBlock, kStages, true,
                                 Profile::kValueRows><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.qhigh),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(value.data),
        static_cast<__nv_bfloat16*>(z.data), Profile::kValueZRows, out_ld, Profile::kHidden, cols,
        weight.padded_shape[1], Profile::kSlabs);
    CUDA_CHECK(cudaGetLastError());
}

template <class Profile>
void launch_q5(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
               cudaStream_t stream) {
    if (x.ne[1] == 1) {
        launch_q5_gemv<Profile>(x, weight, value, z, stream);
        return;
    }
    if (x.ne[1] <= 6) {
        launch_q5_split4_exact<Profile>(x, weight, value, z, stream);
        return;
    }
    if (x.ne[1] <= 15) {
        launch_q5_simt_r8_c8<Profile>(x, weight, value, z, stream);
        return;
    }
    throw std::invalid_argument("Q4/Q5 GDN independent launch requires T in [1,15]");
}

template <class Profile>
void launch_t4_pdl(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                   Tensor& qk, Tensor& value, Tensor& z, cudaStream_t stream) {
    using Q4Schedule         = Q4GdnSimtR8C4Schedule;
    constexpr int kQ5Threads = 4 * 32;
    const dim3 q4_grid(Profile::kQkRows / Q4Schedule::kRowsPerCta, 1u, 1u);
    const dim3 q5_grid(Profile::kValueZRows, 1u, 1u);
    const std::int32_t q4_out_ld = static_cast<std::int32_t>(qk.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t q5_out_ld = static_cast<std::int32_t>(value.nb[1] / sizeof(__nv_bfloat16));

    // Q5 and Q4 publish disjoint row ranges. Q4 can execute while Q5 drains and joins Q5 only at
    // exit, before the following convolution/snapshot kernel becomes runnable.
    q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, 4, Profile::kSlabs,
                                        Profile::kHidden, true, Profile::kValueRows,
                                        Q5Split4StoreEpilogue, true, false>
        <<<q5_grid, kQ5Threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(value_z_weight.qdata),
            static_cast<const std::uint8_t*>(value_z_weight.qhigh),
            static_cast<const std::uint8_t*>(value_z_weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
            Profile::kValueZRows, q5_out_ld, Profile::kHidden, 4,
            value_z_weight.padded_shape[1], Profile::kSlabs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(pdl::launch_dependent(
        {q4_grid, dim3(Q4Schedule::kThreads), 0, stream},
        q4_rowsplit_gemm_simt_kernel<Q4Schedule, true, false, 0, Q4SimtStoreEpilogue, false, true>,
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(qk_weight.qdata),
        static_cast<const std::uint8_t*>(qk_weight.scales), static_cast<__nv_bfloat16*>(qk.data),
        nullptr, q4_out_ld, 0, Profile::kQkRows, Profile::kHidden, 4, qk_weight.padded_shape[1],
        Q4SimtStoreEpilogue{}));
}

template <class Profile>
void launch_independent(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                        Tensor& qk, Tensor& value, Tensor& z, cudaStream_t stream) {
    if (x.ne[1] == 4) {
        launch_t4_pdl<Profile>(x, qk_weight, value_z_weight, qk, value, z, stream);
        return;
    }
    launch_q4<Profile>(x, qk_weight, qk, stream);
    launch_q5<Profile>(x, value_z_weight, value, z, stream);
}

} // namespace

void q4_q5_gdn_input_independent_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, Tensor& qk, Tensor& value,
                                        Tensor& z, cudaStream_t stream) {
    // The fused q/k, value, and z outputs name the same tuple the plan admitted, so the geometry is
    // recovered from the operands rather than passed down a second time.
    const Q4Q5GdnInputGeometry geometry =
        q4_q5_gdn_input_geometry(x.ne[0], qk.ne[0], value.ne[0], z.ne[0]);
    if (same_geometry(geometry, kQ4Q5GdnInput27B)) {
        launch_independent<Gdn27BProfile>(x, qk_weight, value_z_weight, qk, value, z, stream);
        return;
    }
    if (same_geometry(geometry, kQ4Q5GdnInput9B)) {
        launch_independent<Gdn9BProfile>(x, qk_weight, value_z_weight, qk, value, z, stream);
        return;
    }
    throw std::invalid_argument(
        "Q4/Q5 GDN independent launch: the operand shapes name no registered geometry");
}

} // namespace ninfer::ops::detail
