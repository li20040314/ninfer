#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// True when the weight's planes live in host memory, in which case the contraction has to run on
// the CPU rather than being launched as a device kernel.
//
// This is how the offload path gets a CPU contraction without any consumer having to declare where
// its weights ended up: `bind_view` already selects `host_parent()` for objects marked
// `Residency::Host` (the mechanism `--host-embedding` uses), so a host-resident weight arrives here
// carrying a plain host pointer, and a device-resident one carries a CUDA allocation.
//
// The two are distinguishable. A CUDA allocation always resolves and reports Host, Device or
// Managed; ordinary host memory comes back as `cudaMemoryTypeUnregistered`, or on older runtimes
// does not resolve at all. Either of those host shapes selects the CPU route.
[[nodiscard]] bool weight_is_host_resident(const Weight& weight);

// Device activation -> CPU contraction against a host-resident weight -> device output.
//
// The activation crosses the bus once per call, which is 2*K*T bytes; at the shipped hidden sizes
// and T=1 that is a few tens of kilobytes, against the megabytes of weight it replaces on the PCIe
// path.
void linear_host(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

// Fused gate/up projection + SwiGLU against a host-resident weight: out = SiLU(gate) * up.
//
// The contraction is the same one `linear_host` runs -- the full [2M,K] parent, gate rows [0,M)
// ahead of their matching up rows [M,2M) -- and the SwiGLU is folded into the host side of it. That
// is worth doing here rather than uploading the projection for a device `silu_mul`: the gate half
// is consumed by the activation and never needed on the device, so materializing it locally saves
// half the writeback (M*T elements instead of 2*M*T) on every call, and at prefill extents that is
// the difference between one layer's writeback and two.
void linear_swiglu_host(const Tensor& x, const Weight& gate_up, Tensor& out, cudaStream_t stream);

// residual += W @ x against a host-resident weight, with the addition folded into the writeback.
//
// The block being read-modify-written is the residual, not a fresh projection, so the readback of
// the old value is what makes this a separate entry point rather than a call to `linear_host`: the
// two bus crossings (activation and residual) are issued together and retired by one sync, and the
// sum happens in fp32 on the CPU. Rounding once at the end matches what the device epilogue does.
void linear_add_host(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream);

// Paired Q/K + gate/V projection against host-resident weights (the 27B Q4/Q5 form):
//
//   q    = query_key  [0, query_rows) @ x
//   k    = query_key  [query_rows, query_rows + kv_rows) @ x
//   gate = gate_value [0, query_rows) @ x
//   v    = gate_value [query_rows, query_rows + kv_rows) @ x.
//
// One activation readback serves both parents; each parent is contracted whole and the two row
// ranges of each staging buffer are written back to their own output tensor.
void attn_input_proj_host(const Tensor& x, const Weight& query_key, const Weight& gate_value,
                          Tensor& q, Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);

// Paired Q/K + value/Z projection against host-resident weights (the 27B Q4/Q5 form):
//
//   qkv = [ qk @ x ; value_z [0, value_rows) @ x ]
//   z   = value_z [value_rows, value_rows + z_rows) @ x.
void gdn_input_proj_host(const Tensor& x, const Weight& qk, const Weight& value_z, Tensor& qkv,
                         Tensor& z, cudaStream_t stream);

} // namespace ninfer::ops::detail
