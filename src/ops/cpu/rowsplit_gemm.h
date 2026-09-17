#pragma once

// CPU contraction of RowSplit weights. This is the "compute on the CPU" half of the weight
// offload path: a linear whose weight lives in host memory can be evaluated without copying it
// across PCIe at all, turning a per-layer byte stream (~13 GB/s effective) into a local read
// (~34 GB/s measured on this host).
//
// The numeric contract is the engine's A16 one: weights are dequantized exactly as
// (float)q_int * (float)scale_fp16, and the activation enters as the exact fp32 value of its bf16
// encoding. Nothing here re-quantizes the activation.

#include "core/weight.h"

#include <cstdint>

namespace ninfer::ops::cpu {

// True when `weight` is a 2-D RowSplit weight of a format and group size this module contracts
// (q4/q5/q6 with group 64, q8 with group 32). Everything else is rejected rather than approximated.
[[nodiscard]] bool rowsplit_gemm_supported(const Weight& weight) noexcept;

// out[token][n] = dequant(weight[n][:]) . activation[token][:]   (both operands token-major)
//
// All three buffers are HOST memory. `weight` must carry host RowSplit planes (Weight.qdata /
// qhigh / scales). Layout follows the engine's own convention for a [K,T] operand: the activation
// is token-major (`activation[token * k + column]`, the k axis contiguous), and the output bf16 is
// likewise token-major (`out[token * n + row]`). tokens == 1 makes the two orders coincide, which
// is why callers must exercise tokens > 1 before trusting a port of this path.
//
// Throws std::invalid_argument for an unsupported format, a geometry mismatch, or a null plane.
void rowsplit_gemm(const Weight& weight, const void* activation_bf16, void* out_bf16,
                   std::int32_t tokens);

// Worker threads used by the CPU contraction. A value of 0 restores the hardware-derived default
// (physical cores minus one — see rowsplit_gemm.cpp for why the logical count is the wrong number
// to use on a hybrid P/E-core part). Sweeps are expected to call this explicitly.
[[nodiscard]] unsigned rowsplit_worker_threads() noexcept;
void set_rowsplit_worker_threads(unsigned threads) noexcept;

// Whether this host exposes AVX2, i.e. whether the vectorized contraction (rather than the scalar
// reference) will run. Exposed so a caller can report the path it actually took.
[[nodiscard]] bool rowsplit_avx2_available() noexcept;

} // namespace ninfer::ops::cpu
