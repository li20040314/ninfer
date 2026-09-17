#pragma once

// Internal interface between the RowSplit CPU contraction and its vectorized translation unit.
// The AVX2 file is compiled with /arch:AVX2 (or -mavx2), so it must never be entered on a host
// that lacks AVX2 — the dispatcher in rowsplit_gemm.cpp checks rowsplit_avx2_available() first.
// The scalar file is compiled without AVX2 so the fallback stays executable everywhere.

#include <cstdint>
#include <cstring>
#include <limits>

namespace ninfer::ops::cpu::detail {

// IEEE half (the RowSplit scale word) to float. Exact for every finite half, including subnormals,
// so the dequantized weight carries no error of its own.
inline float half_to_float(std::uint16_t half) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(half & 0x8000u) << 16;
    const std::uint32_t exp  = (half >> 10) & 0x1Fu;
    const std::uint32_t mant = half & 0x03FFu;
    std::uint32_t bits       = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // Subnormal: normalize so the exponent field becomes the implied leading shift.
            int shift = 0;
            std::uint32_t m = mant;
            while ((m & 0x0400u) == 0) {
                m <<= 1;
                ++shift;
            }
            bits = sign | static_cast<std::uint32_t>(127 - 15 - shift) << 23 |
                   ((m & 0x03FFu) << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F80'0000u | (mant << 13); // Inf / NaN, payload preserved
    } else {
        bits = sign | (exp + (127 - 15)) << 23 | (mant << 13);
    }
    float out = 0.0F;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// bf16 to the float it encodes exactly (bf16 is a truncated fp32, so this is a shift).
inline float bf16_to_float(std::uint16_t value) noexcept {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float out                = 0.0F;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// fp32 to bf16 with round-to-nearest-even, matching the CUDA conversion the device path stores.
// NaN is quieted rather than truncated into a signalling payload.
inline std::uint16_t float_to_bf16(float value) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7FFF'FFFFu) > 0x7F80'0000u) {
        return static_cast<std::uint16_t>((bits >> 16) | 0x0040u);
    }
    const std::uint32_t rounding = 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>((bits + rounding) >> 16);
}

// RowSplit row geometry, derived once per call from the Weight and shared by both contractions.
struct RowSplitGeometry {
    int bits              = 0;  // code width: 4, 5, 6 or 8
    std::uint32_t group_size = 0;
    std::uint64_t groups      = 0;  // groups per row, over the padded column count

    std::uint64_t code_row_bytes  = 0;
    std::uint64_t high_row_bytes  = 0;  // q5/q6 only, 0 otherwise
    std::uint64_t scale_row_bytes = 0;
    std::uint64_t high_per_group  = 0;
    std::uint64_t high_offset     = 0;
    std::uint64_t scale_offset    = 0;
};

// One row block: out[token][row] for row in [row_begin, row_end), token in [0, tokens).
//   payload       - base of the RowSplit payload; row r's planes start at
//                   payload + r*code_row_bytes, payload + high_offset + r*high_row_bytes and
//                   payload + scale_offset + r*scale_row_bytes.
//   x             - [tokens][padded_columns] fp32 activation of the bf16 input, in storage order
//   out           - [tokens][n] bf16, written as out[token*out_row_stride + row]
//
// Both operands are token-major, and that is the engine's layout rather than a choice made here:
// the device kernels index the activation as `x[token * k + ...]` (bf16_n256_k5120.cuh,
// fp8_a16_ksplit_mma.cuh), so a [K,T] tensor's token axis is its outer one. The distinction only
// shows up for tokens > 1, which is why a decode-shaped (tokens == 1) test proves nothing about it.
void gemm_rows(const RowSplitGeometry& geometry, const std::uint8_t* payload,
               std::uint64_t padded_columns, std::uint64_t row_begin, std::uint64_t row_end,
               const float* x, std::int32_t tokens, std::uint64_t out_row_stride,
               std::uint16_t* out);

// Scalar reference; identical semantics, no requirement on the host ISA. Also the implementation
// used for code widths the vectorized path does not cover (q6).
void gemm_rows_scalar(const RowSplitGeometry& geometry, const std::uint8_t* payload,
                      std::uint64_t padded_columns, std::uint64_t row_begin,
                      std::uint64_t row_end, const float* x, std::int32_t tokens,
                      std::uint64_t out_row_stride, std::uint16_t* out);

// Defined in the AVX2 translation unit; reports the host's AVX2 support, not the build's.
[[nodiscard]] bool host_has_avx2() noexcept;

} // namespace ninfer::ops::cpu::detail
