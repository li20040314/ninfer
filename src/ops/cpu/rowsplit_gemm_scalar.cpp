// RowSplit CPU contraction, scalar reference. Compiled without AVX2 so it stays executable on any
// host, which is what makes it usable both as the fallback for `host_has_avx2() == false` and as
// the implementation for code widths the vectorized path does not cover (q6).

#include "ops/cpu/rowsplit_gemm_detail.h"

#include <cmath>
#include <cstdint>

namespace ninfer::ops::cpu::detail {
namespace {

// The code of one element, in storage order, sign-extended from its width.
//   q4: byte index/2, low nibble for even elements and high nibble for odd ones
//   q5: the same nibble plus bit (index & 7) of high byte (index >> 3), two's complement over 5 bits
//   q6: the same nibble plus bits (index*2, index*2+1) of the high plane, over 6 bits
//   q8: one signed byte per element
inline int decode_code(int bits, const std::uint8_t* codes, const std::uint8_t* high,
                       std::uint64_t index) noexcept {
    if (bits == 8) { return static_cast<std::int8_t>(codes[index]); }

    const std::uint8_t packed = codes[index / 2];
    const std::uint32_t value =
        (index & 1u) != 0u ? static_cast<std::uint32_t>(packed >> 4)
                           : static_cast<std::uint32_t>(packed & 0x0Fu);
    if (bits == 4) { return (value & 0x08u) != 0u ? static_cast<int>(value) - 16
                                                  : static_cast<int>(value); }
    if (bits == 5) {
        const std::uint32_t full =
            value | (static_cast<std::uint32_t>((high[index >> 3] >> (index & 7u)) & 0x01u) << 4);
        return (full & 0x10u) != 0u ? static_cast<int>(full) - 32 : static_cast<int>(full);
    }
    const std::uint64_t position = index * 2;
    const std::uint32_t full =
        value | (static_cast<std::uint32_t>((high[position >> 3] >> (position & 7u)) & 0x03u) << 4);
    return (full & 0x20u) != 0u ? static_cast<int>(full) - 64 : static_cast<int>(full);
}

} // namespace

void gemm_rows_scalar(const RowSplitGeometry& geometry, const std::uint8_t* payload,
                      std::uint64_t padded_columns, std::uint64_t row_begin,
                      std::uint64_t row_end, const float* x, std::int32_t tokens,
                      std::uint64_t out_row_stride, std::uint16_t* out) {
    const std::uint64_t group_size = geometry.group_size;

    for (std::uint64_t row = row_begin; row < row_end; ++row) {
        const std::uint8_t* codes = payload + row * geometry.code_row_bytes;
        const std::uint8_t* high  = geometry.high_row_bytes == 0
                                        ? nullptr
                                        : payload + geometry.high_offset +
                                              row * geometry.high_row_bytes;
        const std::uint8_t* scales =
            payload + geometry.scale_offset + row * geometry.scale_row_bytes;

        for (std::int32_t token = 0; token < tokens; ++token) {
            const float* activation = x + static_cast<std::uint64_t>(token) * padded_columns;
            float accumulator       = 0.0F;

            for (std::uint64_t group = 0; group < geometry.groups; ++group) {
                std::uint16_t scale_bits = 0;
                std::memcpy(&scale_bits, scales + group * 2, 2);
                const float scale = half_to_float(scale_bits);

                // q8 places one element per byte, so its group starts at group*group_size; the
                // nibble formats always pack 64 elements into 32 bytes.
                const std::uint64_t code_origin =
                    geometry.bits == 8 ? group * group_size : group * 32;
                const std::uint8_t* group_high =
                    high == nullptr ? nullptr : high + group * geometry.high_per_group;

                float group_accumulator = 0.0F;
                for (std::uint64_t lane = 0; lane < group_size; ++lane) {
                    const int code =
                        decode_code(geometry.bits, codes + code_origin, group_high, lane);
                    group_accumulator =
                        std::fma(static_cast<float>(code),
                                 activation[group * group_size + lane], group_accumulator);
                }
                accumulator = std::fma(group_accumulator, scale, accumulator);
            }
            out[static_cast<std::uint64_t>(token) * out_row_stride + row] =
                float_to_bf16(accumulator);
        }
    }
}

} // namespace ninfer::ops::cpu::detail
