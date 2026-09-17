// RowSplit CPU contraction, vectorized. Compiled with AVX2 enabled for this translation unit only
// (see src/CMakeLists.txt): the rest of the engine keeps the default host ISA, so the dispatcher in
// rowsplit_gemm.cpp must confirm host_has_avx2() before calling in here.
//
// The expansion below is written out longhand rather than through a `chunks[8]` array on purpose.
// Materializing the unpacked nibbles to the stack and reading them back eight bytes at a time makes
// every chunk a store/load pair of mismatched width, which the store buffer cannot forward: the
// kernel then pays a forwarding stall per chunk and runs several times slower than the instruction
// count suggests. Keeping the codes in registers costs a handful of extra vector moves and removes
// the stall entirely.

#include "ops/cpu/rowsplit_gemm_detail.h"

#include <immintrin.h>

#include <cstdint>
#include <cstring>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace ninfer::ops::cpu::detail {
namespace {

// 2 KiB of bit-plane expansion, L1-resident. The plane's semantics are fixed: high byte c carries
// the 5th bit of element 8c+s in bit s. The *layout* a consumer needs depends on how it re-laces
// the codes, and this one re-laces them into element order --
//
//   `unpacklo_epi8`/`unpackhi_epi8` of the low and high nibble planes interleave byte j's two
//   nibbles back into elements 2j and 2j+1, which is exactly the store order, so the emitted 64
//   values come out in element order and the bit plane must be too: byte s = bit s of the source
//   byte c, i.e. the bit of element 8c+s.
//
// Getting this wrong is silent: the shape stays right and only the 5th bit lands on the wrong
// element, which surfaces as a relative-L2 around 1.0 instead of 1e-3. The table is built
// explicitly rather than derived from bit arithmetic at the use site for exactly that reason.
struct HighTables {
    std::uint8_t slot[256][8];
};

const HighTables& high_tables() {
    static const HighTables tables = [] {
        HighTables built{};
        for (int byte = 0; byte < 256; ++byte) {
            for (int i = 0; i < 8; ++i) {
                built.slot[byte][i] = static_cast<std::uint8_t>((byte >> i) & 1);
            }
        }
        return built;
    }();
    return tables;
}

// Eight codes, one 128-bit lane of a widened nibble pair, against eight activation columns. The
// widening and the float conversion are separate instructions (`vpmovsxbd`, `vcvtdq2ps`) and both
// keep their operands in registers, so a group is a straight chain of dependent FMAs.
inline __m256 contract8(__m256 accumulator, __m128i codes, const float* activation) {
    return _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(codes)),
                           _mm256_loadu_ps(activation), accumulator);
}

// One 8-byte table row as an xmm register, for assembling bit planes without touching memory.
inline __m128i slot_row(const std::uint8_t* table_entry) {
    return _mm_loadl_epi64(reinterpret_cast<const __m128i*>(table_entry));
}

// The RowSplit scale word is an IEEE half, and AVX2 implies F16C, so the hardware conversion is
// available and worth taking: the portable scalar version is a branch chain that the predictor gets
// wrong often enough to matter, once per group per row.
inline float scale_from_half(std::uint16_t bits) {
#if defined(_MSC_VER) || defined(__F16C__)
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(bits))));
#else
    return half_to_float(bits);
#endif
}

} // namespace

bool host_has_avx2() noexcept {
    static const bool supported = [] {
#if defined(_MSC_VER)
        int registers[4] = {0, 0, 0, 0};
        __cpuid(registers, 0);
        if (registers[0] < 7) { return false; }
        __cpuid(registers, 1);
        const bool osxsave = (registers[2] & (1 << 27)) != 0;
        const bool avx     = (registers[2] & (1 << 28)) != 0;
        if (!osxsave || !avx) { return false; }
        // The OS must have enabled YMM state saving, otherwise the registers do not survive a
        // context switch and every AVX2 instruction faults.
        const unsigned long long extended = _xgetbv(0);
        if ((extended & 0x6ull) != 0x6ull) { return false; }
        __cpuidex(registers, 7, 0);
        return (registers[1] & (1 << 5)) != 0;
#else
        return __builtin_cpu_supports("avx2") != 0;
#endif
    }();
    return supported;
}

void gemm_rows(const RowSplitGeometry& geometry, const std::uint8_t* payload,
               std::uint64_t padded_columns, std::uint64_t row_begin, std::uint64_t row_end,
               const float* x, std::int32_t tokens, std::uint64_t out_row_stride,
               std::uint16_t* out) {
    const bool byte_coded        = geometry.bits == 8;
    const bool has_high          = geometry.bits == 5;
    const std::uint64_t group_size = geometry.group_size;

    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i xor4     = _mm256_set1_epi8(static_cast<char>(0x88));
    const __m256i q4_bias  = _mm256_set1_epi8(8);

    const HighTables& tables = high_tables();

    for (std::uint64_t row = row_begin; row < row_end; ++row) {
        const std::uint8_t* codes  = payload + row * geometry.code_row_bytes;
        const std::uint8_t* high   = has_high ? payload + geometry.high_offset +
                                                    row * geometry.high_row_bytes
                                              : nullptr;
        const std::uint8_t* scales = payload + geometry.scale_offset + row * geometry.scale_row_bytes;

        for (std::int32_t token = 0; token < tokens; ++token) {
            __m256 accumulator = _mm256_setzero_ps();
            const float* activation =
                x + static_cast<std::uint64_t>(token) * padded_columns;

            for (std::uint64_t group = 0; group < geometry.groups; ++group) {
                std::uint16_t scale_bits = 0;
                std::memcpy(&scale_bits, scales + group * 2, 2);
                const __m256 scale = _mm256_set1_ps(scale_from_half(scale_bits));
                const float* group_activation = activation + group * group_size;
                __m256 group_accumulator      = _mm256_setzero_ps();

                if (byte_coded) {
                    // One signed byte per element, already in storage order: widen straight through.
                    // Two independent accumulators, not one: an FMA retires in ~4 cycles, so a single
                    // dependency chain through all four chunks would spend 16 cycles per group where
                    // four cycles of issue would do. Halves the chain and doubles the throughput.
                    const __m256i dense = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(codes + group * 32));
                    const __m128i low_half  = _mm256_castsi256_si128(dense);
                    const __m128i high_half = _mm256_extracti128_si256(dense, 1);
                    __m256 pair0 = contract8(_mm256_setzero_ps(), low_half, group_activation);
                    pair0 = contract8(pair0, _mm_srli_si128(low_half, 8), group_activation + 8);
                    __m256 pair1 = contract8(_mm256_setzero_ps(), high_half, group_activation + 16);
                    pair1 = contract8(pair1, _mm_srli_si128(high_half, 8), group_activation + 24);
                    group_accumulator = _mm256_add_ps(pair0, pair1);
                } else {
                    // Expand 32 code bytes into 64 values across two vectors. Each unpack works
                    // within a 128-bit lane, so the two results come out lane-interleaved:
                    //   a = {elements  0..15 | elements 32..47}
                    //   b = {elements 16..31 | elements 48..63}
                    __m256i dense =
                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes + group * 32));
                    if (!has_high) { dense = _mm256_xor_si256(dense, xor4); } // q4's stored-nibble bias
                    const __m256i low_nibbles  = _mm256_and_si256(dense, low_mask);
                    const __m256i high_nibbles =
                        _mm256_and_si256(_mm256_srli_epi16(dense, 4), low_mask);
                    __m256i a = _mm256_unpacklo_epi8(low_nibbles, high_nibbles);
                    __m256i b = _mm256_unpackhi_epi8(low_nibbles, high_nibbles);
                    if (has_high) {
                        // value = n - 16 * bit4, folded in as a 0/16 byte subtraction so the sign
                        // handling stays in the byte domain and four mantissa bits survive the
                        // widening to i32. The three most significant bits of a 5-bit code are 1,
                        // so the stored nibble is biased by 16 and the bit plane is what removes it.
                        const std::uint8_t* group_high = high + group * geometry.high_per_group;
                        const __m256i bits_low = _mm256_set_m128i(
                            _mm_unpacklo_epi64(slot_row(tables.slot[group_high[2]]),
                                               slot_row(tables.slot[group_high[3]])),
                            _mm_unpacklo_epi64(slot_row(tables.slot[group_high[0]]),
                                               slot_row(tables.slot[group_high[1]])));
                        const __m256i bits_high = _mm256_set_m128i(
                            _mm_unpacklo_epi64(slot_row(tables.slot[group_high[6]]),
                                               slot_row(tables.slot[group_high[7]])),
                            _mm_unpacklo_epi64(slot_row(tables.slot[group_high[4]]),
                                               slot_row(tables.slot[group_high[5]])));
                        const __m256i scaled_low  = _mm256_slli_epi16(bits_low, 4);
                        const __m256i scaled_high = _mm256_slli_epi16(bits_high, 4);
                        // `a` holds lanes {0..15 | 32..47} and `b` holds {16..31 | 48..63}, so the
                        // bit plane has to be re-laned to match before subtracting.
                        a = _mm256_sub_epi8(a, _mm256_permute2x128_si256(scaled_low, scaled_high, 0x20));
                        b = _mm256_sub_epi8(b, _mm256_permute2x128_si256(scaled_low, scaled_high, 0x31));
                    } else {
                        a = _mm256_sub_epi8(a, q4_bias);
                        b = _mm256_sub_epi8(b, q4_bias);
                    }
                    const __m128i a_low  = _mm256_castsi256_si128(a);          // elements  0..15
                    const __m128i b_low  = _mm256_castsi256_si128(b);          // elements 16..31
                    const __m128i a_high = _mm256_extracti128_si256(a, 1);     // elements 32..47
                    const __m128i b_high = _mm256_extracti128_si256(b, 1);     // elements 48..63
                    // Four independent chains of two FMAs each, rather than one chain of eight.
                    __m256 lane0 = contract8(_mm256_setzero_ps(), a_low, group_activation);
                    lane0 = contract8(lane0, _mm_srli_si128(a_low, 8), group_activation + 8);
                    __m256 lane1 = contract8(_mm256_setzero_ps(), b_low, group_activation + 16);
                    lane1 = contract8(lane1, _mm_srli_si128(b_low, 8), group_activation + 24);
                    __m256 lane2 = contract8(_mm256_setzero_ps(), a_high, group_activation + 32);
                    lane2 = contract8(lane2, _mm_srli_si128(a_high, 8), group_activation + 40);
                    __m256 lane3 = contract8(_mm256_setzero_ps(), b_high, group_activation + 48);
                    lane3 = contract8(lane3, _mm_srli_si128(b_high, 8), group_activation + 56);
                    group_accumulator = _mm256_add_ps(_mm256_add_ps(lane0, lane1),
                                                      _mm256_add_ps(lane2, lane3));
                }

                // The scale is constant across the group, so it is applied once to the group's
                // partial sum instead of once per chunk: 8 multiplies become 1, and rather than
                // costing accuracy it removes 7 roundings.
                accumulator = _mm256_fmadd_ps(group_accumulator, scale, accumulator);
            }

            __m256 total = _mm256_add_ps(accumulator, _mm256_permute2f128_ps(accumulator, accumulator, 1));
            total        = _mm256_hadd_ps(total, total);
            total        = _mm256_hadd_ps(total, total);
            out[static_cast<std::uint64_t>(token) * out_row_stride + row] =
                float_to_bf16(_mm256_cvtss_f32(total));
        }
    }
}

} // namespace ninfer::ops::cpu::detail
