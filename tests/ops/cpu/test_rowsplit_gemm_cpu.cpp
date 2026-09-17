// Conformance for the CPU contraction of RowSplit weights. The oracle is an fp64 dot product over
// the fixture's own decoder, not the kernel's own arithmetic: `logical_weight_fp64` re-derives every
// weight element from the packed payload, so a re-lacing or bit-plane mistake shows up here rather
// than as a plausible-looking number.
//
// The activation is rounded to bf16 before either side sees it, because that is what the device
// path stores and what the A16 contract is stated against.

#include "ops/cpu/rowsplit_gemm.h"
#include "ops/cpu/rowsplit_gemm_detail.h"
#include "ops/op_check.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using ninfer::QType;
using ninfer::QuantLayout;
using ninfer::Weight;
using ninfer::ops::cpu::detail::bf16_to_float;
using ninfer::ops::cpu::detail::float_to_bf16;
using ninfer::ops::cpu::rowsplit_avx2_available;
using ninfer::ops::cpu::rowsplit_gemm;
using ninfer::ops::cpu::rowsplit_gemm_supported;
using ninfer::ops::cpu::rowsplit_worker_threads;
using ninfer::ops::cpu::set_rowsplit_worker_threads;
using ninfer::test::ReductionCriterion;
using ninfer::test::ReductionStats;
using namespace ninfer::test::quantized_weight;

// The A16 preset from tests/ops/linear/linear_test_common.cpp: one bf16 unit roundoff for the
// relative-L2, the same for the gross absolute bound. Restated rather than linked so this suite
// runs without a CUDA device, and asserted against that file's value below.
constexpr ReductionCriterion kA16Criterion{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};

struct Geometry {
    std::int32_t n;
    std::int32_t k;
    std::uint32_t seed;
};

// K is not always a multiple of 128: the 1000-column entry has to show that the padded columns
// contribute exactly nothing, which is the failure a zero-fill of the activation is there to avoid.
constexpr std::array kGeometries{
    Geometry{8, 1000, 103U},  Geometry{37, 5120, 109U},  Geometry{64, 512, 107U},
    Geometry{128, 17408, 127U}, Geometry{256, 5120, 113U},
};

constexpr std::array kTokenCounts{1, 2, 3, 4, 5, 8};

std::uint64_t mix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

double uniform_from(std::uint64_t& state) {
    state = mix64(state);
    return static_cast<double>(state >> 11) / static_cast<double>(1ULL << 53);
}

// Deterministic activation in the engine's [tokens][k] order. The token axis is the outer one --
// bf16_n256_k5120.cuh reads `x_shared[token * kGroupK + ...]` -- and the test has to follow that
// rather than pick its own convention: a self-consistent but non-engine order would pass here and
// break every batched contraction in the model.
std::vector<std::uint16_t> make_activation(std::int32_t k, std::int32_t tokens, std::uint32_t seed) {
    std::vector<std::uint16_t> out(static_cast<std::size_t>(k) * tokens);
    std::uint64_t state = mix64(seed);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t column = 0; column < k; ++column) {
            out[static_cast<std::size_t>(token) * k + column] =
                float_to_bf16(static_cast<float>(4.0 * (uniform_from(state) - 0.5)));
        }
    }
    return out;
}

// Reads one activation element the way the engine stores it.
inline float activation_at(const std::vector<std::uint16_t>& activation, std::int32_t k,
                           std::int32_t token, std::int32_t column) {
    return bf16_to_float(activation[static_cast<std::size_t>(token) * k + column]);
}

PackedWeight pack(QType qtype, const std::vector<float>& source, std::int32_t n, std::int32_t k) {
    switch (qtype) {
    case QType::Q4_G64_FP16:
        return pack_q4_row_split(source, n, k);
    case QType::Q5_G64_FP16:
        return pack_q5_row_split(source, n, k);
    case QType::Q6_G64_FP16:
        return pack_q6_row_split(source, n, k);
    case QType::Q8_G32_FP16:
        return pack_q8_g32_row_split(source, n, k);
    default:
        throw std::invalid_argument("cpu gemm test: unreachable format");
    }
}

// Every logical element, exactly decoded, row-major over the full N. A code of at most 8 bits times
// an 11-bit fp16 scale is exactly representable in fp32, so accumulating these products in double is
// a faithful oracle.
std::vector<float> dequantize_all(const PackedWeight& packed) {
    std::vector<std::int32_t> rows(static_cast<std::size_t>(packed.weight.n));
    for (std::int32_t row = 0; row < packed.weight.n; ++row) { rows[static_cast<std::size_t>(row)] = row; }
    return materialize_rows_fp32(packed, rows);
}

int run_shape(const char* label, QType qtype, const Geometry& shape) {
    int failures = 0;
    std::vector<float> source(static_cast<std::size_t>(shape.n) * shape.k);
    std::uint64_t state = mix64(shape.seed);
    for (auto& value : source) {
        value = static_cast<float>(2.0 * (uniform_from(state) - 0.5));
    }

    const PackedWeight packed = pack(qtype, source, shape.n, shape.k);
    if (!rowsplit_gemm_supported(packed.weight)) {
        std::cerr << label << ": supported() rejected a well-formed weight\n";
        ++failures;
        return failures;
    }
    const std::vector<float> exact = dequantize_all(packed);

    for (const std::int32_t tokens : kTokenCounts) {
        const std::vector<std::uint16_t> activation = make_activation(shape.k, tokens, shape.seed);
        std::vector<std::uint16_t> produced(static_cast<std::size_t>(shape.n) * tokens, 0);

        rowsplit_gemm(packed.weight, activation.data(), produced.data(), tokens);

        std::vector<double> actual(produced.size());
        std::vector<double> reference(produced.size());
        for (std::int32_t row = 0; row < shape.n; ++row) {
            for (std::int32_t token = 0; token < tokens; ++token) {
                double accumulator = 0.0;
                for (std::int32_t column = 0; column < shape.k; ++column) {
                    const float weight_value =
                        exact[static_cast<std::size_t>(row) * shape.k + column];
                    const float input = activation_at(activation, shape.k, token, column);
                    accumulator += static_cast<double>(weight_value) * static_cast<double>(input);
                }
                const std::size_t index = static_cast<std::size_t>(token) * shape.n + row;
                actual[index]           = static_cast<double>(bf16_to_float(produced[index]));
                reference[index]        = accumulator;
            }
        }

        const ReductionStats stats = ninfer::test::compute_reduction_stats(
            actual.data(), reference.data(), static_cast<std::int64_t>(actual.size()));
        if (!ninfer::test::reduction_passes(stats, static_cast<std::int64_t>(actual.size()),
                                            kA16Criterion)) {
            std::cerr << label << " n=" << shape.n << " k=" << shape.k << " tokens=" << tokens
                      << ": rel_l2=" << stats.relative_l2
                      << " (limit " << kA16Criterion.relative_l2 << ')'
                      << " max_abs=" << stats.maximum_absolute_error << " (limit "
                      << ninfer::test::gross_error_limit(stats, kA16Criterion) << ", max_ref "
                      << stats.maximum_absolute_reference << ')'
                      << " first_non_finite=" << stats.first_non_finite << '\n';
            ++failures;
        }
    }
    return failures;
}

int rejection_cases() {
    int failures = 0;
    const PackedWeight packed =
        pack_q8_g32_row_split(std::vector<float>(64 * 512, 0.25F), 64, 512);
    std::vector<std::uint16_t> activation(512, 0);
    std::vector<std::uint16_t> out(64, 0);

    const auto expect_rejected = [&](const char* what, auto&& call) {
        try {
            call();
            std::cerr << "cpu gemm accepted " << what << '\n';
            ++failures;
        } catch (const std::invalid_argument&) {
        }
    };

    {
        Weight detached = packed.weight;
        detached.qdata          = nullptr;
        if (rowsplit_gemm_supported(detached)) {
            std::cerr << "supported() accepted a weight with no code plane\n";
            ++failures;
        }
        expect_rejected("a weight with no code plane",
                        [&] { rowsplit_gemm(detached, activation.data(), out.data(), 1); });
    }
    expect_rejected("a Contiguous layout", [&] {
        Weight contiguous = packed.weight;
        contiguous.layout         = QuantLayout::Contiguous;
        rowsplit_gemm(contiguous, activation.data(), out.data(), 1);
    });
    expect_rejected("tokens == 0", [&] { rowsplit_gemm(packed.weight, activation.data(), out.data(), 0); });
    expect_rejected("a null output", [&] { rowsplit_gemm(packed.weight, activation.data(), nullptr, 1); });
    // A padded column count that disagrees with the parent would silently change the row stride and
    // read the neighbouring row's codes, so it has to be caught rather than tolerated.
    expect_rejected("a mismatched padded K", [&] {
        Weight skewed  = packed.weight;
        skewed.padded_shape[1] = 640;
        rowsplit_gemm(skewed, activation.data(), out.data(), 1);
    });
    return failures;
}

// Not a criterion: three shipped geometries, swept across worker counts, token counts and two code
// widths. The sweep is the point --
//   * a contraction that is actually memory-bound has to scale with threads;
//   * a q4/q8 pair separates "limited by bytes touched" from "limited by arithmetic", because q4
//     moves 53% of q8's bytes for the same element count;
//   * tokens=1 is the decode shape and tokens=4 the batched one, and they sit on opposite sides of
//     that distinction;
//   * the 17408-column entry is deliberately larger than this host's 24 MiB L3. The 5120-column
//     entries fit in cache, so their bandwidth flatters the kernel in a way no real weight tensor
//     can: the 27B host weights are 13 GB and always stream from DRAM.
void report_throughput() {
    struct Entry {
        const char* label;
        std::int32_t n;
        std::int32_t k;
        double bytes_per_row;
        PackedWeight packed;
    };
    constexpr std::int32_t kMaxN      = 8192;
    constexpr std::int32_t kMaxK      = 5120;
    constexpr std::int32_t kMaxTokens = 64;

    std::vector<Entry> entries;
    entries.push_back(
        Entry{"q8", 8192, 5120, static_cast<double>(5120 + (5120 / 32) * 2),
              pack_q8_g32_row_split(std::vector<float>(std::size_t{8192} * 5120, 0.5F), 8192, 5120)});
    entries.push_back(
        Entry{"q4", 8192, 5120, static_cast<double>((5120 / 2) + (5120 / 64) * 2),
              pack_q4_row_split(std::vector<float>(std::size_t{8192} * 5120, 0.5F), 8192, 5120)});

    std::vector<std::uint16_t> activation(static_cast<std::size_t>(kMaxK) * kMaxTokens, 0);
    std::vector<std::uint16_t> out(static_cast<std::size_t>(kMaxN) * kMaxTokens, 0);

    for (auto& entry : entries) {
        const double bytes = static_cast<double>(entry.n) * entry.bytes_per_row;
        for (const std::int32_t tokens : {1, 4, 16, 64}) {
            // Small token counts get the full worker sweep; the large ones exist to answer a
            // single question -- whether the contraction's arithmetic throughput rises once each
            // decoded weight is reused across many tokens -- so one worker count is enough.
            const std::vector<unsigned> workers =
                tokens <= 4 ? std::vector<unsigned>{1u, 2u, 4u, 8u, 13u} : std::vector<unsigned>{13u};
            for (const unsigned threads : workers) {
                set_rowsplit_worker_threads(threads);
                // One warm pass so the page faults and the first thread wake-up leave the
                // measurement, then the best of several: a single pass at this size is under a
                // millisecond, which is short enough that scheduler noise dominates the mean.
                rowsplit_gemm(entry.packed.weight, activation.data(), out.data(), tokens);
                double seconds = 1e9;
                for (int repeat = 0; repeat < 5; ++repeat) {
                    const auto start = std::chrono::steady_clock::now();
                    rowsplit_gemm(entry.packed.weight, activation.data(), out.data(), tokens);
                    seconds = std::min(
                        seconds,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                            .count());
                }

                const double flops =
                    2.0 * entry.n * static_cast<double>(entry.k) * static_cast<double>(tokens);
                std::cout << entry.label << ' ' << entry.n << 'x' << entry.k << " t=" << tokens
                          << " workers=" << rowsplit_worker_threads() << ": " << (seconds * 1e3)
                          << " ms, " << (bytes / seconds / 1e9) << " GB/s, "
                          << (flops / seconds / 1e9) << " GFLOP/s" << std::endl;
            }
        }
    }
}

} // namespace

int main() {
    // Unbuffered, so a crash mid-suite still shows how far the run got. A failing contraction can
    // take the process down through the allocator's own fast-fail path, and a buffered stream would
    // lose every diagnostic written before it.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);

    int failures = 0;
    try {
        const std::array<QType, 4> formats{QType::Q4_G64_FP16, QType::Q5_G64_FP16,
                                          QType::Q6_G64_FP16, QType::Q8_G32_FP16};
        const std::array<const char*, 4> names{"Q4", "Q5", "Q6", "Q8"};
        for (const auto& shape : kGeometries) {
            for (std::size_t index = 0; index < formats.size(); ++index) {
                std::cout << "case " << names[index] << " n=" << shape.n << " k=" << shape.k
                          << std::endl;
                failures += run_shape(names[index], formats[index], shape);
            }
        }
        std::cout << "case rejections" << std::endl;
        failures += rejection_cases();

        std::cout << "cpu contraction: avx2=" << (rowsplit_avx2_available() ? "yes" : "no")
                  << " threads=" << rowsplit_worker_threads() << '\n';
        report_throughput();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " RowSplit CPU GEMM\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "RowSplit CPU GEMM: " << error.what() << '\n';
        return 1;
    }
}
