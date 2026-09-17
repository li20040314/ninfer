// End-to-end conformance for the host contraction as the engine actually reaches it.
//
// The point of this suite is the *dispatch*, not the arithmetic: the weight is a host allocation, the
// activation and output are device buffers, and the call goes through the ordinary public `linear()`
// entry point. Nothing here tells the engine where the weight lives -- if the CPU route were not
// chosen on its own, the device kernels would read a host pointer and either fault or return
// nonsense, and the fp64 oracle would catch it.
//
// The oracle decodes the packed payload independently, so a mis-routed call cannot pass by agreeing
// with itself.

#include "ninfer/ops/linear.h"
#include "ops/cpu/rowsplit_gemm_detail.h"
#include "ops/linear/host/linear_host.h"
#include "ops/op_check.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::QType;
using ninfer::Tensor;
using ninfer::Weight;
using ninfer::ops::cpu::detail::bf16_to_float;
using ninfer::ops::cpu::detail::float_to_bf16;
using ninfer::test::ReductionCriterion;
using ninfer::test::ReductionStats;
using namespace ninfer::test::quantized_weight;

// The A16 preset, same as the device suites use.
constexpr ReductionCriterion kA16Criterion{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};

struct Geometry {
    std::int32_t n;
    std::int32_t k;
    std::uint32_t seed;
};

constexpr std::array kGeometries{
    Geometry{64, 512, 401U},
    Geometry{37, 5120, 409U},
    Geometry{128, 17408, 419U},
};

constexpr std::array kTokenCounts{1, 3, 8};

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
        throw std::invalid_argument("linear host test: unreachable format");
    }
}

// A device copy of `bytes` from `source`, or a hard failure -- the suite is not meaningful if the
// staging itself silently produced garbage.
void* upload(const void* source, std::size_t bytes) {
    void* device = nullptr;
    if (cudaMalloc(&device, bytes) != cudaSuccess) { throw std::runtime_error("cudaMalloc failed"); }
    if (cudaMemcpy(device, source, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("upload failed");
    }
    return device;
}

int run_case(const char* label, QType qtype, const Geometry& shape, std::int32_t tokens,
             int& failures) {
    std::vector<float> source(static_cast<std::size_t>(shape.n) * shape.k);
    std::uint64_t state = mix64(shape.seed);
    for (auto& value : source) { value = static_cast<float>(2.0 * (uniform_from(state) - 0.5)); }
    const PackedWeight packed = pack(qtype, source, shape.n, shape.k);

    // The engine has to recognise this on its own; assert the precondition explicitly so a failure
    // here is not later mistaken for a wrong contraction.
    if (!ninfer::ops::detail::weight_is_host_resident(packed.weight)) {
        std::cerr << label << " n=" << shape.n << " k=" << shape.k
                  << ": a host weight was not recognised as host-resident\n";
        ++failures;
        return failures;
    }

    // Activation in the engine's [T][K] order: the token axis is the outer one, matching how the
    // device kernels index it. Same convention as make_activation in the CPU suite, and the same
    // trap -- at tokens == 1 either order looks right.
    std::vector<std::uint16_t> activation(static_cast<std::size_t>(shape.k) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t column = 0; column < shape.k; ++column) {
            activation[static_cast<std::size_t>(token) * shape.k + column] =
                float_to_bf16(static_cast<float>(4.0 * (uniform_from(state) - 0.5)));
        }
    }

    void* device_activation = upload(activation.data(), activation.size() * sizeof(std::uint16_t));
    const std::vector<std::uint16_t> zeroed(static_cast<std::size_t>(shape.n) * tokens, 0);
    void* device_output = upload(zeroed.data(), zeroed.size() * sizeof(std::uint16_t));

    Tensor x(device_activation, DType::BF16, {shape.k, tokens});
    Tensor out(device_output, DType::BF16, {shape.n, tokens});
    ninfer::ops::linear(x, packed.weight, out, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        throw std::runtime_error(std::string(label) + ": device synchronisation failed");
    }

    std::vector<std::uint16_t> produced(static_cast<std::size_t>(shape.n) * tokens, 0);
    if (cudaMemcpy(produced.data(), device_output, produced.size() * sizeof(std::uint16_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error(std::string(label) + ": output readback failed");
    }
    cudaFree(device_activation);
    cudaFree(device_output);

    // fp64 oracle over the fixture's own decoder.
    std::vector<std::int32_t> rows(static_cast<std::size_t>(shape.n));
    for (std::int32_t row = 0; row < shape.n; ++row) { rows[static_cast<std::size_t>(row)] = row; }
    const std::vector<float> exact = materialize_rows_fp32(packed, rows);

    std::vector<double> actual(produced.size());
    std::vector<double> reference(produced.size());
    for (std::int32_t row = 0; row < shape.n; ++row) {
        for (std::int32_t token = 0; token < tokens; ++token) {
            double accumulator = 0.0;
            for (std::int32_t column = 0; column < shape.k; ++column) {
                accumulator +=
                    static_cast<double>(exact[static_cast<std::size_t>(row) * shape.k + column]) *
                    static_cast<double>(bf16_to_float(
                        activation[static_cast<std::size_t>(token) * shape.k + column]));
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
                  << ": rel_l2=" << stats.relative_l2 << " max_abs=" << stats.maximum_absolute_error
                  << " first_non_finite=" << stats.first_non_finite << '\n';
        ++failures;
    }
    return failures;
}

// The probe has to answer for both sides. Claiming that a device allocation is host memory would
// route a device weight into a CPU contraction and read a device pointer as host memory, which
// faults rather than merely being slow -- so the negative case is asserted here, not assumed.
int residency_probe_case() {
    int failures = 0;
    const PackedWeight packed =
        pack_q8_g32_row_split(std::vector<float>(64 * 512, 0.25F), 64, 512);

    if (!ninfer::ops::detail::weight_is_host_resident(packed.weight)) {
        std::cerr << "a fixture host weight was not recognised as host-resident\n";
        ++failures;
    }

    void* device_payload         = upload(packed.payload.data(), packed.payload.size());
    const Weight device_weight   = packed.device_weight(device_payload);
    if (ninfer::ops::detail::weight_is_host_resident(device_weight)) {
        std::cerr << "a device weight was mistaken for host memory\n";
        ++failures;
    }
    // And back to the host weight: the remembered verdict must have been replaced, not kept.
    if (!ninfer::ops::detail::weight_is_host_resident(packed.weight)) {
        std::cerr << "a host weight was not recognised after a device weight was probed\n";
        ++failures;
    }
    cudaFree(device_payload);
    return failures;
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);

    int device = 0;
    if (cudaGetDeviceCount(&device) != cudaSuccess || device < 1) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        std::cout << "SKIP: device 0 is not usable\n";
        return 77;
    }

    int failures = 0;
    try {
        const std::array<QType, 4> formats{QType::Q4_G64_FP16, QType::Q5_G64_FP16,
                                           QType::Q6_G64_FP16, QType::Q8_G32_FP16};
        const std::array<const char*, 4> names{"Q4", "Q5", "Q6", "Q8"};
        for (const auto& shape : kGeometries) {
            for (std::size_t index = 0; index < formats.size(); ++index) {
                for (const std::int32_t tokens : kTokenCounts) {
                    std::cout << "case " << names[index] << " n=" << shape.n << " k=" << shape.k
                              << " t=" << tokens << std::endl;
                    failures += run_case(names[index], formats[index], shape, tokens, failures);
                }
            }
        }
        std::cout << "case residency probe" << std::endl;
        failures += residency_probe_case();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Linear host contraction\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Linear host contraction: " << error.what() << '\n';
        return 1;
    }
}
