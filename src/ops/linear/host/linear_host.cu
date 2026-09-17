#include "ops/linear/host/linear_host.h"

#include "ops/cpu/rowsplit_gemm.h"
#include "ops/cpu/rowsplit_gemm_detail.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops::detail {
namespace {

// Staging for the bus crossings. A contraction is a per-layer event rather than a per-token one,
// and the row count is fixed by the model, so these grow to their high-water mark once.
//
// The two groups are resized at different points, and the distinction is what keeps the writebacks
// asynchronous:
//   activation, residual - async-copy *destinations*. Resizing one only affects the copy about to
//                          be issued into it, so it may happen before the sync.
//   projection, result   - async-copy *sources*. The previous call's writeback may still be reading
//                          one of these, so they may only be resized after the sync has retired it.
// Every entry point below therefore issues all of its reads, syncs once, and only then touches the
// second group.
thread_local std::vector<std::uint16_t> t_activation;
thread_local std::vector<std::uint16_t> t_residual;
thread_local std::vector<std::uint16_t> t_projection;
thread_local std::vector<std::uint16_t> t_result;

// Remembered verdicts. A decode loop walks the same handful of weights in the same order every
// token, so a small set of entries turns the per-call query into a per-weight one. The set has to
// hold more than one weight because the fused projections query *two* parents back to back -- a
// single-slot cache would evict its own verdict every call and never hit.
constexpr std::size_t kResidencyCacheEntries = 8;
thread_local const void* t_cached_planes[kResidencyCacheEntries] = {};
thread_local bool t_cached_verdicts[kResidencyCacheEntries]      = {};
thread_local bool t_cached_valid[kResidencyCacheEntries]         = {};
thread_local std::size_t t_cache_cursor                          = 0;

void require_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("linear_host: ") + what + ": " +
                                 cudaGetErrorName(status));
    }
}

// Both operands of a contraction are token-major: one token's row of columns is contiguous and
// nb[1] is its stride. A view into a wider buffer keeps the wider pitch -- matrix_window hands the
// head the leading columns of the logits buffer, and copying that as a flat run would splice
// neighbouring columns into the result.
void require_token_major(const Tensor& tensor, const char* what) {
    if (tensor.nb[0] != static_cast<std::int64_t>(sizeof(std::uint16_t))) {
        throw std::invalid_argument(std::string("linear_host: ") + what +
                                    " rows must be contiguous");
    }
}

void require_host_weight(const Weight& weight, const char* what) {
    if (!cpu::rowsplit_gemm_supported(weight)) {
        throw std::invalid_argument(std::string("linear_host: ") + what +
                                    " is not a RowSplit host weight this contraction covers");
    }
    if (!weight_is_host_resident(weight)) {
        throw std::invalid_argument(std::string("linear_host: ") + what +
                                    " is not host resident");
    }
}

// Reads a [columns,T] bf16 device tensor into `staging`, one token row at a time so the source
// pitch is honoured.
void readback_rows(std::vector<std::uint16_t>& staging, const Tensor& source, std::int32_t columns,
                   std::int32_t tokens, cudaStream_t stream, const char* what) {
    require_token_major(source, what);
    staging.resize(static_cast<std::size_t>(columns) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        require_cuda(cudaMemcpyAsync(
                         staging.data() + static_cast<std::size_t>(token) * columns,
                         static_cast<const std::byte*>(source.data) + source.nb[1] * token,
                         static_cast<std::size_t>(columns) * sizeof(std::uint16_t),
                         cudaMemcpyDeviceToHost, stream),
                     what);
    }
}

// Writes `staging` back to a [columns,T] bf16 device tensor, one token row at a time. Left
// asynchronous: the next launch on this stream is ordered after it, so the CPU never waits for the
// result it just produced -- the following call's sync is what retires it.
void writeback_rows(const std::vector<std::uint16_t>& staging, Tensor& destination,
                    std::int32_t columns, std::int32_t tokens, cudaStream_t stream) {
    require_token_major(destination, "output");
    for (std::int32_t token = 0; token < tokens; ++token) {
        require_cuda(cudaMemcpyAsync(
                         static_cast<std::byte*>(destination.data) + destination.nb[1] * token,
                         staging.data() + static_cast<std::size_t>(token) * columns,
                         static_cast<std::size_t>(columns) * sizeof(std::uint16_t),
                         cudaMemcpyHostToDevice, stream),
                     "output writeback");
    }
}

// Writes one row range `[row_begin, row_begin + row_count)` of a token-major `[total_rows,T]`
// staging buffer to a `[row_count,T]` device tensor. This is how a fused parent's staging splits
// into its named outputs without materializing an intermediate device plane.
void writeback_row_range(const std::vector<std::uint16_t>& staging, std::size_t total_rows,
                         std::size_t row_begin, std::size_t row_count, Tensor& destination,
                         std::int32_t tokens, cudaStream_t stream) {
    require_token_major(destination, "output");
    for (std::int32_t token = 0; token < tokens; ++token) {
        require_cuda(cudaMemcpyAsync(
                         static_cast<std::byte*>(destination.data) + destination.nb[1] * token,
                         staging.data() + static_cast<std::size_t>(token) * total_rows + row_begin,
                         row_count * sizeof(std::uint16_t), cudaMemcpyHostToDevice, stream),
                     "output writeback");
    }
}

// Contracts a host-resident weight into token-major `[T][rows]` staging after the activation sync.
void contract_into(const Weight& weight, const std::vector<std::uint16_t>& activation,
                   std::vector<std::uint16_t>& staging, std::int32_t tokens) {
    staging.resize(static_cast<std::size_t>(weight.n) * tokens);
    cpu::rowsplit_gemm(weight, activation.data(), staging.data(), tokens);
}

// Mirrors the device kernel (`ops/common/math.cuh`): SiLU is evaluated exactly in fp32 rather than
// as a polynomial fit, and the product is rounded to bf16 once.
float silu_f32(float x) { return x / (1.0F + std::exp(-x)); }

// --- attribution timing ---------------------------------------------------------------------------
//
// Sums, across every host-path call in the process, how long the contraction and each bus crossing
// took, and prints the split once at exit. This exists to answer one question -- of the measured
// per-token time, how much is the CPU kernel versus everything around it -- and costs four atomic
// adds per call. It reports unconditionally because the counters only move when a host path is
// wired in, i.e. exactly when the split is interesting.
struct HostPathTiming {
    std::atomic<unsigned long long> calls{0};
    std::atomic<unsigned long long> read_ns{0};   // readback issue + the sync that retires it
    std::atomic<unsigned long long> gemm_ns{0};   // rowsplit_gemm, the CPU contraction itself
    std::atomic<unsigned long long> epi_ns{0};    // host-side epilogue (silu-mul, residual add)
    std::atomic<unsigned long long> wb_ns{0};     // writeback issue (asynchronous, not waited on)

    ~HostPathTiming() {
        const auto calls_total   = calls.load(std::memory_order_relaxed);
        if (calls_total == 0) { return; }
        const auto read  = read_ns.load(std::memory_order_relaxed);
        const auto gemm  = gemm_ns.load(std::memory_order_relaxed);
        const auto epi   = epi_ns.load(std::memory_order_relaxed);
        const auto wb    = wb_ns.load(std::memory_order_relaxed);
        std::fprintf(stderr,
                     "linear_host timing: calls=%llu readback=%.1f ms gemm=%.1f ms "
                     "epilogue=%.1f ms writeback-issue=%.1f ms\n",
                     static_cast<unsigned long long>(calls_total),
                     static_cast<double>(read) / 1e6, static_cast<double>(gemm) / 1e6,
                     static_cast<double>(epi) / 1e6, static_cast<double>(wb) / 1e6);
    }
};

HostPathTiming& host_path_timing() {
    static HostPathTiming timing;
    return timing;
}

double steady_ns_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start)
        .count();
}

} // namespace

bool weight_is_host_resident(const Weight& weight) {
    const void* plane = weight.qdata;
    if (plane == nullptr) { return false; }
    for (std::size_t entry = 0; entry < kResidencyCacheEntries; ++entry) {
        if (t_cached_valid[entry] && plane == t_cached_planes[entry]) {
            return t_cached_verdicts[entry];
        }
    }

    cudaPointerAttributes attributes{};
    const cudaError_t status = cudaPointerGetAttributes(&attributes, plane);
    bool is_host             = false;
    if (status != cudaSuccess) {
        // Older runtimes reject a pointer they never allocated. That is the host case, so clear the
        // pending `cudaErrorInvalidValue` the query just set -- otherwise the next unrelated error
        // check in the engine would report this query instead of its own failure.
        (void)cudaGetLastError();
        is_host = true;
    } else {
        // Current runtimes answer instead of rejecting: ordinary host memory comes back as
        // `cudaMemoryTypeUnregistered`, and only an actual CUDA allocation reports Host, Device or
        // Managed. Anything unrecognised stays on the device route, because reading a device
        // pointer as host memory would fault rather than merely be slow.
        switch (attributes.type) {
        case cudaMemoryTypeHost:
        case cudaMemoryTypeUnregistered:
            is_host = true;
            break;
        default:
            is_host = false;
            break;
        }
    }

    // Fill the next slot in rotation; the loop visits a fixed handful of weights per token, so an
    // eight-entry set is always large enough to keep every verdict resident.
    t_cached_planes[t_cache_cursor]  = plane;
    t_cached_verdicts[t_cache_cursor] = is_host;
    t_cached_valid[t_cache_cursor]   = true;
    t_cache_cursor                   = (t_cache_cursor + 1) % kResidencyCacheEntries;
    return is_host;
}

void linear_host(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    require_host_weight(w, "weight");
    const std::int32_t tokens = x.ne[1];
    if (tokens <= 0 || x.ne[0] != w.k || out.ne[0] != w.n || out.ne[1] != tokens) {
        throw std::invalid_argument("linear_host: expected [K,T] x [N,K] -> [N,T]");
    }

    auto& timing = host_path_timing();
    timing.calls.fetch_add(1, std::memory_order_relaxed);
    const auto read_start = std::chrono::steady_clock::now();
    readback_rows(t_activation, x, w.k, tokens, stream, "activation readback");
    require_cuda(cudaStreamSynchronize(stream), "activation sync");
    timing.read_ns.fetch_add(static_cast<unsigned long long>(steady_ns_since(read_start)),
                             std::memory_order_relaxed);

    const auto gemm_start = std::chrono::steady_clock::now();
    t_projection.resize(static_cast<std::size_t>(w.n) * tokens);
    cpu::rowsplit_gemm(w, t_activation.data(), t_projection.data(), tokens);
    timing.gemm_ns.fetch_add(static_cast<unsigned long long>(steady_ns_since(gemm_start)),
                             std::memory_order_relaxed);

    const auto wb_start = std::chrono::steady_clock::now();
    writeback_rows(t_projection, out, w.n, tokens, stream);
    timing.wb_ns.fetch_add(static_cast<unsigned long long>(steady_ns_since(wb_start)),
                           std::memory_order_relaxed);
}

void linear_swiglu_host(const Tensor& x, const Weight& gate_up, Tensor& out, cudaStream_t stream) {
    require_host_weight(gate_up, "gate/up weight");
    const std::int32_t gate_up_rows = gate_up.n;
    if ((gate_up_rows % 2) != 0) {
        throw std::invalid_argument("linear_swiglu_host: gate/up rows must be even");
    }
    const std::int32_t rows   = gate_up_rows / 2;
    const std::int32_t tokens = x.ne[1];
    if (tokens <= 0 || x.ne[0] != gate_up.k || out.ne[0] != rows || out.ne[1] != tokens) {
        throw std::invalid_argument("linear_swiglu_host: expected [K,T] x [2M,K] -> [M,T]");
    }

    auto& timing = host_path_timing();
    timing.calls.fetch_add(1, std::memory_order_relaxed);
    const auto read_start = std::chrono::steady_clock::now();
    readback_rows(t_activation, x, gate_up.k, tokens, stream, "activation readback");
    require_cuda(cudaStreamSynchronize(stream), "activation sync");
    timing.read_ns.fetch_add(static_cast<unsigned long long>(steady_ns_since(read_start)),
                             std::memory_order_relaxed);

    const auto gemm_start = std::chrono::steady_clock::now();
    // The whole parent is contracted even though only half of it survives: the gate and up rows are
    // interleaved in the same weight, so there is no cheaper way to reach the up rows than to walk
    // the parent. What the fusion saves is the writeback, not the read.
    t_projection.resize(static_cast<std::size_t>(gate_up_rows) * tokens);
    cpu::rowsplit_gemm(gate_up, t_activation.data(), t_projection.data(), tokens);
    timing.gemm_ns.fetch_add(static_cast<unsigned long long>(steady_ns_since(gemm_start)),
                             std::memory_order_relaxed);

    const auto epi_start = std::chrono::steady_clock::now();
    t_result.resize(static_cast<std::size_t>(rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        const auto* gate = t_projection.data() + static_cast<std::size_t>(token) * gate_up_rows;
        const auto* up   = gate + rows;
        auto* destination = t_result.data() + static_cast<std::size_t>(token) * rows;
        for (std::int32_t row = 0; row < rows; ++row) {
            const float value = silu_f32(cpu::detail::bf16_to_float(gate[row])) *
                                cpu::detail::bf16_to_float(up[row]);
            destination[row] = cpu::detail::float_to_bf16(value);
        }
    }
    timing.epi_ns.fetch_add(static_cast<unsigned long long>(steady_ns_since(epi_start)),
                            std::memory_order_relaxed);

    const auto wb_start = std::chrono::steady_clock::now();
    writeback_rows(t_result, out, rows, tokens, stream);
    timing.wb_ns.fetch_add(static_cast<unsigned long long>(steady_ns_since(wb_start)),
                           std::memory_order_relaxed);
}

void linear_add_host(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    require_host_weight(w, "weight");
    const std::int32_t tokens = x.ne[1];
    if (tokens <= 0 || x.ne[0] != w.k || residual_out.ne[0] != w.n ||
        residual_out.ne[1] != tokens) {
        throw std::invalid_argument("linear_add_host: expected [K,T] x [N,K] + [N,T] -> [N,T]");
    }

    auto& timing = host_path_timing();
    timing.calls.fetch_add(1, std::memory_order_relaxed);
    const auto read_start = std::chrono::steady_clock::now();
    // Both reads are independent, so they share one sync: the residual is part of this op's operand
    // set rather than a second round trip.
    readback_rows(t_activation, x, w.k, tokens, stream, "activation readback");
    readback_rows(t_residual, residual_out, w.n, tokens, stream, "residual readback");
    require_cuda(cudaStreamSynchronize(stream), "operand sync");
    timing.read_ns.fetch_add(static_cast<unsigned long long>(steady_ns_since(read_start)),
                             std::memory_order_relaxed);

    const auto gemm_start = std::chrono::steady_clock::now();
    t_projection.resize(static_cast<std::size_t>(w.n) * tokens);
    cpu::rowsplit_gemm(w, t_activation.data(), t_projection.data(), tokens);
    timing.gemm_ns.fetch_add(static_cast<unsigned long long>(steady_ns_since(gemm_start)),
                             std::memory_order_relaxed);

    const auto epi_start = std::chrono::steady_clock::now();
    // The sum is fp32 and rounds once, which is what the device epilogue does with its accumulator.
    for (std::size_t index = 0; index < t_projection.size(); ++index) {
        t_projection[index] = cpu::detail::float_to_bf16(
            cpu::detail::bf16_to_float(t_projection[index]) + cpu::detail::bf16_to_float(t_residual[index]));
    }
    timing.epi_ns.fetch_add(static_cast<unsigned long long>(steady_ns_since(epi_start)),
                            std::memory_order_relaxed);

    const auto wb_start = std::chrono::steady_clock::now();
    writeback_rows(t_projection, residual_out, w.n, tokens, stream);
    timing.wb_ns.fetch_add(static_cast<unsigned long long>(steady_ns_since(wb_start)),
                           std::memory_order_relaxed);
}

void attn_input_proj_host(const Tensor& x, const Weight& query_key, const Weight& gate_value,
                          Tensor& q, Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    require_host_weight(query_key, "query/key weight");
    require_host_weight(gate_value, "gate/value weight");
    const std::int32_t tokens = x.ne[1];
    const std::int32_t query_rows = q.ne[0];
    const std::int32_t kv_rows    = k.ne[0];
    if (tokens <= 0 || x.ne[0] != query_key.k || query_key.k != gate_value.k ||
        query_key.n != query_rows + kv_rows || gate_value.n != gate.ne[0] + v.ne[0] ||
        gate.ne[0] != query_rows || v.ne[0] != kv_rows || q.ne[1] != tokens ||
        gate.ne[1] != tokens || k.ne[1] != tokens || v.ne[1] != tokens) {
        throw std::invalid_argument(
            "attn_input_proj_host: expected [K,T] x (Q4 [Q+KV,K], Q5 [Q+KV,K]) -> q/gate/k/v");
    }

    auto& timing = host_path_timing();
    timing.calls.fetch_add(1, std::memory_order_relaxed);
    const auto read_start = std::chrono::steady_clock::now();
    // One activation readback serves both parents: they consume the same x.
    readback_rows(t_activation, x, query_key.k, tokens, stream, "activation readback");
    require_cuda(cudaStreamSynchronize(stream), "activation sync");
    timing.read_ns.fetch_add(static_cast<unsigned long long>(steady_ns_since(read_start)),
                             std::memory_order_relaxed);

    const auto gemm_start = std::chrono::steady_clock::now();
    contract_into(query_key, t_activation, t_projection, tokens);
    contract_into(gate_value, t_activation, t_result, tokens);
    timing.gemm_ns.fetch_add(static_cast<unsigned long long>(steady_ns_since(gemm_start)),
                             std::memory_order_relaxed);

    const auto wb_start = std::chrono::steady_clock::now();
    const std::size_t fused_rows = static_cast<std::size_t>(query_key.n);
    writeback_row_range(t_projection, fused_rows, 0, static_cast<std::size_t>(query_rows), q,
                        tokens, stream);
    writeback_row_range(t_projection, fused_rows, static_cast<std::size_t>(query_rows),
                        static_cast<std::size_t>(kv_rows), k, tokens, stream);
    writeback_row_range(t_result, fused_rows, 0, static_cast<std::size_t>(gate.ne[0]), gate,
                        tokens, stream);
    writeback_row_range(t_result, fused_rows, static_cast<std::size_t>(gate.ne[0]),
                        static_cast<std::size_t>(v.ne[0]), v, tokens, stream);
    timing.wb_ns.fetch_add(static_cast<unsigned long long>(steady_ns_since(wb_start)),
                           std::memory_order_relaxed);
}

void gdn_input_proj_host(const Tensor& x, const Weight& qk, const Weight& value_z, Tensor& qkv,
                         Tensor& z, cudaStream_t stream) {
    require_host_weight(qk, "qk weight");
    require_host_weight(value_z, "value/z weight");
    const std::int32_t tokens     = x.ne[1];
    const std::int32_t value_rows = qkv.ne[0] - qk.n;
    if (tokens <= 0 || x.ne[0] != qk.k || qk.k != value_z.k || value_rows <= 0 ||
        value_z.n != value_rows + z.ne[0] || qkv.ne[1] != tokens || z.ne[1] != tokens) {
        throw std::invalid_argument(
            "gdn_input_proj_host: expected [K,T] x (Q4 [QK,K], Q5 [V+Z,K]) -> qkv/z");
    }

    auto& timing = host_path_timing();
    timing.calls.fetch_add(1, std::memory_order_relaxed);
    const auto read_start = std::chrono::steady_clock::now();
    readback_rows(t_activation, x, qk.k, tokens, stream, "activation readback");
    require_cuda(cudaStreamSynchronize(stream), "activation sync");
    timing.read_ns.fetch_add(static_cast<unsigned long long>(steady_ns_since(read_start)),
                             std::memory_order_relaxed);

    const auto gemm_start = std::chrono::steady_clock::now();
    contract_into(qk, t_activation, t_projection, tokens);
    contract_into(value_z, t_activation, t_result, tokens);
    timing.gemm_ns.fetch_add(static_cast<unsigned long long>(steady_ns_since(gemm_start)),
                             std::memory_order_relaxed);

    const auto wb_start = std::chrono::steady_clock::now();
    writeback_row_range(t_projection, static_cast<std::size_t>(qk.n), 0,
                        static_cast<std::size_t>(qk.n), qkv, tokens, stream);
    writeback_row_range(t_result, static_cast<std::size_t>(value_z.n), 0,
                        static_cast<std::size_t>(value_rows), qkv, tokens, stream);
    writeback_row_range(t_result, static_cast<std::size_t>(value_z.n),
                        static_cast<std::size_t>(value_rows), static_cast<std::size_t>(z.ne[0]),
                        z, tokens, stream);
    timing.wb_ns.fetch_add(static_cast<unsigned long long>(steady_ns_since(wb_start)),
                           std::memory_order_relaxed);
}

} // namespace ninfer::ops::detail
