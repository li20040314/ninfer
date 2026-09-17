#include "ops/cpu/rowsplit_gemm.h"

#include "core/platform.h"
#include "core/weight_view.h"
#include "ops/cpu/rowsplit_gemm_detail.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ninfer::ops::cpu {
namespace {

namespace detail = ops::cpu::detail;

// Row blocks are cut so each one moves roughly this many bytes of weight. A block that is too small
// pays the cursor hand-off more often than it pays for memory; one that is too large leaves workers
// idle behind whatever block finishes last. 512 KiB was the knee on the development host.
constexpr std::uint64_t kTargetBlockBytes = 512ull * 1024ull;

[[nodiscard]] int code_bits(QType qtype) noexcept {
    switch (qtype) {
    case QType::Q4_G64_FP16:
        return 4;
    case QType::Q5_G64_FP16:
        return 5;
    case QType::Q6_G64_FP16:
        return 6;
    case QType::Q8_G32_FP16:
        return 8;
    default:
        return 0;
    }
}

[[nodiscard]] bool finite_geometry(std::int32_t n, std::int32_t k) noexcept { return n > 0 && k > 0; }

// Derives the row geometry from the engine's own single source of truth, so the CPU contraction
// cannot drift from the layout the device contracts read.
[[nodiscard]] detail::RowSplitGeometry derive_geometry(const Weight& weight) {
    if (weight.layout != QuantLayout::RowSplit) {
        throw std::invalid_argument("cpu rowsplit gemm: weight layout is not RowSplit");
    }
    if (!finite_geometry(weight.n, weight.k)) {
        throw std::invalid_argument("cpu rowsplit gemm: weight extent must be positive");
    }
    if (weight.qdata == nullptr || weight.scales == nullptr) {
        throw std::invalid_argument("cpu rowsplit gemm: weight planes are not resident");
    }

    const int bits = code_bits(weight.qtype);
    if (bits == 0) {
        throw std::invalid_argument("cpu rowsplit gemm: weight format has no CPU contraction");
    }
    // Only the 5- and 6-bit formats carry a high plane; q4's code is a whole nibble and q8's a
    // whole byte, so a null qhigh there is correct rather than missing.
    const bool needs_high = bits == 5 || bits == 6;
    if (needs_high && weight.qhigh == nullptr) {
        throw std::invalid_argument("cpu rowsplit gemm: high plane is not resident");
    }

    const std::uint64_t shape[2] = {static_cast<std::uint64_t>(weight.n),
                                    static_cast<std::uint64_t>(weight.k)};
    const WeightGeometry geo     = weight_geometry(weight.qtype, weight.layout, shape);

    // The parent's padded column count is authoritative for the row stride; a padded_shape that
    // disagrees means the weight was bound against a parent this module does not understand.
    const auto padded = static_cast<std::uint64_t>(weight.padded_shape[1]);
    if (padded < static_cast<std::uint64_t>(weight.k) || padded != geo.padded_columns) {
        throw std::invalid_argument("cpu rowsplit gemm: padded columns disagree with the parent");
    }

    detail::RowSplitGeometry out;
    out.bits             = bits;
    out.group_size       = static_cast<std::uint32_t>(geo.group_size);
    out.groups           = geo.padded_columns / geo.group_size;
    out.code_row_bytes   = geo.code_bytes_per_row;
    out.high_row_bytes   = geo.high_bytes_per_row;
    out.scale_row_bytes  = geo.scale_bytes_per_row;
    out.high_per_group   = geo.group_size == 0 ? 0 : geo.high_bytes_per_row / out.groups;
    out.high_offset      = geo.high_offset;
    out.scale_offset     = geo.scale_offset;
    if (out.groups == 0 || out.code_row_bytes == 0 || out.scale_row_bytes == 0) {
        throw std::invalid_argument("cpu rowsplit gemm: weight geometry is degenerate");
    }
    return out;
}

// --- worker pool --------------------------------------------------------------------------------
//
// A persistent pool with one shared cursor, rather than a thread per call or a static partition.
// Thread-per-call would rebuild 13 threads per layer; a static partition would make every worker
// wait for the slowest block, which on a hybrid P/E-core host is the difference between the memory
// ceiling and two thirds of it.
class RowSplitWorkers {
public:
    static RowSplitWorkers& shared() {
        static RowSplitWorkers instance;
        return instance;
    }

    RowSplitWorkers(const RowSplitWorkers&)            = delete;
    RowSplitWorkers& operator=(const RowSplitWorkers&) = delete;

    // The pool outlives every contraction, so its workers are still parked when the process tears
    // its statics down. Without this, `~vector<thread>` finds twelve joinable threads and calls
    // std::terminate -- which on Windows fast-fails with 0xC0000409, long after the work was done
    // and with its exit code masking whatever the run actually produced.
    ~RowSplitWorkers() {
        std::vector<std::thread> retired;
        {
            std::lock_guard<std::mutex> state(state_mutex_);
            stopping_ = true;
            start_cv_.notify_all();
            retired.swap(threads_);
        }
        for (auto& worker : retired) { worker.join(); }
    }

    void run(std::uint64_t items, std::uint64_t granularity,
             const std::function<void(std::uint64_t, std::uint64_t)>& body) {
        if (items == 0) { return; }
        const std::uint64_t step = std::max<std::uint64_t>(1, granularity);

        // One contraction at a time: `body_` is shared with the pool threads.
        std::lock_guard<std::mutex> call(call_mutex_);
        {
            std::lock_guard<std::mutex> state(state_mutex_);
            body_        = &body;
            cursor_.store(0, std::memory_order_relaxed);
            items_       = items;
            step_        = step;
            finished_    = 0;
            ++generation_;
        }
        start_cv_.notify_all();

        pull();

        {
            std::unique_lock<std::mutex> state(state_mutex_);
            done_cv_.wait(state, [this] { return finished_ == threads_.size(); });
            body_ = nullptr;
        }
    }

    unsigned threads() const noexcept {
        std::lock_guard<std::mutex> state(state_mutex_);
        return static_cast<unsigned>(threads_.size());
    }

    // Resizes the pool. The requested thread count is the total workforce including the caller.
    void set_threads(unsigned requested) {
        const unsigned target = std::max(1u, requested) - 1u;
        std::lock_guard<std::mutex> call(call_mutex_);
        std::vector<std::thread> retired;
        {
            std::lock_guard<std::mutex> state(state_mutex_);
            if (target == threads_.size()) { return; }
            // Retirement has to be signalled and the join performed *outside* the state lock: a
            // thread parked in start_cv_.wait cannot re-acquire a mutex the joiner still holds.
            if (!threads_.empty()) {
                stopping_ = true;
                start_cv_.notify_all();
            }
            retired.swap(threads_);
        }
        for (auto& worker : retired) { worker.join(); }
        {
            std::lock_guard<std::mutex> state(state_mutex_);
            stopping_ = false;
            for (unsigned index = 0; index < target; ++index) {
                threads_.emplace_back([this, index] { worker_loop(index); });
            }
        }
    }

private:
    RowSplitWorkers() { set_threads(default_threads()); }

    [[nodiscard]] static unsigned default_threads() {
        const unsigned cores = platform::physical_core_count(1);
        // Leave one physical core for the caller's own share of the work.
        return cores > 1 ? cores - 1 : 1;
    }

    void pull() {
        for (;;) {
            const std::uint64_t base =
                cursor_.fetch_add(step_, std::memory_order_relaxed);
            if (base >= items_) { return; }
            (*body_)(base, std::min(base + step_, items_));
        }
    }

    void worker_loop(unsigned index) {
        (void)index;
        std::uint64_t seen = 0;
        std::unique_lock<std::mutex> state(state_mutex_);
        for (;;) {
            start_cv_.wait(state, [this, seen] { return stopping_ || generation_ != seen; });
            if (stopping_) { return; }
            seen = generation_;
            state.unlock();
            pull();
            state.lock();
            ++finished_;
            done_cv_.notify_all();
        }
    }

    std::mutex call_mutex_;
    mutable std::mutex state_mutex_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    std::vector<std::thread> threads_;
    const std::function<void(std::uint64_t, std::uint64_t)>* body_ = nullptr;
    std::atomic<std::uint64_t> cursor_{0};
    std::uint64_t items_   = 0;
    std::uint64_t step_    = 1;
    std::uint64_t generation_ = 0;
    unsigned finished_     = 0;
    bool stopping_         = false;
};

// --- activation staging -------------------------------------------------------------------------
//
// The engine stores a [K,T] activation token-major ([tokens][k], k contiguous), and so does this
// contraction; staging exists only to widen bf16 to fp32 once per call so the inner loops do not
// re-decode the half type. The buffer is thread-local and only grows, because a contraction is a
// per-layer event, not a per-token one.
thread_local std::vector<float> t_activation;

// Converts the engine's bf16 activation to fp32 in place, keeping its [tokens][k] order. The
// token axis is the outer one, so a token's row of columns is contiguous -- getting this backwards
// is invisible at tokens == 1, where both orders coincide, and silently wrong above it.
void stage_activation(const std::uint16_t* activation, std::int32_t tokens, std::int32_t k,
                      std::uint64_t padded, const float*& staged) {
    t_activation.resize(static_cast<std::size_t>(tokens) * padded);
    for (std::int32_t token = 0; token < tokens; ++token) {
        float* row = t_activation.data() + static_cast<std::size_t>(token) * padded;
        for (std::int32_t column = 0; column < k; ++column) {
            row[column] = detail::bf16_to_float(
                activation[static_cast<std::size_t>(token) * k + column]);
        }
        // Columns the weight pads to 128 contribute nothing; the scale of a padded group is zero,
        // but leaving the activation zero keeps the product exactly zero regardless.
        std::fill(row + k, row + padded, 0.0F);
    }
    staged = t_activation.data();
}

} // namespace

bool rowsplit_gemm_supported(const Weight& weight) noexcept {
    if (weight.layout != QuantLayout::RowSplit || !finite_geometry(weight.n, weight.k)) {
        return false;
    }
    const int bits = code_bits(weight.qtype);
    if (bits == 0 || weight.qdata == nullptr || weight.scales == nullptr) { return false; }
    // q4 and q8 carry no high plane, so only the 5- and 6-bit formats require one.
    return (bits == 5 || bits == 6) ? weight.qhigh != nullptr : true;
}

void rowsplit_gemm(const Weight& weight, const void* activation_bf16, void* out_bf16,
                   std::int32_t tokens) {
    if (tokens <= 0) {
        throw std::invalid_argument("cpu rowsplit gemm: token count must be positive");
    }
    if (activation_bf16 == nullptr || out_bf16 == nullptr) {
        throw std::invalid_argument("cpu rowsplit gemm: activation and output must be resident");
    }

    const detail::RowSplitGeometry geometry = derive_geometry(weight);
    const std::uint64_t padded              = weight.padded_shape[1] >= weight.k
                                                  ? static_cast<std::uint64_t>(weight.padded_shape[1])
                                                  : static_cast<std::uint64_t>(weight.k);

    // Every code width is contracted in storage order: q8 reads one signed byte per element, and
    // the nibble formats' unpacklo/unpackhi interleave byte j's two nibbles back into elements 2j
    // and 2j+1, which is the same order the scale groups and the activation columns use.
    const float* staged = nullptr;
    stage_activation(static_cast<const std::uint16_t*>(activation_bf16), tokens, weight.k, padded,
                     staged);

    const auto* payload  = static_cast<const std::uint8_t*>(weight.qdata);
    auto* out            = static_cast<std::uint16_t*>(out_bf16);
    const std::uint64_t rows = static_cast<std::uint64_t>(weight.n);
    // The result is token-major, [tokens][n], because that is how the engine stores a [N,T]
    // tensor -- the same convention that makes the activation [tokens][k].
    const std::uint64_t out_row_stride = rows;

    const std::uint64_t bytes_per_row =
        geometry.code_row_bytes + geometry.high_row_bytes + geometry.scale_row_bytes;
    const std::uint64_t granularity = std::max<std::uint64_t>(1, kTargetBlockBytes / bytes_per_row);

    // The vectorized contraction covers q4/q5/q8; q6 stays on the scalar reference until it is
    // needed (no shipped artifact uses it), and so does any host without AVX2.
    const bool vectorized = geometry.bits != 6 && detail::host_has_avx2();

    RowSplitWorkers::shared().run(rows, granularity, [&](std::uint64_t begin, std::uint64_t end) {
        if (vectorized) {
            detail::gemm_rows(geometry, payload, padded, begin, end, staged, tokens, out_row_stride,
                              out);
        } else {
            detail::gemm_rows_scalar(geometry, payload, padded, begin, end, staged, tokens,
                                     out_row_stride, out);
        }
    });
}

unsigned rowsplit_worker_threads() noexcept { return RowSplitWorkers::shared().threads() + 1u; }

void set_rowsplit_worker_threads(unsigned threads) noexcept {
    RowSplitWorkers::shared().set_threads(threads);
}

bool rowsplit_avx2_available() noexcept { return detail::host_has_avx2(); }

} // namespace ninfer::ops::cpu
