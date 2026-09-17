#pragma once

// Host-platform primitives. Feature code calls these instead of spelling out POSIX or Win32 calls
// itself, so exactly one translation unit per platform owns the difference: the Linux/macOS build
// issues the POSIX calls directly and the MSVC build maps the same operations onto the Win32 and
// CRT equivalents. Everything below is small and purpose-driven — process identity and terminal
// geometry for log and temporary-file naming, plus read-only byte-range file access for artifact
// loading. Nothing here is a general portability framework.

#include <cstddef>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <string>

namespace ninfer::platform {

// --- process and terminal queries ------------------------------------------------------------

// Identifier of the running process. Distinct processes never share it, which is what keeps
// temporary file names from colliding between concurrent runs.
[[nodiscard]] int process_id() noexcept;

// True when the process' standard error stream is an interactive terminal. This selects in-place
// progress rendering over line-oriented logging.
[[nodiscard]] bool standard_error_is_interactive() noexcept;

// Column count of the standard error terminal, or `fallback` when it is not a terminal.
[[nodiscard]] std::size_t standard_error_columns(std::size_t fallback) noexcept;

// Number of physical processor cores, or `fallback` when the host does not report them.
//
// This is deliberately not `std::thread::hardware_concurrency()`: on a hybrid part with SMT, the
// logical count is roughly twice the core count, and a CPU-bound kernel sized from it oversubscribes
// the cores it depends on. Measured on a 6P+8E host (14 physical, 20 logical), a memory-bound
// contraction that ran 20 threads was slower than one that ran one per physical core, because the
// efficiency cores stream at a fraction of the performance-core rate and every barrier then waits
// on them. Callers that size a compute pool should start here, not from the logical count.
[[nodiscard]] unsigned physical_core_count(unsigned fallback) noexcept;

// --- wall-clock calendar conversion ------------------------------------------------------------

// Broken-down calendar time, in the shape `std::tm` carries. The host's thread-safe conversion
// routines disagree on both name and argument order: POSIX takes `localtime_r(const time_t*, tm*)`
// and `gmtime_r`, MSVC takes `localtime_s(tm*, const time_t*)` and `gmtime_s`. Callers ask for the
// breakdown by kind rather than naming either spelling.
[[nodiscard]] std::tm local_calendar_time(std::int64_t unix_seconds) noexcept;
[[nodiscard]] std::tm utc_calendar_time(std::int64_t unix_seconds) noexcept;

// --- read-only byte-range files --------------------------------------------------------------

// An open read-only file. The host representation is deliberately opaque: the POSIX build stores a
// descriptor and the MSVC build a file handle. The default-constructed value is unopened, which is
// what `valid()` distinguishes.
struct ReadHandle {
#if defined(_WIN32)
    void* native = nullptr;
    [[nodiscard]] bool valid() const noexcept { return native != nullptr; }
#else
    int native = -1;
    [[nodiscard]] bool valid() const noexcept { return native >= 0; }
#endif
};

// Highest byte offset `read_at` accepts, and the largest single transfer it performs. Both host
// interfaces address files with a signed 64-bit offset and Win32 `ReadFile` takes a 32-bit byte
// count, so one limit covers both and callers need no platform branch of their own.
inline constexpr std::uint64_t kMaxReadOffset  = 0x7fff'ffff'ffff'ffffULL;
inline constexpr std::uint64_t kMaxReadAtBytes = 0xffff'ffffULL;

// Opens `path` read-only. `unbuffered` requests an uncached transfer where the host offers one
// (POSIX `O_DIRECT`, Windows `FILE_FLAG_NO_BUFFERING`); the caller is then responsible for that
// host's alignment requirement. A failure is reported through `last_error_message()`.
[[nodiscard]] ReadHandle open_read_only(const std::filesystem::path& path, bool unbuffered) noexcept;

void close_read(ReadHandle handle) noexcept;

// Result of inspecting an open handle. `Failed` and `NotRegular` are separate because they are
// different operator errors: an unreadable file versus a path that is not a file at all.
enum class FileShape : std::uint8_t {
    Regular,
    NotRegular,
    Failed,
};

// Reports whether the handle refers to a regular file and, when it does, its length as `size`.
[[nodiscard]] FileShape describe_file(ReadHandle handle, std::uint64_t& size) noexcept;

// Reads `count` bytes starting at `offset` without disturbing a shared file position, so concurrent
// readers of one handle cannot disturb each other. Returns the byte count, 0 at end of file, or -1.
// `count` must not exceed `kMaxReadAtBytes` and `offset` must not exceed `kMaxReadOffset`.
[[nodiscard]] std::int64_t read_at(ReadHandle handle,
                                   std::byte* destination,
                                   std::size_t count,
                                   std::uint64_t offset) noexcept;

// True when the most recent `read_at` failure is a host interrupt that should be retried unchanged.
[[nodiscard]] bool retry_after_interrupt() noexcept;

// Human-readable description of the calling thread's most recent failure from this layer.
[[nodiscard]] std::string last_error_message();

} // namespace ninfer::platform
