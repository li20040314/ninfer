#include "core/platform.h"

#include <cerrno>
#include <cstring>
#include <system_error>
#include <thread>
#include <vector>

#if defined(_WIN32)

// `windows.h` must see these before it is parsed: NOMINMAX keeps the min/max macros out of every
// later <algorithm> use and WIN32_LEAN_AND_MEAN avoids the Winsock 1 headers the host code never
// wants. Both are confined to this translation unit.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <io.h>
#include <process.h>

#else

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#endif

namespace ninfer::platform {

int process_id() noexcept {
#if defined(_WIN32)
    return ::_getpid();
#else
    return static_cast<int>(::getpid());
#endif
}

bool standard_error_is_interactive() noexcept {
#if defined(_WIN32)
    return ::_isatty(::_fileno(stderr)) != 0;
#else
    return ::isatty(STDERR_FILENO) == 1;
#endif
}

std::size_t standard_error_columns(std::size_t fallback) noexcept {
#if defined(_WIN32)
    CONSOLE_SCREEN_BUFFER_INFO info{};
    if (::GetConsoleScreenBufferInfo(::GetStdHandle(STD_ERROR_HANDLE), &info) == 0) { return fallback; }
    // The visible window, not the scrollback buffer: a wide buffer behind a narrow window would
    // otherwise render bars that wrap. This matches what TIOCGWINSZ reports on the POSIX side.
    const LONG columns = info.srWindow.Right - info.srWindow.Left + 1;
    return columns > 0 ? static_cast<std::size_t>(columns) : fallback;
#else
    winsize size{};
    if (::ioctl(STDERR_FILENO, TIOCGWINSZ, &size) == 0 && size.ws_col != 0) { return size.ws_col; }
    return fallback;
#endif
}

unsigned physical_core_count(unsigned fallback) noexcept {
#if defined(_WIN32)
    // Relationship `RelationProcessorCore` is reported once per physical core, so counting the
    // records gives the number SMT hides. The first call sizes the buffer and is expected to fail
    // with ERROR_INSUFFICIENT_BUFFER.
    DWORD bytes = 0;
    ::GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
    if (bytes == 0) { return fallback; }
    std::vector<std::byte> storage(bytes);
    if (::GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(storage.data()),
            &bytes) == 0) {
        return fallback;
    }
    unsigned cores = 0;
    for (DWORD offset = 0; offset < bytes;) {
        const auto* entry = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
            storage.data() + offset);
        if (entry->Relationship == RelationProcessorCore) { ++cores; }
        if (entry->Size == 0) { break; }
        offset += entry->Size;
    }
    return cores == 0 ? fallback : cores;
#else
    // Linux and macOS expose no portable physical-core count; the logical count is the only
    // available answer, and on those hosts it is also what the kernel schedules against.
    const unsigned logical = std::thread::hardware_concurrency();
    return logical == 0 ? fallback : logical;
#endif
}

std::tm local_calendar_time(std::int64_t unix_seconds) noexcept {    const std::time_t stamp = static_cast<std::time_t>(unix_seconds);
    std::tm out{};
#if defined(_WIN32)
    // The MSVC routines take the destination first and report failure through errno rather than
    // through the return value, which is why the result is default-constructed above and filled in.
    (void)::localtime_s(&out, &stamp);
#else
    ::localtime_r(&stamp, &out);
#endif
    return out;
}

std::tm utc_calendar_time(std::int64_t unix_seconds) noexcept {
    const std::time_t stamp = static_cast<std::time_t>(unix_seconds);
    std::tm out{};
#if defined(_WIN32)
    (void)::gmtime_s(&out, &stamp);
#else
    ::gmtime_r(&stamp, &out);
#endif
    return out;
}

ReadHandle open_read_only(const std::filesystem::path& path, bool unbuffered) noexcept {
#if defined(_WIN32)
    // Directories are rejected here by the flags rather than by a later inspection: without
    // FILE_FLAG_BACKUP_SEMANTICS, CreateFileW refuses to open a directory at all.
    const DWORD attributes =
        FILE_ATTRIBUTE_NORMAL | (unbuffered ? FILE_FLAG_NO_BUFFERING : 0);
    const HANDLE handle = ::CreateFileW(path.c_str(), GENERIC_READ,
                                        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                        nullptr, OPEN_EXISTING, attributes, nullptr);
    return handle == INVALID_HANDLE_VALUE ? ReadHandle{} : ReadHandle{handle};
#else
    return ReadHandle{
        ::open(path.c_str(), O_RDONLY | O_CLOEXEC | (unbuffered ? O_DIRECT : 0))};
#endif
}

void close_read(ReadHandle handle) noexcept {
    if (!handle.valid()) { return; }
#if defined(_WIN32)
    (void)::CloseHandle(static_cast<HANDLE>(handle.native));
#else
    (void)::close(handle.native);
#endif
}

FileShape describe_file(ReadHandle handle, std::uint64_t& size) noexcept {
    if (!handle.valid()) { return FileShape::Failed; }
#if defined(_WIN32)
    BY_HANDLE_FILE_INFORMATION info{};
    if (::GetFileInformationByHandle(static_cast<HANDLE>(handle.native), &info) == 0) {
        return FileShape::Failed;
    }
    // Consoles, pipes and device namespaces open successfully but carry no byte range to read.
    if ((info.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_DEVICE)) != 0) {
        return FileShape::NotRegular;
    }
    size = (static_cast<std::uint64_t>(info.nFileSizeHigh) << 32U) | info.nFileSizeLow;
    return FileShape::Regular;
#else
    struct stat status {};
    if (::fstat(handle.native, &status) != 0) { return FileShape::Failed; }
    if (status.st_size < 0 || !S_ISREG(status.st_mode)) { return FileShape::NotRegular; }
    size = static_cast<std::uint64_t>(status.st_size);
    return FileShape::Regular;
#endif
}

std::int64_t read_at(ReadHandle handle,
                     std::byte* destination,
                     std::size_t count,
                     std::uint64_t offset) noexcept {
    if (!handle.valid() || count == 0) { return 0; }
#if defined(_WIN32)
    if (count > kMaxReadAtBytes) {
        ::SetLastError(ERROR_INVALID_PARAMETER);
        return -1;
    }
    // A handle opened without FILE_FLAG_OVERLAPPED is synchronous, but supplying an OVERLAPPED is
    // still what selects the offset: the transfer starts at `Offset`/`OffsetHigh` and ReadFile does
    // not return until it completes. Concurrent callers therefore read their own ranges even though
    // the host also advances the shared file position as a side effect nothing here depends on.
    OVERLAPPED overlapped{};
    overlapped.Offset     = static_cast<DWORD>(offset & 0xffff'ffffULL);
    overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32U);
    DWORD transferred     = 0;
    if (::ReadFile(static_cast<HANDLE>(handle.native), destination, static_cast<DWORD>(count),
                   &transferred, &overlapped) == 0) {
        // A read that begins at end of file fails with ERROR_HANDLE_EOF, whereas POSIX pread reports
        // zero bytes. Callers distinguish "no more bytes" from a failure by the sign, so the end of
        // file has to be mapped here rather than leaking out as an error.
        return ::GetLastError() == ERROR_HANDLE_EOF ? 0 : -1;
    }
    return static_cast<std::int64_t>(transferred);
#else
    return static_cast<std::int64_t>(
        ::pread(handle.native, destination, count, static_cast<off_t>(offset)));
#endif
}

bool retry_after_interrupt() noexcept {
#if defined(_WIN32)
    // A synchronous Win32 transfer either completes or fails; there is no restartable interruption.
    return false;
#else
    return errno == EINTR;
#endif
}

std::string last_error_message() {
#if defined(_WIN32)
    return std::system_category().message(static_cast<int>(::GetLastError()));
#else
    return std::strerror(errno);
#endif
}

} // namespace ninfer::platform
