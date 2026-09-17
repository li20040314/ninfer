#include "artifact/file_io.h"

#include "artifact/framing.h"
#include "artifact/schema.h"
#include "core/platform.h"

#include <algorithm>
#include <string>
#include <utility>

namespace ninfer::artifact {
namespace {

// UTF-8 text for diagnostics. The native narrow encoding is not UTF-8 on Windows and paths there
// are UTF-16, so neither `string()` nor copying the native bytes round-trips a non-ASCII name.
std::string path_text(const std::filesystem::path& path) {
    const std::u8string utf8 = path.u8string();
    return {reinterpret_cast<const char*>(utf8.data()), utf8.size()};
}

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw ArtifactError(path_text(path) + ": " + operation + ": " + platform::last_error_message());
}

// Positional reads address the file with a signed 64-bit offset on every host.
void require_readable_offset(std::uint64_t offset) {
    if (offset > platform::kMaxReadOffset) {
        throw ArtifactError("file offset exceeds positional I/O range");
    }
}

} // namespace

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
    fd_ = platform::open_read_only(path_, false);
    if (!fd_.valid()) { fail(path_, "open"); }

    std::uint64_t size = 0;
    switch (platform::describe_file(fd_, size)) {
    case platform::FileShape::Regular:
        break;
    case platform::FileShape::NotRegular:
        platform::close_read(fd_);
        fd_ = {};
        throw ArtifactError(path_text(path_) + ": expected a regular file");
    case platform::FileShape::Failed: {
        // Read the message before closing: the close overwrites the thread's error state.
        const std::string error = platform::last_error_message();
        platform::close_read(fd_);
        fd_ = {};
        throw ArtifactError(path_text(path_) + ": fstat: " + error);
    }
    }
    bytes_ = size;
}

InputFile::~InputFile() {
    platform::close_read(direct_fd_);
    platform::close_read(fd_);
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_text(path_) + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        require_readable_offset(offset);
        const auto read = platform::read_at(fd_, destination.data(), count, offset);
        if (read < 0) {
            if (platform::retry_after_interrupt()) { continue; }
            fail(path_, "pread");
        }
        if (!read) { throw ArtifactError(path_text(path_) + ": unexpected EOF"); }
        offset += static_cast<std::uint64_t>(read);
        destination = destination.subspan(static_cast<std::size_t>(read));
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment ||
        destination.size() > platform::kMaxReadAtBytes) {
        throw ArtifactError(path_text(path_) + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
    if (!direct_fd_.valid()) {
        direct_fd_ = platform::open_read_only(path_, true);
        if (!direct_fd_.valid()) { fail(path_, "open direct"); }
    }
    std::int64_t read;
    do {
        require_readable_offset(offset);
        read = platform::read_at(direct_fd_, destination.data(), destination.size(), offset);
    } while (read < 0 && platform::retry_after_interrupt());
    if (read < 0) { fail(path_, "direct pread"); }
    return static_cast<std::size_t>(read);
}

} // namespace ninfer::artifact
