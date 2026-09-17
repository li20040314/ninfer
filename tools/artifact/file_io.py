"""Keep large offline transfers from retaining whole artifacts in Linux's page cache."""

from __future__ import annotations

import os

IO_CHUNK_BYTES = 8 * 1024 * 1024
WRITEBACK_BYTES = 64 * 1024 * 1024
try:
    _PAGE_BYTES = os.sysconf("SC_PAGE_SIZE")
except AttributeError:  # Windows: os.sysconf is unavailable.
    _PAGE_BYTES = 4096


def discard_cached_pages(fd: int, offset: int = 0, count: int | None = None) -> None:
    if not hasattr(os, "posix_fadvise"):
        return  # Windows: the OS manages the standby list; nothing to drop.
    if count is None:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    elif count > 0:
        begin = offset // _PAGE_BYTES * _PAGE_BYTES
        end = (offset + count + _PAGE_BYTES - 1) // _PAGE_BYTES * _PAGE_BYTES
        os.posix_fadvise(fd, begin, end - begin, os.POSIX_FADV_DONTNEED)


if hasattr(os, "pread"):
    pread = os.pread
    pwrite = os.pwrite
else:
    # Windows: emulate positional I/O with explicit lseek; save/restore the file
    # position and serialize access so concurrent callers stay consistent.
    import threading

    _positional_lock = threading.Lock()

    def pread(fd: int, count: int, offset: int) -> bytes:
        with _positional_lock:
            saved = os.lseek(fd, 0, os.SEEK_CUR)
            try:
                os.lseek(fd, offset, os.SEEK_SET)
                blocks: list[bytes] = []
                remaining = count
                while remaining > 0:
                    block = os.read(fd, min(1 << 20, remaining))
                    if not block:
                        break
                    blocks.append(block)
                    remaining -= len(block)
                return b"".join(blocks)
            finally:
                os.lseek(fd, saved, os.SEEK_SET)

    def pwrite(fd: int, data, offset: int) -> int:
        with _positional_lock:
            saved = os.lseek(fd, 0, os.SEEK_CUR)
            try:
                os.lseek(fd, offset, os.SEEK_SET)
                written = 0
                view = memoryview(data)
                while written < len(data):
                    written += os.write(fd, view[written:])
                return written
            finally:
                os.lseek(fd, saved, os.SEEK_SET)


def fdatasync(fd: int) -> None:
    if hasattr(os, "fdatasync"):
        os.fdatasync(fd)
    else:  # Windows: fdatasync is unavailable; fsync also flushes metadata.
        os.fsync(fd)


class Writeback:
    """Bound dirty output across all open shards; release clean pages after writeback."""

    def __init__(self) -> None:
        self._bytes = 0
        self._fds: set[int] = set()

    def written(self, fd: int, count: int) -> None:
        self._fds.add(fd)
        self._bytes += count
        if self._bytes >= WRITEBACK_BYTES:
            self.flush()

    def flush(self) -> None:
        for fd in self._fds:
            fdatasync(fd)
            discard_cached_pages(fd)
        self._fds.clear()
        self._bytes = 0
