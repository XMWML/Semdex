"""Race-safe opening of regular files beneath configured watch roots."""
from __future__ import annotations

import os
from pathlib import Path
import stat

from .config import Config


def configured_watch_roots(config: Config) -> list[Path]:
    roots: list[Path] = []
    for configured_root in config.folders:
        try:
            root = configured_root.expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if root.is_dir():
            roots.append(root)
    return roots


def _supports_dirfd_nofollow() -> bool:
    """Whether this runtime can use the stronger openat-style path walk."""
    return (
        hasattr(os, "O_NOFOLLOW")
        and hasattr(os, "O_DIRECTORY")
        and os.open in os.supports_dir_fd
    )


def _is_link_or_reparse_point(st: os.stat_result) -> bool:
    """Treat Windows junctions as links as well as ordinary symlinks."""
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(st, "st_file_attributes", 0)
    return stat.S_ISLNK(st.st_mode) or bool(reparse_flag and attributes & reparse_flag)


def _validate_regular_path_without_links(path: Path, root: Path) -> os.stat_result:
    """Validate every component when openat/O_NOFOLLOW is unavailable.

    The returned ``lstat`` result is compared with the opened descriptor below.
    That turns a path-swap race into a rejection while preserving the descriptor
    as the only object the scanner or indexer ever reads from.
    """
    try:
        parts = path.relative_to(root).parts
    except ValueError as e:
        raise OSError("索引路径不在监控目录中") from e
    if not parts:
        raise OSError("索引路径不是监控目录中的普通文件")

    root_stat = root.lstat()
    if _is_link_or_reparse_point(root_stat) or not stat.S_ISDIR(root_stat.st_mode):
        raise OSError("监控目录不是安全的真实目录")

    current = root
    for part in parts[:-1]:
        current = current / part
        current_stat = current.lstat()
        if _is_link_or_reparse_point(current_stat) or not stat.S_ISDIR(current_stat.st_mode):
            raise OSError("索引路径包含符号链接或非目录组件")

    final_path = current / parts[-1]
    final_stat = final_path.lstat()
    if _is_link_or_reparse_point(final_stat) or not stat.S_ISREG(final_stat.st_mode):
        raise OSError("索引路径不是普通文件")
    return final_stat


def _same_open_file(opened: os.stat_result, path_stat: os.stat_result) -> bool:
    """Require a stable inode identity instead of trusting a later path lookup."""
    if not opened.st_ino or not path_stat.st_ino:
        # A platform without stable file IDs cannot prove that a path did not
        # change while it was being opened, so fail closed.
        return False
    return os.path.samestat(opened, path_stat)


def _open_checked_without_dirfd(path: Path, root: Path) -> int:
    """Portable checked-open fallback for Windows and limited Python runtimes."""
    _validate_regular_path_without_links(path, root)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    file_fd = os.open(path, flags)
    try:
        opened_stat = os.fstat(file_fd)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise OSError("索引路径不是普通文件")
        path_stat = _validate_regular_path_without_links(path, root)
        if not _same_open_file(opened_stat, path_stat):
            raise OSError("打开期间路径已变化")
    except Exception:
        os.close(file_fd)
        raise
    return file_fd


def open_regular_file_beneath_root(path: Path, root: Path) -> int:
    """Open a regular in-root file without trusting a mutable pathname.

    POSIX platforms use an ``openat`` walk with ``O_NOFOLLOW``.  Windows and
    other runtimes without those flags use a checked descriptor fallback: it
    rejects symlinks/junctions in every component and verifies that the opened
    descriptor still identifies the same regular file after opening.
    """
    if not _supports_dirfd_nofollow():
        return _open_checked_without_dirfd(path, root)

    parts = path.relative_to(root).parts
    if not parts:
        raise OSError("索引路径不是监控目录中的普通文件")

    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | close_on_exec
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | close_on_exec
    directory_fd = os.open(root, directory_flags)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(parts[-1], file_flags, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)

    try:
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise OSError("索引路径不是普通文件")
    except Exception:
        os.close(file_fd)
        raise
    return file_fd
