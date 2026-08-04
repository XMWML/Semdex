"""扫描监控文件夹，登记新增/变化，清理消失的记录。

变化判定两级：size+mtime 都没变 → 直接跳过（不读文件）；
否则算 sha256，哈希变了才重新进入 pending（避免无谓重跑模型）。
"""
from __future__ import annotations

import hashlib
import os
from fnmatch import fnmatch
from pathlib import Path
import stat
from collections.abc import Callable

from .config import Config
from .db import Database, INVALID_PATH_ERROR_PREFIX
from .models import ScanStats
from .safepath import open_regular_file_beneath_root


def _hash_file(path: Path, root: Path) -> tuple[str, os.stat_result]:
    fd = open_regular_file_beneath_root(path, root)
    h = hashlib.sha256()
    try:
        source = os.fdopen(fd, "rb", closefd=True)
        fd = -1
        with source:
            source_stat = os.fstat(source.fileno())
            while chunk := source.read(1 << 20):
                h.update(chunk)
        return h.hexdigest(), source_stat
    finally:
        if fd != -1:
            os.close(fd)


def _excluded(name: str, patterns: list[str]) -> bool:
    return any(fnmatch(name, p) for p in patterns)


def scan(
    db: Database,
    config: Config,
    log=print,
    progress: Callable[..., None] | None = None,
) -> ScanStats:
    # Keep every caller (CLI, watcher, web and native desktop) on the same
    # persistent invalidation contract before metadata short-circuits unchanged
    # files below.
    from .indexer import synchronize_index_state

    synchronize_index_state(db, config, log=log)
    stats = ScanStats()
    present: set[str] = set()
    scanned_roots: list[Path] = []
    configured_roots: list[Path] = []
    inaccessible_roots: list[Path] = []
    max_bytes = config.max_file_mb * 1024 * 1024

    for configured_root in config.folders:
        root = configured_root.expanduser().resolve()
        configured_roots.append(root)
        if not root.is_dir():
            log(f"⚠ 监控文件夹不存在，跳过: {root}")
            continue
        scanned_roots.append(root)

        def on_walk_error(error: OSError, *, watch_root: Path = root) -> None:
            stats.errors += 1
            failed_path = Path(error.filename) if error.filename else watch_root
            if not failed_path.is_absolute():
                failed_path = watch_root / failed_path
            try:
                inaccessible_roots.append(failed_path.resolve(strict=False))
            except OSError:
                inaccessible_roots.append(failed_path.absolute())
            log(f"⚠ 无法读取目录，保留已有索引: {failed_path}")

        for dirpath, dirnames, filenames in os.walk(
            root, topdown=True, onerror=on_walk_error, followlinks=False
        ):
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith(".") and not _excluded(d, config.exclude)
            ]
            for fn in filenames:
                if fn.startswith(".") or _excluded(fn, config.exclude):
                    continue
                p = Path(dirpath) / fn
                if progress is not None:
                    progress("scanning", current_file=str(p), scanned=stats.scanned)
                try:
                    st = p.lstat()
                except OSError:
                    # A file can transiently be locked, permission-denied, or
                    # disappear between os.walk() and stat().  It must not be
                    # mistaken for a deliberate deletion in remove_missing().
                    present.add(str(p.absolute()))
                    stats.errors += 1
                    continue
                if not stat.S_ISREG(st.st_mode):
                    continue
                path_str = str(p.absolute())
                if st.st_size > max_bytes:
                    # Keep an existing entry from being treated as a deleted file
                    # when the configured size limit is lowered temporarily, but
                    # never leave its prior indexed text searchable.
                    present.add(path_str)
                    row = db.get_file_by_path(path_str)
                    if row is not None and row["index_status"] != "too_large":
                        db.mark_too_large(int(row["id"]), st.st_size, st.st_mtime)
                    stats.too_large += 1
                    continue

                stats.scanned += 1
                present.add(path_str)

                row = db.get_file_by_path(path_str)
                if (
                    row is not None
                    and row["size"] == st.st_size
                    and row["mtime"] == st.st_mtime
                    and row["index_status"] != "too_large"
                    and not (
                        row["index_status"] == "skipped"
                        and (row["error_msg"] or "").startswith(INVALID_PATH_ERROR_PREFIX)
                    )
                ):
                    stats.unchanged += 1
                    continue
                try:
                    content_hash, secure_stat = _hash_file(p, root)
                except OSError:
                    stats.errors += 1
                    continue
                if secure_stat.st_size > max_bytes:
                    row = db.get_file_by_path(path_str)
                    if row is not None and row["index_status"] != "too_large":
                        db.mark_too_large(int(row["id"]), secure_stat.st_size, secure_stat.st_mtime)
                    stats.too_large += 1
                    continue
                _, changed = db.upsert_scan(
                    path_str, fn, p.suffix.lower(), secure_stat.st_size, secure_stat.st_mtime, content_hash
                )
                if changed:
                    stats.new_or_changed += 1
                else:
                    stats.unchanged += 1

    if configured_roots:
        stats.removed = db.remove_missing(
            present, scanned_roots, configured_roots, inaccessible_roots
        )
    if progress is not None:
        progress("scanning", current_file="", scanned=stats.scanned)
    return stats
