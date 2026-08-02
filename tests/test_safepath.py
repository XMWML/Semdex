from __future__ import annotations

import os
from pathlib import Path

import pytest

from semdex import safepath


def test_portable_checked_open_fallback_reads_regular_in_root_file(tmp_path: Path, monkeypatch):
    root = tmp_path / "files"
    root.mkdir()
    path = root / "note.txt"
    path.write_text("safe contents", encoding="utf-8")
    monkeypatch.setattr(safepath, "_supports_dirfd_nofollow", lambda: False)

    fd = safepath.open_regular_file_beneath_root(path, root)
    with os.fdopen(fd, "rb") as source:
        assert source.read() == b"safe contents"


def test_portable_checked_open_fallback_rejects_symlink(tmp_path: Path, monkeypatch):
    root = tmp_path / "files"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("must not be read", encoding="utf-8")
    path = root / "linked.txt"
    try:
        path.symlink_to(outside)
    except OSError as e:
        pytest.skip(f"symlink unavailable on this filesystem: {e}")
    monkeypatch.setattr(safepath, "_supports_dirfd_nofollow", lambda: False)

    with pytest.raises(OSError, match="普通文件|符号链接"):
        safepath.open_regular_file_beneath_root(path, root)
