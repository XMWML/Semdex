from __future__ import annotations

from array import array
import os
from pathlib import Path
import sqlite3
import sys

import pytest

from semdex import indexer, scanner
from semdex.config import Config, ModelCfg, ScriptRule
from semdex.db import Database
from semdex.indexer import index_pending
from semdex.models import ModelUnavailable
from semdex.scanner import scan
from semdex.search import search
from semdex.watcher import DebouncedIndexer


def _env(tmp_path: Path):
    root = tmp_path / "files"
    root.mkdir()
    cfg = Config(db_path=tmp_path / "index.db", folders=[root])
    db = Database(cfg.db_path)
    return cfg, db, root


def test_changed_file_is_hidden_until_new_content_is_indexed(tmp_path: Path):
    cfg, db, root = _env(tmp_path)
    path = root / "note.txt"
    path.write_text("oldneedle", encoding="utf-8")
    scan(db, cfg, log=lambda *_: None)
    index_pending(db, cfg, log=lambda *_: None)
    assert search(db, cfg, "oldneedle", mode="fulltext")

    path.write_text("newneedle", encoding="utf-8")
    scan(db, cfg, log=lambda *_: None)
    assert not search(db, cfg, "oldneedle", mode="fulltext")
    assert not search(db, cfg, "newneedle", mode="fulltext")
    assert db.get_file_by_path(str(path.resolve()))["index_status"] == "pending"
    db.close()


def test_missing_watch_root_does_not_delete_prior_index(tmp_path: Path):
    cfg, db, root = _env(tmp_path)
    path = root / "keep.txt"
    path.write_text("preserve this", encoding="utf-8")
    scan(db, cfg, log=lambda *_: None)
    index_pending(db, cfg, log=lambda *_: None)

    root.rename(tmp_path / "offline")
    stats = scan(db, cfg, log=lambda *_: None)
    assert stats.removed == 0
    assert db.counts()["total"] == 1
    db.close()


def test_records_are_removed_when_a_folder_is_explicitly_unwatched(tmp_path: Path):
    cfg, db, root = _env(tmp_path)
    (root / "old.txt").write_text("old folder", encoding="utf-8")
    scan(db, cfg, log=lambda *_: None)
    index_pending(db, cfg, log=lambda *_: None)

    replacement = tmp_path / "replacement"
    replacement.mkdir()
    cfg.folders = [replacement]
    stats = scan(db, cfg, log=lambda *_: None)
    assert stats.removed == 1
    assert db.counts()["total"] == 0
    db.close()


def test_unreadable_child_directory_does_not_remove_existing_records(tmp_path: Path, monkeypatch):
    cfg, db, root = _env(tmp_path)
    blocked = root / "blocked"
    blocked.mkdir()
    kept = blocked / "keep.txt"
    kept.write_text("keep this indexed record", encoding="utf-8")
    scan(db, cfg, log=lambda *_: None)
    index_pending(db, cfg, log=lambda *_: None)
    stored_path = str(kept.resolve())

    def walk_with_unreadable_child(top, *, topdown=True, onerror=None, followlinks=False):
        assert top == root.resolve()
        assert topdown is True and followlinks is False
        yield str(root), ["blocked"], []
        assert onerror is not None
        onerror(PermissionError(13, "Permission denied", str(blocked)))

    monkeypatch.setattr("semdex.scanner.os.walk", walk_with_unreadable_child)
    stats = scan(db, cfg, log=lambda *_: None)

    assert stats.errors == 1
    assert stats.removed == 0
    assert db.get_file_by_path(stored_path) is not None
    db.close()


def test_unreadable_file_does_not_remove_its_existing_record(tmp_path: Path, monkeypatch):
    cfg, db, root = _env(tmp_path)
    kept = root / "temporarily-locked.txt"
    kept.write_text("preserve this file record", encoding="utf-8")
    scan(db, cfg, log=lambda *_: None)
    index_pending(db, cfg, log=lambda *_: None)
    stored_path = str(kept.resolve())

    original_lstat = Path.lstat

    def denied_lstat(path, *args, **kwargs):
        if path == kept:
            raise PermissionError(13, "Permission denied", str(kept))
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", denied_lstat)
    stats = scan(db, cfg, log=lambda *_: None)

    assert stats.errors == 1
    assert stats.removed == 0
    assert db.get_file_by_path(stored_path) is not None
    db.close()


def test_index_rejects_file_replaced_by_symlink_after_scan(tmp_path: Path):
    cfg, db, root = _env(tmp_path)
    tracked = root / "tracked.txt"
    tracked.write_text("inside watch root", encoding="utf-8")
    scan(db, cfg, log=lambda *_: None)
    stored_path = str(tracked.resolve())
    row = db.get_file_by_path(stored_path)
    assert row is not None
    file_id = int(row["id"])
    # Simulate stale derived rows left behind by a prior interrupted run.  The
    # path rejection must remove them as well as avoid reading the symlink.
    db.save_content(file_id, "stale indexed content", tracked.name)
    db.replace_chunks(file_id, [("stale vector", array("f", [1.0]).tobytes())])
    db.replace_entities(file_id, [{"name": "Stale Project", "type": "project"}])
    db.set_status(file_id, "pending")

    outside = tmp_path / "outside.txt"
    outside.write_text("outside secret must never be indexed", encoding="utf-8")
    tracked.unlink()
    try:
        tracked.symlink_to(outside)
    except OSError as e:
        pytest.skip(f"symlink unavailable on this filesystem: {e}")

    stats = index_pending(db, cfg, log=lambda *_: None)
    row = db.get_file_by_path(stored_path)
    assert row is not None
    assert stats.skipped == 1
    assert row["index_status"] == "skipped"
    assert db.get_content(file_id) is None
    assert db.entities_for_file(file_id) == []
    assert not db.conn.execute("SELECT 1 FROM chunks WHERE file_id=?", (file_id,)).fetchone()
    assert not search(db, cfg, "outside secret", mode="fulltext")
    db.close()


def test_scan_does_not_hash_a_file_swapped_to_symlink_before_secure_open(tmp_path: Path, monkeypatch):
    cfg, db, root = _env(tmp_path)
    tracked = root / "scan-race.txt"
    tracked.write_text("inside watch root", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("OUTSIDE_HASH_INPUT", encoding="utf-8")
    original_hash_file = scanner._hash_file

    def swap_then_hash(path, watch_root):
        tracked.unlink()
        try:
            tracked.symlink_to(outside)
        except OSError as e:
            pytest.skip(f"symlink unavailable on this filesystem: {e}")
        return original_hash_file(path, watch_root)

    monkeypatch.setattr(scanner, "_hash_file", swap_then_hash)
    stats = scan(db, cfg, log=lambda *_: None)

    assert stats.errors == 1
    assert stats.new_or_changed == 0
    assert db.counts()["total"] == 0
    db.close()


def test_index_does_not_follow_a_symlink_swapped_after_validation(tmp_path: Path, monkeypatch):
    cfg, db, root = _env(tmp_path)
    tracked = root / "race.txt"
    tracked.write_text("inside watch root", encoding="utf-8")
    scan(db, cfg, log=lambda *_: None)
    stored_path = str(tracked.resolve())
    file_id = int(db.get_file_by_path(stored_path)["id"])

    outside = tmp_path / "outside.txt"
    outside.write_text("OUTSIDE_SECRET", encoding="utf-8")
    original_validate = indexer._validated_source_path

    def validate_then_swap(path, config):
        result = original_validate(path, config)
        if result[0] is not None:
            tracked.unlink()
            try:
                tracked.symlink_to(outside)
            except OSError as e:
                pytest.skip(f"symlink unavailable on this filesystem: {e}")
        return result

    monkeypatch.setattr(indexer, "_validated_source_path", validate_then_swap)
    stats = index_pending(db, cfg, log=lambda *_: None)

    assert stats.skipped == 1
    assert db.get_file(file_id)["index_status"] == "skipped"
    assert db.get_content(file_id) is None
    assert not search(db, cfg, "OUTSIDE_SECRET", mode="fulltext")
    db.close()


def test_a_safely_rejected_path_requeues_when_the_regular_file_returns(tmp_path: Path):
    cfg, db, root = _env(tmp_path)
    tracked = root / "restored.txt"
    original_text = "the original regular file is safe again"
    tracked.write_text(original_text, encoding="utf-8")
    original_stat = tracked.stat()
    scan(db, cfg, log=lambda *_: None)
    stored_path = str(tracked.resolve())

    outside = tmp_path / "outside.txt"
    outside.write_text("never index this", encoding="utf-8")
    tracked.unlink()
    try:
        tracked.symlink_to(outside)
    except OSError as e:
        pytest.skip(f"symlink unavailable on this filesystem: {e}")
    index_pending(db, cfg, log=lambda *_: None)
    assert db.get_file_by_path(stored_path)["index_status"] == "skipped"

    tracked.unlink()
    tracked.write_text(original_text, encoding="utf-8")
    # Match the original metadata to exercise the fast path as well as the
    # same-content-hash recovery logic.
    os.utime(tracked, (original_stat.st_atime, original_stat.st_mtime))
    scan(db, cfg, log=lambda *_: None)
    assert db.get_file_by_path(stored_path)["index_status"] == "pending"
    index_pending(db, cfg, log=lambda *_: None)
    assert search(db, cfg, "regular file", mode="fulltext")
    db.close()


def test_index_rejects_pending_path_outside_configured_root(tmp_path: Path):
    cfg, db, _ = _env(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside content must never be indexed", encoding="utf-8")
    st = outside.stat()
    file_id, changed = db.upsert_scan(
        str(outside.resolve()), outside.name, outside.suffix, st.st_size, st.st_mtime, "outside"
    )
    assert changed

    stats = index_pending(db, cfg, log=lambda *_: None)
    row = db.get_file(file_id)
    assert row is not None
    assert stats.skipped == 1
    assert row["index_status"] == "skipped"
    assert db.get_content(file_id) is None
    db.close()


def test_lowering_size_limit_hides_stale_content_and_can_be_recovered(tmp_path: Path):
    cfg, db, root = _env(tmp_path)
    path = root / "limited.txt"
    path.write_text("受大小限制的旧内容", encoding="utf-8")
    scan(db, cfg, log=lambda *_: None)
    index_pending(db, cfg, log=lambda *_: None)
    assert search(db, cfg, "旧内容", mode="fulltext")

    cfg.max_file_mb = 0
    scan(db, cfg, log=lambda *_: None)
    assert not search(db, cfg, "旧内容", mode="fulltext")
    assert db.get_file_by_path(str(path.resolve()))["index_status"] == "too_large"

    cfg.max_file_mb = 1
    scan(db, cfg, log=lambda *_: None)
    index_pending(db, cfg, log=lambda *_: None)
    assert search(db, cfg, "旧内容", mode="fulltext")
    db.close()


def test_hybrid_search_falls_back_when_embedding_server_is_unavailable(tmp_path: Path, monkeypatch):
    cfg, db, root = _env(tmp_path)
    path = root / "note.txt"
    path.write_text("广州地铁三号线", encoding="utf-8")
    scan(db, cfg, log=lambda *_: None)
    index_pending(db, cfg, log=lambda *_: None)
    cfg.embedding = ModelCfg(enabled=True, model="mock")
    file_id = int(db.get_file_by_path(str(path.resolve()))["id"])
    db.replace_chunks(file_id, [("广州地铁三号线", array("f", [1.0, 0.0]).tobytes())])
    db.meta_set("embedding_model_id", "http://localhost:1234/v1\nmock")
    db.meta_set("embedding_index_identity", indexer.embedding_index_identity(cfg))

    def unavailable(self, texts):
        raise ModelUnavailable("embedding server unavailable")

    monkeypatch.setattr("semdex.search.ModelClient.embed", unavailable)
    hits = search(db, cfg, "地铁", mode="hybrid")
    assert hits and hits[0].source == "fulltext"
    with pytest.raises(ModelUnavailable):
        search(db, cfg, "地铁", mode="semantic")
    db.close()


def test_watcher_reuses_the_incremental_pipeline(tmp_path: Path):
    cfg, db, root = _env(tmp_path)
    db.close()
    (root / "live.txt").write_text("监听器索引的内容", encoding="utf-8")
    DebouncedIndexer(cfg, log=lambda *_: None).run()
    db = Database(cfg.db_path)
    assert search(db, cfg, "监听器", mode="fulltext")
    db.close()


def test_watcher_periodic_run_retries_failed_sources(tmp_path: Path):
    cfg, db, root = _env(tmp_path)
    source = root / "retry.source"
    source.write_text("original source", encoding="utf-8")
    script = tmp_path / "extract.py"
    script.write_text(
        f"#!{sys.executable}\nimport sys\nsys.exit(1)\n", encoding="utf-8"
    )
    script.chmod(script.stat().st_mode | 0o111)
    cfg.script_rules = [ScriptRule(match="*.source", script=str(script))]
    db.close()

    watcher = DebouncedIndexer(cfg, log=lambda *_: None)
    watcher.run()
    db = Database(cfg.db_path)
    row = db.get_file_by_path(str(source.resolve()))
    assert row is not None and row["index_status"] == "failed"
    db.close()

    script.write_text(
        f"#!{sys.executable}\nprint('periodic retry succeeded')\n", encoding="utf-8"
    )
    watcher.run()
    db = Database(cfg.db_path)
    assert db.get_file_by_path(str(source.resolve()))["index_status"] == "failed"
    db.close()

    watcher.run(retry_failed=True)
    db = Database(cfg.db_path)
    try:
        file_id = int(db.get_file_by_path(str(source.resolve()))["id"])
        assert db.get_file(file_id)["index_status"] == "done"
        assert db.get_content(file_id) == "periodic retry succeeded\n"
    finally:
        db.close()


def test_existing_database_is_migrated_for_entity_status(tmp_path: Path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE files (id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, filename TEXT NOT NULL, "
        "ext TEXT, size INTEGER, mtime REAL, content_hash TEXT, extractor TEXT, "
        "index_status TEXT NOT NULL DEFAULT 'pending', error_msg TEXT, indexed_at REAL)"
    )
    conn.commit()
    conn.close()

    db = Database(path)
    columns = {row["name"] for row in db.conn.execute("PRAGMA table_info(files)")}
    assert {"entity_status", "entity_error"} <= columns
    db.close()
