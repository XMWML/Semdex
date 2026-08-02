from __future__ import annotations

from array import array
from pathlib import Path

import pytest

from semdex.config import Config, ModelCfg
from semdex.db import Database
from semdex.indexer import embed_missing, index_pending
from semdex.models import EmbeddingRebuildRequired, ModelUnavailable
from semdex.search import search


def _indexed_file(db: Database, root: Path, name: str, text: str) -> int:
    path = root / name
    path.write_text(text, encoding="utf-8")
    stat = path.stat()
    file_id, _ = db.upsert_scan(
        str(path), name, path.suffix, stat.st_size, stat.st_mtime, f"hash-{name}"
    )
    db.save_content(file_id, text, name)
    db.replace_chunks(file_id, [(text, array("f", [1.0, 0.0]).tobytes())])
    db.set_status(file_id, "done")
    return file_id


def test_model_change_invalidates_old_vectors_and_hybrid_falls_back(tmp_path: Path):
    root = tmp_path / "files"
    root.mkdir()
    cfg = Config(
        db_path=tmp_path / "index.db",
        folders=[root],
        embedding=ModelCfg(enabled=True, model="new-model"),
    )
    db = Database(cfg.db_path)
    _indexed_file(db, root, "note.txt", "广州地铁三号线")
    db.meta_set("embedding_model", "old-model")

    # The normal incremental path must never mix vectors from both models.
    index_pending(db, cfg, log=lambda *_: None)

    assert db.chunks_with_embeddings() == []
    assert db.meta_get("embedding_rebuild_required") == "1"
    with pytest.raises(EmbeddingRebuildRequired):
        search(db, cfg, "地铁", mode="semantic")
    hits = search(db, cfg, "地铁", mode="hybrid")
    assert hits and hits[0].source == "fulltext"
    db.close()


def test_vectors_without_a_recorded_model_also_require_a_rebuild(tmp_path: Path):
    root = tmp_path / "files"
    root.mkdir()
    cfg = Config(
        db_path=tmp_path / "index.db",
        folders=[root],
        embedding=ModelCfg(enabled=True, model="new-model"),
    )
    db = Database(cfg.db_path)
    _indexed_file(db, root, "legacy.txt", "来源未知的旧向量")

    index_pending(db, cfg, log=lambda *_: None)

    assert db.chunks_with_embeddings() == []
    assert db.meta_get("embedding_rebuild_required") == "1"
    db.close()


def test_embedding_endpoint_change_requires_a_full_rebuild(tmp_path: Path):
    root = tmp_path / "files"
    root.mkdir()
    cfg = Config(
        db_path=tmp_path / "index.db",
        folders=[root],
        embedding=ModelCfg(enabled=True, base_url="http://new-host/v1", model="shared-name"),
    )
    db = Database(cfg.db_path)
    _indexed_file(db, root, "note.txt", "同名模型也可能来自不同服务")
    db.meta_set("embedding_model", "shared-name")
    db.meta_set("embedding_model_id", "http://old-host/v1\nshared-name")

    index_pending(db, cfg, log=lambda *_: None)

    assert db.chunks_with_embeddings() == []
    assert db.meta_get("embedding_rebuild_required") == "1"
    db.close()


def test_semantic_search_rejects_vectors_from_a_different_endpoint_before_querying(tmp_path: Path):
    root = tmp_path / "files"
    root.mkdir()
    cfg = Config(
        db_path=tmp_path / "index.db",
        folders=[root],
        embedding=ModelCfg(enabled=True, base_url="http://new-host/v1", model="shared-name"),
    )
    db = Database(cfg.db_path)
    _indexed_file(db, root, "note.txt", "同名模型的来源也必须一致")
    db.meta_set("embedding_model", "shared-name")
    db.meta_set("embedding_model_id", "http://old-host/v1\nshared-name")

    # A search can run before the next indexing pass.  It must still refuse to
    # compare a query vector from the new endpoint to the old vector space.
    with pytest.raises(EmbeddingRebuildRequired, match="服务地址"):
        search(db, cfg, "来源", mode="semantic")
    assert db.chunks_with_embeddings() == []
    assert db.meta_get("embedding_rebuild_required") == "1"
    db.close()


def test_explicit_rebuild_replaces_every_vector_before_clearing_marker(tmp_path: Path, monkeypatch):
    root = tmp_path / "files"
    root.mkdir()
    cfg = Config(
        db_path=tmp_path / "index.db",
        folders=[root],
        embedding=ModelCfg(enabled=True, model="new-model"),
    )
    db = Database(cfg.db_path)
    _indexed_file(db, root, "one.txt", "第一份内容")
    _indexed_file(db, root, "two.txt", "第二份内容")
    db.meta_set("embedding_model", "old-model")
    index_pending(db, cfg, log=lambda *_: None)

    class FakeEmbeddingClient:
        def __init__(self, _cfg, _kind):
            pass

        def embed(self, texts):
            return [[float(index + 1), 1.0] for index, _ in enumerate(texts)]

    monkeypatch.setattr("semdex.indexer.ModelClient", FakeEmbeddingClient)
    assert embed_missing(db, cfg, log=lambda *_: None, rebuild=True) == 2
    assert len(db.chunks_with_embeddings()) == 2
    assert db.meta_get("embedding_model") == "new-model"
    assert db.meta_get("embedding_model_id") == "http://localhost:1234/v1\nnew-model"
    assert db.meta_get("embedding_rebuild_required") is None
    db.close()


def test_failed_rebuild_keeps_the_database_marked_for_rebuild(tmp_path: Path, monkeypatch):
    root = tmp_path / "files"
    root.mkdir()
    cfg = Config(
        db_path=tmp_path / "index.db",
        folders=[root],
        embedding=ModelCfg(enabled=True, model="new-model"),
    )
    db = Database(cfg.db_path)
    _indexed_file(db, root, "note.txt", "恢复前内容")
    db.meta_set("embedding_model", "old-model")
    index_pending(db, cfg, log=lambda *_: None)

    class OfflineEmbeddingClient:
        def __init__(self, _cfg, _kind):
            pass

        def embed(self, _texts):
            raise ModelUnavailable("模型服务不可用")

    monkeypatch.setattr("semdex.indexer.ModelClient", OfflineEmbeddingClient)
    with pytest.raises(ModelUnavailable):
        embed_missing(db, cfg, log=lambda *_: None, rebuild=True)
    assert db.meta_get("embedding_rebuild_required") == "1"
    assert db.chunks_with_embeddings() == []
    db.close()
