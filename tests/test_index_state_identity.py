from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pytest

from semdex.config import Config, EntityCfg, ExtractorRule, LlmProvider, ModelCfg
from semdex.db import Database
from semdex.indexer import index_pending
from semdex.models import ModelUnavailable
from semdex.scanner import scan


def _file_id(db: Database, path: Path) -> int:
    row = db.get_file_by_path(str(path.resolve()))
    assert row is not None
    return int(row["id"])


def _chunk_rows(db: Database, file_id: int):
    return db.conn.execute(
        "SELECT chunk_index, text, embedding FROM chunks "
        "WHERE file_id=? ORDER BY chunk_index",
        (file_id,),
    ).fetchall()


@pytest.mark.parametrize(
    "change",
    [
        lambda config: config.extractor_rules.reverse(),
        lambda config: setattr(config.ocr, "languages", "eng+deu"),
        lambda config: setattr(config.asr, "compute_type", "float16"),
    ],
    ids=["rule-order", "ocr-options", "asr-options"],
)
def test_primary_runtime_changes_reindex_unchanged_files(
    tmp_path: Path,
    change,
):
    root = tmp_path / "files"
    root.mkdir()
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    path = root / "note.txt"
    path.write_text("unchanged primary text", encoding="utf-8")
    config = Config(
        db_path=tmp_path / "index.db",
        folders=[root],
        extractor_dir=plugin_dir,
    )

    with closing(Database(config.db_path)) as db:
        scan(db, config, log=lambda *_: None)
        assert index_pending(db, config, log=lambda *_: None).indexed == 1
        assert index_pending(db, config, log=lambda *_: None).indexed == 0

        change(config)

        stats = index_pending(db, config, log=lambda *_: None)
        file_id = _file_id(db, path)
        assert stats.indexed == 1
        assert db.get_file(file_id)["index_status"] == "done"
        assert db.get_content(file_id) == "unchanged primary text"


def test_existing_primary_content_without_an_identity_is_rebuilt_once(tmp_path: Path):
    root = tmp_path / "files"
    root.mkdir()
    path = root / "legacy.txt"
    path.write_text("current source text", encoding="utf-8")
    config = Config(db_path=tmp_path / "index.db", folders=[root])

    with closing(Database(config.db_path)) as db:
        stat = path.stat()
        file_id, _ = db.upsert_scan(
            str(path.resolve()),
            path.name,
            path.suffix,
            stat.st_size,
            stat.st_mtime,
            "legacy-hash",
        )
        db.save_content(file_id, "stale text from an older Semdex", path.name)
        db.set_status(file_id, "done")
        assert db.meta_get("primary_index_identity") is None

        assert index_pending(db, config, log=lambda *_: None).indexed == 1
        assert db.get_content(file_id) == "current source text"
        assert db.meta_get("primary_index_identity") is not None
        assert index_pending(db, config, log=lambda *_: None).indexed == 0


def test_llm_rule_prompt_and_provider_changes_reindex_unchanged_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_chat(client, messages, **_kwargs):
        prompt = messages[-1]["content"].splitlines()[0]
        return f"{client.cfg.model}:{prompt}"

    monkeypatch.setattr("semdex.indexer.ModelClient.chat", fake_chat)
    root = tmp_path / "files"
    root.mkdir()
    path = root / "plan.note"
    path.write_text("source body", encoding="utf-8")
    provider = LlmProvider(
        id="summarizer",
        name="Summarizer",
        enabled=True,
        model="model-v1",
    )
    config = Config(
        db_path=tmp_path / "index.db",
        folders=[root],
        extractor_dir=tmp_path / "plugins",
        llm_providers=[provider],
        extractor_rules=[
            ExtractorRule(
                id="notes",
                label="Notes",
                extensions=[".note"],
                kind="llm",
                provider="summarizer",
                prompt="prompt-one",
            )
        ],
    )

    with closing(Database(config.db_path)) as db:
        scan(db, config, log=lambda *_: None)
        assert index_pending(db, config, log=lambda *_: None).indexed == 1
        file_id = _file_id(db, path)
        assert "model-v1:prompt-one" in (db.get_content(file_id) or "")

        rule = next(rule for rule in config.extractor_rules if rule.id == "notes")
        rule.prompt = "prompt-two"
        assert index_pending(db, config, log=lambda *_: None).indexed == 1
        assert "model-v1:prompt-two" in (db.get_content(file_id) or "")

        configured_provider = config.llm_provider("summarizer")
        assert configured_provider is not None
        configured_provider.model = "model-v2"
        assert index_pending(db, config, log=lambda *_: None).indexed == 1
        content = db.get_content(file_id) or ""
        assert "model-v2:prompt-two" in content
        assert "model-v1:prompt-two" not in content


def test_plugin_source_change_is_invalidated_by_scan_and_reextracted(
    tmp_path: Path,
):
    root = tmp_path / "files"
    root.mkdir()
    path = root / "record.custom"
    path.write_text("source", encoding="utf-8")
    plugin_dir = tmp_path / "plugins"
    plugin = plugin_dir / "custom"
    plugin.mkdir(parents=True)
    plugin_path = plugin / "plugin.py"
    plugin_path.write_text(
        "def extract(path):\n"
        "    return 'plugin revision one:' + path.read_text(encoding='utf-8')\n",
        encoding="utf-8",
    )
    config = Config(
        db_path=tmp_path / "index.db",
        folders=[root],
        extractor_dir=plugin_dir,
        extractor_rules=[
            ExtractorRule(
                id="custom",
                label="Custom",
                extensions=[".custom"],
                kind="python",
                plugin="custom",
            )
        ],
    )

    with closing(Database(config.db_path)) as db:
        scan(db, config, log=lambda *_: None)
        assert index_pending(db, config, log=lambda *_: None).indexed == 1
        file_id = _file_id(db, path)
        assert db.get_content(file_id) == "plugin revision one:source"

        plugin_path.write_text(
            "def extract(path):\n"
            "    return 'plugin revision two changed:' + path.read_text(encoding='utf-8')\n",
            encoding="utf-8",
        )

        scan(db, config, log=lambda *_: None)
        assert db.get_file(file_id)["index_status"] == "pending"
        assert db.get_content(file_id) is None
        assert index_pending(db, config, log=lambda *_: None).indexed == 1
        assert db.get_content(file_id) == "plugin revision two changed:source"

        replacement_dir = tmp_path / "replacement-plugins"
        replacement = replacement_dir / "custom"
        replacement.mkdir(parents=True)
        (replacement / "plugin.py").write_text(
            "def extract(path):\n"
            "    return 'replacement plugin root:' + path.read_text(encoding='utf-8')\n",
            encoding="utf-8",
        )
        config.extractor_dir = replacement_dir

        assert index_pending(db, config, log=lambda *_: None).indexed == 1
        assert db.get_content(file_id) == "replacement plugin root:source"


def test_first_embedding_enable_backfills_existing_primary_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[list[str]] = []

    def fake_embed(_client, texts):
        calls.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]

    monkeypatch.setattr("semdex.indexer.ModelClient.embed", fake_embed)
    root = tmp_path / "files"
    root.mkdir()
    path = root / "existing.txt"
    path.write_text("primary content indexed before embeddings", encoding="utf-8")
    config = Config(db_path=tmp_path / "index.db", folders=[root])

    with closing(Database(config.db_path)) as db:
        scan(db, config, log=lambda *_: None)
        assert index_pending(db, config, log=lambda *_: None).indexed == 1
        file_id = _file_id(db, path)
        assert _chunk_rows(db, file_id) == []

        config.embedding = ModelCfg(enabled=True, model="embed-v1")
        stats = index_pending(db, config, log=lambda *_: None)

        rows = _chunk_rows(db, file_id)
        assert stats.indexed == 0
        assert calls
        assert rows and all(row["embedding"] is not None for row in rows)
        assert db.get_content(file_id) == "primary content indexed before embeddings"
        assert db.meta_get("embedding_rebuild_required") is None


def test_chunk_settings_change_automatically_rebuilds_all_vectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[list[str]] = []

    def fake_embed(_client, texts):
        calls.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]

    monkeypatch.setattr("semdex.indexer.ModelClient.embed", fake_embed)
    root = tmp_path / "files"
    root.mkdir()
    path = root / "long.txt"
    source_text = "0123456789" * 8
    path.write_text(source_text, encoding="utf-8")
    config = Config(
        db_path=tmp_path / "index.db",
        folders=[root],
        embedding=ModelCfg(enabled=True, model="embed-v1"),
        chunk_size=16,
        chunk_overlap=4,
    )

    with closing(Database(config.db_path)) as db:
        scan(db, config, log=lambda *_: None)
        assert index_pending(db, config, log=lambda *_: None).indexed == 1
        file_id = _file_id(db, path)
        before = [row["text"] for row in _chunk_rows(db, file_id)]
        calls_before = len(calls)
        assert len(before) > 1

        config.chunk_size = 31
        config.chunk_overlap = 3
        stats = index_pending(db, config, log=lambda *_: None)

        rows = _chunk_rows(db, file_id)
        after = [row["text"] for row in rows]
        assert stats.indexed == 0
        assert len(calls) > calls_before
        assert after != before
        assert len(after) < len(before)
        assert all(row["embedding"] is not None for row in rows)
        assert db.get_content(file_id) == source_text
        assert db.meta_get("embedding_rebuild_required") is None


def test_embedding_outage_keeps_primary_content_and_retries_on_next_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    offline = True

    def fake_embed(_client, texts):
        if offline:
            raise ModelUnavailable("embedding service offline")
        return [[float(len(text)), 1.0] for text in texts]

    monkeypatch.setattr("semdex.indexer.ModelClient.embed", fake_embed)
    root = tmp_path / "files"
    root.mkdir()
    path = root / "durable.txt"
    path.write_text("primary text survives a vector outage", encoding="utf-8")
    config = Config(db_path=tmp_path / "index.db", folders=[root])

    with closing(Database(config.db_path)) as db:
        scan(db, config, log=lambda *_: None)
        assert index_pending(db, config, log=lambda *_: None).indexed == 1
        file_id = _file_id(db, path)
        config.embedding = ModelCfg(enabled=True, model="embed-v1")

        failed = index_pending(db, config, log=lambda *_: None)
        assert failed.embed_errors
        assert db.get_file(file_id)["index_status"] == "done"
        assert db.get_content(file_id) == "primary text survives a vector outage"
        assert db.chunks_with_embeddings() == []
        assert db.meta_get("embedding_rebuild_required") == "1"

        offline = False
        recovered = index_pending(db, config, log=lambda *_: None)
        assert recovered.embed_errors == []
        assert db.chunks_with_embeddings()
        assert db.meta_get("embedding_rebuild_required") is None
        assert db.get_file(file_id)["index_status"] == "done"


def test_entity_parameter_and_model_changes_reextract_existing_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []

    def fake_chat(client, _messages, **_kwargs):
        calls.append(client.cfg.model)
        if client.cfg.model == "entity-v2":
            return '[{"name":"Bob","type":"person","context":"Bob owns it"}]'
        return (
            '[{"name":"Alice","type":"person","context":"Alice owns it"},'
            '{"name":"Project X","type":"project","context":"Project X"}]'
        )

    monkeypatch.setattr("semdex.entities.ModelClient.chat", fake_chat)
    root = tmp_path / "files"
    root.mkdir()
    path = root / "entities.txt"
    path.write_text("Alice owns Project X", encoding="utf-8")
    config = Config(
        db_path=tmp_path / "index.db",
        folders=[root],
        entities=EntityCfg(enabled=True, max_per_file=2),
        entities_model=ModelCfg(enabled=True, model="entity-v1"),
    )

    with closing(Database(config.db_path)) as db:
        scan(db, config, log=lambda *_: None)
        assert index_pending(db, config, log=lambda *_: None).entities_indexed == 1
        file_id = _file_id(db, path)
        assert {item["name"] for item in db.entities_for_file(file_id)} == {
            "Alice",
            "Project X",
        }

        config.entities.max_per_file = 1
        limited = index_pending(db, config, log=lambda *_: None)
        assert limited.indexed == 0
        assert limited.entities_indexed == 1
        assert [item["name"] for item in db.entities_for_file(file_id)] == ["Alice"]

        config.entities_model.model = "entity-v2"
        changed_model = index_pending(db, config, log=lambda *_: None)
        assert changed_model.indexed == 0
        assert changed_model.entities_indexed == 1
        assert [item["name"] for item in db.entities_for_file(file_id)] == ["Bob"]
        assert calls == ["entity-v1", "entity-v1", "entity-v2"]


def test_disabling_entities_removes_relations_from_agent_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "semdex.entities.ModelClient.chat",
        lambda *_args, **_kwargs: (
            '[{"name":"Alice","type":"person","context":"Alice owns it"}]'
        ),
    )
    root = tmp_path / "files"
    root.mkdir()
    path = root / "entities.txt"
    path.write_text("Alice owns the project", encoding="utf-8")
    config = Config(
        db_path=tmp_path / "index.db",
        folders=[root],
        entities=EntityCfg(enabled=True),
        entities_model=ModelCfg(enabled=True, model="entity-v1"),
    )

    with closing(Database(config.db_path)) as db:
        scan(db, config, log=lambda *_: None)
        assert index_pending(db, config, log=lambda *_: None).entities_indexed == 1
        file_id = _file_id(db, path)
        assert db.files_by_entity("alice") == [file_id]

        config.entities.enabled = False
        stats = index_pending(db, config, log=lambda *_: None)

        assert stats.indexed == 0
        assert stats.entities_indexed == 0
        assert db.entities_for_file(file_id) == []
        assert db.files_by_entity("alice") == []
        assert db.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 0
