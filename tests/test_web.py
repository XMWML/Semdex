from __future__ import annotations

import asyncio
import hashlib
import stat
from pathlib import Path

import httpx
import pytest

from semdex.config import AsrCfg, Config, ModelCfg, OcrCfg
from semdex.config import load_config
from semdex.db import Database
from semdex.models import ModelUnavailable
from semdex.web.app import create_app


def test_content_api_returns_complete_indexed_text_and_clamps_search_limit(tmp_path: Path):
    root = tmp_path / "files"
    root.mkdir()
    path = root / "large.txt"
    text = "完整正文" + ("x" * 220_000)
    path.write_text(text, encoding="utf-8")
    cfg = Config(db_path=tmp_path / "index.db", folders=[root])
    db = Database(cfg.db_path)
    file_id, _ = db.upsert_scan(str(path), path.name, ".txt", path.stat().st_size, path.stat().st_mtime, "hash")
    db.save_content(file_id, text, path.name)
    db.set_status(file_id, "done")
    db.meta_set("embedding_rebuild_required", "1")
    db.close()

    async def request():
        transport = httpx.ASGITransport(app=create_app(cfg))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            content = await client.get("/api/content", params={"file_id": file_id})
            search = await client.get("/api/search", params={"q": "完整正文", "limit": -1})
            status = await client.get("/api/status")
        return content, search, status

    content, search, status = asyncio.run(request())
    assert content.status_code == 200
    assert len(content.json()["text"]) == len(text)
    assert search.status_code == 200
    assert len(search.json()["hits"]) == 1
    assert status.status_code == 200
    assert status.json()["embedding_rebuild_required"] is True


def test_full_rebuild_api_reextracts_unchanged_files(tmp_path: Path, monkeypatch):
    root = tmp_path / "files"
    root.mkdir()
    path = root / "note.txt"
    path.write_text("refreshed indexed text", encoding="utf-8")
    cfg = Config(db_path=tmp_path / "index.db", folders=[root])
    db = Database(cfg.db_path)
    stat = path.stat()
    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    file_id, _ = db.upsert_scan(
        str(path), path.name, path.suffix, stat.st_size, stat.st_mtime, content_hash
    )
    db.save_content(file_id, "stale indexed text", path.name)
    db.set_status(file_id, "done")
    db.close()

    class InlineThread:
        def __init__(self, *, target, daemon):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr("semdex.web.app.threading.Thread", InlineThread)

    async def request():
        transport = httpx.ASGITransport(app=create_app(cfg))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/api/rebuild")

    response = asyncio.run(request())
    assert response.status_code == 200
    assert response.json() == {"ok": True, "started": True, "full_rebuild": True}

    db = Database(cfg.db_path)
    try:
        assert db.get_content(file_id) == "refreshed indexed text"
        assert db.get_file(file_id)["index_status"] == "done"
    finally:
        db.close()


def test_full_rebuild_keeps_fulltext_when_embedding_is_offline(tmp_path: Path, monkeypatch):
    root = tmp_path / "files"
    root.mkdir()
    path = root / "note.txt"
    path.write_text("full text must still rebuild", encoding="utf-8")
    cfg = Config(
        db_path=tmp_path / "index.db",
        folders=[root],
        embedding=ModelCfg(enabled=True, model="offline"),
    )

    class InlineThread:
        def __init__(self, *, target, daemon):
            self.target = target

        def start(self):
            self.target()

    def unavailable_embedding(*_args, **_kwargs):
        raise ModelUnavailable("embedding service offline")

    monkeypatch.setattr("semdex.web.app.threading.Thread", InlineThread)
    monkeypatch.setattr("semdex.indexer.embed_missing", unavailable_embedding)

    async def request():
        transport = httpx.ASGITransport(app=create_app(cfg))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            rebuild = await client.post("/api/rebuild")
            status = await client.get("/api/status")
        return rebuild, status

    rebuild, status = asyncio.run(request())
    assert rebuild.status_code == 200
    last_run = status.json()["last_run"]
    assert last_run["embedding_error"] == "embedding service offline"
    assert "error" not in last_run

    db = Database(cfg.db_path)
    try:
        row = db.get_file_by_path(str(path.resolve()))
        assert row is not None
        assert db.get_content(int(row["id"])) == "full text must still rebuild"
        assert db.meta_get("embedding_rebuild_required") == "1"
    finally:
        db.close()


def test_settings_api_saves_multiple_folders_without_returning_secrets(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    config_path = tmp_path / "config.toml"
    cfg = Config(
        db_path=tmp_path / "index.db",
        folders=[first],
        llm=ModelCfg(enabled=True, base_url="http://127.0.0.1:1234/v1", api_key="keep-secret", model="old"),
        config_path=config_path,
    )

    async def request():
        transport = httpx.ASGITransport(app=create_app(cfg))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            before = await client.get("/api/settings")
            updated = await client.put("/api/settings", json={"settings": {
                "folders": [str(first), str(second)],
                "models": {
                    "agent": {
                        "enabled": True,
                        "base_url": "http://127.0.0.1:11434/v1",
                        "model": "qwen-agent",
                        "api_key": "agent-secret",
                    },
                    "entities": {
                        "enabled": True,
                        "base_url": "http://127.0.0.1:1234/v1",
                        "model": "entity-model",
                    },
                },
                "ocr": {
                    "enabled": True,
                    "provider": "local_http",
                    "endpoint": "http://127.0.0.1:8080/ocr",
                    "response_path": "result.text",
                },
                "asr": {
                    "enabled": True,
                    "provider": "openai_compatible",
                    "base_url": "http://127.0.0.1:9000/v1",
                    "endpoint": "",
                    "model": "whisper-large-v3",
                    "response_path": "data.text",
                },
                "entities": {"enabled": True, "max_chars": 3000, "max_per_file": 9},
                "agent": {"max_steps": 4, "max_results": 7},
                "agent_fallback": {"enabled": True, "max_bytes": 4096},
            }})
            status = await client.get("/api/status")
            page = await client.get("/settings")
        return before, updated, status, page

    before, updated, status, page = asyncio.run(request())
    assert before.status_code == 200
    before_settings = before.json()["settings"]
    assert "api_key" not in before_settings["models"]["llm"]
    assert before_settings["models"]["llm"]["api_key_configured"] is True

    assert updated.status_code == 200
    returned = updated.json()["settings"]
    assert returned["folders"] == [str(first), str(second)]
    assert returned["models"]["agent"]["model"] == "qwen-agent"
    assert returned["models"]["agent"]["api_key_configured"] is True
    assert returned["asr"]["response_path"] == "data.text"
    assert status.json()["folders"] == [str(first), str(second)]
    assert page.status_code == 200
    assert "索引目录、模型与本地能力" in page.text

    saved = load_config(config_path)
    assert saved.folders == [first, second]
    assert saved.llm.api_key == "keep-secret"
    assert saved.agent_model.model == "qwen-agent"
    assert saved.agent_model.api_key == "agent-secret"
    assert saved.entities_model.model == "entity-model"
    assert saved.ocr.endpoint == "http://127.0.0.1:8080/ocr"
    assert saved.asr.response_path == "data.text"
    assert saved.entities.enabled is True
    assert saved.agent_fallback.enabled is True
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600


def test_settings_api_rejects_missing_folder_without_overwriting_config(tmp_path: Path):
    root = tmp_path / "files"
    root.mkdir()
    config_path = tmp_path / "config.toml"
    cfg = Config(db_path=tmp_path / "index.db", folders=[root], config_path=config_path)

    async def request():
        transport = httpx.ASGITransport(app=create_app(cfg))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.put("/api/settings", json={"settings": {
                "folders": [str(tmp_path / "does-not-exist")],
            }})

    response = asyncio.run(request())
    assert response.status_code == 422
    assert "不是目录" in response.json()["detail"]
    assert not config_path.exists()


def test_settings_api_rejects_a_directory_as_the_database_path(tmp_path: Path):
    root = tmp_path / "files"
    root.mkdir()
    config_path = tmp_path / "config.toml"
    cfg = Config(db_path=tmp_path / "index.db", folders=[root], config_path=config_path)

    async def request():
        transport = httpx.ASGITransport(app=create_app(cfg))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.put("/api/settings", json={"settings": {
                "db_path": str(tmp_path),
            }})

    response = asyncio.run(request())
    assert response.status_code == 422
    assert "不能是目录" in response.json()["detail"]
    assert not config_path.exists()


def test_enabling_fallback_requeues_files_skipped_without_an_extractor(tmp_path: Path):
    root = tmp_path / "files"
    root.mkdir()
    path = root / "notes.unknown"
    path.write_text("需要使用兜底模型的内容", encoding="utf-8")
    config_path = tmp_path / "config.toml"
    cfg = Config(db_path=tmp_path / "index.db", folders=[root], config_path=config_path)
    db = Database(cfg.db_path)
    stat_result = path.stat()
    file_id, _ = db.upsert_scan(
        str(path), path.name, path.suffix, stat_result.st_size, stat_result.st_mtime, "hash"
    )
    db.set_status(file_id, "skipped", error="没有适用的提取器")
    db.close()

    async def request():
        transport = httpx.ASGITransport(app=create_app(cfg))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.put("/api/settings", json={"settings": {
                "agent_fallback": {"enabled": True},
            }})

    response = asyncio.run(request())
    assert response.status_code == 200

    db = Database(cfg.db_path)
    try:
        assert db.get_file(file_id)["index_status"] == "pending"
        assert db.get_file(file_id)["error_msg"] is None
    finally:
        db.close()


def test_settings_api_can_clear_saved_model_and_capability_keys(tmp_path: Path):
    root = tmp_path / "files"
    root.mkdir()
    config_path = tmp_path / "config.toml"
    cfg = Config(
        db_path=tmp_path / "index.db",
        folders=[root],
        config_path=config_path,
        agent_model=ModelCfg(api_key="agent-secret"),
        ocr=OcrCfg(api_key="ocr-secret"),
        asr=AsrCfg(api_key="asr-secret"),
    )

    async def request():
        transport = httpx.ASGITransport(app=create_app(cfg))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.put("/api/settings", json={"settings": {
                "models": {"agent": {"clear_api_key": True}},
                "ocr": {"clear_api_key": True},
                "asr": {"clear_api_key": True},
            }})

    response = asyncio.run(request())
    assert response.status_code == 200
    returned = response.json()["settings"]
    assert returned["models"]["agent"]["api_key_configured"] is False
    assert returned["ocr"]["api_key_configured"] is False
    assert returned["asr"]["api_key_configured"] is False

    saved = load_config(config_path)
    assert saved.agent_model.api_key == ""
    assert saved.ocr.api_key == ""
    assert saved.asr.api_key == ""


@pytest.mark.parametrize(
    ("base_url", "model"),
    [
        ("http://127.0.0.1:1234/v1", "embedding-v2"),
        ("http://127.0.0.1:8080/v1", "embedding-v1"),
    ],
)
def test_settings_api_immediately_invalidates_vectors_for_embedding_identity_change(
    tmp_path: Path, base_url: str, model: str
):
    root = tmp_path / "files"
    root.mkdir()
    path = root / "note.txt"
    path.write_text("需要重新建立向量的内容", encoding="utf-8")
    cfg = Config(
        db_path=tmp_path / "index.db",
        folders=[root],
        config_path=tmp_path / "config.toml",
        embedding=ModelCfg(
            enabled=True,
            base_url="http://127.0.0.1:1234/v1",
            model="embedding-v1",
        ),
    )
    db = Database(cfg.db_path)
    stat_result = path.stat()
    file_id, _ = db.upsert_scan(
        str(path), path.name, path.suffix, stat_result.st_size, stat_result.st_mtime, "hash"
    )
    db.save_content(file_id, path.read_text(encoding="utf-8"), path.name)
    db.replace_chunks(file_id, [("需要重新建立向量的内容", b"old-vector")])
    db.set_status(file_id, "done")
    db.meta_set("embedding_model_id", "http://127.0.0.1:1234/v1\nembedding-v1")
    db.close()

    async def request():
        transport = httpx.ASGITransport(app=create_app(cfg))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.put("/api/settings", json={"settings": {
                "models": {"embedding": {
                    "enabled": True,
                    "base_url": base_url,
                    "model": model,
                }},
            }})

    response = asyncio.run(request())
    assert response.status_code == 200

    db = Database(cfg.db_path)
    try:
        assert db.chunks_with_embeddings() == []
        assert db.meta_get("embedding_rebuild_required") == "1"
    finally:
        db.close()
