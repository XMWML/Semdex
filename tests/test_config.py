from __future__ import annotations

import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest

from semdex.chunker import chunk_text
from semdex import cli
from semdex.config import load_config, write_default_config
from semdex.db import Database
from semdex.settings import save_settings, settings_dict


def test_invalid_chunk_overlap_is_rejected_when_loading_config(tmp_path: Path):
    config = tmp_path / "config.toml"
    config.write_text("[chunking]\nchunk_size = 100\nchunk_overlap = 100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="chunk_overlap"):
        load_config(config)


def test_chunker_rejects_invalid_direct_arguments():
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_text("text", chunk_size=0)
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_text("text", chunk_size=10, overlap=10)


def test_watch_reconcile_interval_can_be_disabled(tmp_path: Path):
    config = tmp_path / "config.toml"
    config.write_text("[watch]\nreconcile_sec = 0\n", encoding="utf-8")

    assert load_config(config).watch_reconcile_sec == 0.0


def test_default_config_and_database_are_created_owner_only(tmp_path: Path, monkeypatch):
    import semdex.config as config_module
    import semdex.db as db_module

    state_dir = tmp_path / ".semdex"
    state_dir.mkdir(mode=0o755)
    os.chmod(state_dir, 0o755)
    config_path = state_dir / "config.toml"
    db_path = state_dir / "index.db"
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", db_path)

    assert write_default_config() == config_path
    db = Database(db_path)
    db.meta_set("private", "yes")
    db.close()

    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(state_dir.stat().st_mode) == 0o700
    for sidecar in (db_path.with_name("index.db-wal"), db_path.with_name("index.db-shm")):
        if sidecar.exists():
            assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600


def test_serve_creates_a_default_config_for_the_settings_page(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "settings.toml"
    calls = []
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(
        run=lambda app, **kwargs: calls.append((app, kwargs)),
    ))

    result = cli.cmd_serve(SimpleNamespace(
        config=str(config_path), host="127.0.0.1", port=8787, no_open=True,
    ))

    assert result == 0
    assert config_path.exists()
    assert len(calls) == 1


def test_serve_rejects_a_non_loopback_host(tmp_path: Path):
    config_path = tmp_path / "settings.toml"

    result = cli.cmd_serve(SimpleNamespace(
        config=str(config_path), host="0.0.0.0", port=8787, no_open=True,
    ))

    assert result == 1
    assert not config_path.exists()


def test_stable_builtin_extension_rule_can_switch_to_llm_and_cannot_be_deleted(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    write_default_config(config_path)
    current = load_config(config_path)

    saved = save_settings(current, {
        "extractors": {
            "rules": [{
                "id": "pdf",
                "kind": "llm",
                "enabled": True,
                "extensions": [".pdf"],
                "model": "fallback",
            }],
        },
    })
    pdf = next(rule for rule in saved.extractor_rules if rule.id == "pdf")
    assert (pdf.label, pdf.kind, pdf.model) == ("PDF 文档", "llm", "fallback")
    settings_pdf = next(
        rule for rule in settings_dict(saved)["extractors"]["rules"] if rule["id"] == "pdf"
    )
    assert settings_pdf["builtin"] is True
    assert 'kind = "llm"' in config_path.read_text(encoding="utf-8")
    assert 'model = "fallback"' in config_path.read_text(encoding="utf-8")

    # Omitting a stable built-in ID from a later payload retains the configured
    # route instead of treating it as a deletable custom rule.
    preserved = save_settings(saved, {"extractors": {"rules": []}})
    assert next(rule for rule in preserved.extractor_rules if rule.id == "pdf").kind == "llm"


def test_llm_extension_rule_rejects_unknown_model_purpose(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    write_default_config(config_path)
    current = load_config(config_path)

    with pytest.raises(ValueError, match="model"):
        save_settings(current, {
            "extractors": {
                "rules": [{
                    "id": "notes",
                    "label": "笔记",
                    "kind": "llm",
                    "enabled": True,
                    "extensions": [".note"],
                    "model": "not-a-model",
                }],
            },
        })
