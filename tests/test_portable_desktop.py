from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import semdex.bootstrap as bootstrap_module
import semdex.desktop as desktop_module
from semdex.bootstrap import ensure_portable_directories, portable_environment
from semdex.config import AsrCfg, Config, load_config, write_default_config
from semdex.desktop import DesktopController
from semdex.extractors.base import ExtractContext
from semdex.extractors.legacy_office import LegacyOfficeExtractor
from semdex.extractors.media import MediaExtractor
from semdex.gui import SearchRequestGuard, _normalize_extractor_input_mode
from semdex.modelclient import ModelClient
from semdex.models import ExtractError
from semdex.paths import (
    configure_cache_environment,
    default_config_path,
    default_database_path,
    project_root,
)
from semdex.settings import save_settings


def test_project_paths_follow_semdex_home_without_cwd_dependency(tmp_path: Path, monkeypatch):
    portable_root = tmp_path / "portable-semdex"
    other_cwd = tmp_path / "elsewhere"
    portable_root.mkdir()
    other_cwd.mkdir()
    monkeypatch.setenv("SEMDEX_HOME", str(portable_root))
    monkeypatch.chdir(other_cwd)

    assert project_root() == portable_root
    assert default_config_path() == portable_root / ".semdex" / "config.toml"
    assert default_database_path() == portable_root / ".semdex" / "index.db"


def test_portable_cache_environment_includes_uv_managed_python(tmp_path: Path, monkeypatch):
    portable_root = tmp_path / "portable-semdex"
    portable_root.mkdir()
    monkeypatch.setenv("SEMDEX_HOME", str(portable_root))
    for name in ("UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "PIP_CACHE_DIR", "HF_HOME"):
        monkeypatch.delenv(name, raising=False)

    configure_cache_environment()
    assert Path(os.environ["UV_CACHE_DIR"]) == portable_root / ".uv-cache"
    assert Path(os.environ["UV_PYTHON_INSTALL_DIR"]) == portable_root / ".uv-python"
    assert Path(os.environ["PIP_CACHE_DIR"]) == portable_root / ".uv-cache" / "pip"
    assert Path(os.environ["HF_HOME"]) == portable_root / ".semdex" / "models" / "huggingface"


def test_bootstrap_sets_uv_paths_before_sync(tmp_path: Path):
    portable_root = tmp_path / "portable-semdex"
    environment = portable_environment(
        portable_root,
        {"PATH": "/usr/bin", "UV_PYTHON_INSTALL_DIR": "/old-user-cache"},
    )
    ensure_portable_directories(environment)

    assert Path(environment["SEMDEX_HOME"]) == portable_root
    assert Path(environment["UV_CACHE_DIR"]) == portable_root / ".uv-cache"
    assert Path(environment["UV_PYTHON_INSTALL_DIR"]) == portable_root / ".uv-python"
    assert Path(environment["HF_HOME"]) == portable_root / ".semdex" / "models" / "huggingface"
    assert Path(environment["TMPDIR"]) == portable_root / ".semdex" / "tmp"
    assert Path(environment["UV_PYTHON_INSTALL_DIR"]).is_dir()


def test_bootstrap_passes_project_local_environment_to_uv_before_sync(tmp_path: Path, monkeypatch):
    portable_root = tmp_path / "portable-semdex"
    calls: list[tuple[list[str], dict[str, object]]] = []

    monkeypatch.setattr(bootstrap_module, "application_root", lambda: portable_root)
    monkeypatch.setattr(bootstrap_module.shutil, "which", lambda *_args, **_kwargs: "uv")
    monkeypatch.setattr(
        bootstrap_module.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)) or SimpleNamespace(returncode=0),
    )

    assert bootstrap_module.main(["--sync-only"]) == 0
    command, kwargs = calls[0]
    assert command == ["uv", "sync", "--extra", "gui"]
    assert Path(str(kwargs["cwd"])) == portable_root
    environment = kwargs["env"]
    assert isinstance(environment, dict)
    assert Path(environment["UV_PYTHON_INSTALL_DIR"]) == portable_root / ".uv-python"


def test_bootstrap_can_launch_webui_with_selected_local_model_runtime(tmp_path: Path, monkeypatch):
    portable_root = tmp_path / "portable-semdex"
    calls: list[list[str]] = []

    monkeypatch.setattr(bootstrap_module, "application_root", lambda: portable_root)
    monkeypatch.setattr(bootstrap_module.shutil, "which", lambda *_args, **_kwargs: "uv")
    monkeypatch.setattr(
        bootstrap_module.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command) or SimpleNamespace(returncode=0),
    )

    assert bootstrap_module.main(["--web", "--with-asr", "--with-gguf"]) == 0
    assert calls == [
        ["uv", "sync", "--extra", "asr", "--extra", "gguf"],
        ["uv", "run", "--extra", "asr", "--extra", "gguf", "semdex", "serve"],
    ]


def test_gui_search_guard_discards_old_searches():
    guard = SearchRequestGuard()
    first = guard.begin()
    second = guard.begin()
    assert not guard.is_current(first)
    assert guard.is_current(second)


def test_gui_raw_image_mode_requires_only_supported_image_extensions():
    assert _normalize_extractor_input_mode(".png, JPG", "image") == "image"
    assert _normalize_extractor_input_mode(".txt", "image") == "text"
    assert _normalize_extractor_input_mode(".png, .pdf", "image") == "text"
    assert _normalize_extractor_input_mode(".png", "text") == "text"


def test_template_storage_paths_move_with_the_config_directory(tmp_path: Path):
    original = tmp_path / "original" / ".semdex" / "config.toml"
    write_default_config(original)
    before = load_config(original)
    assert before.db_path == original.parent / "index.db"
    assert before.temp_dir == original.parent / "tmp"
    assert before.model_dir == original.parent / "models"

    moved = tmp_path / "external-drive-copy" / ".semdex" / "config.toml"
    moved.parent.mkdir(parents=True)
    shutil.copy2(original, moved)
    after = load_config(moved)
    assert after.db_path == moved.parent / "index.db"
    assert after.temp_dir == moved.parent / "tmp"
    assert after.model_dir == moved.parent / "models"


def test_saving_project_managed_storage_keeps_relative_paths(tmp_path: Path):
    state = tmp_path / ".semdex"
    config_path = state / "config.toml"
    write_default_config(config_path)
    current = load_config(config_path)

    saved = save_settings(current, {"db_path": str(state / "index.db")})
    assert saved.db_path == state / "index.db"
    text = config_path.read_text(encoding="utf-8")
    assert 'db_path = "index.db"' in text
    assert 'temp_dir = "tmp"' in text
    assert 'model_dir = "models"' in text


def test_local_whisper_uses_project_model_directory(tmp_path: Path, monkeypatch):
    observed: dict[str, object] = {}

    class FakeWhisper:
        def __init__(self, *args, **kwargs):
            observed["args"] = args
            observed["kwargs"] = kwargs

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisper))
    MediaExtractor._models.clear()
    config = Config(
        db_path=tmp_path / "index.db",
        model_dir=tmp_path / ".semdex" / "models",
        asr=AsrCfg(enabled=True, provider="faster_whisper", model="base"),
    )
    ctx = ExtractContext(
        config=config,
        vision=ModelClient(config.vision, "vision"),
        llm=ModelClient(config.llm, "llm"),
    )

    MediaExtractor()._model(ctx)
    assert observed["kwargs"] == {
        "device": "auto",
        "compute_type": "int8",
        "download_root": str(config.model_dir / "whisper"),
    }
    assert (config.model_dir / "whisper").is_dir()


def test_legacy_office_uses_a_project_local_libreoffice_profile(tmp_path: Path, monkeypatch):
    source = tmp_path / "legacy.doc"
    source.write_bytes(b"legacy")
    commands: list[list[str]] = []

    def fake_run(args, **_kwargs):
        commands.append(args)
        return SimpleNamespace(returncode=1, stdout="", stderr="expected failure")

    monkeypatch.setattr(LegacyOfficeExtractor, "_converter", lambda _self: "fake-soffice")
    monkeypatch.setattr("semdex.extractors.legacy_office.subprocess.run", fake_run)
    config = Config(db_path=tmp_path / "index.db", temp_dir=tmp_path / ".semdex" / "tmp")
    ctx = ExtractContext(
        config=config,
        vision=ModelClient(config.vision, "vision"),
        llm=ModelClient(config.llm, "llm"),
    )

    with pytest.raises(ExtractError, match="转换失败"):
        LegacyOfficeExtractor().extract(source, ctx)
    profile_arg = next(arg for arg in commands[0] if arg.startswith("-env:UserInstallation="))
    assert str(config.temp_dir) in profile_arg


def test_desktop_controller_indexes_and_searches_without_a_gui_runtime(tmp_path: Path):
    source = tmp_path / "files"
    source.mkdir()
    (source / "note.txt").write_text("portable desktop search text", encoding="utf-8")
    config_path = tmp_path / ".semdex" / "config.toml"
    controller = DesktopController(config_path)
    controller.save_settings({"folders": [str(source)]})

    worker = controller.start_index()
    worker.join(timeout=10)
    assert not worker.is_alive()
    status = controller.status()
    assert status["indexing"] is False
    assert status["files"]["by_status"]["done"] == 1
    assert controller.config.temp_dir == config_path.parent / "tmp"
    assert controller.config.temp_dir.is_dir()
    assert list(controller.config.temp_dir.iterdir()) == []
    hits = controller.search("desktop search", "fulltext")
    assert len(hits) == 1
    assert controller.content(hits[0]["file_id"])["text"] == "portable desktop search text"


def test_desktop_reveal_uses_the_linux_file_manager_for_an_indexed_path(tmp_path: Path, monkeypatch):
    source = tmp_path / "files"
    source.mkdir()
    note = source / "note.txt"
    note.write_text("reveal me", encoding="utf-8")
    controller = DesktopController(tmp_path / ".semdex" / "config.toml")
    controller.save_settings({"folders": [str(source)]})
    worker = controller.start_index()
    worker.join(timeout=10)
    commands: list[list[str]] = []
    monkeypatch.setattr(desktop_module.sys, "platform", "linux")
    monkeypatch.setattr(desktop_module.subprocess, "Popen", lambda args: commands.append(args))

    controller.open_path(str(note.resolve()), reveal=True)
    assert commands == [["xdg-open", str(source)]]
