from pathlib import Path
from types import SimpleNamespace

import pytest

from semdex.extractors.script import (
    PythonFunctionExtractor,
    SHIPPED_PLUGIN_DIR,
    discover_python_plugins,
    resolve_python_plugin,
    seed_shipped_plugins,
)
from semdex.models import ExtractError


def test_folder_plugin_accepts_path_and_context(tmp_path: Path):
    plugin_dir = tmp_path / "plugins"
    folder = plugin_dir / "notes"
    folder.mkdir(parents=True)
    (folder / "plugin.py").write_text(
        "def extract(path, ctx):\n"
        "    return f'{ctx.prefix}:{path.read_text(encoding=\"utf-8\")}'\n",
        encoding="utf-8",
    )
    source = tmp_path / "note.txt"
    source.write_text("content", encoding="utf-8")

    extractor = PythonFunctionExtractor.from_plugin(plugin_dir, "notes")

    assert extractor.extract(source, SimpleNamespace(prefix="ctx")) == "ctx:content"


def test_folder_plugin_accepts_path_only_and_legacy_file_still_works(tmp_path: Path):
    plugin_dir = tmp_path / "plugins"
    folder = plugin_dir / "folder_style"
    folder.mkdir(parents=True)
    (folder / "plugin.py").write_text(
        "def convert(path):\n    return path.name\n",
        encoding="utf-8",
    )
    legacy = plugin_dir / "legacy.py"
    legacy.write_text("def extract(path):\n    return path.suffix\n", encoding="utf-8")
    source = tmp_path / "record.custom"
    source.touch()

    assert PythonFunctionExtractor(folder, "convert").extract(source, object()) == "record.custom"
    assert PythonFunctionExtractor(legacy).extract(source, object()) == ".custom"
    assert resolve_python_plugin(plugin_dir, "legacy.py") == legacy


def test_plugin_module_is_cached_until_its_source_changes(tmp_path: Path):
    plugin = tmp_path / "cached"
    plugin.mkdir()
    (plugin / "plugin.py").write_text(
        "loads = 0\n"
        "loads += 1\n"
        "def extract(path):\n    return loads\n",
        encoding="utf-8",
    )
    first = PythonFunctionExtractor(plugin)
    second = PythonFunctionExtractor(plugin)

    assert first.extract(tmp_path / "one", object()) == "1"
    assert second.extract(tmp_path / "two", object()) == "1"
    assert first._load_module() is second._load_module()


def test_plugin_references_cannot_escape_the_configured_directory(tmp_path: Path):
    with pytest.raises(ExtractError, match="单个文件夹名"):
        resolve_python_plugin(tmp_path, "../outside")


def test_discovery_reads_metadata_without_executing_plugin(tmp_path: Path):
    plugin_dir = tmp_path / "plugins"
    good = plugin_dir / "safe"
    good.mkdir(parents=True)
    marker = tmp_path / "imported"
    (good / "plugin.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "PLUGIN_METADATA = {\n"
        "    'name': 'Safe metadata',\n"
        "    'description': 'Read statically',\n"
        "    'function': 'convert',\n"
        "}\n"
        "def convert(path, ctx):\n    return path.name\n",
        encoding="utf-8",
    )
    broken = plugin_dir / "broken"
    broken.mkdir()

    discovered = {plugin.folder: plugin for plugin in discover_python_plugins(plugin_dir)}

    assert not marker.exists()
    assert discovered["safe"].as_dict() == {
        "id": "safe",
        "folder": "safe",
        "name": "Safe metadata",
        "description": "Read statically",
        "function": "convert",
        "builtin": False,
        "path": str(good.resolve()),
        "available": True,
        "error": "",
        "legacy": False,
    }
    assert discovered["broken"].available is False
    assert "plugin.py" in discovered["broken"].error


def test_seed_shipped_plugins_copies_missing_folders_without_overwriting(tmp_path: Path):
    destination = tmp_path / "plugins"
    custom_ocr = destination / "ocr"
    custom_ocr.mkdir(parents=True)
    sentinel = custom_ocr / "plugin.py"
    sentinel.write_text("# user-owned\n", encoding="utf-8")

    copied = seed_shipped_plugins(destination)

    assert {path.name for path in copied} == {"asr"}
    assert sentinel.read_text(encoding="utf-8") == "# user-owned\n"
    assert (destination / "asr" / "plugin.py").is_file()
    seeded_asr = next(
        plugin for plugin in discover_python_plugins(destination) if plugin.id == "asr"
    )
    assert seeded_asr.builtin is True
    assert seeded_asr.path == str((destination / "asr").resolve())
    assert seed_shipped_plugins(destination) == []


def test_shipped_plugins_have_static_available_metadata():
    discovered = {plugin.folder: plugin for plugin in discover_python_plugins(SHIPPED_PLUGIN_DIR)}

    assert discovered["ocr"].available is True
    assert discovered["ocr"].function == "extract"
    assert discovered["ocr"].builtin is True
    assert discovered["ocr"].path == str((SHIPPED_PLUGIN_DIR / "ocr").resolve())
    assert discovered["asr"].available is True
    assert discovered["asr"].description


def test_shipped_ocr_plugin_wraps_image_and_pdf_adapters(tmp_path: Path, monkeypatch):
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "semdex.ocr.ocr_image",
        lambda path, _cfg: calls.append(("image", path.name)) or "image text",
    )
    monkeypatch.setattr(
        "semdex.ocr.ocr_pdf",
        lambda path, _cfg: calls.append(("pdf", path.name)) or "pdf text",
    )
    PythonFunctionExtractor._module_cache.clear()
    extractor = PythonFunctionExtractor(SHIPPED_PLUGIN_DIR / "ocr")
    ctx = SimpleNamespace(config=SimpleNamespace(ocr=object()))

    assert extractor.extract(tmp_path / "scan.png", ctx) == "[OCR 文字识别]\nimage text"
    assert extractor.extract(tmp_path / "scan.pdf", ctx) == "[OCR 文字识别]\npdf text"
    assert calls == [("image", "scan.png"), ("pdf", "scan.pdf")]


def test_shipped_asr_plugin_wraps_media_extractor(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "semdex.extractors.media.MediaExtractor.extract",
        lambda _self, path, ctx: f"{ctx.marker}:{path.name}",
    )
    PythonFunctionExtractor._module_cache.clear()
    extractor = PythonFunctionExtractor(SHIPPED_PLUGIN_DIR / "asr")

    assert extractor.extract(tmp_path / "voice.mp3", SimpleNamespace(marker="asr")) == "asr:voice.mp3"
