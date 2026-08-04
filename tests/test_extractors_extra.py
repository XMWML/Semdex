from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

import pytest
from pypdf import PdfWriter

from semdex.config import Config, ExtractorRule, ModelCfg, OcrCfg, ScriptRule
from semdex.extractors import ExtractContext, resolve
from semdex.extractors.archive import ZipExtractor
from semdex.extractors.image import ImageExtractor
from semdex.extractors.mail import EmlExtractor
from semdex.extractors.pdf import PdfExtractor
from semdex.extractors.script import PythonFunctionExtractor
from semdex.indexer import index_pending
from semdex.modelclient import ModelClient
from semdex.models import ExtractError
from semdex.scanner import scan
from semdex.db import Database


def _ctx(tmp_path: Path, *, ocr: OcrCfg | None = None) -> ExtractContext:
    cfg = Config(db_path=tmp_path / "index.db", ocr=ocr or OcrCfg())
    return ExtractContext(cfg, ModelClient(cfg.vision, "vision"), ModelClient(cfg.llm, "llm"))


def test_eml_extractor_reads_headers_body_and_attachment_names(tmp_path: Path):
    path = tmp_path / "message.eml"
    path.write_bytes(
        b"Subject: Project Update\nFrom: alice@example.com\nTo: bob@example.com\n"
        b"MIME-Version: 1.0\nContent-Type: multipart/mixed; boundary=sep\n\n"
        b"--sep\nContent-Type: text/plain; charset=utf-8\n\nMeeting with Zhang San on Monday.\n"
        b"--sep\nContent-Type: text/plain\nContent-Disposition: attachment; filename=plan.txt\n\nattached\n--sep--\n"
    )
    text = EmlExtractor().extract(path, _ctx(tmp_path))
    assert "Project Update" in text
    assert "Meeting with Zhang San" in text
    assert "plan.txt" in text


def test_zip_extractor_recursively_extracts_text_members(tmp_path: Path):
    path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("folder/meeting.txt", "广州地铁项目会议记录")
        archive.writestr("ignored.bin", b"\x00\x01")
    text = ZipExtractor().extract(path, _ctx(tmp_path))
    assert "folder/meeting.txt" in text
    assert "广州地铁项目会议记录" in text


def test_zip_extractor_recursively_extracts_nested_archives(tmp_path: Path):
    inner = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner, "w") as archive:
        archive.writestr("nested/meeting.txt", "嵌套压缩包中的会议记录")

    outer = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer, "w") as archive:
        archive.writestr("exports/inner.zip", inner.read_bytes())

    text = ZipExtractor().extract(outer, _ctx(tmp_path))

    assert "exports/inner.zip" in text
    assert "nested/meeting.txt" in text
    assert "嵌套压缩包中的会议记录" in text


def test_zip_extractor_keeps_other_members_after_extract_error(tmp_path: Path):
    path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("notes.txt", "仍然应该被索引的内容")
        archive.writestr("broken.pdf", b"not-a-pdf")

    text = ZipExtractor().extract(path, _ctx(tmp_path))

    assert "仍然应该被索引的内容" in text
    assert "# broken.pdf" in text
    assert "提取失败" in text


@pytest.mark.parametrize("suffix", [".zip", ".cbz"])
def test_archive_waiting_ocr_plugin_member_is_retried(tmp_path: Path, monkeypatch, suffix: str):
    root = tmp_path / "files"
    root.mkdir()
    path = root / f"album{suffix}"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("cover.png", b"not-a-real-image")

    cfg = Config(db_path=tmp_path / "index.db", folders=[root])
    PythonFunctionExtractor._module_cache.clear()
    db = Database(cfg.db_path)
    try:
        scan(db, cfg, log=lambda *_: None)
        first = index_pending(db, cfg, log=lambda *_: None)
        assert first.indexed == 0
        assert first.waiting_capability == 1

        row = db.iter_files(["waiting_capability"])[0]
        assert row["filename"] == path.name
        assert db.get_content(int(row["id"])) is None

        cfg.ocr.enabled = True
        PythonFunctionExtractor._module_cache.clear()
        monkeypatch.setattr("semdex.ocr.ocr_image", lambda _path, _cfg: "封面上的文字")
        retry = index_pending(db, cfg, log=lambda *_: None)

        assert retry.indexed == 1
        row = db.get_file(int(row["id"]))
        assert row is not None and row["index_status"] == "done"
        assert "cover.png" in (db.get_content(int(row["id"])) or "")
        assert "封面上的文字" in (db.get_content(int(row["id"])) or "")
    finally:
        db.close()


def test_scanned_pdf_requires_explicit_ocr_plugin(tmp_path: Path):
    path = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with path.open("wb") as output:
        writer.write(output)
    with pytest.raises(ExtractError, match="OCR Python 插件"):
        PdfExtractor().extract(path, _ctx(tmp_path, ocr=OcrCfg(enabled=True)))


def test_short_pdf_text_layer_does_not_require_ocr(tmp_path: Path, monkeypatch):
    path = tmp_path / "short.pdf"
    path.write_bytes(b"placeholder")

    class FakePage:
        def extract_text(self):
            return "短文"

    class FakeReader:
        is_encrypted = False
        pages = [FakePage()]

        def __init__(self, _path):
            pass

    monkeypatch.setattr("pypdf.PdfReader", FakeReader)
    assert PdfExtractor().extract(path, _ctx(tmp_path)) == "短文"


def test_image_uses_ocr_without_a_vision_model(tmp_path: Path, monkeypatch):
    path = tmp_path / "image.png"
    path.write_bytes(b"not-a-real-image")
    monkeypatch.setattr("semdex.extractors.image.ocr_image", lambda _path, _cfg: "截图中的文字")
    text = ImageExtractor().extract(path, _ctx(tmp_path, ocr=OcrCfg(enabled=True)))
    assert "截图中的文字" in text


def test_audio_without_asr_waits_for_local_capability(tmp_path: Path):
    root = tmp_path / "files"
    root.mkdir()
    (root / "recording.mp3").write_bytes(b"not-an-audio-file")
    cfg = Config(db_path=tmp_path / "index.db", folders=[root])
    db = Database(cfg.db_path)
    scan(db, cfg, log=lambda *_: None)
    stats = index_pending(db, cfg, log=lambda *_: None)
    assert stats.waiting_capability == 1
    assert db.counts()["by_status"]["waiting_capability"] == 1
    db.close()


def test_custom_script_receives_a_safe_snapshot_with_the_original_filename(tmp_path: Path):
    root = tmp_path / "files"
    root.mkdir()
    source = root / "report.custom"
    source.write_text("script extractor source text", encoding="utf-8")
    script = tmp_path / "extract.py"
    script.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        "import sys\n"
        "path = Path(sys.argv[1])\n"
        "print(path.name)\n"
        "print(path.read_text(encoding='utf-8'))\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | 0o111)

    cfg = Config(
        db_path=tmp_path / "index.db",
        folders=[root],
        script_rules=[ScriptRule(match="*.custom", script=str(script))],
    )
    db = Database(cfg.db_path)
    scan(db, cfg, log=lambda *_: None)
    stats = index_pending(db, cfg, log=lambda *_: None)

    assert stats.indexed == 1
    file_id = int(db.get_file_by_path(str(source.resolve()))["id"])
    text = db.get_content(file_id)
    assert "report.custom" in text
    assert "script extractor source text" in text
    assert str(root) not in text
    db.close()


def test_python_function_rule_returns_a_value_and_builtin_rules_are_preserved(tmp_path: Path):
    root = tmp_path / "files"
    root.mkdir()
    source = root / "report.custom"
    source.write_text("function result", encoding="utf-8")
    extractor_dir = tmp_path / "extractors"
    extractor_dir.mkdir()
    (extractor_dir / "custom.py").write_text(
        "def extract(path):\n"
        "    return {'name': path.name, 'text': path.read_text(encoding='utf-8')}\n",
        encoding="utf-8",
    )
    cfg = Config(
        db_path=tmp_path / "index.db",
        folders=[root],
        extractor_dir=extractor_dir,
        extractor_rules=[ExtractorRule(
            id="custom", label="自定义", extensions=[".custom"], kind="python",
            script="custom.py", function="extract",
        )],
    )
    db = Database(cfg.db_path)
    scan(db, cfg, log=lambda *_: None)
    stats = index_pending(db, cfg, log=lambda *_: None)
    assert stats.indexed == 1
    file_id = int(db.get_file_by_path(str(source.resolve()))["id"])
    assert "report.custom" in (db.get_content(file_id) or "")
    assert "function result" in (db.get_content(file_id) or "")
    assert {rule.id for rule in cfg.extractor_rules} >= {"text", "pdf", "image", "custom"}
    db.close()


def test_explicit_extension_route_precedes_legacy_command_rule(tmp_path: Path):
    legacy_script = tmp_path / "legacy-extractor"
    legacy_script.write_text("#!/bin/sh\nprintf legacy\n", encoding="utf-8")
    legacy_script.chmod(legacy_script.stat().st_mode | 0o111)
    cfg = Config(
        db_path=tmp_path / "index.db",
        script_rules=[ScriptRule(match="*.pdf", script=str(legacy_script))],
        extractor_rules=[ExtractorRule(
            id="pdf", label="PDF 文档", extensions=[".pdf"], kind="llm", model="agent",
        )],
    )

    extractor = resolve(tmp_path / "report.pdf", cfg)

    assert extractor is not None
    assert extractor.name == "llm:agent"


def test_llm_extension_rule_uses_the_selected_model_and_keeps_readable_text(tmp_path: Path, monkeypatch):
    root = tmp_path / "files"
    root.mkdir()
    source = root / "plan.note"
    source.write_text("火星项目于 2026-08-03 启动，负责人是李雷。", encoding="utf-8")
    cfg = Config(
        db_path=tmp_path / "index.db",
        folders=[root],
        llm=ModelCfg(enabled=True, model="default-chat"),
        agent_model=ModelCfg(enabled=True, model="selected-agent"),
        extractor_rules=[ExtractorRule(
            id="notes", label="项目笔记", extensions=[".note"], kind="llm", model="agent",
        )],
    )
    selected: list[tuple[str, str]] = []

    class FakeClient:
        def __init__(self, model_cfg, kind):
            selected.append((kind, model_cfg.model))
            self.enabled = model_cfg.enabled
            self.kind = kind

        def chat(self, messages, **kwargs):
            assert self.kind == "agent"
            assert kwargs["max_tokens"] > 0
            assert "火星项目" in messages[-1]["content"]
            return "火星项目；李雷；2026-08-03"

    monkeypatch.setattr("semdex.indexer.ModelClient", FakeClient)
    db = Database(cfg.db_path)
    try:
        scan(db, cfg, log=lambda *_: None)
        assert index_pending(db, cfg, log=lambda *_: None).indexed == 1
        row = db.get_file_by_path(str(source.resolve()))
        assert row is not None and row["extractor"] == "llm:agent"
        text = db.get_content(int(row["id"])) or ""
        assert "火星项目于 2026-08-03" in text
        assert "火星项目；李雷；2026-08-03" in text
        assert ("agent", "selected-agent") in selected
    finally:
        db.close()


def test_llm_extension_rule_rejects_binary_snapshots(tmp_path: Path, monkeypatch):
    root = tmp_path / "files"
    root.mkdir()
    source = root / "opaque.note"
    source.write_bytes(b"\x00\x01not text")
    cfg = Config(
        db_path=tmp_path / "index.db",
        folders=[root],
        llm=ModelCfg(enabled=True, model="chat"),
        extractor_rules=[ExtractorRule(
            id="opaque", label="不透明文件", extensions=[".note"], kind="llm",
        )],
    )

    class FakeClient:
        def __init__(self, model_cfg, _kind):
            self.enabled = model_cfg.enabled

        def chat(self, *_args, **_kwargs):
            pytest.fail("二进制快照不应发送给 LLM")

    monkeypatch.setattr("semdex.indexer.ModelClient", FakeClient)
    db = Database(cfg.db_path)
    try:
        scan(db, cfg, log=lambda *_: None)
        assert index_pending(db, cfg, log=lambda *_: None).failed == 1
        row = db.get_file_by_path(str(source.resolve()))
        assert row is not None and row["index_status"] == "failed"
        assert "只支持可读文本" in str(row["error_msg"])
    finally:
        db.close()
