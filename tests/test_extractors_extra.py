from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

import pytest
from pypdf import PdfWriter

from semdex.config import Config, OcrCfg, ScriptRule
from semdex.extractors import ExtractContext
from semdex.extractors.archive import ZipExtractor
from semdex.extractors.image import ImageExtractor
from semdex.extractors.mail import EmlExtractor
from semdex.extractors.pdf import PdfExtractor
from semdex.indexer import index_pending
from semdex.modelclient import ModelClient
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
def test_archive_waiting_model_member_is_retried(tmp_path: Path, monkeypatch, suffix: str):
    root = tmp_path / "files"
    root.mkdir()
    path = root / f"album{suffix}"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("cover.png", b"not-a-real-image")

    cfg = Config(db_path=tmp_path / "index.db", folders=[root])
    db = Database(cfg.db_path)
    try:
        scan(db, cfg, log=lambda *_: None)
        first = index_pending(db, cfg, log=lambda *_: None)
        assert first.indexed == 0
        assert first.waiting_model == 1

        row = db.iter_files(["waiting_model"])[0]
        assert row["filename"] == path.name
        assert db.get_content(int(row["id"])) is None

        cfg.vision.enabled = True
        monkeypatch.setattr(ModelClient, "describe_image", lambda _self, _path: "封面上的文字")
        retry = index_pending(db, cfg, log=lambda *_: None)

        assert retry.indexed == 1
        row = db.get_file(int(row["id"]))
        assert row is not None and row["index_status"] == "done"
        assert "cover.png" in (db.get_content(int(row["id"])) or "")
        assert "封面上的文字" in (db.get_content(int(row["id"])) or "")
    finally:
        db.close()


def test_scanned_pdf_uses_ocr_fallback(tmp_path: Path, monkeypatch):
    path = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with path.open("wb") as output:
        writer.write(output)
    monkeypatch.setattr("semdex.extractors.pdf.ocr_pdf", lambda _path, _cfg: "扫描件识别文字")
    text = PdfExtractor().extract(path, _ctx(tmp_path, ocr=OcrCfg(enabled=True)))
    assert "扫描件识别文字" in text


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
    monkeypatch.setattr(
        "semdex.extractors.pdf.ocr_pdf",
        lambda *_args: pytest.fail("短文本层不应触发 OCR"),
    )

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
