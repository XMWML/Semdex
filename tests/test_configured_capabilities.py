from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from semdex import agent, entities, indexer, remote
from semdex.config import (
    AgentFallbackCfg,
    AsrCfg,
    CONFIG_TEMPLATE,
    Config,
    EntityCfg,
    ModelCfg,
    OcrCfg,
    load_config,
)
from semdex.db import Database
from semdex.extractors import ExtractContext
from semdex.extractors.media import MediaExtractor
from semdex.models import ExtractError
from semdex.ocr import ocr_image, ocr_pdf
from semdex.scanner import scan


class _Response:
    def __init__(self, payload: object):
        self.payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


def test_default_template_separates_primary_providers_from_downstream_models(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(CONFIG_TEMPLATE, encoding="utf-8")

    cfg = load_config(path)

    assert cfg.llm_providers and cfg.llm_providers[0].id == "default"
    assert cfg.agent_model is not None and cfg.agent_model is not cfg.llm
    assert cfg.entities_model is not None and cfg.entities_model is not cfg.llm
    assert cfg.embedding is not cfg.llm


def test_legacy_llm_config_is_copied_for_per_purpose_models(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[models.llm]
enabled = true
base_url = "http://legacy.test/v1/"
api_key = "legacy-key"
model = "legacy-model"

[models.agent]
enabled = true
base_url = "http://agent.test/v1"
api_key = "agent-key"
model = "agent-model"
""",
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg.agent_model is not None and cfg.agent_model.model == "agent-model"
    assert cfg.entities_model is not None and cfg.entities_model.model == "legacy-model"
    assert cfg.fallback_model is not None and cfg.fallback_model.model == "legacy-model"
    assert cfg.entities_model is not cfg.llm
    assert cfg.fallback_model is not cfg.llm
    cfg.llm.model = "changed-after-load"
    assert cfg.entities_model.model == "legacy-model"


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ('[ocr]\nenabled = true\nprovider = "local_http"\n', "endpoint"),
        ('[ocr]\nprovider = "unknown"\n', "provider"),
        ('[asr]\nprovider = "unknown"\n', "provider"),
    ],
)
def test_provider_config_is_validated(tmp_path: Path, contents: str, message: str):
    path = tmp_path / "config.toml"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_agent_and_entities_choose_their_dedicated_models(tmp_path: Path, monkeypatch):
    root = tmp_path / "files"
    root.mkdir()
    path = root / "meeting.txt"
    path.write_text("张三讨论项目", encoding="utf-8")
    cfg = Config(
        db_path=tmp_path / "index.db",
        agent_model=ModelCfg(enabled=True, model="agent-only"),
        entities_model=ModelCfg(enabled=True, model="entities-only"),
        entities=EntityCfg(enabled=True),
    )
    db = Database(cfg.db_path)
    file_id, _ = db.upsert_scan(
        str(path), path.name, path.suffix, path.stat().st_size, path.stat().st_mtime, path.name
    )
    db.save_content(file_id, path.read_text(encoding="utf-8"), path.name)
    db.set_status(file_id, "done")
    selected: list[tuple[str, str]] = []

    class AgentClient:
        def __init__(self, model_cfg, kind):
            selected.append((kind, model_cfg.model))
            self.turn = 0

        def chat_with_tools(self, *_args, **_kwargs):
            self.turn += 1
            if self.turn == 1:
                return (
                    {"role": "assistant", "content": None},
                    "",
                    [{"id": "search", "name": "search_fulltext", "arguments": '{"query":"张三"}'}],
                )
            return ({"role": "assistant", "content": "已找到。"}, "已找到。", [])

        def chat(self, *_args, **_kwargs):
            return "[]"

    monkeypatch.setattr(agent, "ModelClient", AgentClient)
    assert agent.ask(db, cfg, "张三在哪").answer == "已找到。"

    class EntityClient:
        def __init__(self, model_cfg, kind):
            selected.append((kind, model_cfg.model))

        def chat(self, *_args, **_kwargs):
            return "[]"

    monkeypatch.setattr(entities, "ModelClient", EntityClient)
    assert entities.index_entities(db, cfg, log=lambda *_: None).indexed == 1
    assert ("agent", "agent-only") in selected
    assert ("entities", "entities-only") in selected
    db.close()


def test_legacy_unknown_text_fallback_is_not_an_active_primary_route(tmp_path: Path, monkeypatch):
    root = tmp_path / "files"
    root.mkdir()
    (root / "note.unknowntext").write_text("未知格式中的项目记录", encoding="utf-8")
    cfg = Config(
        db_path=tmp_path / "index.db",
        folders=[root],
        fallback_model=ModelCfg(enabled=True, model="fallback-only"),
        agent_fallback=AgentFallbackCfg(enabled=True),
    )
    selected: list[tuple[str, str]] = []

    class FakeClient:
        def __init__(self, model_cfg, kind):
            selected.append((kind, model_cfg.model))
            self.enabled = model_cfg.enabled

        def chat(self, *_args, **_kwargs):
            return "摘要"

    monkeypatch.setattr(indexer, "ModelClient", FakeClient)
    db = Database(cfg.db_path)
    try:
        scan(db, cfg, log=lambda *_: None)
        stats = indexer.index_pending(db, cfg, log=lambda *_: None)
        assert stats.skipped == 1
        assert ("fallback", "fallback-only") not in selected
    finally:
        db.close()


def test_local_http_ocr_uploads_file_languages_and_reads_dot_path(tmp_path: Path, monkeypatch):
    image = tmp_path / "screen.png"
    image.write_bytes(b"image bytes")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _Response({"result": {"text": "截图中的文字"}})

    monkeypatch.setattr(remote, "urlopen", fake_urlopen)
    text = ocr_image(
        image,
        OcrCfg(
            enabled=True,
            provider="local_http",
            endpoint="http://127.0.0.1:8080/ocr",
            api_key="secret",
            languages="chi_sim",
            response_path="result.text",
            timeout_sec=12,
        ),
    )

    headers = {name.lower(): value for name, value in captured["request"].header_items()}
    assert text == "截图中的文字"
    assert captured["request"].full_url == "http://127.0.0.1:8080/ocr"
    assert captured["timeout"] == 12
    assert headers["authorization"] == "Bearer secret"
    assert b'name="file"' in captured["request"].data
    assert b'name="languages"' in captured["request"].data
    assert b"chi_sim" in captured["request"].data


def test_local_http_ocr_is_invoked_for_each_rendered_pdf_page(tmp_path: Path, monkeypatch):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"placeholder")
    uploads: list[bytes] = []

    def fake_find_command(_command, _label):
        return "renderer"

    def fake_run(args, _label, _timeout_sec=180):
        prefix = Path(args[-1])
        (prefix.parent / "page-1.png").write_bytes(b"page one")
        (prefix.parent / "page-2.png").write_bytes(b"page two")
        return ""

    def fake_urlopen(request, timeout):
        uploads.append(request.data)
        return _Response({"text": f"page {len(uploads)}"})

    import semdex.ocr as ocr_module

    monkeypatch.setattr(ocr_module, "_find_command", fake_find_command)
    monkeypatch.setattr(ocr_module, "_run", fake_run)
    monkeypatch.setattr(remote, "urlopen", fake_urlopen)

    text = ocr_pdf(
        pdf,
        OcrCfg(enabled=True, provider="local_http", endpoint="http://127.0.0.1:8080/ocr"),
    )

    assert text == "page 1\n\npage 2"
    assert len(uploads) == 2
    assert b"page one" in uploads[0]
    assert b"page two" in uploads[1]


def test_tesseract_uses_the_configured_timeout(tmp_path: Path, monkeypatch):
    image = tmp_path / "screen.png"
    image.write_bytes(b"image bytes")
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["timeout"] = kwargs["timeout"]
        return SimpleNamespace(returncode=0, stdout="截图文字", stderr="")

    monkeypatch.setattr("semdex.ocr._find_command", lambda *_args: "tesseract")
    monkeypatch.setattr("semdex.ocr.subprocess.run", fake_run)

    assert ocr_image(image, OcrCfg(enabled=True, timeout_sec=17)) == "截图文字"
    assert captured["timeout"] == 17
    assert captured["args"][-3:] == ["stdout", "-l", "eng+chi_sim"]


def test_openai_compatible_asr_uses_default_transcription_endpoint(tmp_path: Path, monkeypatch):
    audio = tmp_path / "recording.mp3"
    audio.write_bytes(b"audio bytes")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _Response({"text": "这是转写结果"})

    monkeypatch.setattr(remote, "urlopen", fake_urlopen)
    cfg = Config(
        asr=AsrCfg(
            enabled=True,
            provider="openai_compatible",
            base_url="http://127.0.0.1:1234/v1/",
            api_key="asr-key",
            model="",
            language="zh",
            timeout_sec=15,
        )
    )
    ctx = ExtractContext(cfg, vision=None, llm=None)  # type: ignore[arg-type]

    assert MediaExtractor().extract(audio, ctx) == "[音视频转写]\n这是转写结果"
    headers = {name.lower(): value for name, value in captured["request"].header_items()}
    assert captured["request"].full_url == "http://127.0.0.1:1234/v1/audio/transcriptions"
    assert captured["timeout"] == 15
    assert headers["authorization"] == "Bearer asr-key"
    assert b'name="file"' in captured["request"].data
    assert b'name="language"' in captured["request"].data
    assert b"zh" in captured["request"].data
    assert b'name="model"' not in captured["request"].data


def test_openai_compatible_asr_requires_a_text_response(tmp_path: Path, monkeypatch):
    audio = tmp_path / "recording.mp3"
    audio.write_bytes(b"audio bytes")
    monkeypatch.setattr(remote, "urlopen", lambda *_args, **_kwargs: _Response({"result": "missing"}))
    cfg = Config(asr=AsrCfg(enabled=True, provider="openai_compatible"))
    ctx = ExtractContext(cfg, vision=None, llm=None)  # type: ignore[arg-type]

    with pytest.raises(ExtractError, match="text"):
        MediaExtractor().extract(audio, ctx)


def test_openai_compatible_asr_reads_a_configured_response_path(tmp_path: Path, monkeypatch):
    audio = tmp_path / "recording.mp3"
    audio.write_bytes(b"audio bytes")
    monkeypatch.setattr(remote, "urlopen", lambda *_args, **_kwargs: _Response({"data": {"text": "嵌套转写"}}))
    cfg = Config(
        asr=AsrCfg(
            enabled=True,
            provider="openai_compatible",
            response_path="data.text",
        )
    )
    ctx = ExtractContext(cfg, vision=None, llm=None)  # type: ignore[arg-type]

    assert MediaExtractor().extract(audio, ctx) == "[音视频转写]\n嵌套转写"
