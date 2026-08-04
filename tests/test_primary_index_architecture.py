from __future__ import annotations

from pathlib import Path

import pytest

from semdex.config import Config, ExtractorRule, LlmProvider, load_config, write_default_config
from semdex.db import Database
from semdex.indexer import index_pending
from semdex.scanner import scan
from semdex.settings import save_settings, settings_dict


def test_defaults_seed_folder_plugins_and_use_three_primary_rule_kinds(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    write_default_config(config_path)

    config = load_config(config_path)
    payload = settings_dict(config)
    rules = {rule["id"]: rule for rule in payload["extractors"]["rules"]}
    plugins = {plugin["id"]: plugin for plugin in payload["extractors"]["plugins"]}

    assert rules["text"]["kind"] == "text"
    assert rules["pdf"]["kind"] == "text"
    assert (rules["image"]["kind"], rules["image"]["plugin"]) == ("python", "ocr")
    assert (rules["asr"]["kind"], rules["asr"]["plugin"]) == ("python", "asr")
    assert {rule["kind"] for rule in rules.values()} <= {"text", "llm", "python"}
    assert payload["extractors"]["plugin_dir"] == str(tmp_path / "extractors")
    for plugin_id in ("ocr", "asr"):
        assert plugins[plugin_id]["builtin"] is True
        assert plugins[plugin_id]["available"] is True
        assert set(plugins[plugin_id]) >= {
            "id", "name", "description", "function", "builtin", "path", "available",
        }
        assert (tmp_path / "extractors" / plugin_id / "plugin.py").is_file()


def test_enabled_primary_rules_reject_duplicate_extensions():
    with pytest.raises(ValueError, match=r"扩展名 \.pdf.*PDF 文档.*重复 PDF"):
        Config(extractor_rules=[
            ExtractorRule(
                id="duplicate-pdf",
                label="重复 PDF",
                extensions=[".pdf"],
                kind="text",
            ),
        ])

    config = Config(extractor_rules=[
        ExtractorRule(
            id="disabled-pdf",
            label="停用 PDF",
            extensions=[".pdf"],
            kind="text",
            enabled=False,
        ),
    ])
    assert any(rule.id == "disabled-pdf" for rule in config.extractor_rules)


def test_raw_image_input_rejects_non_image_extensions():
    provider = LlmProvider(id="vision", name="视觉模型")
    with pytest.raises(ValueError, match="原始图片输入不支持 \\.pdf"):
        Config(
            llm_providers=[provider],
            extractor_rules=[ExtractorRule(
                id="custom-pdf",
                label="PDF 图片输入",
                extensions=[".pdf"],
                kind="llm",
                provider="vision",
                input_mode="image",
            )],
        )


def test_llm_provider_crud_preserves_secrets_and_rejects_dangling_rules(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    write_default_config(config_path)
    current = load_config(config_path)

    saved = save_settings(current, {
        "llm_providers": [
            {"id": "default", "name": "主模型"},
            {
                "id": "cloud-vision",
                "name": "云端多模态",
                "enabled": True,
                "mode": "openai",
                "base_url": "https://example.invalid/v1",
                "api_key": "provider-secret",
                "model": "vision-chat",
            },
        ],
        "extractors": {"rules": [
            {"id": "image", "enabled": False},
            {
                "id": "screenshots",
                "label": "产品截图",
                "kind": "llm",
                "enabled": True,
                "extensions": [".png"],
                "provider": "cloud-vision",
                "input_mode": "image",
                "prompt": "提取界面中的字段和值",
            },
        ]},
    })

    providers = {provider.id: provider for provider in saved.llm_providers or []}
    assert providers["default"].name == "主模型"
    assert providers["default"].api_key == "lm-studio"
    assert providers["cloud-vision"].api_key == "provider-secret"
    api_provider = next(
        item for item in settings_dict(saved)["llm_providers"] if item["id"] == "cloud-vision"
    )
    assert "api_key" not in api_provider
    assert api_provider["api_key_configured"] is True

    cleared = save_settings(saved, {"llm_providers": [
        {"id": "default", "name": "主模型"},
        {"id": "cloud-vision", "name": "云端多模态", "clear_api_key": True},
    ]})
    assert next(
        provider for provider in cleared.llm_providers or [] if provider.id == "cloud-vision"
    ).api_key == ""

    with pytest.raises(ValueError, match="不存在的 LLM 供应商"):
        save_settings(cleared, {"llm_providers": [{"id": "default", "name": "主模型"}]})

    without_custom = save_settings(cleared, {
        "llm_providers": [{"id": "default", "name": "主模型"}],
        "extractors": {"rules": []},
    })
    empty = save_settings(without_custom, {"llm_providers": []})
    assert empty.llm_providers == []
    assert load_config(config_path).llm_providers == []


def test_legacy_models_and_rule_aliases_migrate_to_providers(tmp_path: Path):
    config_path = tmp_path / "legacy.toml"
    config_path.write_text(
        """
[models.llm]
enabled = true
base_url = "http://legacy.test/v1"
api_key = "legacy-key"
model = "legacy-chat"

[models.fallback]
enabled = true
base_url = "http://fallback.test/v1"
api_key = "fallback-key"
model = "fallback-chat"

[models.vision]
enabled = true
base_url = "http://vision.test/v1"
api_key = "vision-key"
model = "old-vision"

[[extractors.rules]]
id = "notes"
label = "旧笔记"
kind = "llm"
extensions = [".note"]
model = "fallback"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)
    providers = {provider.id: provider for provider in config.llm_providers or []}
    rule = next(rule for rule in config.extractor_rules if rule.id == "notes")

    assert providers["default"].model == "legacy-chat"
    assert providers["legacy-fallback"].model == "fallback-chat"
    assert rule.provider == "legacy-fallback"
    assert config.vision.model == "old-vision"  # tolerated, but no active route uses it
    assert (tmp_path / "extractors" / "ocr" / "plugin.py").is_file()


def test_llm_rules_use_selected_provider_prompt_and_input_mode(tmp_path: Path, monkeypatch):
    calls: list[tuple[str, str, str]] = []

    class FakeClient:
        def __init__(self, model_cfg, kind):
            self.enabled = model_cfg.enabled
            self.kind = kind
            self.model = model_cfg.model

        def chat(self, messages, **_kwargs):
            calls.append(("text", self.model, messages[-1]["content"]))
            return "文本模型整理结果"

        def describe_image(self, path, prompt):
            calls.append(("image", self.model, f"{prompt}|{path.suffix}"))
            return "图片模型整理结果"

    monkeypatch.setattr("semdex.indexer.ModelClient", FakeClient)
    provider = LlmProvider(
        id="chosen", name="选择的模型", enabled=True, mode="openai",
        base_url="http://model.test/v1", api_key="x", model="chosen-model",
    )

    text_root = tmp_path / "text-files"
    text_root.mkdir()
    text_path = text_root / "plan.note"
    text_path.write_text("一级索引原文", encoding="utf-8")
    text_config = Config(
        db_path=tmp_path / "text.db",
        folders=[text_root],
        llm_providers=[provider],
        extractor_rules=[ExtractorRule(
            id="notes", label="笔记", extensions=[".note"], kind="llm",
            provider="chosen", input_mode="text", prompt="保留里程碑",
        )],
    )
    text_db = Database(text_config.db_path)
    try:
        scan(text_db, text_config, log=lambda *_: None)
        assert index_pending(text_db, text_config, log=lambda *_: None).indexed == 1
        row = text_db.get_file_by_path(str(text_path.resolve()))
        content = text_db.get_content(int(row["id"])) if row is not None else ""
        assert "一级索引原文" in (content or "")
        assert "文本模型整理结果" in (content or "")
    finally:
        text_db.close()

    image_root = tmp_path / "image-files"
    image_root.mkdir()
    image_path = image_root / "screen.png"
    image_path.write_bytes(b"fake image")
    image_config = Config(
        db_path=tmp_path / "image.db",
        folders=[image_root],
        llm_providers=[provider],
        extractor_rules=[ExtractorRule(
            id="image", label="图片", extensions=[".png"], kind="llm",
            provider="chosen", input_mode="image", prompt="读取表格字段",
        )],
    )
    image_db = Database(image_config.db_path)
    try:
        scan(image_db, image_config, log=lambda *_: None)
        assert index_pending(image_db, image_config, log=lambda *_: None).indexed == 1
    finally:
        image_db.close()

    assert ("text", "chosen-model") in {(kind, model) for kind, model, _ in calls}
    assert any(kind == "text" and "保留里程碑" in prompt for kind, _, prompt in calls)
    assert any(kind == "image" and "读取表格字段" in prompt for kind, _, prompt in calls)


def test_large_llm_primary_index_keeps_model_output_before_text_limit(tmp_path: Path, monkeypatch):
    class FakeClient:
        def __init__(self, model_cfg, kind):
            self.enabled = model_cfg.enabled

        def chat(self, *_args, **_kwargs):
            return "必须保留的模型结果"

    monkeypatch.setattr("semdex.indexer.ModelClient", FakeClient)
    root = tmp_path / "files"
    root.mkdir()
    path = root / "large.txt"
    path.write_text("原" * 1_000_100, encoding="utf-8")
    provider = LlmProvider(
        id="chosen", name="选择的模型", enabled=True, mode="openai",
        base_url="http://model.test/v1", api_key="x", model="chosen-model",
    )
    config = Config(
        db_path=tmp_path / "index.db",
        folders=[root],
        llm_providers=[provider],
        extractor_rules=[ExtractorRule(
            id="text", label="文本与代码", extensions=[".txt"], kind="llm",
            provider="chosen", input_mode="text", prompt="整理正文",
        )],
    )
    db = Database(config.db_path)
    try:
        scan(db, config, log=lambda *_: None)
        assert index_pending(db, config, log=lambda *_: None).indexed == 1
        row = db.get_file_by_path(str(path.resolve()))
        content = db.get_content(int(row["id"])) if row is not None else ""
        assert (content or "").startswith("[LLM 一级索引")
        assert "必须保留的模型结果" in (content or "")
        assert len(content or "") == 1_000_000
    finally:
        db.close()
