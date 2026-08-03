from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from semdex.config import Config, ModelCfg
from semdex.localmodels import LocalModelManager
from semdex.modelclient import ModelClient
from semdex.models import ModelNotConfigured


def test_local_model_manager_discovers_gguf_and_mlx_layouts(tmp_path: Path):
    gguf = tmp_path / "chat" / "qwen.gguf"
    gguf.parent.mkdir()
    gguf.write_bytes(b"GGUF")
    mlx = tmp_path / "mlx-community" / "Qwen3-1.7B"
    mlx.mkdir(parents=True)
    (mlx / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
    (mlx / "tokenizer.json").write_text("{}", encoding="utf-8")
    (mlx / "model.safetensors").write_bytes(b"weights")

    records = LocalModelManager(tmp_path).discover()

    assert [record.id for record in records] == ["chat/qwen.gguf", "mlx-community/Qwen3-1.7B"]
    assert records[0].format == "gguf"
    assert "chat" in records[0].capabilities
    assert records[1].format == "mlx"
    assert "chat" in records[1].capabilities


def test_local_model_manager_rejects_paths_outside_model_directory(tmp_path: Path):
    manager = LocalModelManager(tmp_path / "models")
    with pytest.raises(ModelNotConfigured):
        manager.load("../outside.gguf", "chat")
    with pytest.raises(ModelNotConfigured):
        manager.load(str(tmp_path / "outside.gguf"), "chat")


def test_local_model_manager_load_and_unload_without_native_runtime(tmp_path: Path, monkeypatch):
    model = tmp_path / "chat.gguf"
    model.write_bytes(b"GGUF")
    manager = LocalModelManager(tmp_path)

    class FakeLlama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def close(self):
            self.closed = True

    monkeypatch.setitem(__import__("sys").modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    loaded = manager.load("chat.gguf", "chat")
    assert loaded["loaded"] == ["chat"]
    unloaded = manager.unload("chat.gguf", "chat")
    assert unloaded["loaded"] == []


def test_model_client_uses_local_manager_for_chat(tmp_path: Path, monkeypatch):
    model = tmp_path / "chat.gguf"
    model.write_bytes(b"GGUF")
    calls: list[tuple[str, list[dict]]] = []

    class FakeManager:
        def chat(self, model_id, messages, **kwargs):
            calls.append((model_id, messages))
            return "本地回答"

    monkeypatch.setattr("semdex.modelclient.get_local_model_manager", lambda _path: FakeManager())
    cfg = Config(
        model_dir=tmp_path,
        llm=ModelCfg(enabled=True, mode="local", local_model="chat.gguf"),
    )
    assert ModelClient(cfg.llm, "llm").chat([{"role": "user", "content": "你好"}]) == "本地回答"
    assert calls[0][0] == "chat.gguf"
