"""Project-local model discovery and lifecycle management.

The core never starts a model HTTP server for files placed in ``model_dir``.
Instead it discovers supported layouts and keeps explicitly loaded runtime
objects in this process.  Optional runtimes are imported only when a user
loads a matching model, so a normal OpenAI-compatible installation remains
lightweight on macOS and Linux.
"""
from __future__ import annotations

import gc
import importlib.util
import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from .models import ModelNotConfigured, ModelUnavailable


CAPABILITIES = ("chat", "embedding", "vision", "asr")


@dataclass(frozen=True)
class LocalModelRecord:
    """A model file or directory below a configured project model directory."""

    id: str
    name: str
    path: Path
    format: str
    size_bytes: int
    capabilities: tuple[str, ...]


@dataclass
class _LoadedModel:
    backend: str
    value: Any
    release: Callable[[], None] | None = None


def _module_available(name: str) -> bool:
    """Check an optional package without importing native extensions."""
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _safe_size(path: Path) -> int:
    """Return a best-effort file-tree size without following symlinked dirs."""
    try:
        if path.is_file():
            return path.stat().st_size
    except OSError:
        return 0

    total = 0
    try:
        for current, dirs, files in os.walk(path, followlinks=False):
            dirs[:] = [name for name in dirs if not Path(current, name).is_symlink()]
            for name in files:
                candidate = Path(current, name)
                try:
                    if candidate.is_file() and not candidate.is_symlink():
                        total += candidate.stat().st_size
                except OSError:
                    continue
    except OSError:
        return total
    return total


def _read_model_config(path: Path) -> dict[str, Any]:
    config = path / "config.json"
    try:
        if not config.is_file() or config.stat().st_size > 2_000_000:
            return {}
        parsed = json.loads(config.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def _model_description(path: Path, config: dict[str, Any]) -> str:
    parts = [path.name]
    for key in ("model_type", "architectures", "_name_or_path"):
        value = config.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value:
            parts.append(str(value))
    return " ".join(parts).casefold()


class LocalModelManager:
    """Discover, load and release models that live beneath one ``model_dir``.

    ``model_id`` is always a slash-separated relative path.  Keeping IDs
    relative gives copied projects stable configuration while rejecting paths
    outside the model directory, including symlink escapes.
    """

    def __init__(self, model_dir: str | os.PathLike[str]):
        self.model_dir = Path(model_dir).expanduser().resolve(strict=False)
        self._loaded: dict[tuple[str, str, str], _LoadedModel] = {}
        self._lock = threading.RLock()

    def _root(self) -> Path:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        return self.model_dir.resolve(strict=False)

    def _id_for(self, path: Path, root: Path) -> str:
        return path.relative_to(root).as_posix()

    def _directory_record(self, path: Path, root: Path) -> LocalModelRecord | None:
        try:
            names = {child.name for child in path.iterdir() if not child.is_symlink()}
        except OSError:
            return None

        config = _read_model_config(path)
        # faster-whisper uses a CTranslate2 model.bin directory.  Check this
        # before generic config.json layouts, which it also contains.
        if "model.bin" in names and "config.json" in names:
            return LocalModelRecord(
                id=self._id_for(path, root),
                name=path.name,
                path=path,
                format="faster-whisper",
                size_bytes=_safe_size(path),
                capabilities=("asr",),
            )

        has_weights = any(
            name.endswith(".safetensors") or name.startswith("consolidated.")
            for name in names
        )
        if not ("config.json" in names and has_weights):
            return None

        description = _model_description(path, config)
        if "whisper" in description:
            capabilities = ("asr",)
        else:
            capabilities_list = ["chat"]
            if any(token in description for token in ("vision", "vl", "llava", "pixtral", "qwen2_5_vl")):
                capabilities_list.append("vision")
            if any(token in description for token in ("embedding", "bge", "e5", "nomic", "gte", "jina")):
                capabilities_list.append("embedding")
            capabilities = tuple(capabilities_list)
        return LocalModelRecord(
            id=self._id_for(path, root),
            name=path.name,
            path=path,
            format="mlx",
            size_bytes=_safe_size(path),
            capabilities=capabilities,
        )

    def discover(self) -> list[LocalModelRecord]:
        """Return supported files/layouts, sorted by their stable relative ID."""
        root = self._root()
        records: list[LocalModelRecord] = []
        try:
            walker = os.walk(root, followlinks=False)
            for current, dirs, files in walker:
                current_path = Path(current)
                dirs[:] = sorted(
                    [name for name in dirs if not (current_path / name).is_symlink()],
                    key=str.casefold,
                )
                directory_record = self._directory_record(current_path, root)
                if directory_record is not None:
                    records.append(directory_record)
                    # A recognized model directory is a single selectable model,
                    # not a container whose weight shards should be rediscovered.
                    dirs[:] = []
                    continue

                for name in sorted(files, key=str.casefold):
                    candidate = current_path / name
                    try:
                        if candidate.is_symlink() or not candidate.is_file():
                            continue
                    except OSError:
                        continue
                    suffix = candidate.suffix.casefold()
                    lowered = candidate.name.casefold()
                    if suffix == ".gguf":
                        caps = ("chat", "embedding")
                        if "whisper" in lowered:
                            caps += ("asr",)
                        records.append(LocalModelRecord(
                            id=self._id_for(candidate, root),
                            name=candidate.name,
                            path=candidate,
                            format="gguf",
                            size_bytes=_safe_size(candidate),
                            capabilities=caps,
                        ))
                    elif suffix == ".bin" and (lowered.startswith("ggml-") or "whisper" in lowered):
                        records.append(LocalModelRecord(
                            id=self._id_for(candidate, root),
                            name=candidate.name,
                            path=candidate,
                            format="whisper-ggml",
                            size_bytes=_safe_size(candidate),
                            capabilities=("asr",),
                        ))
        except OSError:
            pass
        return sorted(records, key=lambda record: record.id.casefold())

    def _record(self, model_id: str) -> LocalModelRecord:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ModelNotConfigured("请先在项目 models 目录中选择一个本地模型")
        try:
            parts = PurePosixPath(model_id.strip()).parts
        except TypeError as exc:
            raise ModelNotConfigured("本地模型标识无效") from exc
        if not parts or PurePosixPath(model_id).is_absolute() or any(part in {"", ".", ".."} for part in parts):
            raise ModelNotConfigured("本地模型必须是 models 目录内的相对路径")

        root = self._root()
        candidate = (root.joinpath(*parts)).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ModelNotConfigured("本地模型必须位于项目 models 目录内") from exc

        for record in self.discover():
            if record.id == PurePosixPath(*parts).as_posix() and record.path.resolve(strict=False) == candidate:
                return record
        raise ModelNotConfigured(f"在项目 models 目录中找不到本地模型: {model_id}")

    @staticmethod
    def runtimes() -> list[dict[str, Any]]:
        """Describe optional local runtimes without importing them eagerly."""
        is_macos = sys.platform == "darwin"
        return [
            {
                "id": "llama_cpp",
                "available": _module_available("llama_cpp"),
                "detail": "GGUF 文本生成和向量化（llama-cpp-python）",
            },
            {
                "id": "faster_whisper",
                "available": _module_available("faster_whisper"),
                "detail": "本地 CTranslate2 Whisper 转写（faster-whisper）",
            },
            {
                "id": "mlx_lm",
                "available": is_macos and _module_available("mlx_lm"),
                "detail": "MLX 文本模型（仅 macOS）",
            },
            {
                "id": "mlx_embeddings",
                "available": is_macos and _module_available("mlx_embeddings"),
                "detail": "MLX 向量模型（仅 macOS）",
            },
            {
                "id": "mlx_vlm",
                "available": is_macos and _module_available("mlx_vlm"),
                "detail": "MLX 视觉语言模型（仅 macOS）",
            },
            {
                "id": "mlx_whisper",
                "available": is_macos and _module_available("mlx_whisper"),
                "detail": "MLX Whisper 转写（仅 macOS）",
            },
            {
                "id": "whisper_cpp",
                "available": _module_available("pywhispercpp"),
                "detail": "GGML/GGUF Whisper 转写（pywhispercpp，可选）",
            },
        ]

    def _runtime_available(self, runtime: str) -> bool:
        return any(item["id"] == runtime and item["available"] for item in self.runtimes())

    def _default_backend(self, record: LocalModelRecord, capability: str, requested: str = "auto") -> str:
        requested = (requested or "auto").strip().lower()
        if requested != "auto":
            return requested
        if capability == "asr":
            if record.format == "faster-whisper":
                return "faster_whisper"
            if record.format == "mlx":
                return "mlx_whisper"
            if record.format in {"whisper-ggml", "gguf"}:
                return "whisper_cpp"
        if record.format == "gguf":
            return "llama_cpp"
        if record.format == "mlx":
            if capability == "vision":
                return "mlx_vlm"
            if capability == "embedding":
                return "mlx_embeddings"
            return "mlx_lm"
        raise ModelUnavailable(f"本地模型 {record.name} 不支持 {capability} 加载")

    def _key(self, model_id: str, capability: str, backend: str) -> tuple[str, str, str]:
        return (model_id, capability, backend)

    def _load_gguf(self, record: LocalModelRecord, capability: str) -> _LoadedModel:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ModelUnavailable(
                "未安装 GGUF 运行时。执行 `uv sync --extra gguf` 后重新加载模型"
            ) from exc
        kwargs: dict[str, Any] = {"model_path": str(record.path), "verbose": False}
        if capability == "embedding":
            kwargs["embedding"] = True
        try:
            return _LoadedModel("llama_cpp", Llama(**kwargs))
        except Exception as exc:
            raise ModelUnavailable(f"无法加载 GGUF 模型 {record.name}: {exc}") from exc

    def _load_mlx_lm(self, record: LocalModelRecord) -> _LoadedModel:
        if sys.platform != "darwin":
            raise ModelUnavailable("MLX 仅支持 macOS；Linux 请为该用途选择 GGUF 或 OpenAI API")
        try:
            from mlx_lm import load
        except ImportError as exc:
            raise ModelUnavailable(
                "未安装 MLX 文本运行时。执行 `uv sync --extra mlx` 后重新加载模型"
            ) from exc
        try:
            return _LoadedModel("mlx_lm", load(str(record.path)))
        except Exception as exc:
            raise ModelUnavailable(f"无法加载 MLX 模型 {record.name}: {exc}") from exc

    def _load_mlx_embeddings(self, record: LocalModelRecord) -> _LoadedModel:
        if sys.platform != "darwin":
            raise ModelUnavailable("MLX 仅支持 macOS；Linux 请为该用途选择 GGUF 或 OpenAI API")
        try:
            from mlx_embeddings import load
        except ImportError as exc:
            raise ModelUnavailable(
                "该 MLX 模型需要 mlx-embeddings。安装对应运行时，或为向量选择 GGUF/OpenAI API"
            ) from exc
        try:
            return _LoadedModel("mlx_embeddings", load(str(record.path)))
        except Exception as exc:
            raise ModelUnavailable(f"无法加载 MLX 向量模型 {record.name}: {exc}") from exc

    def _load_mlx_vlm(self, record: LocalModelRecord) -> _LoadedModel:
        if sys.platform != "darwin":
            raise ModelUnavailable("MLX 仅支持 macOS；Linux 请为视觉选择 OpenAI API")
        try:
            from mlx_vlm import load
        except ImportError as exc:
            raise ModelUnavailable(
                "未安装 MLX 视觉运行时。执行 `uv sync --extra mlx` 后重新加载模型"
            ) from exc
        try:
            return _LoadedModel("mlx_vlm", load(str(record.path)))
        except Exception as exc:
            raise ModelUnavailable(f"无法加载 MLX 视觉模型 {record.name}: {exc}") from exc

    def _load_asr(
        self,
        record: LocalModelRecord,
        backend: str,
        *,
        device: str,
        compute_type: str,
    ) -> _LoadedModel:
        if backend == "faster_whisper":
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise ModelUnavailable(
                    "未安装 faster-whisper。执行 `uv sync --extra asr` 后重新加载模型"
                ) from exc
            try:
                return _LoadedModel(
                    "faster_whisper",
                    WhisperModel(str(record.path), device=device, compute_type=compute_type),
                )
            except Exception as exc:
                raise ModelUnavailable(f"无法加载本地 Whisper 模型 {record.name}: {exc}") from exc
        if backend == "mlx_whisper":
            if sys.platform != "darwin":
                raise ModelUnavailable("MLX Whisper 仅支持 macOS")
            try:
                import mlx_whisper
            except ImportError as exc:
                raise ModelUnavailable(
                    "未安装 MLX Whisper。执行 `uv sync --extra mlx` 后重新加载模型"
                ) from exc
            # mlx-whisper versions differ: some expose a model loader while
            # others load inside transcribe().  Keep either object cached.
            try:
                loader = getattr(mlx_whisper, "load_model", None)
                value = loader(str(record.path)) if callable(loader) else mlx_whisper
                return _LoadedModel("mlx_whisper", value)
            except Exception as exc:
                raise ModelUnavailable(f"无法加载 MLX Whisper 模型 {record.name}: {exc}") from exc
        if backend == "whisper_cpp":
            try:
                from pywhispercpp.model import Model
            except ImportError as exc:
                raise ModelUnavailable(
                    "未安装 pywhispercpp。请安装该可选运行时后重新加载 GGML/GGUF Whisper 模型"
                ) from exc
            try:
                return _LoadedModel("whisper_cpp", Model(str(record.path)))
            except Exception as exc:
                raise ModelUnavailable(f"无法加载本地 Whisper 模型 {record.name}: {exc}") from exc
        raise ModelUnavailable(f"不支持的本地 ASR 运行时: {backend}")

    def _load(
        self,
        record: LocalModelRecord,
        capability: str,
        backend: str,
        *,
        device: str = "auto",
        compute_type: str = "int8",
    ) -> _LoadedModel:
        key = self._key(record.id, capability, backend)
        loaded = self._loaded.get(key)
        if loaded is not None:
            return loaded

        if backend == "llama_cpp":
            loaded = self._load_gguf(record, capability)
        elif backend == "mlx_lm":
            loaded = self._load_mlx_lm(record)
        elif backend == "mlx_embeddings":
            loaded = self._load_mlx_embeddings(record)
        elif backend == "mlx_vlm":
            loaded = self._load_mlx_vlm(record)
        elif capability == "asr":
            loaded = self._load_asr(record, backend, device=device, compute_type=compute_type)
        else:
            raise ModelUnavailable(f"本地模型 {record.name} 不支持 {capability} 运行时 {backend}")
        self._loaded[key] = loaded
        return loaded

    def load(
        self,
        model_id: str,
        capability: str,
        *,
        backend: str = "auto",
        device: str = "auto",
        compute_type: str = "int8",
    ) -> dict[str, Any]:
        """Load a selectable model into this process and return its status."""
        capability = capability.strip().lower()
        if capability not in CAPABILITIES:
            raise ValueError(f"未知模型能力: {capability}")
        with self._lock:
            record = self._record(model_id)
            if capability not in record.capabilities:
                raise ModelUnavailable(f"本地模型 {record.name} 不支持 {capability}")
            selected_backend = self._default_backend(record, capability, backend)
            self._load(
                record,
                capability,
                selected_backend,
                device=device,
                compute_type=compute_type,
            )
            return self._record_dict(record)

    def _release(self, loaded: _LoadedModel) -> None:
        try:
            if loaded.release is not None:
                loaded.release()
            close = getattr(loaded.value, "close", None)
            if callable(close):
                close()
        finally:
            # Native backends can retain substantial buffers until collection.
            del loaded.value
            gc.collect()
            if loaded.backend.startswith("mlx"):
                try:
                    import mlx.core as mx

                    mx.metal.clear_cache()
                except Exception:
                    pass

    def unload(self, model_id: str, capability: str | None = None) -> dict[str, Any]:
        """Unload one capability or every cached capability for a model."""
        normalized = capability.strip().lower() if capability is not None else None
        if normalized is not None and normalized not in CAPABILITIES:
            raise ValueError(f"未知模型能力: {normalized}")
        with self._lock:
            record = self._record(model_id)
            keys = [
                key for key in self._loaded
                if key[0] == record.id and (normalized is None or key[1] == normalized)
            ]
            for key in keys:
                loaded = self._loaded.pop(key)
                self._release(loaded)
            return self._record_dict(record)

    def unload_all(self) -> None:
        """Release every local model managed by this process."""
        with self._lock:
            values = list(self._loaded.values())
            self._loaded.clear()
            for loaded in values:
                self._release(loaded)

    def _loadable(self, record: LocalModelRecord) -> tuple[bool, str]:
        reasons: list[str] = []
        for capability in record.capabilities:
            try:
                backend = self._default_backend(record, capability)
            except ModelUnavailable as exc:
                reasons.append(str(exc))
                continue
            runtime = {
                "llama_cpp": "llama_cpp",
                "faster_whisper": "faster_whisper",
                "mlx_lm": "mlx_lm",
                "mlx_embeddings": "mlx_embeddings",
                "mlx_vlm": "mlx_vlm",
                "mlx_whisper": "mlx_whisper",
                "whisper_cpp": "whisper_cpp",
            }.get(backend, backend)
            if self._runtime_available(runtime):
                return True, ""
            reasons.append(f"缺少 {runtime} 运行时")
        return False, reasons[0] if reasons else "没有可用运行时"

    def _record_dict(self, record: LocalModelRecord) -> dict[str, Any]:
        loaded_capabilities = sorted({key[1] for key in self._loaded if key[0] == record.id})
        loadable, reason = self._loadable(record)
        return {
            "id": record.id,
            "name": record.name,
            "format": record.format,
            "size_bytes": record.size_bytes,
            "capabilities": list(record.capabilities),
            "loaded": loaded_capabilities,
            "loadable": loadable,
            "reason": reason,
        }

    def catalog(self) -> dict[str, Any]:
        """Return the stable API-friendly local-model catalog and runtime state."""
        with self._lock:
            return {
                "model_dir": str(self._root()),
                "models": [self._record_dict(record) for record in self.discover()],
                "runtimes": self.runtimes(),
            }

    def status(self) -> dict[str, Any]:
        """Alias for callers that only need current discovery/load status."""
        return self.catalog()

    @staticmethod
    def _text_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            if isinstance(content, list):
                text = "\n".join(
                    str(item.get("text", "")) for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            elif content is None:
                text = ""
            else:
                text = str(content)
            normalized.append({"role": role, "content": text})
        return normalized

    @classmethod
    def _mlx_prompt(cls, loaded: _LoadedModel, messages: list[dict[str, Any]]) -> str:
        model, tokenizer = loaded.value
        normalized = cls._text_messages(messages)
        template = getattr(tokenizer, "apply_chat_template", None)
        if callable(template):
            try:
                return template(normalized, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        return "\n\n".join(f"{item['role']}: {item['content']}" for item in normalized) + "\nassistant:"

    def _mlx_generate(self, loaded: _LoadedModel, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        try:
            from mlx_lm import generate
        except ImportError as exc:
            raise ModelUnavailable("MLX 文本运行时在加载后不可用") from exc
        model, tokenizer = loaded.value
        prompt = self._mlx_prompt(loaded, messages)
        max_tokens = int(kwargs.get("max_tokens", 1_024))
        temperature = kwargs.get("temperature")
        try:
            result = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
                temp=temperature,
            )
        except TypeError:
            result = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        return str(result)

    @staticmethod
    def _choice_content(response: Any) -> str:
        try:
            choices = response["choices"] if isinstance(response, dict) else response.choices
            choice = choices[0]
            message = choice["message"] if isinstance(choice, dict) else choice.message
            content = message.get("content") if isinstance(message, dict) else message.content
            return str(content or "")
        except (AttributeError, IndexError, KeyError, TypeError):
            raise ModelUnavailable("本地模型返回了无法识别的聊天响应")

    def chat(self, model_id: str, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        with self._lock:
            record = self._record(model_id)
            if "chat" not in record.capabilities:
                raise ModelUnavailable(f"本地模型 {record.name} 不是文本对话模型")
            backend = self._default_backend(record, "chat")
            loaded = self._load(record, "chat", backend)
            try:
                if loaded.backend == "llama_cpp":
                    response = loaded.value.create_chat_completion(
                        messages=self._text_messages(messages), **kwargs
                    )
                    return self._choice_content(response)
                if loaded.backend == "mlx_lm":
                    return self._mlx_generate(loaded, messages, **kwargs)
            except ModelUnavailable:
                raise
            except Exception as exc:
                raise ModelUnavailable(f"本地模型 {record.name} 推理失败: {exc}") from exc
            raise ModelUnavailable(f"本地模型 {record.name} 没有可用的聊天运行时")

    @staticmethod
    def _tool_calls_from_message(message: Any) -> tuple[dict[str, Any], str, list[dict[str, str]]]:
        if not isinstance(message, dict):
            raise ModelUnavailable("本地模型工具调用响应无效")
        content = str(message.get("content") or "")
        calls: list[dict[str, str]] = []
        for raw in message.get("tool_calls") or []:
            function = raw.get("function", {}) if isinstance(raw, dict) else {}
            name = function.get("name", "") if isinstance(function, dict) else ""
            arguments = function.get("arguments", "{}") if isinstance(function, dict) else "{}"
            call_id = raw.get("id", "local-tool") if isinstance(raw, dict) else "local-tool"
            if name:
                calls.append({"id": str(call_id), "name": str(name), "arguments": str(arguments)})
        return message, content, calls

    def chat_with_tools(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[dict[str, Any], str, list[dict[str, str]]]:
        """Use native GGUF tool calls, with a JSON protocol fallback for MLX."""
        with self._lock:
            record = self._record(model_id)
            if "chat" not in record.capabilities:
                raise ModelUnavailable(f"本地模型 {record.name} 不是文本对话模型")
            backend = self._default_backend(record, "chat")
            loaded = self._load(record, "chat", backend)
            if loaded.backend == "llama_cpp":
                try:
                    response = loaded.value.create_chat_completion(
                        messages=self._text_messages(messages),
                        tools=tools,
                        tool_choice="auto",
                        **kwargs,
                    )
                    choices = response["choices"] if isinstance(response, dict) else response.choices
                    choice = choices[0]
                    message = choice["message"] if isinstance(choice, dict) else choice.message
                    if not isinstance(message, dict):
                        dump = getattr(message, "model_dump", None)
                        message = dump(exclude_none=True) if callable(dump) else {"role": "assistant", "content": getattr(message, "content", "")}
                    return self._tool_calls_from_message(message)
                except TypeError:
                    # Older llama-cpp builds have no native tool schema support.
                    pass
                except Exception as exc:
                    raise ModelUnavailable(f"本地模型 {record.name} 工具调用失败: {exc}") from exc

        protocol = (
            "你可以调用以下工具。需要调用时，只输出 JSON 对象，格式为 "
            "{\"tool_calls\":[{\"name\":\"工具名\",\"arguments\":{...}}]}。"
            "不需要工具时直接回答。工具定义：\n" + json.dumps(tools, ensure_ascii=False)
        )
        text = self.chat(model_id, [{"role": "system", "content": protocol}, *messages], **kwargs)
        candidate = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return {"role": "assistant", "content": text}, text, []
        raw_calls = payload.get("tool_calls") if isinstance(payload, dict) else None
        if not isinstance(raw_calls, list):
            return {"role": "assistant", "content": text}, text, []
        known = {
            str(item.get("function", {}).get("name"))
            for item in tools if isinstance(item, dict) and isinstance(item.get("function"), dict)
        }
        calls: list[dict[str, str]] = []
        for index, raw in enumerate(raw_calls):
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name", ""))
            if name not in known:
                continue
            arguments = raw.get("arguments", {})
            calls.append({
                "id": str(raw.get("id", f"local-{index}")),
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False) if isinstance(arguments, dict) else str(arguments),
            })
        return {"role": "assistant", "content": None if calls else text}, text, calls

    @staticmethod
    def _vectors(value: Any) -> list[list[float]]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, dict):
            value = value.get("data", value.get("embeddings", value))
        if not isinstance(value, list):
            raise ModelUnavailable("本地向量模型返回了无法识别的数据")
        vectors: list[list[float]] = []
        for item in value:
            if isinstance(item, dict):
                item = item.get("embedding")
            if hasattr(item, "tolist"):
                item = item.tolist()
            if not isinstance(item, (list, tuple)):
                raise ModelUnavailable("本地向量模型返回了无效向量")
            vectors.append([float(number) for number in item])
        return vectors

    def embed(self, model_id: str, texts: list[str]) -> list[list[float]]:
        with self._lock:
            record = self._record(model_id)
            if "embedding" not in record.capabilities:
                raise ModelUnavailable(f"本地模型 {record.name} 不支持向量化")
            backend = self._default_backend(record, "embedding")
            loaded = self._load(record, "embedding", backend)
            try:
                if loaded.backend == "llama_cpp":
                    embed = getattr(loaded.value, "embed", None)
                    if callable(embed):
                        return self._vectors(embed(texts))
                    create = getattr(loaded.value, "create_embedding", None)
                    if callable(create):
                        return self._vectors(create(texts))
                if loaded.backend == "mlx_embeddings":
                    encode = getattr(loaded.value, "encode", None) or getattr(loaded.value, "embed", None)
                    if callable(encode):
                        return self._vectors(encode(texts))
            except ModelUnavailable:
                raise
            except Exception as exc:
                raise ModelUnavailable(f"本地模型 {record.name} 向量化失败: {exc}") from exc
            raise ModelUnavailable(f"本地模型 {record.name} 的运行时不提供向量接口")

    def describe_image(self, model_id: str, path: Path, prompt: str) -> str:
        with self._lock:
            record = self._record(model_id)
            if "vision" not in record.capabilities:
                raise ModelUnavailable(f"本地模型 {record.name} 不是视觉语言模型")
            backend = self._default_backend(record, "vision")
            loaded = self._load(record, "vision", backend)
            if loaded.backend != "mlx_vlm":
                raise ModelUnavailable(f"本地模型 {record.name} 没有可用的视觉运行时")
            try:
                from mlx_vlm import generate

                model, processor = loaded.value
                try:
                    output = generate(
                        model, processor, prompt, image=str(path), max_tokens=1_024, verbose=False
                    )
                except TypeError:
                    output = generate(model, processor, prompt, str(path), max_tokens=1_024, verbose=False)
                return str(output).strip()
            except Exception as exc:
                raise ModelUnavailable(f"本地视觉模型 {record.name} 推理失败: {exc}") from exc

    @staticmethod
    def _asr_text(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            text = value.get("text")
            if isinstance(text, str):
                return text.strip()
        if isinstance(value, (list, tuple)):
            parts = []
            for item in value:
                text = getattr(item, "text", None)
                if text is None and isinstance(item, dict):
                    text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            return "\n".join(parts)
        return ""

    def transcribe(
        self,
        model_id: str,
        path: Path,
        *,
        backend: str = "auto",
        device: str = "auto",
        compute_type: str = "int8",
        language: str = "",
    ) -> tuple[str, str]:
        """Transcribe local audio and return ``(text, detected_language)``."""
        with self._lock:
            record = self._record(model_id)
            if "asr" not in record.capabilities:
                raise ModelUnavailable(f"本地模型 {record.name} 不是 Whisper/ASR 模型")
            selected_backend = self._default_backend(record, "asr", backend)
            loaded = self._load(
                record,
                "asr",
                selected_backend,
                device=device,
                compute_type=compute_type,
            )
            try:
                if loaded.backend == "faster_whisper":
                    options: dict[str, Any] = {"vad_filter": True}
                    if language:
                        options["language"] = language
                    segments, info = loaded.value.transcribe(str(path), **options)
                    return self._asr_text(list(segments)), str(getattr(info, "language", "") or "")
                if loaded.backend == "mlx_whisper":
                    transcribe = getattr(loaded.value, "transcribe", None)
                    if not callable(transcribe):
                        import mlx_whisper

                        transcribe = mlx_whisper.transcribe
                    kwargs: dict[str, Any] = {"path_or_hf_repo": str(record.path)}
                    if language:
                        kwargs["language"] = language
                    try:
                        result = transcribe(str(path), **kwargs)
                    except TypeError:
                        result = transcribe(str(path), model=str(record.path), **({"language": language} if language else {}))
                    return self._asr_text(result), language
                if loaded.backend == "whisper_cpp":
                    result = loaded.value.transcribe(str(path))
                    return self._asr_text(result), language
            except ModelUnavailable:
                raise
            except Exception as exc:
                raise ModelUnavailable(f"本地 Whisper 模型 {record.name} 转写失败: {exc}") from exc
            raise ModelUnavailable(f"本地 Whisper 模型 {record.name} 没有可用的转写运行时")


_MANAGERS: dict[str, LocalModelManager] = {}
_MANAGERS_LOCK = threading.RLock()


def get_local_model_manager(model_dir: str | os.PathLike[str]) -> LocalModelManager:
    """Return the process-wide manager for one project-local model directory."""
    root = str(Path(model_dir).expanduser().resolve(strict=False))
    with _MANAGERS_LOCK:
        manager = _MANAGERS.get(root)
        if manager is None:
            manager = LocalModelManager(root)
            _MANAGERS[root] = manager
        return manager
