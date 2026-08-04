"""Audio/video transcription through local or OpenAI-compatible ASR services."""
from __future__ import annotations

from pathlib import Path

from ..localmodels import get_local_model_manager
from ..models import (
    CapabilityNotConfigured,
    CapabilityUnavailable,
    ExtractError,
    ModelNotConfigured,
    ModelUnavailable,
)
from ..paths import ensure_private_directory
from ..remote import RemoteRequestError, RemoteResponseError, post_multipart_json
from .base import ExtractContext, Extractor


class MediaExtractor(Extractor):
    name = "asr"
    exts = (
        ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus",
        ".mp4", ".mov", ".mkv", ".webm", ".avi",
    )
    _models: dict[tuple[str, str, str, str], object] = {}

    def _model(self, ctx: ExtractContext):
        cfg = ctx.config.asr
        if not cfg.enabled:
            raise CapabilityNotConfigured(
                "ASR 未启用（请在设置中启用；本地 faster-whisper 还需退出 Semdex 后运行 "
                '`python3 "Start Semdex.py" --with-asr --sync-only`）'
            )
        if cfg.provider not in {"faster_whisper", "local"}:
            raise CapabilityNotConfigured(
                "当前 ASR provider 不是 faster_whisper，不能加载本地 Whisper 模型"
            )
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise CapabilityUnavailable(
                "未安装 faster-whisper。请退出 Semdex，在项目目录运行 "
                '`python3 "Start Semdex.py" --with-asr --sync-only`，重新启动后再运行索引'
            ) from e

        download_root = ensure_private_directory(ctx.config.model_dir / "whisper")
        # ``model`` is retained as the legacy faster-whisper identifier.  New
        # local file selections are handled by LocalModelManager in extract().
        legacy_model = cfg.model.strip() or cfg.local_model.strip()
        key = (legacy_model, cfg.device, cfg.compute_type, str(download_root))
        model = self._models.get(key)
        if model is None:
            try:
                model = WhisperModel(
                    legacy_model,
                    device=cfg.device,
                    compute_type=cfg.compute_type,
                    download_root=str(download_root),
                )
            except Exception as e:
                raise CapabilityUnavailable(f"无法加载本地 Whisper 模型 {cfg.model}: {e}") from e
            self._models[key] = model
        return model

    @staticmethod
    def _local_manager(ctx: ExtractContext):
        model_dir = ctx.config.asr.local_model_dir or ctx.config.model_dir
        return get_local_model_manager(model_dir)

    @staticmethod
    def _known_legacy_model(model_id: str) -> bool:
        # Existing configurations used names that faster-whisper downloads from
        # Hugging Face.  Keep those names working while requiring new models to
        # be files/directories discovered under project models/.
        return model_id.strip().lower() in {
            "tiny", "tiny.en", "base", "base.en", "small", "small.en",
            "medium", "medium.en", "large-v1", "large-v2", "large-v3",
            "large", "distil-large-v2", "distil-large-v3",
        }

    @staticmethod
    def _remote_endpoint(ctx: ExtractContext) -> str:
        cfg = ctx.config.asr
        endpoint = cfg.endpoint.strip()
        if endpoint:
            return endpoint
        base_url = cfg.base_url.rstrip("/")
        if not base_url:
            raise CapabilityNotConfigured(
                "ASR openai_compatible 需要配置 endpoint 或 base_url"
            )
        return base_url + "/audio/transcriptions"

    def _transcribe_openai_compatible(self, path: Path, ctx: ExtractContext) -> str:
        cfg = ctx.config.asr
        fields: dict[str, str] = {}
        if cfg.model.strip():
            fields["model"] = cfg.model.strip()
        if cfg.language.strip():
            fields["language"] = cfg.language.strip()
        try:
            payload = post_multipart_json(
                self._remote_endpoint(ctx),
                file_path=path,
                fields=fields,
                api_key=cfg.api_key,
                timeout_sec=cfg.timeout_sec,
                label="ASR 服务",
            )
        except RemoteRequestError as e:
            raise CapabilityUnavailable(str(e)) from e
        except RemoteResponseError as e:
            raise ExtractError(str(e)) from e
        return self._response_text(payload, cfg.response_path)

    @staticmethod
    def _response_text(payload: object, response_path: str) -> str:
        """Read a textual transcript from a standard or compatible response."""
        value = payload
        for part in response_path.split("."):
            if isinstance(value, dict):
                if part not in value:
                    raise ExtractError(f"ASR 响应中找不到 response_path `{response_path}`")
                value = value[part]
            elif isinstance(value, list):
                try:
                    value = value[int(part)]
                except (ValueError, IndexError) as e:
                    raise ExtractError(f"ASR 响应中找不到 response_path `{response_path}`") from e
            else:
                raise ExtractError(f"ASR 响应中找不到 response_path `{response_path}`")
        if not isinstance(value, str):
            raise ExtractError(f"ASR response_path `{response_path}` 的值不是文本")
        return value.strip()

    def extract(self, path: Path, ctx: ExtractContext) -> str:
        cfg = ctx.config.asr
        if not cfg.enabled:
            raise CapabilityNotConfigured(
                "ASR 未启用（在配置中设置 [asr] enabled = true）"
            )

        if cfg.provider in {"faster_whisper", "local"}:
            model_id = (cfg.local_model or cfg.model).strip()
            if not model_id:
                raise CapabilityNotConfigured(
                    "本地 ASR 未选择模型（请在设置中选择 models 目录内的 local_model）"
                )
            manager = self._local_manager(ctx)
            discovered_ids = {record.id for record in manager.discover()}
            if model_id in discovered_ids:
                try:
                    text, language = manager.transcribe(
                        model_id,
                        path,
                        backend=cfg.local_backend,
                        device=cfg.device,
                        compute_type=cfg.compute_type,
                        language=cfg.language,
                    )
                except (ModelNotConfigured, ModelUnavailable):
                    raise
                except Exception as e:
                    raise ExtractError(f"音视频转写失败: {e}") from e
            elif cfg.provider == "faster_whisper" or self._known_legacy_model(model_id):
                model = self._model(ctx)
                try:
                    options = {"vad_filter": True}
                    if cfg.language:
                        options["language"] = cfg.language
                    segments, info = model.transcribe(str(path), **options)
                    text = "\n".join(segment.text.strip() for segment in segments if segment.text.strip())
                except Exception as e:
                    raise ExtractError(f"音视频转写失败: {e}") from e
                language = getattr(info, "language", "")
            else:
                # Let the manager produce the precise relative-path error.
                text, language = manager.transcribe(
                    model_id,
                    path,
                    backend=cfg.local_backend,
                    device=cfg.device,
                    compute_type=cfg.compute_type,
                    language=cfg.language,
                )
            prefix = f"[音视频转写{f'，语言: {language}' if language else ''}]"
        elif cfg.provider == "openai_compatible":
            text = self._transcribe_openai_compatible(path, ctx)
            prefix = "[音视频转写]"
        else:
            raise CapabilityNotConfigured(f"不支持的 ASR provider: {cfg.provider}")

        if not text:
            raise ExtractError("音视频中没有识别到可索引的语音")
        return prefix + "\n" + text
