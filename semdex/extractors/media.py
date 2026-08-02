"""Audio/video transcription through local or OpenAI-compatible ASR services."""
from __future__ import annotations

from pathlib import Path

from ..models import CapabilityNotConfigured, CapabilityUnavailable, ExtractError
from ..remote import RemoteRequestError, RemoteResponseError, post_multipart_json
from .base import ExtractContext, Extractor


class MediaExtractor(Extractor):
    name = "asr"
    exts = (
        ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus",
        ".mp4", ".mov", ".mkv", ".webm", ".avi",
    )
    _models: dict[tuple[str, str, str], object] = {}

    def _model(self, ctx: ExtractContext):
        cfg = ctx.config.asr
        if not cfg.enabled:
            raise CapabilityNotConfigured(
                "ASR 未启用（在配置中设置 [asr] enabled = true，并执行 `uv sync --extra asr`）"
            )
        if cfg.provider != "faster_whisper":
            raise CapabilityNotConfigured(
                "当前 ASR provider 不是 faster_whisper，不能加载本地 Whisper 模型"
            )
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise CapabilityUnavailable(
                "未安装 faster-whisper。执行 `uv sync --extra asr` 后重新运行 `semdex index`"
            ) from e

        key = (cfg.model, cfg.device, cfg.compute_type)
        model = self._models.get(key)
        if model is None:
            try:
                model = WhisperModel(
                    cfg.model, device=cfg.device, compute_type=cfg.compute_type
                )
            except Exception as e:
                raise CapabilityUnavailable(f"无法加载本地 Whisper 模型 {cfg.model}: {e}") from e
            self._models[key] = model
        return model

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

        if cfg.provider == "faster_whisper":
            model = self._model(ctx)
            try:
                segments, info = model.transcribe(str(path), vad_filter=True)
                text = "\n".join(segment.text.strip() for segment in segments if segment.text.strip())
            except Exception as e:
                raise ExtractError(f"音视频转写失败: {e}") from e
            language = getattr(info, "language", "")
            prefix = f"[音视频转写{f'，语言: {language}' if language else ''}]"
        elif cfg.provider == "openai_compatible":
            text = self._transcribe_openai_compatible(path, ctx)
            prefix = "[音视频转写]"
        else:
            raise CapabilityNotConfigured(f"不支持的 ASR provider: {cfg.provider}")

        if not text:
            raise ExtractError("音视频中没有识别到可索引的语音")
        return prefix + "\n" + text
