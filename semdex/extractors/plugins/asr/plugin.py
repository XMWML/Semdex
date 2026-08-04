"""Shipped speech-recognition plugin backed by Semdex's ASR adapter."""
from pathlib import Path

from semdex.extractors.media import MediaExtractor

PLUGIN_METADATA = {
    "name": "音视频语音识别",
    "description": "使用 Semdex 已配置的本地或兼容云端 ASR 转写音视频。",
    "function": "extract",
    "builtin": True,
}

_EXTRACTOR = MediaExtractor()


def extract(path: Path, ctx) -> str:
    return _EXTRACTOR.extract(path, ctx)
