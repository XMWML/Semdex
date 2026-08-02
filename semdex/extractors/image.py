"""图片：确定性 OCR 优先，随后可补充本地视觉模型描述。"""
from __future__ import annotations

from pathlib import Path

from .base import ExtractContext, Extractor
from ..models import CapabilityNotConfigured, CapabilityUnavailable, ModelNotConfigured, ModelUnavailable
from ..ocr import ocr_image


class ImageExtractor(Extractor):
    name = "image"
    exts = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")

    def extract(self, path: Path, ctx: ExtractContext) -> str:
        parts: list[str] = []
        ocr_error: CapabilityNotConfigured | CapabilityUnavailable | None = None

        if ctx.config.ocr.enabled:
            try:
                text = ocr_image(path, ctx.config.ocr)
                parts.append(f"[图片 OCR]\n{text or '（未识别到文字）'}")
            except (CapabilityNotConfigured, CapabilityUnavailable) as e:
                ocr_error = e

        if ctx.vision.enabled:
            try:
                desc = ctx.vision.describe_image(path)
                if desc.strip():
                    parts.append(f"[图片描述]\n{desc.strip()}")
            except (ModelNotConfigured, ModelUnavailable):
                if not parts:
                    raise

        if parts:
            return "\n\n".join(parts)
        if ocr_error is not None:
            raise ocr_error
        raise ModelNotConfigured(
            "图片需要启用 [ocr] 或 [models.vision] 中至少一种本地识别能力"
        )
