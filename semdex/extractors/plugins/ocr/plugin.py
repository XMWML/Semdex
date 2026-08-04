"""Shipped OCR plugin for images and scanned PDFs."""
from pathlib import Path

from semdex.models import ExtractError
from semdex import ocr as ocr_adapter

PLUGIN_METADATA = {
    "name": "OCR 文字识别",
    "description": "使用 Semdex 已配置的 OCR 服务识别图片或扫描 PDF。",
    "function": "extract",
    "builtin": True,
}

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


def extract(path: Path, ctx) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = ocr_adapter.ocr_pdf(path, ctx.config.ocr)
    elif suffix in _IMAGE_EXTENSIONS:
        text = ocr_adapter.ocr_image(path, ctx.config.ocr)
    else:
        raise ExtractError(f"OCR 插件不支持此扩展名: {suffix or '无扩展名'}")
    if not text.strip():
        raise ExtractError("OCR 没有识别到可索引的文字")
    return "[OCR 文字识别]\n" + text.strip()
