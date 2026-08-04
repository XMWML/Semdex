"""Image formats accepted by raw-image LLM inputs."""
from __future__ import annotations


IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}
SUPPORTED_IMAGE_EXTENSIONS = frozenset(IMAGE_MIME_TYPES)


def matches_image_signature(extension: str, header: bytes) -> bool:
    """Return whether a file header matches its supported image extension."""
    extension = extension.strip().lower()
    if extension == ".png":
        return header.startswith(b"\x89PNG\r\n\x1a\n")
    if extension in {".jpg", ".jpeg"}:
        return header.startswith(b"\xff\xd8\xff")
    if extension == ".webp":
        return len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP"
    if extension == ".gif":
        return header.startswith((b"GIF87a", b"GIF89a"))
    if extension == ".bmp":
        return header.startswith(b"BM")
    return False


def image_format_error(extension: str) -> str:
    supported = "、".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
    return (
        f"原始图片输入不支持 {extension or '无扩展名'}；仅支持 {supported}。"
        "PDF 或文档请改用文本输入或 Python 插件。"
    )
