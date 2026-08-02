"""纯文本/代码类文件：直接读入。"""
from __future__ import annotations

from pathlib import Path

from ..models import ExtractError
from .base import ExtractContext, Extractor

TEXT_EXTS = (
    ".txt", ".md", ".markdown", ".rst", ".org", ".tex",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".mjs", ".c", ".h", ".cpp", ".hpp",
    ".java", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".sql", ".sh",
    ".zsh", ".bash", ".lua", ".r",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env",
    ".csv", ".tsv", ".log",
    ".html", ".htm", ".css", ".xml", ".svg",
)


def decode_text_best_effort(raw: bytes) -> str:
    for enc in ("utf-8", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def read_text_best_effort(path: Path) -> str:
    return decode_text_best_effort(path.read_bytes())


class TextExtractor(Extractor):
    name = "text"
    exts = TEXT_EXTS

    def extract(self, path: Path, ctx: ExtractContext) -> str:
        try:
            return read_text_best_effort(path)
        except OSError as e:
            raise ExtractError(f"读取失败: {e}") from e
