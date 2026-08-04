"""Deterministic extraction of a PDF text layer."""
from __future__ import annotations

from pathlib import Path

from ..models import ExtractError
from .base import ExtractContext, Extractor


class PdfExtractor(Extractor):
    name = "pdf"
    exts = (".pdf",)

    def extract(self, path: Path, ctx: ExtractContext) -> str:
        from pypdf import PdfReader

        try:
            reader = PdfReader(str(path))
            if reader.is_encrypted:
                try:
                    if not reader.decrypt(""):
                        raise ExtractError("PDF 已加密，无法读取")
                except Exception:
                    raise ExtractError("PDF 已加密，无法读取")
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")
        except ExtractError:
            raise
        except Exception as e:
            raise ExtractError(f"PDF 解析失败: {e}") from e

        text = "\n\n".join(pages).strip()
        # A short but valid text layer is still useful for retrieval.  Treating
        # it as a scan would otherwise hide one-line PDFs until OCR is enabled.
        if text:
            return text

        raise ExtractError("PDF 没有可提取的文本层；请把该扩展名改用 OCR Python 插件")
