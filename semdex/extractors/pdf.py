"""PDF：优先抽取文本层，扫描件回退到配置的本地 OCR。"""
from __future__ import annotations

from pathlib import Path

from ..models import ExtractError
from ..ocr import ocr_pdf
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

        ocr_text = ocr_pdf(path, ctx.config.ocr)
        if len(ocr_text) < 3:
            raise ExtractError("PDF 没有可提取的文本，OCR 也没有识别出文字")
        return f"[扫描 PDF OCR]\n{ocr_text}"
