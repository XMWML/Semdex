"""提取器注册与路由。

路由优先级：启用的扩展名规则 > 旧版可执行脚本规则（仅兼容）。每条
规则明确选择 text、llm 或 python；OCR/ASR 也通过文件夹插件进入同一
路由，不再是特殊的一级索引分支。
新增内置提取器：写一个 Extractor 子类，填好 name/exts，在下面 _BUILTIN
列表里加上即可。
"""
from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

from .base import ExtractContext, Extractor
from .image import ImageExtractor
from .archive import ZipExtractor
from .mail import EmlExtractor, MboxExtractor
from .media import MediaExtractor
from .legacy_office import LegacyOfficeExtractor
from .office import DocxExtractor, PptxExtractor, XlsxExtractor
from .pdf import PdfExtractor
from .llm import LlmTextExtractor
from .script import PythonFunctionExtractor, ScriptExtractor
from .text import TextExtractor

if TYPE_CHECKING:
    from ..config import Config

_BUILTIN: list[Extractor] = [
    TextExtractor(),
    PdfExtractor(),
    DocxExtractor(),
    XlsxExtractor(),
    PptxExtractor(),
    ImageExtractor(),
    ZipExtractor(),
    EmlExtractor(),
    MboxExtractor(),
    MediaExtractor(),
    LegacyOfficeExtractor(),
]

EXT_MAP: dict[str, Extractor] = {
    ext: ex for ex in _BUILTIN for ext in ex.exts
}
BUILTIN_BY_ID: dict[str, Extractor] = {extractor.name: extractor for extractor in _BUILTIN}


def resolve(path: Path, config: "Config") -> Extractor | None:
    """为文件挑选提取器；返回 None 表示没有适用的（文件将标记 skipped）。"""
    suffix = path.suffix.lower()
    for rule in config.extractor_rules:
        if not rule.enabled or suffix not in rule.extensions:
            continue
        if rule.kind == "text":
            if rule.id in {"image", "asr"}:
                return TextExtractor()
            return BUILTIN_BY_ID.get(rule.id, TextExtractor())
        if rule.kind == "llm":
            source: Extractor
            if rule.id == "image":
                source = PythonFunctionExtractor.from_plugin(config.extractor_dir, "ocr")
            elif rule.id == "asr":
                source = PythonFunctionExtractor.from_plugin(config.extractor_dir, "asr")
            else:
                source = BUILTIN_BY_ID.get(rule.id, TextExtractor())
            return LlmTextExtractor(
                provider=rule.provider,
                input_mode=rule.input_mode,
                prompt=rule.prompt,
                source=source,
            )
        if rule.kind == "python":
            extractor = PythonFunctionExtractor.from_plugin(
                config.extractor_dir, rule.plugin, rule.function,
            )
            extractor.name = f"python:{Path(rule.plugin).stem}"
            return extractor
    # Keep hand-written command rules usable for old configurations when no
    # explicit per-extension route was selected in the settings page.
    for rule in config.script_rules:
        if fnmatch(path.name, rule.match):
            return ScriptExtractor(rule.script)
    return None


__all__ = ["Extractor", "ExtractContext", "resolve", "EXT_MAP", "BUILTIN_BY_ID"]
