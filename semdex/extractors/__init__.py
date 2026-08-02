"""提取器注册与路由。

路由优先级：配置里的自定义脚本规则（按顺序第一条命中） > 内置扩展名映射。
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
from .agent_fallback import AgentFallbackExtractor
from .office import DocxExtractor, PptxExtractor, XlsxExtractor
from .pdf import PdfExtractor
from .script import ScriptExtractor
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


def resolve(path: Path, config: "Config") -> Extractor | None:
    """为文件挑选提取器；返回 None 表示没有适用的（文件将标记 skipped）。"""
    for rule in config.script_rules:
        if fnmatch(path.name, rule.match):
            return ScriptExtractor(rule.script)
    extractor = EXT_MAP.get(path.suffix.lower())
    if extractor is not None:
        return extractor
    if config.agent_fallback.enabled:
        return AgentFallbackExtractor()
    return None


__all__ = ["Extractor", "ExtractContext", "resolve", "EXT_MAP"]
