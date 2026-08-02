"""提取器基类与上下文。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from ..config import Config
    from ..modelclient import ModelClient


@dataclass
class ExtractContext:
    config: "Config"
    vision: "ModelClient"
    llm: "ModelClient"


class Extractor(ABC):
    """把一个文件转成可索引的纯文本。

    失败时抛 ExtractError（文件问题）/ ModelNotConfigured / ModelUnavailable
    （模型问题，可等待重试），见 semdex.models。
    """

    name: ClassVar[str] = ""
    exts: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def extract(self, path: Path, ctx: ExtractContext) -> str: ...
