"""提取器基类与上下文。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from ..models import ModelNotConfigured

if TYPE_CHECKING:
    from ..config import Config
    from ..modelclient import ModelClient


@dataclass
class ExtractContext:
    config: "Config"
    vision: "ModelClient"
    llm: "ModelClient"
    # Stable provider ids are the active first-level LLM contract.
    llm_providers: dict[str, "ModelClient"] = field(default_factory=dict)
    # Compatibility alias for integrations from the purpose-model prototype.
    llm_models: dict[str, "ModelClient"] = field(default_factory=dict)

    def llm_for(self, provider_id: str) -> "ModelClient":
        client = self.llm_providers.get(provider_id) or self.llm_models.get(provider_id)
        if client is not None:
            return client
        # Preserve direct ExtractContext construction from older integrations.
        if provider_id in {"default", "llm", "fallback"} and self.llm is not None:
            return self.llm
        raise ModelNotConfigured(f"LLM 提取器没有可用的供应商配置: {provider_id}")


class Extractor(ABC):
    """把一个文件转成可索引的纯文本。

    失败时抛 ExtractError（文件问题）/ ModelNotConfigured / ModelUnavailable
    （模型问题，可等待重试），见 semdex.models。
    """

    name: ClassVar[str] = ""
    exts: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def extract(self, path: Path, ctx: ExtractContext) -> str: ...
