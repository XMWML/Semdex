"""共享数据类型与异常。此模块不依赖包内其他模块。"""
from __future__ import annotations

from dataclasses import dataclass, field


class ExtractError(Exception):
    """文件内容提取失败（文件损坏、格式不支持等），文件标记为 failed。"""


class ModelNotConfigured(Exception):
    """所需模型能力未在配置中启用，文件标记为 waiting_model，启用后重跑即可。"""


class ModelUnavailable(Exception):
    """模型已配置但服务无法连接（如 LM Studio 未启动），文件标记为 waiting_model。"""


class EmbeddingRebuildRequired(RuntimeError):
    """Embedding 模型变更后，旧向量已失效，必须完整重建。"""


class CapabilityNotConfigured(Exception):
    """可选本地能力未启用（例如 OCR 或 ASR）。"""


class CapabilityUnavailable(Exception):
    """可选本地能力已启用，但其运行时或命令不可用。"""


# files.index_status 的合法取值
STATUS_PENDING = "pending"            # 等待索引
STATUS_DONE = "done"                  # 已索引
STATUS_FAILED = "failed"              # 提取失败
STATUS_SKIPPED = "skipped"            # 无适用提取器
STATUS_WAITING_MODEL = "waiting_model"  # 等待模型服务可用
STATUS_WAITING_CAPABILITY = "waiting_capability"  # 等待 OCR / ASR 等本地能力
STATUS_TOO_LARGE = "too_large"          # 因当前大小限制而未索引

ALL_STATUSES = (
    STATUS_PENDING,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_WAITING_MODEL,
    STATUS_WAITING_CAPABILITY,
    STATUS_TOO_LARGE,
)


@dataclass
class SearchHit:
    file_id: int
    path: str
    filename: str
    ext: str
    mtime: float
    score: float
    snippet: str
    source: str = "fulltext"  # fulltext / semantic / hybrid / like

    def to_dict(self) -> dict:
        return {
            "file_id": self.file_id,
            "path": self.path,
            "filename": self.filename,
            "ext": self.ext,
            "mtime": self.mtime,
            "score": round(self.score, 6),
            "snippet": self.snippet,
            "source": self.source,
        }


@dataclass
class ScanStats:
    scanned: int = 0
    new_or_changed: int = 0
    unchanged: int = 0
    removed: int = 0
    too_large: int = 0
    errors: int = 0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class IndexStats:
    indexed: int = 0
    failed: int = 0
    skipped: int = 0
    waiting_model: int = 0
    waiting_capability: int = 0
    embedded_files: int = 0
    embed_errors: list[str] = field(default_factory=list)
    entities_indexed: int = 0
    entity_failed: int = 0
    entity_waiting_model: int = 0

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["embed_errors"] = list(self.embed_errors)
        return d
