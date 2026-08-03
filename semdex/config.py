"""Configuration loading for the portable, project-local Semdex state."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .paths import (
    default_config_path,
    default_database_path,
    default_model_dir,
    default_temp_dir,
    ensure_private_directory,
)

DEFAULT_CONFIG_PATH = default_config_path()
DEFAULT_DB_PATH = default_database_path()
DEFAULT_TEMP_DIR = default_temp_dir()
DEFAULT_MODEL_DIR = default_model_dir()

DEFAULT_EXCLUDE = [
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".semdex", ".uv-cache", ".uv-python", ".pytest_cache", ".DS_Store", "*.app", ".Trash", "*.pyc",
]

CONFIG_TEMPLATE = """\
# Semdex 本地文件索引 配置文件
# 修改后重新运行 `semdex index` 生效

[storage]
# Relative paths are resolved next to this config file, so copying the whole
# Semdex folder to another disk keeps state and downloads with it.
db_path = "index.db"
temp_dir = "tmp"
model_dir = "models"

[watch]
# 要索引的文件夹（绝对路径，~ 会被展开），可以写多个
folders = [
    # "~/Desktop/软考资料",
]
# 排除规则：匹配任意一级目录名/文件名（fnmatch 通配符）
exclude = [".git", "node_modules", "__pycache__", ".venv", "venv", ".semdex", ".uv-cache", ".uv-python", ".pytest_cache", ".DS_Store", "*.app", ".Trash", "*.pyc"]
# 超过此大小的文件跳过（MB）
max_file_mb = 50
# `semdex watch` 在文件事件稳定多久后启动一次增量索引
debounce_sec = 1.5
# 长时间监听时的全量对账间隔（秒）；会同时重试失败/跳过项，0 表示关闭
reconcile_sec = 86400

# ── 本地模型（OpenAI 兼容接口）──────────────────────────────
# LM Studio 默认地址是 http://localhost:1234/v1，api_key 随便填
# model 填 LM Studio 里实际加载的模型名（可在其 Local Server 页面看到）
# 模型能力独立开关：没启用的能力对应的文件会标记为 waiting_model，
# 启用后重新 `semdex index` 即可补索引。

[models.llm]        # 旧版通用对话模型；未单独配置时，下面三项会继承它
enabled = false
mode = "openai"      # "openai"（OpenAI 兼容 API）或 "local"（项目 models 目录）
base_url = "http://localhost:1234/v1"
api_key = "lm-studio"
model = "qwen2.5-7b-instruct"
local_model = ""     # mode = "local" 时填写 models/ 下的相对路径

# 三个使用 LLM 的功能可分别选择模型。保持小节缺失会回退到 [models.llm]，
# 所以旧版配置不需要迁移，且只启用上面的 llm 仍可供三项功能使用。
# 要单独覆盖某一功能时，取消下面对应五行的注释并填写实际模型：
# [models.agent]      # 自然语言检索助手
# enabled = true
# mode = "local"
# base_url = "http://localhost:1234/v1"
# api_key = "lm-studio"
# model = "qwen2.5-7b-instruct"
# local_model = "Qwen3-1.7B-MLX-8bit"
#
# [models.entities]   # 实体抽取
# enabled = true
# mode = "openai"
# base_url = "http://localhost:1234/v1"
# api_key = "lm-studio"
# model = "qwen2.5-7b-instruct"
# local_model = ""
#
# [models.fallback]   # 未知文本格式的摘要兜底
# enabled = true
# mode = "openai"
# base_url = "http://localhost:1234/v1"
# api_key = "lm-studio"
# model = "qwen2.5-7b-instruct"
# local_model = ""

[models.vision]     # 视觉模型（图片 → 文字描述）
enabled = false
mode = "openai"
base_url = "http://localhost:1234/v1"
api_key = "lm-studio"
model = "qwen2-vl-7b-instruct"
local_model = ""

[models.embedding]  # 向量模型（语义搜索）
enabled = false
mode = "openai"
base_url = "http://localhost:1234/v1"
api_key = "lm-studio"
model = "text-embedding-bge-m3"
local_model = ""

[chunking]
chunk_size = 800     # 每块字符数
chunk_overlap = 100  # 相邻块重叠字符数

# ── 可选本地能力 ────────────────────────────────────────────
# OCR 可使用系统 tesseract，也可调用本机部署的 HTTP 服务。
[ocr]
enabled = false
provider = "tesseract" # "tesseract" 或 "local_http"
command = "tesseract"
pdf_renderer = "pdftoppm"  # 扫描 PDF 渲染为图片，Poppler 提供
languages = "eng+chi_sim"
dpi = 200
# local_http 时：将图片以 multipart 的 file 字段上传，languages 同样作为表单字段。
# endpoint = "http://127.0.0.1:8000/ocr"
# api_key = "" # 可选，非空时发送 Authorization: Bearer <api_key>
# response_path = "text" # JSON 响应中的点路径，例如 result.text 或 data.0.text
# timeout_sec = 180

# ASR 可使用项目 models 目录内的 Whisper 模型，或 OpenAI 兼容转写接口。
# provider = "openai_compatible" 时，endpoint 为空会使用 base_url + /audio/transcriptions。
[asr]
enabled = false
provider = "local" # "local" 或 "openai_compatible"（兼容旧值 faster_whisper）
local_backend = "auto" # auto / faster_whisper / mlx_whisper / whisper_cpp
local_model = "" # models/ 下的 Whisper 模型相对路径
model = "" # OpenAI 兼容接口的 model 名称；local 模式可留空
device = "auto"
compute_type = "int8"
# base_url = "http://localhost:1234/v1"
# endpoint = ""
# api_key = ""
# language = "" # 可选，例如 "zh"
# response_path = "text" # JSON 响应中的点路径；OpenAI 兼容接口默认 text
# timeout_sec = 180

# 对已提取的正文建立人名、项目、机构、日期等关系。需要启用 entities 模型。
[entities]
enabled = false
max_chars = 12000
max_per_file = 32

# 自然语言问答的工具调用上限，避免本地模型意外陷入循环。
[agent]
max_steps = 6
max_results = 12

# 对没有内置提取器的、可读为文本的文件启用本地 LLM 兜底。
# 二进制格式不会执行命令或离开本机，仍建议为其配置专用脚本提取器。
[agent_fallback]
enabled = false
max_bytes = 262144

# ── 自定义脚本提取器 ────────────────────────────────────────
# 匹配到的文件会执行 `script <安全快照路径>`，stdout 作为索引文本。
# 快照保留原文件名和扩展名，但不保留原目录或相邻文件。
# 规则优先级高于内置提取器，从上到下第一条命中的生效。
#
# [[extractors.rules]]
# match = "*.xyz"
# script = "/path/to/extract_xyz.sh"
"""


@dataclass
class ModelCfg:
    enabled: bool = False
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    model: str = ""
    # ``openai`` keeps the historical OpenAI-compatible path; ``local`` uses
    # ``local_model`` below ``Config.model_dir``.  ``provider`` is accepted by
    # settings/API callers as an alias, but ``mode`` is the persisted field.
    mode: str = "openai"
    local_model: str = ""
    # Filled by Config.__post_init__ so existing ModelClient(cfg, kind) calls
    # continue to work even when a caller chooses a custom model directory.
    local_model_dir: Path | None = field(default=None, repr=False, compare=False)


@dataclass
class ScriptRule:
    match: str
    script: str


@dataclass
class OcrCfg:
    enabled: bool = False
    provider: str = "tesseract"
    command: str = "tesseract"
    pdf_renderer: str = "pdftoppm"
    languages: str = "eng+chi_sim"
    dpi: int = 200
    endpoint: str = ""
    api_key: str = ""
    response_path: str = "text"
    timeout_sec: int = 180
    temp_dir: Path = DEFAULT_TEMP_DIR


@dataclass
class AsrCfg:
    enabled: bool = False
    provider: str = "faster_whisper"
    model: str = "base"
    device: str = "auto"
    compute_type: str = "int8"
    base_url: str = "http://localhost:1234/v1"
    endpoint: str = ""
    api_key: str = ""
    language: str = ""
    response_path: str = "text"
    timeout_sec: int = 180
    # New local-model selection; ``model`` remains the OpenAI API model name
    # (and is used as a legacy faster-whisper model identifier when needed).
    local_model: str = ""
    local_backend: str = "auto"
    local_model_dir: Path | None = field(default=None, repr=False, compare=False)


@dataclass
class EntityCfg:
    enabled: bool = False
    max_chars: int = 12_000
    max_per_file: int = 32


@dataclass
class AgentCfg:
    max_steps: int = 6
    max_results: int = 12


@dataclass
class AgentFallbackCfg:
    enabled: bool = False
    max_bytes: int = 262_144


@dataclass
class Config:
    db_path: Path = DEFAULT_DB_PATH
    temp_dir: Path = DEFAULT_TEMP_DIR
    model_dir: Path = DEFAULT_MODEL_DIR
    folders: list[Path] = field(default_factory=list)
    exclude: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE))
    max_file_mb: int = 50
    llm: ModelCfg = field(default_factory=ModelCfg)
    # These are deliberately named differently from ``agent`` / ``entities``
    # below, which hold feature behaviour rather than model connection data.
    # Missing [models.<purpose>] sections copy the legacy [models.llm] config.
    agent_model: ModelCfg | None = None
    entities_model: ModelCfg | None = None
    fallback_model: ModelCfg | None = None
    vision: ModelCfg = field(default_factory=ModelCfg)
    embedding: ModelCfg = field(default_factory=ModelCfg)
    script_rules: list[ScriptRule] = field(default_factory=list)
    chunk_size: int = 800
    chunk_overlap: int = 100
    watch_debounce_sec: float = 1.5
    watch_reconcile_sec: float = 86_400.0
    ocr: OcrCfg = field(default_factory=OcrCfg)
    asr: AsrCfg = field(default_factory=AsrCfg)
    entities: EntityCfg = field(default_factory=EntityCfg)
    agent: AgentCfg = field(default_factory=AgentCfg)
    agent_fallback: AgentFallbackCfg = field(default_factory=AgentFallbackCfg)
    config_path: Path | None = None

    def __post_init__(self) -> None:
        # Programmatic callers have historically only set ``llm``.  Preserve
        # that useful shorthand while allowing an explicit disabled per-purpose
        # ModelCfg to opt one feature out.
        if self.agent_model is None:
            self.agent_model = _copy_model_cfg(self.llm)
        if self.entities_model is None:
            self.entities_model = _copy_model_cfg(self.llm)
        if self.fallback_model is None:
            self.fallback_model = _copy_model_cfg(self.llm)
        for model in (
            self.llm,
            self.agent_model,
            self.entities_model,
            self.fallback_model,
            self.vision,
            self.embedding,
        ):
            if model is not None:
                model.local_model_dir = self.model_dir
        self.asr.local_model_dir = self.model_dir
        self.ocr.temp_dir = self.temp_dir


def _section(data: dict, name: str) -> dict:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] 必须是 TOML 小节")
    return value


def _bool(d: dict, key: str, default: bool, section: str) -> bool:
    value = d.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"[{section}] {key} 必须是 true 或 false")
    return value


def _string(d: dict, key: str, default: str, section: str) -> str:
    value = d.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"[{section}] {key} 必须是字符串")
    return value


def _positive_int(d: dict, key: str, default: int, section: str, minimum: int = 1) -> int:
    value = d.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"[{section}] {key} 必须是整数")
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError) as e:
        raise ValueError(f"[{section}] {key} 必须是整数") from e


def _model_cfg(d: dict, section: str = "models.llm") -> ModelCfg:
    raw_mode = d.get("mode", d.get("provider", "openai"))
    if not isinstance(raw_mode, str):
        raise ValueError(f"[{section}] mode 必须是 openai 或 local")
    mode = raw_mode.strip().lower()
    if mode in {"openai_compatible", "api", "openai-compatible"}:
        mode = "openai"
    if mode not in {"openai", "local"}:
        raise ValueError(f"[{section}] mode 必须是 openai 或 local")
    enabled = _bool(d, "enabled", False, section)
    local_model = _string(d, "local_model", "", section).strip()
    if enabled and mode == "local" and not local_model:
        raise ValueError(f"[{section}] local 模式启用时必须配置 local_model")
    return ModelCfg(
        enabled=enabled,
        base_url=_string(d, "base_url", "http://localhost:1234/v1", section).rstrip("/"),
        api_key=_string(d, "api_key", "lm-studio", section),
        model=_string(d, "model", "", section),
        mode=mode,
        local_model=local_model,
    )


def _copy_model_cfg(cfg: ModelCfg) -> ModelCfg:
    """Make fallback model config independent of the legacy source object."""
    return ModelCfg(
        enabled=cfg.enabled,
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        model=cfg.model,
        mode=cfg.mode,
        local_model=cfg.local_model,
    )


def _ocr_cfg(d: dict) -> OcrCfg:
    provider = _string(d, "provider", "tesseract", "ocr").strip().lower()
    if provider not in {"tesseract", "local_http"}:
        raise ValueError("[ocr] provider 必须是 tesseract 或 local_http")
    enabled = _bool(d, "enabled", False, "ocr")
    endpoint = _string(d, "endpoint", "", "ocr").strip()
    if enabled and provider == "local_http" and not endpoint:
        raise ValueError("[ocr] provider = local_http 且已启用时必须配置 endpoint")
    return OcrCfg(
        enabled=enabled,
        provider=provider,
        command=_string(d, "command", "tesseract", "ocr"),
        pdf_renderer=_string(d, "pdf_renderer", "pdftoppm", "ocr"),
        languages=_string(d, "languages", "eng+chi_sim", "ocr"),
        dpi=max(72, _positive_int(d, "dpi", 200, "ocr")),
        endpoint=endpoint,
        api_key=_string(d, "api_key", "", "ocr"),
        response_path=_string(d, "response_path", "text", "ocr").strip() or "text",
        timeout_sec=_positive_int(d, "timeout_sec", 180, "ocr"),
    )


def _asr_cfg(d: dict) -> AsrCfg:
    provider = _string(d, "provider", "faster_whisper", "asr").strip().lower()
    original_provider = provider
    legacy_local = provider in {"faster_whisper", "faster-whisper", "mlx_whisper", "mlx-whisper", "whisper-cpp", "whisper_cpp", "gguf"}
    if legacy_local:
        legacy_local = True
        provider = "local"
    if provider not in {"local", "openai_compatible"}:
        raise ValueError("[asr] provider 必须是 local 或 openai_compatible")
    enabled = _bool(d, "enabled", False, "asr")
    endpoint = _string(d, "endpoint", "", "asr").strip()
    base_url = _string(d, "base_url", "http://localhost:1234/v1", "asr").strip().rstrip("/")
    if enabled and provider == "openai_compatible" and not endpoint and not base_url:
        raise ValueError("[asr] openai_compatible 需要配置 endpoint 或 base_url")
    legacy_backend = {
        "mlx_whisper": "mlx_whisper",
        "mlx-whisper": "mlx_whisper",
        "whisper-cpp": "whisper_cpp",
        "whisper_cpp": "whisper_cpp",
        "gguf": "whisper_cpp",
    }.get(original_provider, "faster_whisper")
    backend = _string(d, "local_backend", legacy_backend if legacy_local else "auto", "asr").strip().lower()
    backend_aliases = {
        "faster-whisper": "faster_whisper",
        "mlx-whisper": "mlx_whisper",
        "whisper-cpp": "whisper_cpp",
        "gguf": "whisper_cpp",
    }
    backend = backend_aliases.get(backend, backend)
    if backend not in {"auto", "faster_whisper", "mlx_whisper", "whisper_cpp"}:
        raise ValueError("[asr] local_backend 必须是 auto、faster_whisper、mlx_whisper 或 whisper_cpp")
    model = _string(d, "model", "base", "asr")
    local_model = _string(d, "local_model", model if legacy_local else "", "asr").strip()
    return AsrCfg(
        enabled=enabled,
        provider=provider,
        model=model,
        device=_string(d, "device", "auto", "asr"),
        compute_type=_string(d, "compute_type", "int8", "asr"),
        base_url=base_url,
        endpoint=endpoint,
        api_key=_string(d, "api_key", "", "asr"),
        language=_string(d, "language", "", "asr"),
        response_path=_string(d, "response_path", "text", "asr").strip() or "text",
        timeout_sec=_positive_int(d, "timeout_sec", 180, "asr"),
        local_model=local_model,
        local_backend=backend,
    )


def _storage_path(value: object, default: Path, config_dir: Path) -> Path:
    """Resolve portable storage values relative to their config file."""
    raw = default if value is None else Path(str(value)).expanduser()
    if not raw.is_absolute():
        raw = config_dir / raw
    return raw.resolve(strict=False)


def resolve_config_path(explicit: str | os.PathLike | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve(strict=False)
    env = os.environ.get("SEMDEX_CONFIG")
    if env:
        return Path(env).expanduser().resolve(strict=False)
    return DEFAULT_CONFIG_PATH


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg_path = resolve_config_path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"配置文件不存在: {cfg_path}\n先运行 `semdex init` 生成，再把要索引的文件夹填进 [watch] folders"
        )
    with open(cfg_path, "rb") as f:
        data = tomllib.load(f)

    storage = _section(data, "storage")
    watch = _section(data, "watch")
    models = _section(data, "models")
    chunking = _section(data, "chunking")
    extractors = _section(data, "extractors")
    ocr = _section(data, "ocr")
    asr = _section(data, "asr")
    entities = _section(data, "entities")
    agent = _section(data, "agent")
    agent_fallback = _section(data, "agent_fallback")

    chunk_size = max(1, int(chunking.get("chunk_size", 800)))
    chunk_overlap = max(0, int(chunking.get("chunk_overlap", 100)))
    if chunk_overlap >= chunk_size:
        raise ValueError("[chunking] chunk_overlap 必须小于 chunk_size")

    rules = []
    for r in extractors.get("rules", []):
        if r.get("match") and r.get("script"):
            rules.append(ScriptRule(match=str(r["match"]), script=str(Path(r["script"]).expanduser())))

    llm = _model_cfg(_section(models, "llm"), "models.llm")
    agent_model = (
        _model_cfg(_section(models, "agent"), "models.agent")
        if "agent" in models else _copy_model_cfg(llm)
    )
    entities_model = (
        _model_cfg(_section(models, "entities"), "models.entities")
        if "entities" in models else _copy_model_cfg(llm)
    )
    fallback_model = (
        _model_cfg(_section(models, "fallback"), "models.fallback")
        if "fallback" in models else _copy_model_cfg(llm)
    )
    db_path = _storage_path(storage.get("db_path"), DEFAULT_DB_PATH, cfg_path.parent)
    temp_dir = _storage_path(storage.get("temp_dir"), DEFAULT_TEMP_DIR, cfg_path.parent)
    model_dir = _storage_path(storage.get("model_dir"), DEFAULT_MODEL_DIR, cfg_path.parent)
    ocr_cfg = _ocr_cfg(ocr)
    ocr_cfg.temp_dir = temp_dir

    return Config(
        db_path=db_path,
        temp_dir=temp_dir,
        model_dir=model_dir,
        folders=[Path(p).expanduser() for p in watch.get("folders", [])],
        exclude=list(watch.get("exclude", DEFAULT_EXCLUDE)),
        max_file_mb=max(0, int(watch.get("max_file_mb", 50))),
        llm=llm,
        agent_model=agent_model,
        entities_model=entities_model,
        fallback_model=fallback_model,
        vision=_model_cfg(_section(models, "vision"), "models.vision"),
        embedding=_model_cfg(_section(models, "embedding"), "models.embedding"),
        script_rules=rules,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        watch_debounce_sec=max(0.1, float(watch.get("debounce_sec", 1.5))),
        watch_reconcile_sec=max(0.0, float(watch.get("reconcile_sec", 86_400))),
        ocr=ocr_cfg,
        asr=_asr_cfg(asr),
        entities=EntityCfg(
            enabled=bool(entities.get("enabled", False)),
            max_chars=max(500, int(entities.get("max_chars", 12_000))),
            max_per_file=max(1, int(entities.get("max_per_file", 32))),
        ),
        agent=AgentCfg(
            max_steps=max(1, int(agent.get("max_steps", 6))),
            max_results=max(1, int(agent.get("max_results", 12))),
        ),
        agent_fallback=AgentFallbackCfg(
            enabled=bool(agent_fallback.get("enabled", False)),
            max_bytes=max(1_024, int(agent_fallback.get("max_bytes", 262_144))),
        ),
        config_path=cfg_path,
    )


def write_default_config(path: str | os.PathLike | None = None) -> Path:
    cfg_path = resolve_config_path(path)
    if cfg_path == DEFAULT_CONFIG_PATH:
        # The template may contain an API key, so the managed state directory
        # should not disclose its names or contents to other local accounts.
        ensure_private_directory(cfg_path.parent)
    else:
        cfg_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(cfg_path, flags, 0o600)
    except FileExistsError as e:
        raise FileExistsError(f"配置文件已存在: {cfg_path}") from e
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(CONFIG_TEMPLATE)
    return cfg_path
