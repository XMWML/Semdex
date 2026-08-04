"""Configuration loading for the portable, project-local Semdex state."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .imagetypes import SUPPORTED_IMAGE_EXTENSIONS, image_format_error
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
DEFAULT_EXTRACTOR_DIR = DEFAULT_CONFIG_PATH.parent / "extractors"

DEFAULT_LLM_PROVIDER_ID = "default"
LEGACY_LLM_PROVIDER_IDS = {
    "llm": DEFAULT_LLM_PROVIDER_ID,
    "fallback": "legacy-fallback",
    "agent": "legacy-agent",
    "entities": "legacy-entities",
}

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

# ── 一级索引可选 LLM 供应商 ────────────────────────────────
# 可在设置页添加、重命名或删除；扩展名规则始终通过稳定 id 引用。
[llm_providers]
configured = true

[[llm_providers.items]]
id = "default"
name = "默认 LLM"
enabled = false
mode = "openai"      # openai（OpenAI 兼容 API）或 local（models 目录）
base_url = "http://localhost:1234/v1"
api_key = "lm-studio"
model = "qwen2.5-7b-instruct"
local_model = ""

# 检索 Agent、实体抽取和语义嵌入是一级正文之后的独立能力。
[models.agent]
enabled = false
mode = "openai"
base_url = "http://localhost:1234/v1"
api_key = "lm-studio"
model = "qwen2.5-7b-instruct"
local_model = ""

[models.entities]
enabled = false
mode = "openai"
base_url = "http://localhost:1234/v1"
api_key = "lm-studio"
model = "qwen2.5-7b-instruct"
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
# LLM 可以调用受限的检索/文件详情工具；模型连接在 [models.agent] 配置。
enabled = true
max_steps = 6
max_results = 12

# RAG 语义检索使用 [models.embedding]。关闭后“混合”会退化为关键词检索，
# “语义”模式会提示先开启此开关。
[rag]
enabled = true
max_context_chunks = 8

# ── 文件索引方式 ────────────────────────────────────────────
# 内置规则会自动显示在设置页：可修改扩展名或开关，但不能删除。
# 每条规则可选 text（直接提取正文）、llm 或 python。每个 Python 插件
# 使用独立文件夹，其中入口固定为 plugin.py，并提供 extract(path) 或
# extract(path, ctx)。初始化时会在该目录放入 OCR 与语音识别插件。
[extractors]
plugin_dir = "extractors"
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
class LlmProvider:
    """A reusable chat-model connection selected by extension rules."""

    id: str = DEFAULT_LLM_PROVIDER_ID
    name: str = "默认 LLM"
    enabled: bool = False
    mode: str = "openai"
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    model: str = ""
    local_model: str = ""
    local_model_dir: Path | None = field(default=None, repr=False, compare=False)

    def model_config(self) -> ModelCfg:
        return ModelCfg(
            enabled=self.enabled,
            mode=self.mode,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            local_model=self.local_model,
            local_model_dir=self.local_model_dir,
        )


@dataclass
class ScriptRule:
    """Legacy command rule retained for old hand-written TOML files.

    New settings use ``ExtractorRule(kind='python')`` instead.  Keeping this
    small compatibility layer avoids silently changing existing indexes.
    """
    match: str
    script: str


@dataclass
class ExtractorRule:
    """A configurable extension to extractor mapping.

    Built-in rules are identified by a stable id and are always restored if
    missing from a settings payload.  Their extension list and route may be
    changed, but their id/label remain stable. Python rules refer to
    ``<plugin_dir>/<plugin>/plugin.py``; LLM rules select a provider by stable
    id and independently choose text or image input.
    """
    id: str
    label: str
    extensions: list[str]
    kind: str = "text"  # text / llm / python; legacy "builtin" loads as text
    enabled: bool = True
    provider: str = ""
    input_mode: str = "text"
    prompt: str = ""
    plugin: str = ""
    function: str = "extract"
    # Compatibility aliases accepted from the previous settings contract.
    script: str = ""
    model: str = "llm"


BUILTIN_EXTRACTOR_RULES: tuple[ExtractorRule, ...] = (
    ExtractorRule("text", "文本与代码", [
        ".txt", ".md", ".markdown", ".rst", ".org", ".tex", ".py", ".js", ".ts",
        ".tsx", ".jsx", ".mjs", ".c", ".h", ".cpp", ".hpp", ".java", ".go", ".rs",
        ".rb", ".php", ".swift", ".kt", ".sql", ".sh", ".zsh", ".bash", ".lua", ".r",
        ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env", ".csv",
        ".tsv", ".log", ".html", ".htm", ".css", ".xml", ".svg",
    ]),
    ExtractorRule("pdf", "PDF 文档", [".pdf"]),
    ExtractorRule("docx", "Word 文档", [".docx"]),
    ExtractorRule("xlsx", "Excel 表格", [".xlsx", ".xlsm"]),
    ExtractorRule("pptx", "PowerPoint 演示文稿", [".pptx"]),
    ExtractorRule("legacy_office", "旧版 Office 文档", [".doc", ".xls", ".ppt"]),
    ExtractorRule(
        "image", "图片（OCR 插件）",
        [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"],
        kind="python", plugin="ocr",
    ),
    ExtractorRule("zip", "ZIP 压缩包", [".zip", ".cbz"]),
    ExtractorRule("eml", "邮件 .eml", [".eml"]),
    ExtractorRule("mbox", "邮件归档 .mbox", [".mbox"]),
    ExtractorRule(
        "asr", "音频与视频（语音识别插件）", [
            ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus",
            ".mp4", ".mov", ".mkv", ".webm", ".avi",
        ],
        kind="python", plugin="asr",
    ),
)


def default_extractor_rules() -> list[ExtractorRule]:
    """Return independent built-in rule values for a Config instance."""
    return [
        ExtractorRule(
            id=rule.id, label=rule.label, extensions=list(rule.extensions),
            kind=rule.kind, enabled=rule.enabled, provider=rule.provider,
            input_mode=rule.input_mode, prompt=rule.prompt, plugin=rule.plugin,
            function=rule.function, script=rule.script, model=rule.model,
        )
        for rule in BUILTIN_EXTRACTOR_RULES
    ]


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
    enabled: bool = True
    max_steps: int = 6
    max_results: int = 12


@dataclass
class RagCfg:
    enabled: bool = True
    max_context_chunks: int = 8


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
    # ``llm`` remains an in-memory alias for integrations written before
    # reusable providers were introduced. New extension rules use
    # ``llm_providers`` exclusively.
    llm: ModelCfg = field(default_factory=ModelCfg)
    llm_providers: list[LlmProvider] | None = None
    # These are deliberately named differently from ``agent`` / ``entities``
    # below, which hold feature behaviour rather than model connection data.
    # Missing [models.<purpose>] sections copy the legacy [models.llm] config.
    agent_model: ModelCfg | None = None
    entities_model: ModelCfg | None = None
    fallback_model: ModelCfg | None = None
    vision: ModelCfg = field(default_factory=ModelCfg)
    embedding: ModelCfg = field(default_factory=ModelCfg)
    extractor_dir: Path = DEFAULT_EXTRACTOR_DIR
    extractor_rules: list[ExtractorRule] = field(default_factory=default_extractor_rules)
    script_rules: list[ScriptRule] = field(default_factory=list)
    chunk_size: int = 800
    chunk_overlap: int = 100
    watch_debounce_sec: float = 1.5
    watch_reconcile_sec: float = 86_400.0
    ocr: OcrCfg = field(default_factory=OcrCfg)
    asr: AsrCfg = field(default_factory=AsrCfg)
    entities: EntityCfg = field(default_factory=EntityCfg)
    agent: AgentCfg = field(default_factory=AgentCfg)
    rag: RagCfg = field(default_factory=RagCfg)
    agent_fallback: AgentFallbackCfg = field(default_factory=AgentFallbackCfg)
    config_path: Path | None = None

    def __post_init__(self) -> None:
        if self.config_path is not None and self.extractor_dir == DEFAULT_EXTRACTOR_DIR:
            self.extractor_dir = self.config_path.expanduser().resolve(strict=False).parent / "extractors"
        if self.llm_providers is None:
            self.llm_providers = [_provider_from_model(
                DEFAULT_LLM_PROVIDER_ID, "默认 LLM", self.llm,
            )]
        else:
            self.llm_providers = _normalise_llm_providers(self.llm_providers)
        if self.llm_providers:
            self.llm = self.llm_providers[0].model_config()
        # Programmatic callers have historically only set ``llm``. Preserve
        # that shorthand for independent downstream models when their section
        # is absent, after resolving the first provider alias.
        if self.agent_model is None:
            self.agent_model = _copy_model_cfg(self.llm)
        if self.entities_model is None:
            self.entities_model = _copy_model_cfg(self.llm)
        if self.fallback_model is None:
            self.fallback_model = _copy_model_cfg(self.llm)
        self._add_legacy_rule_providers()
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
        for provider in self.llm_providers:
            provider.local_model_dir = self.model_dir
        self.asr.local_model_dir = self.model_dir
        self.ocr.temp_dir = self.temp_dir
        self.extractor_rules = _merge_builtin_extractor_rules(
            self.extractor_rules, {provider.id for provider in self.llm_providers},
        )

    def llm_provider(self, provider_id: str) -> LlmProvider | None:
        return next((item for item in self.llm_providers or [] if item.id == provider_id), None)

    def _add_legacy_rule_providers(self) -> None:
        """Map old ``model = llm/fallback/...`` rules to provider ids."""
        assert self.llm_providers is not None
        existing = {provider.id for provider in self.llm_providers}
        legacy_models = {
            "llm": self.llm,
            "fallback": self.fallback_model or self.llm,
            "agent": self.agent_model or self.llm,
            "entities": self.entities_model or self.llm,
        }
        for rule in self.extractor_rules:
            if str(rule.kind).strip().lower() != "llm" or rule.provider.strip():
                continue
            legacy_name = rule.model.strip().lower() or "llm"
            provider_id = LEGACY_LLM_PROVIDER_IDS.get(legacy_name, legacy_name)
            rule.provider = provider_id
            if provider_id not in existing and legacy_name in legacy_models:
                self.llm_providers.append(_provider_from_model(
                    provider_id,
                    {
                        "llm": "默认 LLM",
                        "fallback": "旧版提取兜底",
                        "agent": "旧版检索 Agent",
                        "entities": "旧版实体抽取",
                    }[legacy_name],
                    legacy_models[legacy_name],
                ))
                existing.add(provider_id)


def _normalise_extension(value: object) -> str:
    extension = str(value).strip().lower()
    if not extension:
        raise ValueError("提取器扩展名不能为空")
    if not extension.startswith("."):
        extension = f".{extension}"
    if extension == "." or any(char.isspace() for char in extension):
        raise ValueError(f"无效的提取器扩展名: {value}")
    return extension


def _normalise_identifier(value: object, label: str) -> str:
    identifier = str(value).strip()
    if not identifier or not all(char.isalnum() or char in {"_", "-"} for char in identifier):
        raise ValueError(f"{label} 只能包含字母、数字、下划线或连字符: {value}")
    return identifier


def _normalise_plugin_reference(value: object) -> str:
    reference = str(value).strip()
    path = Path(reference)
    valid_legacy = path.name == reference and path.suffix.lower() == ".py"
    valid_folder = path.name == reference and "." not in reference
    if not reference or not (valid_legacy or valid_folder):
        raise ValueError(
            "Python 插件必须是插件文件夹名（其中包含 plugin.py），或旧版单个 .py 文件名"
        )
    return reference


def _normalise_rule(rule: ExtractorRule, *, rule_id: str, label: str,
                    provider_ids: set[str]) -> ExtractorRule:
    kind = str(rule.kind).strip().lower()
    if kind == "builtin":
        kind = "text"
    if kind not in {"text", "python", "llm"}:
        raise ValueError(f"提取器 {label} 的索引方式必须是 text、llm 或 python")
    extensions: list[str] = []
    for extension in rule.extensions:
        normalized = _normalise_extension(extension)
        if normalized not in extensions:
            extensions.append(normalized)
    normalized_rule = ExtractorRule(
        id=rule_id,
        label=label,
        extensions=extensions,
        kind=kind,
        enabled=bool(rule.enabled),
    )
    if kind == "python":
        normalized_rule.plugin = _normalise_plugin_reference(rule.plugin or rule.script)
        if normalized_rule.plugin.endswith(".py"):
            normalized_rule.script = normalized_rule.plugin
        normalized_rule.function = rule.function.strip() or "extract"
        if not normalized_rule.function.isidentifier():
            raise ValueError(f"提取器 {label} 的 function 必须是 Python 函数名")
    elif kind == "llm":
        provider = rule.provider.strip()
        if not provider or provider not in provider_ids:
            raise ValueError(f"提取器 {label} 引用了不存在的 LLM 供应商: {provider or '（空）'}")
        input_mode = rule.input_mode.strip().lower() or "text"
        if input_mode not in {"text", "image"}:
            raise ValueError(f"提取器 {label} 的 input_mode 必须是 text 或 image")
        if input_mode == "image":
            unsupported = [
                extension for extension in extensions
                if extension not in SUPPORTED_IMAGE_EXTENSIONS
            ]
            if unsupported:
                raise ValueError(
                    f"提取器 {label}: {image_format_error(unsupported[0])}"
                )
        normalized_rule.provider = provider
        normalized_rule.model = {
            provider_id: legacy_name
            for legacy_name, provider_id in LEGACY_LLM_PROVIDER_IDS.items()
        }.get(provider, provider)
        normalized_rule.input_mode = input_mode
        normalized_rule.prompt = rule.prompt.strip()
    return normalized_rule


def _merge_builtin_extractor_rules(
    rules: list[ExtractorRule], provider_ids: set[str],
) -> list[ExtractorRule]:
    """Normalize configured rules and guarantee every built-in remains present.

    A built-in row is identified by its stable id, not its current route.  That
    lets a user change ``pdf`` (for example) from ``builtin`` to ``llm`` or
    ``python`` without making it deletable or losing it on the next save.
    """
    builtin_ids = {rule.id for rule in BUILTIN_EXTRACTOR_RULES}
    supplied: dict[str, ExtractorRule] = {}
    for rule in rules:
        if rule.id not in builtin_ids:
            continue
        if rule.id in supplied:
            raise ValueError(f"内置提取器 ID 重复: {rule.id}")
        supplied[rule.id] = rule
    merged: list[ExtractorRule] = []
    for default in BUILTIN_EXTRACTOR_RULES:
        rule = supplied.get(default.id, default)
        merged.append(_normalise_rule(
            rule, rule_id=default.id, label=default.label, provider_ids=provider_ids,
        ))
    seen_ids = {rule.id for rule in merged}
    for rule in rules:
        if rule.id in builtin_ids:
            continue
        rule_id = _normalise_identifier(rule.id, "自定义提取器 ID")
        if rule_id in seen_ids:
            raise ValueError(f"自定义提取器 ID 无效或重复: {rule.id}")
        normalized_rule = _normalise_rule(
            rule,
            rule_id=rule_id,
            label=rule.label.strip() or rule_id,
            provider_ids=provider_ids,
        )
        if not normalized_rule.extensions:
            raise ValueError(f"自定义提取器 {rule_id} 至少需要一个扩展名")
        merged.append(normalized_rule)
        seen_ids.add(rule_id)
    enabled_extensions: dict[str, str] = {}
    for rule in merged:
        if not rule.enabled:
            continue
        for extension in rule.extensions:
            owner = enabled_extensions.get(extension)
            if owner is not None:
                raise ValueError(
                    f"扩展名 {extension} 同时出现在已启用规则“{owner}”和“{rule.label}”中"
                )
            enabled_extensions[extension] = rule.label
    return merged


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


def _provider_from_model(provider_id: str, name: str, cfg: ModelCfg) -> LlmProvider:
    return LlmProvider(
        id=provider_id,
        name=name,
        enabled=cfg.enabled,
        mode=cfg.mode,
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        model=cfg.model,
        local_model=cfg.local_model,
        local_model_dir=cfg.local_model_dir,
    )


def _normalise_llm_providers(providers: list[LlmProvider]) -> list[LlmProvider]:
    result: list[LlmProvider] = []
    seen: set[str] = set()
    for provider in providers:
        provider_id = _normalise_identifier(provider.id, "LLM 供应商 ID")
        if provider_id in seen:
            raise ValueError(f"LLM 供应商 ID 重复: {provider_id}")
        mode = str(provider.mode).strip().lower()
        if mode in {"openai_compatible", "api", "openai-compatible"}:
            mode = "openai"
        if mode not in {"openai", "local"}:
            raise ValueError(f"LLM 供应商 {provider_id} 的 mode 必须是 openai 或 local")
        local_model = str(provider.local_model).strip()
        if provider.enabled and mode == "local" and not local_model:
            raise ValueError(f"LLM 供应商 {provider_id} 启用本地模式时必须配置 local_model")
        base_url = str(provider.base_url).strip().rstrip("/")
        model = str(provider.model).strip()
        if provider.enabled and mode == "openai" and (not base_url or not model):
            raise ValueError(f"LLM 供应商 {provider_id} 启用云端/API 模式时必须配置 base_url 和 model")
        result.append(LlmProvider(
            id=provider_id,
            name=str(provider.name).strip() or provider_id,
            enabled=bool(provider.enabled),
            mode=mode,
            base_url=base_url,
            api_key=str(provider.api_key),
            model=model,
            local_model=local_model,
            local_model_dir=provider.local_model_dir,
        ))
        seen.add(provider_id)
    return result


def _llm_provider_cfg(d: dict, section: str) -> LlmProvider:
    model_cfg = _model_cfg(d, section)
    return LlmProvider(
        id=_normalise_identifier(d.get("id", ""), f"[{section}] id"),
        name=_string(d, "name", str(d.get("id", "")), section).strip(),
        enabled=model_cfg.enabled,
        mode=model_cfg.mode,
        base_url=model_cfg.base_url,
        api_key=model_cfg.api_key,
        model=model_cfg.model,
        local_model=model_cfg.local_model,
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
    rag = _section(data, "rag")

    legacy_llm = _model_cfg(_section(models, "llm"), "models.llm")
    raw_provider_section = data.get("llm_providers")
    if raw_provider_section is None:
        llm_providers: list[LlmProvider] = [
            _provider_from_model(DEFAULT_LLM_PROVIDER_ID, "默认 LLM", legacy_llm)
        ]
    else:
        if isinstance(raw_provider_section, list):
            # Accept the early ``[[llm_providers]]`` draft for compatibility.
            raw_providers = raw_provider_section
        elif isinstance(raw_provider_section, dict):
            raw_providers = raw_provider_section.get("items", [])
        else:
            raise ValueError("[llm_providers] 必须是 TOML 小节")
        if not isinstance(raw_providers, list):
            raise ValueError("[[llm_providers.items]] 必须是 TOML 小节列表")
        llm_providers = []
        for position, raw_provider in enumerate(raw_providers, start=1):
            if not isinstance(raw_provider, dict):
                raise ValueError("[[llm_providers.items]] 必须是 TOML 小节")
            llm_providers.append(_llm_provider_cfg(
                raw_provider, f"llm_providers.items#{position}",
            ))

    chunk_size = max(1, int(chunking.get("chunk_size", 800)))
    chunk_overlap = max(0, int(chunking.get("chunk_overlap", 100)))
    if chunk_overlap >= chunk_size:
        raise ValueError("[chunking] chunk_overlap 必须小于 chunk_size")

    raw_rules = extractors.get("rules", [])
    if not isinstance(raw_rules, list):
        raise ValueError("[extractors] rules 必须是列表")
    extractor_rules: list[ExtractorRule] = []
    legacy_rules: list[ScriptRule] = []
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            raise ValueError("[[extractors.rules]] 必须是 TOML 小节")
        # The historical ``match`` + executable ``script`` form stays valid.
        if "kind" not in raw_rule and raw_rule.get("match") and raw_rule.get("script"):
            legacy_rules.append(ScriptRule(
                match=str(raw_rule["match"]), script=str(Path(raw_rule["script"]).expanduser())
            ))
            continue
        kind = str(raw_rule.get("kind", "")).strip().lower()
        if kind == "builtin":
            kind = "text"
        rule_id = str(raw_rule.get("id", "")).strip()
        if kind not in {"text", "python", "llm"} or not rule_id:
            raise ValueError("[[extractors.rules]] 需要有效的 kind（text/llm/python）和 id")
        raw_extensions = raw_rule.get("extensions", [])
        if not isinstance(raw_extensions, list):
            raise ValueError("[[extractors.rules]] extensions 必须是列表")
        extractor_rules.append(ExtractorRule(
            id=rule_id,
            label=str(raw_rule.get("label", rule_id)),
            extensions=[_normalise_extension(value) for value in raw_extensions],
            kind=kind,
            enabled=bool(raw_rule.get("enabled", True)),
            provider=str(raw_rule.get("provider", "")),
            input_mode=str(raw_rule.get("input_mode", "text")),
            prompt=str(raw_rule.get("prompt", "")),
            plugin=str(raw_rule.get("plugin", "")),
            function=str(raw_rule.get("function", "extract")),
            script=str(raw_rule.get("script", "")),
            model=str(raw_rule.get("model", "llm")),
        ))

    llm = legacy_llm
    provider_default_model = llm_providers[0].model_config() if llm_providers else llm
    agent_model = (
        _model_cfg(_section(models, "agent"), "models.agent")
        if "agent" in models else _copy_model_cfg(provider_default_model)
    )
    entities_model = (
        _model_cfg(_section(models, "entities"), "models.entities")
        if "entities" in models else _copy_model_cfg(provider_default_model)
    )
    fallback_model = (
        _model_cfg(_section(models, "fallback"), "models.fallback")
        if "fallback" in models else _copy_model_cfg(provider_default_model)
    )
    db_path = _storage_path(storage.get("db_path"), DEFAULT_DB_PATH, cfg_path.parent)
    temp_dir = _storage_path(storage.get("temp_dir"), DEFAULT_TEMP_DIR, cfg_path.parent)
    model_dir = _storage_path(storage.get("model_dir"), DEFAULT_MODEL_DIR, cfg_path.parent)
    extractor_dir = _storage_path(
        extractors.get("plugin_dir", extractors.get("custom_dir")),
        DEFAULT_EXTRACTOR_DIR,
        cfg_path.parent,
    )
    ocr_cfg = _ocr_cfg(ocr)
    ocr_cfg.temp_dir = temp_dir

    config = Config(
        db_path=db_path,
        temp_dir=temp_dir,
        model_dir=model_dir,
        folders=[Path(p).expanduser() for p in watch.get("folders", [])],
        exclude=list(watch.get("exclude", DEFAULT_EXCLUDE)),
        max_file_mb=max(0, int(watch.get("max_file_mb", 50))),
        llm=llm,
        llm_providers=llm_providers,
        agent_model=agent_model,
        entities_model=entities_model,
        fallback_model=fallback_model,
        vision=_model_cfg(_section(models, "vision"), "models.vision"),
        embedding=_model_cfg(_section(models, "embedding"), "models.embedding"),
        extractor_dir=extractor_dir,
        extractor_rules=extractor_rules or default_extractor_rules(),
        script_rules=legacy_rules,
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
            enabled=bool(agent.get("enabled", True)),
            max_steps=max(1, int(agent.get("max_steps", 6))),
            max_results=max(1, int(agent.get("max_results", 12))),
        ),
        rag=RagCfg(
            enabled=bool(rag.get("enabled", True)),
            max_context_chunks=max(1, int(rag.get("max_context_chunks", 8))),
        ),
        agent_fallback=AgentFallbackCfg(
            enabled=bool(agent_fallback.get("enabled", False)),
            max_bytes=max(1_024, int(agent_fallback.get("max_bytes", 262_144))),
        ),
        config_path=cfg_path,
    )
    # Loading an older configuration is the migration point: shipped plugins
    # become real, inspectable folders without overwriting user content.
    from .extractors.script import seed_shipped_plugins

    seed_shipped_plugins(config.extractor_dir)
    return config


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
    extractor_dir = cfg_path.parent / "extractors"
    extractor_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(extractor_dir, 0o700)
    except OSError:
        pass
    from .extractors.script import seed_shipped_plugins

    seed_shipped_plugins(extractor_dir)
    return cfg_path
