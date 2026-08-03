"""Web-facing configuration serialization and atomic persistence.

The runtime configuration is TOML so the CLI stays easy to use by hand.  This
module presents the same values as JSON to the local settings page without
ever returning secrets, then validates a complete replacement before atomically
writing the TOML file.
"""
from __future__ import annotations

import math
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from .config import (
    AgentCfg,
    AgentFallbackCfg,
    AsrCfg,
    Config,
    EntityCfg,
    ModelCfg,
    OcrCfg,
    ScriptRule,
    load_config,
)
from .db import Database


MODEL_FIELDS: dict[str, str] = {
    "llm": "llm",
    "agent": "agent_model",
    "entities": "entities_model",
    "fallback": "fallback_model",
    "vision": "vision",
    "embedding": "embedding",
}


def _model_to_dict(cfg: ModelCfg) -> dict[str, Any]:
    """Return a connection description without exposing its API key."""
    return {
        "enabled": cfg.enabled,
        "mode": cfg.mode,
        "base_url": cfg.base_url,
        "model": cfg.model,
        "local_model": cfg.local_model,
        "api_key_configured": bool(cfg.api_key),
    }


def settings_dict(config: Config) -> dict[str, Any]:
    """Build the stable JSON contract consumed by ``settings.html``."""
    models: dict[str, dict[str, Any]] = {}
    for name, attr in MODEL_FIELDS.items():
        model = getattr(config, attr)
        # Config.__post_init__ supplies purpose models for programmatic users.
        models[name] = _model_to_dict(model if model is not None else config.llm)

    return {
        "db_path": str(config.db_path),
        "temp_dir": str(config.temp_dir),
        "model_dir": str(config.model_dir),
        "folders": [str(folder) for folder in config.folders],
        "exclude": list(config.exclude),
        "max_file_mb": config.max_file_mb,
        "watch_debounce_sec": config.watch_debounce_sec,
        "watch_reconcile_sec": config.watch_reconcile_sec,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "models": models,
        "ocr": {
            "enabled": config.ocr.enabled,
            "provider": config.ocr.provider,
            "command": config.ocr.command,
            "pdf_renderer": config.ocr.pdf_renderer,
            "languages": config.ocr.languages,
            "dpi": config.ocr.dpi,
            "endpoint": config.ocr.endpoint,
            "response_path": config.ocr.response_path,
            "timeout_sec": config.ocr.timeout_sec,
            "api_key_configured": bool(config.ocr.api_key),
        },
        "asr": {
            "enabled": config.asr.enabled,
            "provider": config.asr.provider,
            "model": config.asr.model,
            "local_model": config.asr.local_model,
            "local_backend": config.asr.local_backend,
            "device": config.asr.device,
            "compute_type": config.asr.compute_type,
            "base_url": config.asr.base_url,
            "endpoint": config.asr.endpoint,
            "language": config.asr.language,
            "response_path": config.asr.response_path,
            "timeout_sec": config.asr.timeout_sec,
            "api_key_configured": bool(config.asr.api_key),
        },
        "entities": {
            "enabled": config.entities.enabled,
            "max_chars": config.entities.max_chars,
            "max_per_file": config.entities.max_per_file,
        },
        "agent": {
            "max_steps": config.agent.max_steps,
            "max_results": config.agent.max_results,
        },
        "agent_fallback": {
            "enabled": config.agent_fallback.enabled,
            "max_bytes": config.agent_fallback.max_bytes,
        },
        "extractors": {
            "rules": [
                {"match": rule.match, "script": rule.script}
                for rule in config.script_rules
            ]
        },
        "config_path": str(config.config_path) if config.config_path else None,
    }


def _mapping(value: object, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} 必须是对象")
    return value


def _string(value: object, label: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} 必须是文本")
    value = value.strip()
    if not allow_empty and not value:
        raise ValueError(f"{label} 不能为空")
    return value


def _bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} 必须是开关值")
    return value


def _integer(value: object, label: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} 必须是整数")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是整数") from exc
    if result < minimum:
        raise ValueError(f"{label} 不能小于 {minimum}")
    return result


def _number(value: object, label: str, minimum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} 必须是数字")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是数字") from exc
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{label} 不能小于 {minimum}")
    return result


def _setting(part: dict[str, Any], key: str, current: Any) -> Any:
    return part[key] if key in part else current


def _path(
    value: object,
    label: str,
    *,
    directory: bool = False,
    base_dir: Path | None = None,
) -> Path:
    text = _string(value, label, allow_empty=False)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = ((base_dir or Path.cwd()) / path).resolve(strict=False)
    if directory and not path.is_dir():
        raise ValueError(f"{label} 不存在或不是目录: {path}")
    return path


def _validate_database_path(path: Path) -> None:
    """Ensure a saved database location can be opened before it goes live."""
    if path.exists() and path.is_dir():
        raise ValueError(f"索引数据库不能是目录: {path}")

    database: Database | None = None
    try:
        # Opening the actual location verifies its parent, SQLite support and
        # access permissions.  Creating a fresh shadow database here is also
        # intentional: a successful save must not leave /api/status pointing
        # to a location that only fails on its next request.
        database = Database(path)
    except (OSError, RuntimeError, sqlite3.Error) as exc:
        raise ValueError(f"无法打开索引数据库 {path}: {exc}") from exc
    finally:
        if database is not None:
            database.close()


def _folder_list(value: object) -> list[Path]:
    if not isinstance(value, list):
        raise ValueError("索引目录必须是列表")
    folders: list[Path] = []
    seen: set[str] = set()
    for raw in value:
        path = _path(raw, "索引目录", directory=True)
        key = str(path)
        if key not in seen:
            folders.append(path)
            seen.add(key)
    return folders


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} 必须是列表")
    result: list[str] = []
    seen: set[str] = set()
    for raw in value:
        item = _string(raw, label, allow_empty=False)
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _model_from_payload(current: ModelCfg, value: object, label: str) -> ModelCfg:
    data = _mapping(value, label)
    enabled = _bool(_setting(data, "enabled", current.enabled), f"{label}.enabled")
    raw_mode = _setting(data, "mode", _setting(data, "provider", current.mode))
    mode = _string(raw_mode, f"{label}.mode", allow_empty=False).lower()
    if mode in {"openai_compatible", "api", "openai-compatible"}:
        mode = "openai"
    if mode not in {"openai", "local"}:
        raise ValueError(f"{label}.mode 必须是 openai 或 local")
    local_model = _string(
        _setting(data, "local_model", current.local_model),
        f"{label}.local_model",
    )
    if enabled and mode == "local" and not local_model:
        raise ValueError(f"{label}.local_model 不能为空")
    base_url = _string(
        _setting(data, "base_url", current.base_url),
        f"{label}.base_url",
        allow_empty=not enabled or mode == "local",
    ).rstrip("/")
    model = _string(
        _setting(data, "model", current.model),
        f"{label}.model",
        allow_empty=not enabled or mode == "local",
    )
    api_key = current.api_key
    if "api_key" in data:
        candidate = _string(data["api_key"], f"{label}.api_key")
        if candidate:
            api_key = candidate
    if data.get("clear_api_key") is True:
        api_key = ""
    return ModelCfg(
        enabled=enabled,
        base_url=base_url,
        api_key=api_key,
        model=model,
        mode=mode,
        local_model=local_model,
    )


def _build_config(current: Config, payload: object) -> Config:
    data = _mapping(payload, "settings")
    models = _mapping(data.get("models"), "models")
    ocr = _mapping(data.get("ocr"), "ocr")
    asr = _mapping(data.get("asr"), "asr")
    entities = _mapping(data.get("entities"), "entities")
    agent = _mapping(data.get("agent"), "agent")
    fallback = _mapping(data.get("agent_fallback"), "agent_fallback")

    config_dir = current.config_path.parent if current.config_path else None
    db_path = _path(
        _setting(data, "db_path", str(current.db_path)),
        "索引数据库",
        base_dir=config_dir,
    )
    folders = _folder_list(_setting(data, "folders", [str(folder) for folder in current.folders]))
    exclude = _string_list(_setting(data, "exclude", current.exclude), "排除规则")
    max_file_mb = _integer(_setting(data, "max_file_mb", current.max_file_mb), "单文件大小上限", 0)
    debounce = _number(_setting(data, "watch_debounce_sec", current.watch_debounce_sec), "事件防抖", 0.1)
    reconcile = _number(_setting(data, "watch_reconcile_sec", current.watch_reconcile_sec), "全量对账间隔", 0)

    chunk_size = _integer(_setting(data, "chunk_size", current.chunk_size), "分块大小", 1)
    chunk_overlap = _integer(_setting(data, "chunk_overlap", current.chunk_overlap), "分块重叠", 0)
    if chunk_overlap >= chunk_size:
        raise ValueError("分块重叠必须小于分块大小")

    current_models: dict[str, ModelCfg] = {}
    for name, attr in MODEL_FIELDS.items():
        source = getattr(current, attr) or current.llm
        current_models[name] = source
    resolved_models = {
        name: _model_from_payload(current_models[name], models.get(name), f"models.{name}")
        for name in MODEL_FIELDS
    }

    ocr_cfg = OcrCfg(
        enabled=_bool(_setting(ocr, "enabled", current.ocr.enabled), "ocr.enabled"),
        provider=_string(_setting(ocr, "provider", current.ocr.provider), "ocr.provider", allow_empty=False).lower(),
        command=_string(_setting(ocr, "command", current.ocr.command), "ocr.command", allow_empty=False),
        pdf_renderer=_string(_setting(ocr, "pdf_renderer", current.ocr.pdf_renderer), "ocr.pdf_renderer", allow_empty=False),
        languages=_string(_setting(ocr, "languages", current.ocr.languages), "ocr.languages", allow_empty=False),
        dpi=_integer(_setting(ocr, "dpi", current.ocr.dpi), "ocr.dpi", 72),
        endpoint=_string(_setting(ocr, "endpoint", current.ocr.endpoint), "ocr.endpoint"),
        api_key=current.ocr.api_key,
        response_path=_string(_setting(ocr, "response_path", current.ocr.response_path), "ocr.response_path", allow_empty=False),
        timeout_sec=_integer(_setting(ocr, "timeout_sec", current.ocr.timeout_sec), "ocr.timeout_sec", 1),
        temp_dir=current.temp_dir,
    )
    if "api_key" in ocr:
        candidate = _string(ocr["api_key"], "ocr.api_key")
        if candidate:
            ocr_cfg.api_key = candidate
    if ocr.get("clear_api_key") is True:
        ocr_cfg.api_key = ""

    asr_provider = _string(
        _setting(asr, "provider", current.asr.provider), "asr.provider", allow_empty=False
    ).lower()
    legacy_asr_provider = asr_provider
    if asr_provider in {
        "faster_whisper",
        "faster-whisper",
        "mlx_whisper",
        "mlx-whisper",
        "whisper_cpp",
        "whisper-cpp",
        "gguf",
    }:
        asr_provider = "local"
    if asr_provider not in {"local", "openai_compatible"}:
        raise ValueError("asr.provider 必须是 local 或 openai_compatible")
    legacy_backend = {
        "mlx_whisper": "mlx_whisper",
        "mlx-whisper": "mlx_whisper",
        "whisper_cpp": "whisper_cpp",
        "whisper-cpp": "whisper_cpp",
        "gguf": "whisper_cpp",
        "faster_whisper": "faster_whisper",
        "faster-whisper": "faster_whisper",
    }.get(legacy_asr_provider, "auto")
    local_backend = _string(
        _setting(asr, "local_backend", current.asr.local_backend or legacy_backend),
        "asr.local_backend",
        allow_empty=False,
    ).lower()
    local_backend = {
        "faster-whisper": "faster_whisper",
        "mlx-whisper": "mlx_whisper",
        "whisper-cpp": "whisper_cpp",
        "gguf": "whisper_cpp",
    }.get(local_backend, local_backend)
    if local_backend not in {"auto", "faster_whisper", "mlx_whisper", "whisper_cpp"}:
        raise ValueError("asr.local_backend 必须是 auto、faster_whisper、mlx_whisper 或 whisper_cpp")
    asr_enabled = _bool(_setting(asr, "enabled", current.asr.enabled), "asr.enabled")
    asr_model = _string(
        _setting(asr, "model", current.asr.model),
        "asr.model",
        allow_empty=not asr_enabled or asr_provider in {"local", "openai_compatible"},
    )
    asr_local_model = _string(
        _setting(asr, "local_model", current.asr.local_model),
        "asr.local_model",
    )
    if not asr_local_model and asr_provider == "local" and legacy_asr_provider != "local":
        # Keep old [asr] provider = faster_whisper / model = "base" configs
        # usable while new configs select an explicit file path.
        asr_local_model = asr_model
    asr_cfg = AsrCfg(
        enabled=asr_enabled,
        provider=asr_provider,
        # OpenAI-compatible transcription endpoints can choose a server-side
        # default model, while faster-whisper always needs a local model id.
        model=asr_model,
        device=_string(_setting(asr, "device", current.asr.device), "asr.device", allow_empty=False),
        compute_type=_string(_setting(asr, "compute_type", current.asr.compute_type), "asr.compute_type", allow_empty=False),
        base_url=_string(
            _setting(asr, "base_url", current.asr.base_url),
            "asr.base_url",
            allow_empty=not asr_enabled or asr_provider == "local",
        ).rstrip("/"),
        endpoint=_string(_setting(asr, "endpoint", current.asr.endpoint), "asr.endpoint"),
        api_key=current.asr.api_key,
        language=_string(_setting(asr, "language", current.asr.language), "asr.language"),
        response_path=_string(_setting(asr, "response_path", current.asr.response_path), "asr.response_path", allow_empty=False),
        timeout_sec=_integer(_setting(asr, "timeout_sec", current.asr.timeout_sec), "asr.timeout_sec", 1),
        local_model=asr_local_model,
        local_backend=local_backend,
    )
    if "api_key" in asr:
        candidate = _string(asr["api_key"], "asr.api_key")
        if candidate:
            asr_cfg.api_key = candidate
    if asr.get("clear_api_key") is True:
        asr_cfg.api_key = ""

    return Config(
        db_path=db_path,
        temp_dir=current.temp_dir,
        model_dir=current.model_dir,
        folders=folders,
        exclude=exclude,
        max_file_mb=max_file_mb,
        llm=resolved_models["llm"],
        agent_model=resolved_models["agent"],
        entities_model=resolved_models["entities"],
        fallback_model=resolved_models["fallback"],
        vision=resolved_models["vision"],
        embedding=resolved_models["embedding"],
        script_rules=list(current.script_rules),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        watch_debounce_sec=debounce,
        watch_reconcile_sec=reconcile,
        ocr=ocr_cfg,
        asr=asr_cfg,
        entities=EntityCfg(
            enabled=_bool(_setting(entities, "enabled", current.entities.enabled), "entities.enabled"),
            max_chars=_integer(_setting(entities, "max_chars", current.entities.max_chars), "entities.max_chars", 500),
            max_per_file=_integer(_setting(entities, "max_per_file", current.entities.max_per_file), "entities.max_per_file", 1),
        ),
        agent=AgentCfg(
            max_steps=_integer(_setting(agent, "max_steps", current.agent.max_steps), "agent.max_steps", 1),
            max_results=_integer(_setting(agent, "max_results", current.agent.max_results), "agent.max_results", 1),
        ),
        agent_fallback=AgentFallbackCfg(
            enabled=_bool(_setting(fallback, "enabled", current.agent_fallback.enabled), "agent_fallback.enabled"),
            max_bytes=_integer(_setting(fallback, "max_bytes", current.agent_fallback.max_bytes), "agent_fallback.max_bytes", 1_024),
        ),
        config_path=current.config_path,
    )


def _toml_string(value: str) -> str:
    # JSON's string escaping is valid TOML basic-string escaping for the
    # values Semdex writes, including Chinese paths and API keys.
    import json

    return json.dumps(value, ensure_ascii=False)


def _toml_bool(value: bool) -> str:
    return "true" if value else "false"


def _toml_list(values: list[str]) -> str:
    return "[" + ", ".join(_toml_string(item) for item in values) + "]"


def _storage_path_for_toml(path: Path, config: Config) -> str:
    """Keep managed storage relative so a copied bundle remains self-contained."""
    if config.config_path is None:
        return str(path)
    try:
        return str(path.resolve(strict=False).relative_to(config.config_path.parent.resolve(strict=False)))
    except ValueError:
        return str(path)


def _model_toml(name: str, cfg: ModelCfg) -> list[str]:
    return [
        f"[models.{name}]",
        f"enabled = {_toml_bool(cfg.enabled)}",
        f"mode = {_toml_string(cfg.mode)}",
        f"base_url = {_toml_string(cfg.base_url)}",
        f"api_key = {_toml_string(cfg.api_key)}",
        f"model = {_toml_string(cfg.model)}",
        f"local_model = {_toml_string(cfg.local_model)}",
        "",
    ]


def _to_toml(config: Config) -> str:
    lines = [
        "# Semdex 配置文件。此文件可由网页设置页或手工编辑。",
        "",
        "[storage]",
        f"db_path = {_toml_string(_storage_path_for_toml(config.db_path, config))}",
        f"temp_dir = {_toml_string(_storage_path_for_toml(config.temp_dir, config))}",
        f"model_dir = {_toml_string(_storage_path_for_toml(config.model_dir, config))}",
        "",
        "[watch]",
        f"folders = {_toml_list([str(folder) for folder in config.folders])}",
        f"exclude = {_toml_list(config.exclude)}",
        f"max_file_mb = {config.max_file_mb}",
        f"debounce_sec = {config.watch_debounce_sec:g}",
        f"reconcile_sec = {config.watch_reconcile_sec:g}",
        "",
    ]
    for name, attr in MODEL_FIELDS.items():
        model = getattr(config, attr) or config.llm
        lines.extend(_model_toml(name, model))
    lines.extend([
        "[chunking]",
        f"chunk_size = {config.chunk_size}",
        f"chunk_overlap = {config.chunk_overlap}",
        "",
        "[ocr]",
        f"enabled = {_toml_bool(config.ocr.enabled)}",
        f"provider = {_toml_string(config.ocr.provider)}",
        f"command = {_toml_string(config.ocr.command)}",
        f"pdf_renderer = {_toml_string(config.ocr.pdf_renderer)}",
        f"languages = {_toml_string(config.ocr.languages)}",
        f"dpi = {config.ocr.dpi}",
        f"endpoint = {_toml_string(config.ocr.endpoint)}",
        f"api_key = {_toml_string(config.ocr.api_key)}",
        f"response_path = {_toml_string(config.ocr.response_path)}",
        f"timeout_sec = {config.ocr.timeout_sec}",
        "",
        "[asr]",
        f"enabled = {_toml_bool(config.asr.enabled)}",
        f"provider = {_toml_string(config.asr.provider)}",
        f"local_backend = {_toml_string(config.asr.local_backend)}",
        f"local_model = {_toml_string(config.asr.local_model)}",
        f"model = {_toml_string(config.asr.model)}",
        f"device = {_toml_string(config.asr.device)}",
        f"compute_type = {_toml_string(config.asr.compute_type)}",
        f"base_url = {_toml_string(config.asr.base_url)}",
        f"endpoint = {_toml_string(config.asr.endpoint)}",
        f"api_key = {_toml_string(config.asr.api_key)}",
        f"language = {_toml_string(config.asr.language)}",
        f"response_path = {_toml_string(config.asr.response_path)}",
        f"timeout_sec = {config.asr.timeout_sec}",
        "",
        "[entities]",
        f"enabled = {_toml_bool(config.entities.enabled)}",
        f"max_chars = {config.entities.max_chars}",
        f"max_per_file = {config.entities.max_per_file}",
        "",
        "[agent]",
        f"max_steps = {config.agent.max_steps}",
        f"max_results = {config.agent.max_results}",
        "",
        "[agent_fallback]",
        f"enabled = {_toml_bool(config.agent_fallback.enabled)}",
        f"max_bytes = {config.agent_fallback.max_bytes}",
        "",
    ])
    for rule in config.script_rules:
        lines.extend([
            "[[extractors.rules]]",
            f"match = {_toml_string(rule.match)}",
            f"script = {_toml_string(rule.script)}",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def save_settings(current: Config, payload: object) -> Config:
    """Validate and atomically save settings, returning the reloaded config."""
    if current.config_path is None:
        raise ValueError("当前服务没有关联配置文件，不能从网页保存设置")

    candidate = _build_config(current, payload)
    _validate_database_path(candidate.db_path)
    config_path = current.config_path.expanduser()
    config_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    text = _to_toml(candidate)

    # Validate exactly the form that will hit disk before replacing the live
    # configuration.  A temporary file in the same directory lets os.replace
    # stay atomic even when the state directory is on another volume.
    fd, temp_name = tempfile.mkstemp(prefix=f".{config_path.name}.", suffix=".tmp", dir=config_path.parent)
    temp_path = Path(temp_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        load_config(temp_path)
        os.replace(temp_path, config_path)
        try:
            os.chmod(config_path, 0o600)
        except OSError:
            pass
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return load_config(config_path)
