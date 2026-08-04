"""索引管线：提取正文 → 写 contents/FTS → （启用 embedding 时）分块向量化。"""
from __future__ import annotations

from array import array
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from collections.abc import Iterator
from collections.abc import Callable
from typing import Any

from .chunker import chunk_text
from .config import Config
from .db import Database
from .extractors import ExtractContext, resolve
from .extractors.script import PLUGIN_FILENAME, resolve_python_plugin
from .modelclient import ModelClient, embedding_identity
from .paths import ensure_private_directory
from .safepath import configured_watch_roots, open_regular_file_beneath_root
from .models import (
    STATUS_DONE, STATUS_FAILED, STATUS_PENDING, STATUS_SKIPPED,
    STATUS_WAITING_CAPABILITY, STATUS_WAITING_MODEL, CapabilityNotConfigured,
    CapabilityUnavailable, ExtractError, IndexStats,
    ModelNotConfigured, ModelUnavailable,
)

MAX_TEXT_CHARS = 1_000_000
EMBED_BATCH = 16
PRIMARY_IDENTITY_KEY = "primary_index_identity"
EMBEDDING_IDENTITY_KEY = "embedding_index_identity"
ENTITY_IDENTITY_KEY = "entity_index_identity"


def _vec_to_blob(vec: list[float]) -> bytes:
    return array("f", vec).tobytes()


def _identity_value(value: Any) -> Any:
    """Convert runtime configuration into deterministic JSON-compatible data."""
    if isinstance(value, Path):
        return str(value.expanduser().resolve(strict=False))
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _identity_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _identity_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_identity_value(item) for item in value]
    return value


def _identity_hash(payload: object) -> str:
    canonical = json.dumps(
        _identity_value(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _plugin_source_identity(path: Path) -> dict[str, str]:
    canonical = str(path.expanduser().resolve(strict=False))
    try:
        source = path.read_bytes()
    except FileNotFoundError:
        return {"path": canonical, "state": "missing"}
    except (OSError, ValueError) as exc:
        return {
            "path": canonical,
            "state": "unreadable",
            "error": type(exc).__name__,
        }
    return {
        "path": canonical,
        "state": "file",
        "sha256": hashlib.sha256(source).hexdigest(),
    }


def _plugin_runtime_identity(config: Config) -> dict[str, object]:
    """Fingerprint discoverable plugin entrypoints and every configured target."""
    root = config.extractor_dir.expanduser().resolve(strict=False)
    inventory: list[dict[str, object]] = []
    try:
        entries = sorted(root.iterdir(), key=lambda item: item.name.casefold())
    except FileNotFoundError:
        entries = []
        inventory.append({"state": "directory-missing"})
    except OSError as exc:
        entries = []
        inventory.append({"state": "directory-unreadable", "error": type(exc).__name__})

    for entry in entries:
        if entry.name.startswith(".") or entry.name == "__pycache__":
            continue
        if entry.is_dir():
            inventory.append({
                "reference": entry.name,
                "kind": "folder",
                "source": _plugin_source_identity(entry / PLUGIN_FILENAME),
            })
        elif entry.is_file() and entry.suffix.lower() == ".py" and entry.name != "__init__.py":
            inventory.append({
                "reference": entry.name,
                "kind": "legacy-file",
                "source": _plugin_source_identity(entry),
            })

    configured: list[dict[str, object]] = []
    for position, rule in enumerate(config.extractor_rules):
        if rule.kind != "python":
            continue
        try:
            source = resolve_python_plugin(config.extractor_dir, rule.plugin)
        except Exception as exc:
            configured.append({
                "position": position,
                "rule": rule.id,
                "plugin": rule.plugin,
                "state": "invalid-reference",
                "error": type(exc).__name__,
            })
            continue
        configured.append({
            "position": position,
            "rule": rule.id,
            "plugin": rule.plugin,
            "function": rule.function,
            "source": _plugin_source_identity(source),
        })

    return {
        "directory": str(root),
        "inventory": inventory,
        "configured": configured,
    }


def primary_index_identity(config: Config) -> str:
    """Identify every configured input that can change primary indexed text."""
    return _identity_hash({
        "schema": 1,
        "rules": config.extractor_rules,
        "legacy_script_rules": config.script_rules,
        "llm_providers": config.llm_providers or [],
        "legacy_llm": config.llm,
        "plugins": _plugin_runtime_identity(config),
        "ocr": config.ocr,
        "asr": config.asr,
    })


def embedding_index_identity(config: Config) -> str:
    """Identify the complete chunking and vector-space contract."""
    return _identity_hash({
        "schema": 1,
        "enabled": config.embedding.enabled,
        "model": embedding_identity(config.embedding),
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
    })


def entity_index_identity(config: Config) -> str:
    """Identify the entity model and extraction limits independently of primary text."""
    return _identity_hash({
        "schema": 1,
        "settings": config.entities,
        "model": config.entities_model,
    })


def synchronize_index_state(db: Database, config: Config, log=print) -> bool:
    """Invalidate stale primary and derived indexes for the active configuration.

    The operation is idempotent. Identities are opaque hashes, so credentials
    contribute to invalidation without ever being persisted in plaintext.
    """
    current_primary = primary_index_identity(config)
    stored_primary = db.meta_get(PRIMARY_IDENTITY_KEY)
    primary_requeued = False
    if stored_primary is None:
        if db.has_indexed_content():
            count = db.requeue_all_for_full_reindex(
                invalidate_embeddings=config.embedding.enabled,
            )
            primary_requeued = True
            log(f"⚠ 一级索引配置首次建立身份，已重排 {count} 个已有文件")
        db.meta_set(PRIMARY_IDENTITY_KEY, current_primary)
    elif stored_primary != current_primary:
        count = db.requeue_all_for_full_reindex(
            invalidate_embeddings=config.embedding.enabled,
        )
        primary_requeued = True
        db.meta_set(PRIMARY_IDENTITY_KEY, current_primary)
        log(f"↻ 一级索引规则、LLM 或插件已变化，已重排 {count} 个文件")

    current_embedding = embedding_index_identity(config)
    stored_embedding = db.meta_get(EMBEDDING_IDENTITY_KEY)
    rebuild_required = db.meta_get("embedding_rebuild_required") == "1"
    if not config.embedding.enabled:
        if stored_embedding != current_embedding or rebuild_required:
            db.clear_chunks()
            db.meta_delete("embedding_rebuild_required")
            db.meta_delete("embedding_model")
            db.meta_delete("embedding_model_id")
            db.meta_set(EMBEDDING_IDENTITY_KEY, current_embedding)
        rebuild_required = False
    elif stored_embedding is None:
        if db.has_indexed_content() or rebuild_required or primary_requeued:
            db.clear_chunks()
            db.require_embedding_rebuild()
            rebuild_required = True
        else:
            db.meta_set(EMBEDDING_IDENTITY_KEY, current_embedding)
    elif stored_embedding != current_embedding:
        db.clear_chunks()
        db.require_embedding_rebuild()
        rebuild_required = True
        log("↻ Embedding 模型或分块参数已变化，将在本次索引中自动重建向量")

    current_entity = entity_index_identity(config)
    stored_entity = db.meta_get(ENTITY_IDENTITY_KEY)
    if not config.entities.enabled:
        if stored_entity != current_entity:
            db.clear_entities()
            db.meta_set(ENTITY_IDENTITY_KEY, current_entity)
    elif stored_entity != current_entity:
        db.clear_entities()
        db.meta_set(ENTITY_IDENTITY_KEY, current_entity)
        if stored_entity is not None:
            log("↻ 实体抽取模型或参数已变化，将重新抽取已有正文")

    return rebuild_required


def _configured_watch_roots(config: Config) -> list[Path]:
    return configured_watch_roots(config)


def _validated_source_path(path: Path, config: Config) -> tuple[Path | None, str | None]:
    """Return a canonical in-scope path before opening a trusted snapshot.

    Scanner rows can outlive a filesystem change.  In particular, a regular
    file can become a symlink after scanning.  The snapshot helper below
    repeats the containment check atomically while opening the file.
    """
    if path.is_symlink():
        return None, "拒绝符号链接，避免读取监控目录外的内容"

    try:
        resolved_path = path.resolve(strict=True)
    except FileNotFoundError:
        return None, None
    except (OSError, RuntimeError) as e:
        return None, f"无法验证索引路径: {e}"

    watch_roots = _configured_watch_roots(config)
    if not any(resolved_path.is_relative_to(root) for root in watch_roots):
        return None, "拒绝监控目录外的路径"
    return resolved_path, None


@contextmanager
def _trusted_source_snapshot(path: Path, config: Config) -> Iterator[tuple[Path | None, str | None]]:
    """Yield a private, suffix-preserving copy acquired through a trusted FD."""
    roots = [root for root in _configured_watch_roots(config) if path.is_relative_to(root)]
    if not roots:
        yield None, "拒绝监控目录外的路径"
        return

    root = max(roots, key=lambda item: len(item.parts))
    try:
        source_fd = open_regular_file_beneath_root(path, root)
    except FileNotFoundError:
        yield None, None
        return
    except OSError as e:
        yield None, f"拒绝不安全的索引路径: {e}"
        return

    max_bytes = config.max_file_mb * 1024 * 1024
    try:
        temp_dir = tempfile.TemporaryDirectory(
            prefix="semdex-source-",
            dir=str(ensure_private_directory(config.temp_dir)),
        )
    except OSError as e:
        yield None, f"无法创建安全读取快照: {e}"
        os.close(source_fd)
        return

    try:
        with temp_dir as tmp:
            snapshot = Path(tmp) / path.name
            total = 0
            try:
                source = os.fdopen(source_fd, "rb", closefd=True)
                source_fd = -1
                with source, snapshot.open("xb") as target:
                    while chunk := source.read(1 << 20):
                        total += len(chunk)
                        if total > max_bytes:
                            raise OSError("文件在提取前超过当前 max_file_mb 限制")
                        target.write(chunk)
            except OSError as e:
                yield None, f"无法创建安全读取快照: {e}"
                return
            yield snapshot, None
    finally:
        if source_fd != -1:
            os.close(source_fd)


def _embed_file(db: Database, config: Config, emb: ModelClient,
                file_id: int, text: str) -> int | None:
    """分块并向量化一个文件的正文，写入 chunks。失败向上抛。"""
    pieces = chunk_text(text, config.chunk_size, config.chunk_overlap)
    if not pieces:
        db.replace_chunks(file_id, [])
        return None
    blobs: list[bytes] = []
    dimension: int | None = None
    for i in range(0, len(pieces), EMBED_BATCH):
        vectors = emb.embed(pieces[i:i + EMBED_BATCH])
        if len(vectors) != len(pieces[i:i + EMBED_BATCH]):
            raise ValueError("embedding 服务返回的向量数量与输入不一致")
        for vector in vectors:
            if not vector:
                raise ValueError("embedding 服务返回了空向量")
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise ValueError("embedding 服务返回的向量维度不一致")
            blobs.append(_vec_to_blob(vector))
    db.replace_chunks(file_id, list(zip(pieces, blobs)))
    return dimension


def _require_embedding_rebuild(db: Database) -> None:
    """Invalidate vector metadata and retain a marker until full rebuild succeeds."""
    db.require_embedding_rebuild()


def _embedding_identity(config: Config) -> str:
    """Identify the vector space, not just its display model name."""
    return embedding_identity(config.embedding)


def _set_embedding_identity(db: Database, config: Config) -> None:
    db.meta_set("embedding_model", config.embedding.model)
    db.meta_set("embedding_model_id", _embedding_identity(config))
    db.meta_set(EMBEDDING_IDENTITY_KEY, embedding_index_identity(config))


def check_embedding_model(db: Database, config: Config, log=print) -> bool:
    """Compatibility wrapper for callers that only need the rebuild flag."""
    return synchronize_index_state(db, config, log=log)


def _llm_extractor_clients(config: Config) -> dict[str, ModelClient]:
    """Create only reusable providers referenced by active primary rules."""
    selected = {
        rule.provider
        for rule in config.extractor_rules
        if rule.enabled and rule.kind == "llm"
    }
    clients: dict[str, ModelClient] = {}
    for provider_id in selected:
        provider = config.llm_provider(provider_id)
        if provider is not None:
            client_kind = (
                provider_id.removeprefix("legacy-")
                if provider_id.startswith("legacy-")
                else f"llm-provider:{provider_id}"
            )
            clients[provider_id] = ModelClient(
                provider.model_config(), client_kind,
            )
    return clients


def index_pending(
    db: Database,
    config: Config,
    log=print,
    retry_failed: bool = False,
    progress: Callable[..., None] | None = None,
) -> IndexStats:
    stats = IndexStats()
    embedding_rebuild_required = synchronize_index_state(db, config, log=log)
    legacy_llm = ModelClient(config.llm, "llm")
    provider_clients = _llm_extractor_clients(config)
    ctx = ExtractContext(
        config=config,
        vision=ModelClient(config.vision, "vision"),
        llm=legacy_llm,
        llm_providers=provider_clients,
        llm_models=provider_clients,
    )

    statuses = [STATUS_PENDING, STATUS_WAITING_MODEL, STATUS_WAITING_CAPABILITY]
    if retry_failed:
        statuses += [STATUS_FAILED, STATUS_SKIPPED]

    rows = db.iter_files(statuses)
    total = len(rows)
    if progress is not None:
        progress("indexing", current=0, total=total, current_file="")
    for position, row in enumerate(rows, start=1):
        file_id, path = int(row["id"]), Path(row["path"])
        if progress is not None:
            progress("indexing", current=position - 1, total=total, current_file=str(path))
        path, invalid_path_error = _validated_source_path(path, config)
        if path is None:
            if invalid_path_error is not None:
                db.mark_skipped_invalid_path(file_id, invalid_path_error)
                stats.skipped += 1
                log(f"⚠ {row['filename']}: {invalid_path_error}")
                if progress is not None:
                    progress("indexing", current=position, total=total, current_file="")
                continue
            db.remove_file(file_id)
            if progress is not None:
                progress("indexing", current=position, total=total, current_file="")
            continue

        source_name = path.name
        with _trusted_source_snapshot(path, config) as (snapshot, snapshot_error):
            if snapshot is None:
                if snapshot_error is not None:
                    db.mark_skipped_invalid_path(file_id, snapshot_error)
                    stats.skipped += 1
                    log(f"⚠ {source_name}: {snapshot_error}")
                    if progress is not None:
                        progress("indexing", current=position, total=total, current_file="")
                    continue
                db.remove_file(file_id)
                if progress is not None:
                    progress("indexing", current=position, total=total, current_file="")
                continue

            extractor = resolve(snapshot, config)
            if extractor is None:
                db.set_status(file_id, STATUS_SKIPPED, error="没有适用的提取器")
                stats.skipped += 1
                if progress is not None:
                    progress("indexing", current=position, total=total, current_file="")
                continue

            try:
                text = extractor.extract(snapshot, ctx)
            except (ModelNotConfigured, ModelUnavailable) as e:
                db.set_status(file_id, STATUS_WAITING_MODEL, error=str(e), extractor=extractor.name)
                stats.waiting_model += 1
                log(f"⏳ {source_name}: {e}")
                if progress is not None:
                    progress("indexing", current=position, total=total, current_file="")
                continue
            except (CapabilityNotConfigured, CapabilityUnavailable) as e:
                db.set_status(file_id, STATUS_WAITING_CAPABILITY, error=str(e), extractor=extractor.name)
                stats.waiting_capability += 1
                log(f"⏳ {source_name}: {e}")
                if progress is not None:
                    progress("indexing", current=position, total=total, current_file="")
                continue
            except ExtractError as e:
                db.set_status(file_id, STATUS_FAILED, error=str(e), extractor=extractor.name)
                stats.failed += 1
                log(f"✗ {source_name}: {e}")
                if progress is not None:
                    progress("indexing", current=position, total=total, current_file="")
                continue
            except Exception as e:
                db.set_status(file_id, STATUS_FAILED, error=repr(e), extractor=extractor.name)
                stats.failed += 1
                log(f"✗ {source_name}: {e!r}")
                if progress is not None:
                    progress("indexing", current=position, total=total, current_file="")
                continue

        text = text[:MAX_TEXT_CHARS]
        db.save_content(file_id, text, row["filename"])
        db.set_status(file_id, STATUS_DONE, extractor=extractor.name)
        stats.indexed += 1
        log(f"✓ [{extractor.name}] {path.name}")
        if progress is not None:
            progress("indexing", current=position, total=total, current_file="")

    # Embedding is a secondary index over successfully extracted primary text.
    # A normal index pass also repairs missing vectors and completes any full
    # rebuild requested by a configuration identity change.
    if config.embedding.enabled:
        try:
            stats.embedded_files = embed_missing(
                db,
                config,
                log=log,
                rebuild=embedding_rebuild_required,
                progress=progress,
            )
        except Exception as exc:
            stats.embed_errors.append(str(exc))
            log(f"⚠ 向量索引未完成（一级正文不受影响，将在下次索引重试）: {exc}")

    # Entity extraction is secondary: a model outage must never invalidate
    # successful content indexing.  Newly changed files stay pending until the
    # user enables the optional entity feature and its LLM is available.
    if config.entities.enabled and config.entities_model.enabled:
        if progress is not None:
            progress("entities", current=total, total=total, current_file="")
        from .entities import index_entities

        entity_stats = index_entities(db, config, log=log, retry_failed=retry_failed)
        stats.entities_indexed = entity_stats.indexed
        stats.entity_failed = entity_stats.failed
        stats.entity_waiting_model = entity_stats.waiting_model

    return stats


def embed_missing(
    db: Database,
    config: Config,
    log=print,
    rebuild: bool = False,
    progress: Callable[..., None] | None = None,
) -> int:
    """Fill missing vectors, automatically honoring a pending full rebuild."""
    if not config.embedding.enabled:
        raise ModelNotConfigured("embedding 模型未启用（配置 [models.embedding] enabled = true）")

    current_identity = embedding_index_identity(config)
    previous_identity = db.meta_get(EMBEDDING_IDENTITY_KEY)
    rebuild_required = db.meta_get("embedding_rebuild_required") == "1"
    if rebuild or previous_identity != current_identity:
        db.clear_chunks()
        _require_embedding_rebuild(db)
        rebuild_required = True

    if rebuild_required:
        # A full rebuild is all-or-nothing from the searcher's point of view.
        # Drop partial retry output before starting, and retain the marker if
        # the model becomes unavailable midway.
        db.clear_chunks()
        _require_embedding_rebuild(db)
        rows = db.iter_files([STATUS_DONE])
    else:
        rows = db.files_without_chunks()

    emb = ModelClient(config.embedding, "embedding")
    done = 0
    dimension: int | None = None
    total = len(rows)
    if progress is not None:
        progress("embedding", current=0, total=total, current_file="")
    for position, row in enumerate(rows, start=1):
        if progress is not None:
            progress("embedding", current=position - 1, total=total, current_file=str(row["filename"]))
        text = db.get_content(int(row["id"]))
        if not text:
            if progress is not None:
                progress("embedding", current=position, total=total, current_file="")
            continue
        file_dimension = _embed_file(db, config, emb, int(row["id"]), text)
        if file_dimension is not None:
            if dimension is None:
                dimension = file_dimension
            elif dimension != file_dimension:
                raise ValueError("embedding 服务在不同文件间返回了不同维度的向量")
        done += 1
        log(f"✓ 已向量化: {row['filename']}")
        if progress is not None:
            progress("embedding", current=position, total=total, current_file="")
    _set_embedding_identity(db, config)
    if dimension is not None:
        db.meta_set("embedding_dim", str(dimension))
    if rebuild_required:
        db.meta_delete("embedding_rebuild_required")
    return done
