"""索引管线：提取正文 → 写 contents/FTS → （启用 embedding 时）分块向量化。"""
from __future__ import annotations

from array import array
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from collections.abc import Iterator

from .chunker import chunk_text
from .config import Config
from .db import Database
from .extractors import ExtractContext, resolve
from .modelclient import ModelClient, embedding_identity
from .paths import ensure_private_directory
from .safepath import configured_watch_roots, open_regular_file_beneath_root
from .models import (
    STATUS_DONE, STATUS_FAILED, STATUS_PENDING, STATUS_SKIPPED,
    STATUS_WAITING_CAPABILITY, STATUS_WAITING_MODEL, CapabilityNotConfigured,
    CapabilityUnavailable, EmbeddingRebuildRequired, ExtractError, IndexStats,
    ModelNotConfigured, ModelUnavailable,
)

MAX_TEXT_CHARS = 1_000_000
EMBED_BATCH = 16


def _vec_to_blob(vec: list[float]) -> bytes:
    return array("f", vec).tobytes()


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


def check_embedding_model(db: Database, config: Config, log=print) -> bool:
    """Return whether incremental indexing must suppress embedding writes."""
    pending_rebuild = db.meta_get("embedding_rebuild_required") == "1"
    if not config.embedding.enabled:
        return pending_rebuild
    previous_model = db.meta_get("embedding_model")
    previous_identity = db.meta_get("embedding_model_id")
    has_vectors = bool(db.chunks_with_embeddings())
    identity_changed = has_vectors and previous_identity != _embedding_identity(config)
    if identity_changed and not pending_rebuild:
        _require_embedding_rebuild(db)
        pending_rebuild = True
        if previous_identity:
            log(
                f"⚠ embedding 模型或服务地址从 {previous_model or '未知'} 变更，"
                "已清除旧向量。运行 `semdex embed --rebuild` 后再使用语义搜索"
            )
        else:
            log("⚠ 现有向量缺少完整模型来源，已清除。运行 `semdex embed --rebuild` 后再使用语义搜索")
    elif pending_rebuild:
        log("⚠ 向量重建尚未完成；已跳过本次增量向量化。运行 `semdex embed --rebuild`")
    return pending_rebuild


def index_pending(db: Database, config: Config, log=print,
                  retry_failed: bool = False) -> IndexStats:
    stats = IndexStats()
    ctx = ExtractContext(
        config=config,
        vision=ModelClient(config.vision, "vision"),
        llm=ModelClient(config.fallback_model, "fallback"),
    )
    emb = ModelClient(config.embedding, "embedding")
    embedding_rebuild_required = check_embedding_model(db, config, log)

    statuses = [STATUS_PENDING, STATUS_WAITING_MODEL, STATUS_WAITING_CAPABILITY]
    if retry_failed:
        statuses += [STATUS_FAILED, STATUS_SKIPPED]

    for row in db.iter_files(statuses):
        file_id, path = int(row["id"]), Path(row["path"])
        path, invalid_path_error = _validated_source_path(path, config)
        if path is None:
            if invalid_path_error is not None:
                db.mark_skipped_invalid_path(file_id, invalid_path_error)
                stats.skipped += 1
                log(f"⚠ {row['filename']}: {invalid_path_error}")
                continue
            db.remove_file(file_id)
            continue

        source_name = path.name
        with _trusted_source_snapshot(path, config) as (snapshot, snapshot_error):
            if snapshot is None:
                if snapshot_error is not None:
                    db.mark_skipped_invalid_path(file_id, snapshot_error)
                    stats.skipped += 1
                    log(f"⚠ {source_name}: {snapshot_error}")
                    continue
                db.remove_file(file_id)
                continue

            extractor = resolve(snapshot, config)
            if extractor is None:
                db.set_status(file_id, STATUS_SKIPPED, error="没有适用的提取器")
                stats.skipped += 1
                continue

            try:
                text = extractor.extract(snapshot, ctx)
            except (ModelNotConfigured, ModelUnavailable) as e:
                db.set_status(file_id, STATUS_WAITING_MODEL, error=str(e), extractor=extractor.name)
                stats.waiting_model += 1
                log(f"⏳ {source_name}: {e}")
                continue
            except (CapabilityNotConfigured, CapabilityUnavailable) as e:
                db.set_status(file_id, STATUS_WAITING_CAPABILITY, error=str(e), extractor=extractor.name)
                stats.waiting_capability += 1
                log(f"⏳ {source_name}: {e}")
                continue
            except ExtractError as e:
                db.set_status(file_id, STATUS_FAILED, error=str(e), extractor=extractor.name)
                stats.failed += 1
                log(f"✗ {source_name}: {e}")
                continue
            except Exception as e:
                db.set_status(file_id, STATUS_FAILED, error=repr(e), extractor=extractor.name)
                stats.failed += 1
                log(f"✗ {source_name}: {e!r}")
                continue

        text = text[:MAX_TEXT_CHARS]
        db.save_content(file_id, text, row["filename"])

        if emb.enabled and not embedding_rebuild_required:
            try:
                dimension = _embed_file(db, config, emb, file_id, text)
                _set_embedding_identity(db, config)
                if dimension is not None:
                    db.meta_set("embedding_dim", str(dimension))
                stats.embedded_files += 1
            except Exception as e:
                # 全文索引已成功，向量之后可用 `semdex embed` 补
                db.replace_chunks(file_id, [])
                stats.embed_errors.append(f"{path.name}: {e}")
                log(f"⚠ {path.name} 向量化失败（全文索引不受影响）: {e}")
        else:
            db.replace_chunks(file_id, [])

        db.set_status(file_id, STATUS_DONE, extractor=extractor.name)
        stats.indexed += 1
        log(f"✓ [{extractor.name}] {path.name}")

    # Entity extraction is secondary: a model outage must never invalidate
    # successful content indexing.  Newly changed files stay pending until the
    # user enables the optional entity feature and its LLM is available.
    if config.entities.enabled and config.entities_model.enabled:
        from .entities import index_entities

        entity_stats = index_entities(db, config, log=log, retry_failed=retry_failed)
        stats.entities_indexed = entity_stats.indexed
        stats.entity_failed = entity_stats.failed
        stats.entity_waiting_model = entity_stats.waiting_model

    return stats


def embed_missing(db: Database, config: Config, log=print, rebuild: bool = False) -> int:
    """给缺向量的已索引文件补 embedding；--rebuild 时全部重建。"""
    if not config.embedding.enabled:
        raise ModelNotConfigured("embedding 模型未启用（配置 [models.embedding] enabled = true）")

    previous_identity = db.meta_get("embedding_model_id")
    rebuild_required = db.meta_get("embedding_rebuild_required") == "1"
    has_vectors = bool(db.chunks_with_embeddings())
    if has_vectors and previous_identity != _embedding_identity(config) and not rebuild_required:
        _require_embedding_rebuild(db)
        rebuild_required = True

    if rebuild:
        # A manual rebuild is always all-or-nothing from the searcher's point
        # of view.  If the model goes away midway, the marker remains set and
        # semantic retrieval stays disabled until a later successful run.
        _require_embedding_rebuild(db)
        rows = db.iter_files([STATUS_DONE])
    elif rebuild_required:
        raise EmbeddingRebuildRequired(
            "embedding 模型已变更，旧向量已清除；请运行 `semdex embed --rebuild`"
        )
    else:
        rows = db.files_without_chunks()

    emb = ModelClient(config.embedding, "embedding")
    done = 0
    dimension: int | None = None
    for row in rows:
        text = db.get_content(int(row["id"]))
        if not text:
            continue
        file_dimension = _embed_file(db, config, emb, int(row["id"]), text)
        if file_dimension is not None:
            if dimension is None:
                dimension = file_dimension
            elif dimension != file_dimension:
                raise ValueError("embedding 服务在不同文件间返回了不同维度的向量")
        done += 1
        log(f"✓ 已向量化: {row['filename']}")
    _set_embedding_identity(db, config)
    if dimension is not None:
        db.meta_set("embedding_dim", str(dimension))
    if rebuild:
        db.meta_delete("embedding_rebuild_required")
    return done
