"""检索：FTS5 全文（BM25）、向量语义、RRF 混合。

- fulltext：中文按字短语匹配（见 textutil），FTS 无结果时退化为 LIKE 子串兜底
- semantic：embedding 余弦相似度，按文件取最高分块
- hybrid：两路结果 RRF 融合；embedding 未启用时自动退化为 fulltext
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .db import Database
from .modelclient import ModelClient, embedding_identity
from .models import EmbeddingRebuildRequired, ModelNotConfigured, ModelUnavailable, SearchHit
from .textutil import build_fts_query, make_snippet

RRF_K = 60


def _hit(db: Database, file_id: int, score: float, query: str, source: str) -> SearchHit | None:
    row = db.get_file(file_id)
    if row is None or row["index_status"] != "done":
        return None
    text = db.get_content(file_id) or ""
    return SearchHit(
        file_id=file_id,
        path=row["path"],
        filename=row["filename"],
        ext=row["ext"] or "",
        mtime=row["mtime"] or 0.0,
        score=score,
        snippet=make_snippet(text, query),
        source=source,
    )


def _fulltext_ids(db: Database, query: str, limit: int) -> list[tuple[int, float]]:
    fts_query = build_fts_query(query)
    if not fts_query:
        return []
    results = db.fts_search(fts_query, limit)
    if results:
        return results
    # FTS 未命中（如查询里全是标点/特殊符号），LIKE 子串兜底
    return [(fid, 0.0) for fid in db.like_search(query.strip(), limit)]


def _semantic_ids(db: Database, config: Config, query: str, limit: int) -> list[tuple[int, float]]:
    if not config.embedding.enabled:
        raise ModelNotConfigured("语义搜索需要启用 embedding 模型（配置 [models.embedding]）")
    if db.meta_get("embedding_rebuild_required") == "1":
        raise EmbeddingRebuildRequired(
            "embedding 模型变更后正在等待完整重建；请运行 `semdex embed --rebuild`"
        )
    rows = db.chunks_with_embeddings()
    if not rows:
        return []
    stored_identity = db.meta_get("embedding_model_id")
    if stored_identity != embedding_identity(config.embedding):
        # This path can run before the next indexing pass, so persist the same
        # safety state that index_pending() would establish on a config change.
        db.require_embedding_rebuild()
        raise EmbeddingRebuildRequired(
            "当前 embedding 模型或服务地址与库中向量不一致；"
            "请运行 `semdex embed --rebuild` 重建"
        )
    emb = ModelClient(config.embedding, "embedding")
    qv = np.asarray(emb.embed([query])[0], dtype=np.float32)

    try:
        matrix = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    except ValueError as e:
        raise EmbeddingRebuildRequired(
            "索引中的向量维度不一致；请运行 `semdex embed --rebuild`"
        ) from e
    if matrix.shape[1] != qv.shape[0]:
        raise EmbeddingRebuildRequired(
            f"向量维度不匹配（库内 {matrix.shape[1]}，查询 {qv.shape[0]}）；"
            "请运行 `semdex embed --rebuild` 重建"
        )
    qn = qv / (np.linalg.norm(qv) + 1e-9)
    mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    scores = mn @ qn

    best_per_file: dict[int, float] = {}
    for r, s in zip(rows, scores):
        fid = int(r["file_id"])
        if s > best_per_file.get(fid, -2.0):
            best_per_file[fid] = float(s)
    ranked = sorted(best_per_file.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:limit]


def _rrf_merge(*ranked_lists: list[tuple[int, float]]) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, (fid, _) in enumerate(lst):
            scores[fid] = scores.get(fid, 0.0) + 1.0 / (RRF_K + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def search(db: Database, config: Config, query: str,
           mode: str = "hybrid", limit: int = 20) -> list[SearchHit]:
    query = query.strip()
    limit = max(1, min(int(limit), 100))
    if not query:
        return []
    if mode not in ("fulltext", "semantic", "hybrid"):
        raise ValueError(f"未知搜索模式: {mode}（可选 fulltext / semantic / hybrid）")

    if mode == "hybrid" and not config.embedding.enabled:
        mode = "fulltext"  # 优雅降级

    if mode == "fulltext":
        ranked = _fulltext_ids(db, query, limit)
        source = "fulltext"
    elif mode == "semantic":
        ranked = _semantic_ids(db, config, query, limit)
        source = "semantic"
    else:
        ft = _fulltext_ids(db, query, limit * 2)
        try:
            sem = _semantic_ids(db, config, query, limit * 2)
        except (EmbeddingRebuildRequired, ModelNotConfigured, ModelUnavailable):
            # Hybrid remains useful when a previously configured local embedding
            # server is temporarily offline; explicit semantic mode still reports
            # the failure to make configuration problems visible.
            ranked = ft[:limit]
            source = "fulltext"
            sem = None
        if sem is not None:
            ranked = _rrf_merge(ft, sem)[:limit]
            source = "hybrid"

    hits = []
    for fid, score in ranked[:limit]:
        h = _hit(db, fid, score, query, source)
        if h:
            hits.append(h)
    return hits
