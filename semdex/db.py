"""SQLite 存储层：files / contents / contents_fts(FTS5) / chunks / meta。

FTS 表存的是 CJK 按字切分（插空格）后的文本，只用于匹配和排序；
展示用的原文在 contents 表，摘要片段由 textutil.make_snippet 从原文生成。
向量存 chunks.embedding（float32 BLOB），MVP 规模下 numpy 暴力余弦足够，
后续可无缝换 sqlite-vec。
"""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

from .config import DEFAULT_DB_PATH
from .textutil import cjk_spaced

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id            INTEGER PRIMARY KEY,
    path          TEXT UNIQUE NOT NULL,
    filename      TEXT NOT NULL,
    ext           TEXT,
    size          INTEGER,
    mtime         REAL,
    content_hash  TEXT,
    extractor     TEXT,
    index_status  TEXT NOT NULL DEFAULT 'pending',
    error_msg     TEXT,
    indexed_at    REAL,
    entity_status TEXT NOT NULL DEFAULT 'pending',
    entity_error  TEXT
);

CREATE TABLE IF NOT EXISTS contents (
    file_id       INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
    text          TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY,
    file_id       INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    text          TEXT NOT NULL,
    embedding     BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);

CREATE TABLE IF NOT EXISTS entities (
    id              INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    type            TEXT NOT NULL,
    UNIQUE(normalized_name, type)
);

CREATE TABLE IF NOT EXISTS file_entities (
    file_id   INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    context   TEXT,
    PRIMARY KEY(file_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_file_entities_entity ON file_entities(entity_id);

CREATE TABLE IF NOT EXISTS meta (
    key           TEXT PRIMARY KEY,
    value         TEXT
);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS contents_fts USING fts5(
    text, filename, tokenize='unicode61'
);
"""

INVALID_PATH_ERROR_PREFIX = "路径校验失败: "


class Database:
    def __init__(self, path: str | Path):
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if p.parent == DEFAULT_DB_PATH.parent:
            os.chmod(p.parent, 0o700)
        self._create_private_database_file(p)
        self.path = p
        self.conn = sqlite3.connect(str(p))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._ensure_schema()
        self._restrict_permissions()

    @staticmethod
    def _create_private_database_file(path: Path) -> None:
        """Pre-create new SQLite files with owner-only permissions.

        SQLite otherwise creates its database with the process umask, which is
        commonly group/world-readable and would expose indexed file contents.
        """
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        try:
            fd = os.open(path, flags, 0o600)
        except FileExistsError:
            return
        os.close(fd)

    def _restrict_permissions(self) -> None:
        """Keep the database and SQLite sidecars private after writes."""
        for path in (
            self.path,
            self.path.with_name(f"{self.path.name}-wal"),
            self.path.with_name(f"{self.path.name}-shm"),
        ):
            try:
                os.chmod(path, 0o600)
            except FileNotFoundError:
                continue

    def _ensure_schema(self) -> None:
        self.conn.executescript(SCHEMA)
        # Existing Semdex databases predate entity extraction.  SQLite only
        # supports additive column migrations, which is all this schema needs.
        self._ensure_column("files", "entity_status", "entity_status TEXT NOT NULL DEFAULT 'pending'")
        self._ensure_column("files", "entity_error", "entity_error TEXT")
        try:
            self.conn.executescript(FTS_SCHEMA)
        except sqlite3.OperationalError as e:
            raise RuntimeError(
                f"当前 SQLite 不支持 FTS5（{e}）。请使用带 FTS5 的 Python 构建（Homebrew 的 python3 默认支持）。"
            ) from e
        self.conn.commit()

    def _ensure_column(self, table: str, name: str, definition: str) -> None:
        columns = {row["name"] for row in self.conn.execute(f"PRAGMA table_info({table})")}
        if name not in columns:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")

    def close(self) -> None:
        self.conn.close()
        self._restrict_permissions()

    # ── files ─────────────────────────────────────────────

    def get_file_by_path(self, path: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM files WHERE path=?", (path,)).fetchone()

    def get_file(self, file_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM files WHERE id=?", (file_id,)).fetchone()

    def upsert_scan(self, path: str, filename: str, ext: str, size: int,
                    mtime: float, content_hash: str) -> tuple[int, bool]:
        """扫描时登记文件。返回 (file_id, 是否新增或内容变化)。"""
        row = self.get_file_by_path(path)
        if row is None:
            cur = self.conn.execute(
                "INSERT INTO files(path, filename, ext, size, mtime, content_hash, index_status) "
                "VALUES(?,?,?,?,?,?, 'pending')",
                (path, filename, ext, size, mtime, content_hash),
            )
            self.conn.commit()
            return int(cur.lastrowid), True
        if row["content_hash"] != content_hash:
            # A changed file must disappear from results until the new content has
            # been extracted.  Keeping the prior FTS/chunk rows here makes stale
            # text searchable while the file is pending (or when extraction fails).
            self._clear_index_data(int(row["id"]))
            self.conn.execute(
                "UPDATE files SET filename=?, ext=?, size=?, mtime=?, content_hash=?, "
                "index_status='pending', error_msg=NULL, entity_status='pending', entity_error=NULL "
                "WHERE id=?",
                (filename, ext, size, mtime, content_hash, row["id"]),
            )
            self.conn.commit()
            return int(row["id"]), True
        if row["index_status"] == "too_large":
            # The configured size limit may have been raised.  Derived data was
            # deliberately cleared when the file became too large, so restore it
            # to the normal extraction queue even when its hash never changed.
            self.conn.execute(
                "UPDATE files SET filename=?, ext=?, size=?, mtime=?, index_status='pending', "
                "error_msg=NULL, entity_status='pending', entity_error=NULL WHERE id=?",
                (filename, ext, size, mtime, row["id"]),
            )
            self.conn.commit()
            return int(row["id"]), True
        if (
            row["index_status"] == "skipped"
            and (row["error_msg"] or "").startswith(INVALID_PATH_ERROR_PREFIX)
        ):
            # A source can be rejected while it is a symlink and later return
            # as the original regular file with an identical content hash.
            # Requeue it instead of leaving the safety skip permanent.
            self.conn.execute(
                "UPDATE files SET filename=?, ext=?, size=?, mtime=?, index_status='pending', "
                "error_msg=NULL, entity_status='pending', entity_error=NULL WHERE id=?",
                (filename, ext, size, mtime, row["id"]),
            )
            self.conn.commit()
            return int(row["id"]), True
        # 内容没变，只更新元信息
        self.conn.execute(
            "UPDATE files SET size=?, mtime=? WHERE id=?", (size, mtime, row["id"])
        )
        self.conn.commit()
        return int(row["id"]), False

    def touch_meta(self, file_id: int, size: int, mtime: float) -> None:
        self.conn.execute("UPDATE files SET size=?, mtime=? WHERE id=?", (size, mtime, file_id))
        self.conn.commit()

    def _clear_index_data(self, file_id: int) -> None:
        """Remove derived rows for a file whose source content has changed."""
        self.conn.execute("DELETE FROM contents_fts WHERE rowid=?", (file_id,))
        self.conn.execute("DELETE FROM chunks WHERE file_id=?", (file_id,))
        self.conn.execute("DELETE FROM contents WHERE file_id=?", (file_id,))
        self.conn.execute("DELETE FROM file_entities WHERE file_id=?", (file_id,))
        self._remove_orphan_entities()

    def mark_too_large(self, file_id: int, size: int, mtime: float) -> None:
        """Hide stale derived content when a source exceeds the configured limit."""
        self._clear_index_data(file_id)
        self.conn.execute(
            "UPDATE files SET size=?, mtime=?, index_status='too_large', "
            "error_msg='超过当前 max_file_mb 限制' WHERE id=?",
            (size, mtime, file_id),
        )
        self.conn.commit()

    def mark_skipped_invalid_path(self, file_id: int, error: str) -> None:
        """Discard derived data for a source path rejected at index time."""
        self._clear_index_data(file_id)
        self.conn.execute(
            "UPDATE files SET index_status='skipped', error_msg=?, indexed_at=NULL, "
            "entity_status='pending', entity_error=NULL WHERE id=?",
            (f"{INVALID_PATH_ERROR_PREFIX}{error}", file_id),
        )
        self.conn.commit()

    def remove_file(self, file_id: int) -> None:
        self.conn.execute("DELETE FROM contents_fts WHERE rowid=?", (file_id,))
        self.conn.execute("DELETE FROM files WHERE id=?", (file_id,))
        self._remove_orphan_entities()
        self.conn.commit()

    def remove_missing(
        self,
        present_paths: set[str],
        scanned_roots: list[Path] | None = None,
        configured_roots: list[Path] | None = None,
        inaccessible_roots: list[Path] | None = None,
    ) -> int:
        """删除已不在已成功扫描目录中的记录。

        A temporarily unavailable configured directory must not erase its prior
        index.  ``scanned_roots`` limits deletion to roots that were actually
        traversed during this scan, while ``configured_roots`` still removes
        records for folders that the user has deliberately stopped watching.
        ``inaccessible_roots`` preserves descendants of child directories that
        ``os.walk`` reported as unreadable during this scan.
        """
        rows = self.conn.execute("SELECT id, path FROM files").fetchall()
        removed = 0
        for row in rows:
            path = Path(row["path"])
            in_configured_root = (
                configured_roots is None
                or any(path.is_relative_to(root) for root in configured_roots)
            )
            in_scanned_root = (
                scanned_roots is None
                or any(path.is_relative_to(root) for root in scanned_roots)
            )
            under_inaccessible_root = (
                inaccessible_roots is not None
                and any(path.is_relative_to(root) for root in inaccessible_roots)
            )
            should_remove = not in_configured_root or (
                in_scanned_root
                and row["path"] not in present_paths
                and not under_inaccessible_root
            )
            if should_remove:
                self.remove_file(row["id"])
                removed += 1
        return removed

    def requeue_all_for_full_reindex(self, *, invalidate_embeddings: bool = False) -> int:
        """Discard derived data and return every known source to ``pending``.

        A deliberate full rebuild must not leave old text searchable while the
        refreshed extraction is running.  It only changes the local shadow
        index; original files remain untouched.
        """
        total = int(self.conn.execute("SELECT COUNT(*) AS n FROM files").fetchone()["n"])
        self.conn.execute("DELETE FROM contents_fts")
        self.conn.execute("DELETE FROM chunks")
        self.conn.execute("DELETE FROM meta WHERE key='embedding_dim'")
        self.conn.execute("DELETE FROM contents")
        self.conn.execute("DELETE FROM file_entities")
        self._remove_orphan_entities()
        self.conn.execute(
            "UPDATE files SET index_status='pending', error_msg=NULL, extractor=NULL, indexed_at=NULL, "
            "entity_status='pending', entity_error=NULL"
        )
        if invalidate_embeddings:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES('embedding_rebuild_required', '1') "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value"
            )
        self.conn.commit()
        return total

    def requeue_files_without_extractor(self) -> int:
        """Retry sources skipped only because no extractor was configured.

        This deliberately excludes safety-related ``skipped`` states, such as
        paths rejected after validation.  It is used when the optional text
        fallback is turned on from the settings page.
        """
        cursor = self.conn.execute(
            "UPDATE files SET index_status='pending', error_msg=NULL "
            "WHERE index_status='skipped' AND error_msg='没有适用的提取器'"
        )
        self.conn.commit()
        return cursor.rowcount

    def set_status(self, file_id: int, status: str, error: str | None = None,
                   extractor: str | None = None) -> None:
        self.conn.execute(
            "UPDATE files SET index_status=?, error_msg=?, extractor=COALESCE(?, extractor), "
            "indexed_at=? WHERE id=?",
            (status, error, extractor, time.time() if status == "done" else None, file_id),
        )
        self.conn.commit()

    def iter_files(self, statuses: list[str]) -> list[sqlite3.Row]:
        marks = ",".join("?" * len(statuses))
        return self.conn.execute(
            f"SELECT * FROM files WHERE index_status IN ({marks}) ORDER BY id", statuses
        ).fetchall()

    def counts(self) -> dict:
        rows = self.conn.execute(
            "SELECT index_status, COUNT(*) AS n FROM files GROUP BY index_status"
        ).fetchall()
        by_status = {r["index_status"]: r["n"] for r in rows}
        total = sum(by_status.values())
        entity_rows = self.conn.execute(
            "SELECT entity_status, COUNT(*) AS n FROM files GROUP BY entity_status"
        ).fetchall()
        by_entity_status = {r["entity_status"]: r["n"] for r in entity_rows}
        entities = self.conn.execute("SELECT COUNT(*) AS n FROM entities").fetchone()["n"]
        return {
            "total": total,
            "by_status": by_status,
            "entities": int(entities),
            "by_entity_status": by_entity_status,
        }

    # ── contents / FTS ────────────────────────────────────

    def save_content(self, file_id: int, text: str, filename: str) -> None:
        self.conn.execute(
            "INSERT INTO contents(file_id, text) VALUES(?,?) "
            "ON CONFLICT(file_id) DO UPDATE SET text=excluded.text",
            (file_id, text),
        )
        self.conn.execute("DELETE FROM contents_fts WHERE rowid=?", (file_id,))
        self.conn.execute(
            "INSERT INTO contents_fts(rowid, text, filename) VALUES(?,?,?)",
            (file_id, cjk_spaced(text), cjk_spaced(filename)),
        )
        self.conn.commit()

    def get_content(self, file_id: int) -> str | None:
        row = self.conn.execute("SELECT text FROM contents WHERE file_id=?", (file_id,)).fetchone()
        return row["text"] if row else None

    def has_indexed_content(self) -> bool:
        """Return whether the database contains any primary extracted body."""
        return self.conn.execute("SELECT 1 FROM contents LIMIT 1").fetchone() is not None

    # ── entities / relations ──────────────────────────────

    def set_entity_status(self, file_id: int, status: str, error: str | None = None) -> None:
        self.conn.execute(
            "UPDATE files SET entity_status=?, entity_error=? WHERE id=?",
            (status, error, file_id),
        )
        self.conn.commit()

    def files_for_entities(self, statuses: list[str]) -> list[sqlite3.Row]:
        marks = ",".join("?" * len(statuses))
        return self.conn.execute(
            f"SELECT f.* FROM files f JOIN contents c ON c.file_id=f.id "
            f"WHERE f.index_status='done' AND f.entity_status IN ({marks}) ORDER BY f.id",
            statuses,
        ).fetchall()

    def replace_entities(self, file_id: int, entities: list[dict[str, str]]) -> None:
        """Replace a file's entity links while retaining shared entities."""
        self.conn.execute("DELETE FROM file_entities WHERE file_id=?", (file_id,))
        for entity in entities:
            name = entity["name"].strip()
            entity_type = entity["type"].strip().lower()
            if not name or not entity_type:
                continue
            normalized = name.casefold()
            self.conn.execute(
                "INSERT INTO entities(name, normalized_name, type) VALUES(?,?,?) "
                "ON CONFLICT(normalized_name, type) DO UPDATE SET name=excluded.name",
                (name, normalized, entity_type),
            )
            entity_id = self.conn.execute(
                "SELECT id FROM entities WHERE normalized_name=? AND type=?",
                (normalized, entity_type),
            ).fetchone()["id"]
            self.conn.execute(
                "INSERT OR REPLACE INTO file_entities(file_id, entity_id, context) VALUES(?,?,?)",
                (file_id, entity_id, entity.get("context", "")[:500]),
            )
        self._remove_orphan_entities()
        self.conn.commit()

    def clear_entities(self) -> None:
        """Remove all entity relations and queue every source for extraction again."""
        self.conn.execute("DELETE FROM file_entities")
        self.conn.execute("DELETE FROM entities")
        self.conn.execute(
            "UPDATE files SET entity_status='pending', entity_error=NULL"
        )
        self.conn.commit()

    def _remove_orphan_entities(self) -> None:
        self.conn.execute(
            "DELETE FROM entities WHERE NOT EXISTS "
            "(SELECT 1 FROM file_entities fe WHERE fe.entity_id=entities.id)"
        )

    def entities_for_file(self, file_id: int) -> list[dict[str, str]]:
        rows = self.conn.execute(
            "SELECT e.name, e.type, fe.context FROM file_entities fe "
            "JOIN entities e ON e.id=fe.entity_id WHERE fe.file_id=? "
            "ORDER BY e.type, e.name",
            (file_id,),
        ).fetchall()
        return [{"name": r["name"], "type": r["type"], "context": r["context"] or ""} for r in rows]

    def files_by_entity(
        self,
        query: str,
        limit: int = 20,
        file_ids: list[int] | None = None,
    ) -> list[int]:
        """Return entity matches, optionally restricted to an existing candidate set."""
        term = query.strip().casefold()
        if not term:
            return []
        pattern = f"%{term.replace('%', '')}%"
        clauses = ["f.index_status='done'", "e.normalized_name LIKE ?"]
        params: list[object] = [pattern]
        if file_ids is not None:
            ids = list(dict.fromkeys(file_id for file_id in file_ids if isinstance(file_id, int)))[:100]
            if not ids:
                return []
            clauses.append(f"f.id IN ({','.join('?' * len(ids))})")
            params.extend(ids)
        params.append(max(1, min(limit, 100)))
        rows = self.conn.execute(
            "SELECT f.id FROM entities e JOIN file_entities fe ON fe.entity_id=e.id "
            "JOIN files f ON f.id=fe.file_id "
            f"WHERE {' AND '.join(clauses)} ORDER BY f.mtime DESC LIMIT ?",
            params,
        ).fetchall()
        return [int(r["id"]) for r in rows]

    def filter_files(
        self,
        *,
        ext: str | None = None,
        path_prefix: str | None = None,
        mtime_after: float | None = None,
        mtime_before: float | None = None,
        min_size: int | None = None,
        max_size: int | None = None,
        file_ids: list[int] | None = None,
        limit: int = 20,
    ) -> list[sqlite3.Row]:
        """Return indexed files that match a restricted metadata predicate."""
        clauses = ["index_status='done'"]
        params: list[object] = []
        if file_ids is not None:
            ids = list(dict.fromkeys(file_id for file_id in file_ids if isinstance(file_id, int)))[:100]
            if not ids:
                return []
            clauses.append(f"id IN ({','.join('?' * len(ids))})")
            params.extend(ids)
        if ext:
            clean_ext = ext.strip().lower()
            if clean_ext and not clean_ext.startswith("."):
                clean_ext = f".{clean_ext}"
            if clean_ext:
                clauses.append("ext=?")
                params.append(clean_ext)
        if path_prefix:
            clauses.append("path LIKE ? ESCAPE '\\'")
            escaped = path_prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            params.append(f"{escaped}%")
        if mtime_after is not None:
            clauses.append("mtime>=?")
            params.append(mtime_after)
        if mtime_before is not None:
            clauses.append("mtime<=?")
            params.append(mtime_before)
        if min_size is not None:
            clauses.append("size>=?")
            params.append(min_size)
        if max_size is not None:
            clauses.append("size<=?")
            params.append(max_size)
        params.append(max(1, min(limit, 100)))
        return self.conn.execute(
            f"SELECT * FROM files WHERE {' AND '.join(clauses)} ORDER BY mtime DESC LIMIT ?", params
        ).fetchall()

    def fts_search(self, fts_query: str, limit: int) -> list[tuple[int, float]]:
        """返回 [(file_id, score)]，score 越大越相关（bm25 取负）。"""
        try:
            rows = self.conn.execute(
                "SELECT contents_fts.rowid, rank FROM contents_fts "
                "JOIN files f ON f.id=contents_fts.rowid "
                "WHERE contents_fts MATCH ? AND f.index_status='done' "
                "ORDER BY rank LIMIT ?",
                (fts_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(int(r["rowid"]), -float(r["rank"])) for r in rows]

    def like_search(self, term: str, limit: int) -> list[int]:
        """FTS 未命中时的兜底：正文/文件名子串匹配。"""
        esc = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{esc}%"
        rows = self.conn.execute(
            "SELECT f.id FROM files f LEFT JOIN contents c ON c.file_id=f.id "
            "WHERE f.index_status='done' AND "
            "(f.filename LIKE ? ESCAPE '\\' OR c.text LIKE ? ESCAPE '\\') "
            "ORDER BY f.mtime DESC LIMIT ?",
            (pattern, pattern, limit),
        ).fetchall()
        return [int(r["id"]) for r in rows]

    # ── chunks / embeddings ───────────────────────────────

    def replace_chunks(self, file_id: int, chunks: list[tuple[str, bytes | None]]) -> None:
        self.conn.execute("DELETE FROM chunks WHERE file_id=?", (file_id,))
        self.conn.executemany(
            "INSERT INTO chunks(file_id, chunk_index, text, embedding) VALUES(?,?,?,?)",
            [(file_id, i, text, emb) for i, (text, emb) in enumerate(chunks)],
        )
        self.conn.commit()

    def chunks_with_embeddings(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, file_id, text, embedding FROM chunks WHERE embedding IS NOT NULL"
        ).fetchall()

    def clear_embeddings(self) -> None:
        """Invalidate every vector while retaining chunk text for diagnostics."""
        self.conn.execute("UPDATE chunks SET embedding=NULL WHERE embedding IS NOT NULL")
        self.conn.commit()

    def clear_chunks(self) -> None:
        """Remove all chunk text/vectors and their now-invalid dimension metadata."""
        self.conn.execute("DELETE FROM chunks")
        self.conn.execute("DELETE FROM meta WHERE key='embedding_dim'")
        self.conn.commit()

    def require_embedding_rebuild(self) -> None:
        """Atomically hide vectors until a complete rebuild has succeeded."""
        self.conn.execute("UPDATE chunks SET embedding=NULL WHERE embedding IS NOT NULL")
        self.conn.execute("DELETE FROM meta WHERE key='embedding_dim'")
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES('embedding_rebuild_required', '1') "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value"
        )
        self.conn.commit()

    def files_without_chunks(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT f.* FROM files f JOIN contents body ON body.file_id=f.id "
            "WHERE f.index_status='done' AND NOT EXISTS "
            "(SELECT 1 FROM chunks c WHERE c.file_id=f.id AND c.embedding IS NOT NULL) "
            "ORDER BY f.id"
        ).fetchall()

    # ── meta ──────────────────────────────────────────────

    def meta_get(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def meta_set(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.conn.commit()

    def meta_delete(self, key: str) -> None:
        self.conn.execute("DELETE FROM meta WHERE key=?", (key,))
        self.conn.commit()
