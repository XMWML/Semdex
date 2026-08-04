"""Non-UI desktop operations shared by native GUI front ends.

The controller deliberately calls the existing indexing/search core directly
instead of starting a private HTTP service.  It is importable in headless
environments, which keeps desktop behaviour testable without a GUI runtime.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from .agent import ask as do_ask
from .config import Config, load_config, resolve_config_path, write_default_config
from .db import Database
from .indexer import embed_missing, index_pending, synchronize_index_state
from .models import ModelNotConfigured, ModelUnavailable
from .progress import IndexProgress
from .scanner import scan
from .search import search as do_search
from .settings import save_settings, settings_dict


class DesktopController:
    """Thread-safe facade over Semdex's local core services."""

    def __init__(self, config_path: str | os.PathLike | None = None):
        resolved = resolve_config_path(config_path)
        if not resolved.exists():
            write_default_config(resolved)
        self._config = load_config(resolved)
        self._state_lock = threading.RLock()
        self._config_lock = threading.RLock()
        self._indexing = False
        self._last_run: dict[str, Any] | None = None
        self._progress = IndexProgress()

    @property
    def config(self) -> Config:
        with self._config_lock:
            return self._config

    @property
    def indexing(self) -> bool:
        with self._state_lock:
            return self._indexing

    def _open_db(self, config: Config | None = None) -> Database:
        return Database((config or self.config).db_path)

    def settings(self) -> dict[str, Any]:
        return settings_dict(self.config)

    def save_settings(self, payload: object) -> dict[str, Any]:
        with self._state_lock:
            if self._indexing:
                raise RuntimeError("索引正在进行中，完成后再保存设置")
            with self._config_lock:
                before = self._config
                after = save_settings(before, payload)
                self._config = after
                db = self._open_db(after)
                try:
                    synchronize_index_state(db, after, log=lambda *_: None)
                finally:
                    db.close()
                return settings_dict(after)

    def local_model_catalog(self) -> dict[str, Any]:
        """Return local model files and in-memory runtime status for the GUI."""
        from .localmodels import get_local_model_manager

        return get_local_model_manager(self.config.model_dir).catalog()

    def load_local_model(self, model_id: str, capability: str) -> dict[str, Any]:
        """Load one discovered local model for a specific capability."""
        from .localmodels import get_local_model_manager

        return get_local_model_manager(self.config.model_dir).load(model_id, capability)

    def unload_local_model(
        self,
        model_id: str,
        capability: str | None = None,
    ) -> dict[str, Any]:
        """Release one capability, or every runtime for the selected model."""
        from .localmodels import get_local_model_manager

        return get_local_model_manager(self.config.model_dir).unload(model_id, capability)

    def status(self) -> dict[str, Any]:
        config = self.config
        db = self._open_db(config)
        try:
            counts = db.counts()
            needs_rebuild = db.meta_get("embedding_rebuild_required") == "1"
        finally:
            db.close()
        with self._state_lock:
            return {
                "files": counts,
                "indexing": self._indexing,
                "progress": self._progress.snapshot(),
                "last_run": self._last_run,
                "folders": [str(folder) for folder in config.folders],
                "models": {
                    "llm": any(provider.enabled for provider in config.llm_providers or []),
                    "llm_providers": sum(
                        1 for provider in config.llm_providers or [] if provider.enabled
                    ),
                    "embedding": config.embedding.enabled,
                    "rag": config.rag.enabled and config.embedding.enabled,
                    "agent": config.agent.enabled and config.agent_model.enabled,
                    "entities": config.entities.enabled and config.entities_model.enabled,
                },
                "embedding_rebuild_required": needs_rebuild,
                "capabilities": {
                    "ocr": config.ocr.enabled,
                    "asr": config.asr.enabled,
                    "entities": config.entities.enabled,
                    "plugins": sum(
                        1 for rule in config.extractor_rules
                        if rule.enabled and rule.kind == "python"
                    ),
                },
            }

    def search(self, query: str, mode: str = "hybrid", limit: int = 30) -> list[dict[str, Any]]:
        config = self.config
        db = self._open_db(config)
        try:
            return [hit.to_dict() for hit in do_search(db, config, query, mode=mode, limit=limit)]
        finally:
            db.close()

    def ask(self, query: str) -> dict[str, Any]:
        config = self.config
        db = self._open_db(config)
        try:
            return do_ask(db, config, query).to_dict()
        finally:
            db.close()

    def content(self, file_id: int) -> dict[str, Any]:
        db = self._open_db()
        try:
            row = db.get_file(file_id)
            if row is None:
                raise FileNotFoundError("文件不存在")
            return {
                "path": row["path"],
                "extractor": row["extractor"],
                "text": db.get_content(file_id) or "",
                "entities": db.entities_for_file(file_id),
            }
        finally:
            db.close()

    def start_index(self, *, full_rebuild: bool = False) -> threading.Thread:
        with self._state_lock:
            if self._indexing:
                raise RuntimeError("索引正在进行中")
            with self._config_lock:
                config = self._config
            self._indexing = True
            self._progress.begin(full_rebuild=full_rebuild)

        def run() -> None:
            db: Database | None = None
            try:
                db = self._open_db(config)
                requeued = 0
                if full_rebuild:
                    requeued = db.requeue_all_for_full_reindex(
                        invalidate_embeddings=config.embedding.enabled
                    )
                scan_stats = scan(db, config, log=lambda *_: None, progress=self._progress.update)
                index_stats = index_pending(db, config, log=lambda *_: None, progress=self._progress.update)
                result: dict[str, Any] = {
                    "full_rebuild": full_rebuild,
                    "requeued": requeued,
                    "scan": scan_stats.to_dict(),
                    "index": index_stats.to_dict(),
                }
                if (
                    full_rebuild
                    and config.embedding.enabled
                    and db.meta_get("embedding_rebuild_required") == "1"
                ):
                    if index_stats.embed_errors:
                        result["embedding_error"] = index_stats.embed_errors[-1]
                    else:
                        try:
                            result["embedded_files"] = embed_missing(
                                db, config, log=lambda *_: None, rebuild=True,
                                progress=self._progress.update,
                            )
                        except (ModelNotConfigured, ModelUnavailable, ValueError, RuntimeError) as exc:
                            result["embedding_error"] = str(exc)
            except Exception as exc:  # Keep failures visible through status/UI.
                result = {"error": str(exc)}
            finally:
                if db is not None:
                    db.close()
                with self._state_lock:
                    self._indexing = False
                    self._last_run = result
                    self._progress.finish(failed="error" in result)

        worker = threading.Thread(target=run, daemon=True, name="semdex-index")
        worker.start()
        return worker

    def open_path(self, path: str, *, reveal: bool = False) -> None:
        """Open an indexed file, or show it in the host file manager."""
        db = self._open_db()
        try:
            if db.get_file_by_path(path) is None:
                raise FileNotFoundError("路径不在索引中")
        finally:
            db.close()

        target = Path(path)
        if sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(target)] if reveal else ["open", str(target)])
        elif sys.platform == "win32":
            if reveal:
                subprocess.Popen(["explorer.exe", f"/select,{target}"])
            else:
                os.startfile(str(target))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(target.parent if reveal else target)])
