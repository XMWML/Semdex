"""Web 层：REST API + 静态单页界面。

接口（均返回 JSON，外部程序也可直接调）：
  GET  /api/search?q=&mode=&limit=   搜索
  GET  /api/ask?q=                   自然语言工具调用检索
  GET  /api/status                   索引状态
  POST /api/index                    触发后台增量索引（已在跑则 409）
  POST /api/rebuild                  清空派生索引并完整重建（已在跑则 409）
  GET  /api/content?file_id=         查看某文件提取出的全文
  POST /api/open   {path, reveal}    用系统默认程序打开 / 在访达中显示
  GET  /api/models                  扫描项目模型目录并返回加载状态
  POST /api/models/load             将一个本地模型加载到当前进程内存
  POST /api/models/unload           卸载一个本地模型或其用途运行时
  GET  /api/settings                 返回设置（不含密钥）
  PUT  /api/settings                 校验、原子保存并热加载设置
"""
from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ..config import Config
from ..db import Database
from ..models import ModelNotConfigured, ModelUnavailable
from ..modelclient import embedding_identity
from ..localmodels import get_local_model_manager
from ..search import search as do_search
from ..agent import ask as do_ask
from ..settings import save_settings, settings_dict

STATIC_DIR = Path(__file__).parent / "static"


class OpenReq(BaseModel):
    path: str
    reveal: bool = False


class SettingsReq(BaseModel):
    settings: dict[str, Any]


class LocalModelReq(BaseModel):
    model_id: str
    capability: str | None = None
    backend: str = "auto"


def create_app(config: Config) -> FastAPI:
    app = FastAPI(title="Semdex", docs_url=None, redoc_url=None)
    state = {"indexing": False, "last_run": None}
    state_lock = threading.RLock()
    config_lock = threading.RLock()

    def current_config() -> Config:
        with config_lock:
            return config

    def open_db(active_config: Config | None = None) -> Database:
        return Database((active_config or current_config()).db_path)

    def invalidate_changed_embeddings(before: Config, after: Config) -> bool:
        """Drop vectors immediately when settings switch their vector space."""
        if before.db_path != after.db_path:
            return False
        db = open_db(before)
        try:
            has_vectors = bool(db.chunks_with_embeddings())
            stored_identity = db.meta_get("embedding_model_id")
            if has_vectors and stored_identity != embedding_identity(after.embedding):
                db.require_embedding_rebuild()
                return True
            return False
        finally:
            db.close()

    def requeue_files_when_fallback_is_enabled(before: Config, after: Config) -> int:
        """Let newly enabled text fallback revisit otherwise unchanged files."""
        if (
            before.db_path != after.db_path
            or before.agent_fallback.enabled
            or not after.agent_fallback.enabled
        ):
            return 0
        db = open_db(after)
        try:
            return db.requeue_files_without_extractor()
        finally:
            db.close()

    @app.get("/")
    def home():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/settings")
    def settings_page():
        return FileResponse(STATIC_DIR / "settings.html")

    @app.get("/api/search")
    def api_search(q: str = "", mode: str = "hybrid", limit: int = 20):
        active_config = current_config()
        db = open_db(active_config)
        try:
            hits = do_search(db, active_config, q, mode=mode, limit=max(1, min(limit, 100)))
            return {"ok": True, "hits": [h.to_dict() for h in hits]}
        except (ModelNotConfigured, ModelUnavailable, ValueError, RuntimeError) as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
        finally:
            db.close()

    @app.get("/api/ask")
    def api_ask(q: str = ""):
        active_config = current_config()
        db = open_db(active_config)
        try:
            result = do_ask(db, active_config, q)
            return {"ok": True, **result.to_dict()}
        except (ModelNotConfigured, ModelUnavailable, ValueError, RuntimeError) as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
        finally:
            db.close()

    @app.get("/api/status")
    def api_status():
        active_config = current_config()
        db = open_db(active_config)
        try:
            counts = db.counts()
            embedding_rebuild_required = db.meta_get("embedding_rebuild_required") == "1"
        finally:
            db.close()
        with state_lock:
            indexing = state["indexing"]
            last_run = state["last_run"]
        return {
            "ok": True,
            "files": counts,
            "indexing": indexing,
            "last_run": last_run,
            "folders": [str(p) for p in active_config.folders],
            "models": {
                "llm": active_config.agent_model.enabled,
                "vision": active_config.vision.enabled,
                "embedding": active_config.embedding.enabled,
            },
            "embedding_rebuild_required": embedding_rebuild_required,
            "capabilities": {
                "ocr": active_config.ocr.enabled,
                "asr": active_config.asr.enabled,
                "entities": active_config.entities.enabled,
            },
        }

    @app.get("/api/settings")
    def api_settings():
        return {"ok": True, "settings": settings_dict(current_config())}

    @app.get("/api/models")
    def api_models():
        """Discover project-local model files without importing native runtimes."""
        try:
            return {"ok": True, **get_local_model_manager(current_config().model_dir).catalog()}
        except (OSError, ValueError) as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    @app.post("/api/models/load")
    def api_model_load(req: LocalModelReq):
        active_config = current_config()
        capability = (req.capability or "chat").strip().lower()
        kwargs: dict[str, Any] = {"backend": req.backend}
        if capability == "asr":
            kwargs.update({"device": active_config.asr.device, "compute_type": active_config.asr.compute_type})
        try:
            result = get_local_model_manager(active_config.model_dir).load(
                req.model_id, capability, **kwargs
            )
            return {"ok": True, "model": result}
        except (ModelNotConfigured, ModelUnavailable, ValueError, OSError) as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    @app.post("/api/models/unload")
    def api_model_unload(req: LocalModelReq):
        try:
            result = get_local_model_manager(current_config().model_dir).unload(
                req.model_id, req.capability
            )
            return {"ok": True, "model": result}
        except (ModelNotConfigured, ModelUnavailable, ValueError, OSError) as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    @app.put("/api/settings")
    def api_save_settings(req: SettingsReq):
        nonlocal config
        # A running worker keeps a consistent Config instance for its whole
        # pass.  Block edits rather than silently applying them halfway through.
        with state_lock:
            if state["indexing"]:
                raise HTTPException(409, "索引正在进行中，完成后再保存设置")
            with config_lock:
                try:
                    previous_config = config
                    config = save_settings(previous_config, req.settings)
                    invalidate_changed_embeddings(previous_config, config)
                    requeue_files_when_fallback_is_enabled(previous_config, config)
                except (OSError, ValueError) as e:
                    raise HTTPException(422, str(e)) from e
                return {"ok": True, "settings": settings_dict(config)}

    def start_index(*, full_rebuild: bool = False):
        with state_lock:
            if state["indexing"]:
                raise HTTPException(409, "索引正在进行中")
            with config_lock:
                active_config = config
            state["indexing"] = True

        def run():
            from ..indexer import embed_missing, index_pending
            from ..scanner import scan
            db = None
            try:
                db = open_db(active_config)
                requeued = 0
                if full_rebuild:
                    requeued = db.requeue_all_for_full_reindex(
                        invalidate_embeddings=active_config.embedding.enabled
                    )
                scan_stats = scan(db, active_config, log=lambda *_: None)
                index_stats = index_pending(db, active_config, log=lambda *_: None)
                result = {
                    "full_rebuild": full_rebuild,
                    "requeued": requeued,
                    "scan": scan_stats.to_dict(),
                    "index": index_stats.to_dict(),
                }
                if full_rebuild and active_config.embedding.enabled:
                    try:
                        result["embedded_files"] = embed_missing(
                            db, active_config, log=lambda *_: None, rebuild=True
                        )
                    except (ModelNotConfigured, ModelUnavailable, ValueError, RuntimeError) as e:
                        # Full-text rebuilding remains useful even when the
                        # optional vector service is offline.  The rebuild
                        # marker stays set, so semantic search remains safe.
                        result["embedding_error"] = str(e)
            except Exception as e:
                result = {"error": str(e)}
            finally:
                if db is not None:
                    db.close()
                with state_lock:
                    state["indexing"] = False
                    state["last_run"] = result

        threading.Thread(target=run, daemon=True).start()
        return {"ok": True, "started": True, "full_rebuild": full_rebuild}

    @app.post("/api/index")
    def api_index():
        return start_index()

    @app.post("/api/rebuild")
    def api_rebuild():
        return start_index(full_rebuild=True)

    @app.get("/api/content")
    def api_content(file_id: int):
        db = open_db(current_config())
        try:
            row = db.get_file(file_id)
            if row is None:
                raise HTTPException(404, "文件不存在")
            text = db.get_content(file_id) or ""
            return {"ok": True, "path": row["path"], "extractor": row["extractor"],
                    "text": text, "entities": db.entities_for_file(file_id)}
        finally:
            db.close()

    @app.post("/api/open")
    def api_open(req: OpenReq):
        db = open_db(current_config())
        try:
            row = db.get_file_by_path(req.path)
        finally:
            db.close()
        if row is None:
            raise HTTPException(404, "路径不在索引中")  # 防任意路径
        if sys.platform == "darwin":
            cmd = ["open", "-R", req.path] if req.reveal else ["open", req.path]
        elif sys.platform.startswith("linux"):
            cmd = ["xdg-open", str(Path(req.path).parent if req.reveal else Path(req.path))]
        elif sys.platform == "win32":
            cmd = ["explorer.exe", f"/select,{req.path}"] if req.reveal else ["explorer.exe", req.path]
        else:
            raise HTTPException(501, "当前系统没有可用的文件管理器适配")
        subprocess.Popen(cmd)
        return {"ok": True}

    return app
