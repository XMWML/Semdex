"""Cross-platform, debounced filesystem watching backed by watchdog.

On macOS watchdog uses FSEvents; Linux and Windows use their native observers.
Every event triggers the existing idempotent scanner, so rename/delete handling
and hash-based de-duplication remain centralized in one pipeline.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable

from .config import Config
from .db import Database


class DebouncedIndexer:
    """Coalesce bursts of filesystem events into one serial index run."""

    def __init__(self, config: Config, log: Callable[..., None] = print):
        self.config = config
        self.log = log
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._running = False
        self._rerun = False
        self._scheduled_retry_failed = False
        self._rerun_retry_failed = False

    def schedule(self, *, retry_failed: bool = False) -> None:
        with self._lock:
            if self._running:
                self._rerun = True
                self._rerun_retry_failed = self._rerun_retry_failed or retry_failed
                return
            self._scheduled_retry_failed = self._scheduled_retry_failed or retry_failed
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.config.watch_debounce_sec, self.run)
            self._timer.daemon = True
            self._timer.start()

    def run(self, *, retry_failed: bool = False) -> None:
        with self._lock:
            if self._running:
                self._rerun = True
                self._rerun_retry_failed = self._rerun_retry_failed or retry_failed
                return
            if self._timer is not None:
                # A periodic reconciliation can arrive while an event debounce
                # timer is pending.  This run already covers that event.
                self._timer.cancel()
            self._timer = None
            self._running = True
            retry_failed = retry_failed or self._scheduled_retry_failed
            self._scheduled_retry_failed = False
        try:
            from .indexer import index_pending
            from .scanner import scan

            db = Database(self.config.db_path)
            try:
                scan_stats = scan(db, self.config, log=self.log)
                index_stats = index_pending(
                    db, self.config, log=self.log, retry_failed=retry_failed
                )
            finally:
                db.close()
            self.log(
                f"↻ 监听索引完成: 变化 {scan_stats.new_or_changed}，移除 {scan_stats.removed}，"
                f"完成 {index_stats.indexed}"
            )
        except Exception as e:
            self.log(f"⚠ 监听索引失败: {e}")
        finally:
            with self._lock:
                self._running = False
                rerun = self._rerun
                rerun_retry_failed = self._rerun_retry_failed
                self._rerun = False
                self._rerun_retry_failed = False
            if rerun:
                self.schedule(retry_failed=rerun_retry_failed)

    def close(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._scheduled_retry_failed = False
            self._rerun_retry_failed = False


def run_watcher(config: Config, log: Callable[..., None] = print) -> None:
    """Run the initial scan then keep it current until Ctrl+C."""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError as e:
        raise RuntimeError("未安装 watchdog。执行 `uv sync` 后再运行 `semdex watch`") from e

    roots = [path.expanduser().resolve() for path in config.folders if path.expanduser().is_dir()]
    if not roots:
        raise ValueError("没有可监听的监控文件夹，请先在 [watch] folders 中配置存在的目录")

    indexer = DebouncedIndexer(config, log=log)
    indexer.run()
    db_path = str(config.db_path.expanduser().resolve())
    ignored_paths = {db_path, f"{db_path}-wal", f"{db_path}-shm"}

    class Handler(FileSystemEventHandler):
        def on_any_event(self, event) -> None:
            paths = {getattr(event, "src_path", ""), getattr(event, "dest_path", "")}
            if paths & ignored_paths:
                return
            if event.event_type in {"created", "modified", "deleted", "moved"}:
                indexer.schedule()

    observer = Observer()
    handler = Handler()
    for root in roots:
        observer.schedule(handler, str(root), recursive=True)
    observer.start()
    log(f"正在监听 {len(roots)} 个文件夹（Ctrl+C 退出）")
    reconcile_at = (
        time.monotonic() + config.watch_reconcile_sec
        if config.watch_reconcile_sec > 0
        else None
    )
    try:
        while True:
            time.sleep(0.5)
            if reconcile_at is not None and time.monotonic() >= reconcile_at:
                log("↻ 开始定期全量对账并重试失败项")
                indexer.run(retry_failed=True)
                reconcile_at = time.monotonic() + config.watch_reconcile_sec
    except KeyboardInterrupt:
        pass
    finally:
        indexer.close()
        observer.stop()
        observer.join()
