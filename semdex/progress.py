"""Thread-safe progress snapshots shared by Web and native interfaces."""
from __future__ import annotations

import threading
import time
from typing import Any


class IndexProgress:
    """Keep a small, JSON-safe view of the currently running index pass."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state: dict[str, Any] = {
            "phase": "idle",
            "current": 0,
            "total": 0,
            "current_file": "",
            "scanned": 0,
            "started_at": None,
            "updated_at": None,
        }

    def begin(self, *, full_rebuild: bool) -> None:
        with self._lock:
            now = time.time()
            self._state = {
                "phase": "preparing" if full_rebuild else "scanning",
                "current": 0,
                "total": 0,
                "current_file": "",
                "scanned": 0,
                "started_at": now,
                "updated_at": now,
            }

    def update(self, phase: str, *, current: int | None = None,
               total: int | None = None, current_file: str | None = None,
               scanned: int | None = None) -> None:
        with self._lock:
            self._state["phase"] = phase
            if current is not None:
                self._state["current"] = max(0, current)
            if total is not None:
                self._state["total"] = max(0, total)
            if current_file is not None:
                self._state["current_file"] = current_file
            if scanned is not None:
                self._state["scanned"] = max(0, scanned)
            self._state["updated_at"] = time.time()

    def finish(self, *, failed: bool = False) -> None:
        with self._lock:
            self._state["phase"] = "failed" if failed else "complete"
            self._state["current_file"] = ""
            self._state["updated_at"] = time.time()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            data = dict(self._state)
        total = int(data["total"])
        current = int(data["current"])
        data["percent"] = int((current * 100) / total) if total else 0
        return data
