"""Executable-script compatibility and folder-based Python plugins.

The current plugin contract is one directory per plugin::

    my_plugin/
        plugin.py

``plugin.py`` exposes a named function that accepts either ``path`` or
``path, ctx``.  A single ``.py`` file is still accepted so configurations from
older Semdex versions keep working.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import shutil
import subprocess
import sys
import threading
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from ..models import (
    CapabilityNotConfigured,
    CapabilityUnavailable,
    ExtractError,
    ModelNotConfigured,
    ModelUnavailable,
)
from .base import ExtractContext, Extractor

SCRIPT_TIMEOUT = 180
PLUGIN_FILENAME = "plugin.py"
PLUGIN_METADATA_NAMES = ("PLUGIN_METADATA", "PLUGIN")
SHIPPED_PLUGIN_DIR = Path(__file__).with_name("plugins")


@dataclass(frozen=True)
class PluginInfo:
    """Static information about one folder plugin or legacy Python file."""

    id: str
    folder: str
    name: str
    description: str
    function: str
    builtin: bool
    path: str
    available: bool
    error: str = ""
    legacy: bool = False

    def as_dict(self) -> dict[str, object]:
        """Return JSON-ready metadata for settings frontends."""
        return asdict(self)


def _metadata_from_tree(tree: ast.Module) -> dict[str, object]:
    """Read literal plugin metadata without importing or executing the file."""
    for node in tree.body:
        value: ast.expr | None = None
        names: list[str] = []
        if isinstance(node, ast.Assign):
            value = node.value
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value = node.value
            names = [node.target.id]
        if value is None or not any(name in PLUGIN_METADATA_NAMES for name in names):
            continue
        try:
            metadata = ast.literal_eval(value)
        except (ValueError, TypeError, SyntaxError):
            return {}
        return metadata if isinstance(metadata, dict) else {}
    return {}


def _inspect_plugin(folder: str, module_path: Path, *, legacy: bool) -> PluginInfo:
    default_name = module_path.stem if legacy else folder
    plugin_id = module_path.stem if legacy else folder
    absolute_path = str((module_path if legacy else module_path.parent).resolve(strict=False))
    if not module_path.is_file():
        return PluginInfo(
            id=plugin_id,
            folder=folder,
            name=default_name,
            description="",
            function="extract",
            builtin=False,
            path=absolute_path,
            available=False,
            error=f"缺少 {PLUGIN_FILENAME}",
            legacy=legacy,
        )
    try:
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(module_path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        return PluginInfo(
            id=plugin_id,
            folder=folder,
            name=default_name,
            description="",
            function="extract",
            builtin=False,
            path=absolute_path,
            available=False,
            error=f"无法解析插件: {exc}",
            legacy=legacy,
        )

    metadata = _metadata_from_tree(tree)
    raw_name = metadata.get("name", default_name)
    raw_description = metadata.get("description", "")
    raw_function = metadata.get("function", "extract")
    builtin = metadata.get("builtin", False) is True
    name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else default_name
    description = raw_description.strip() if isinstance(raw_description, str) else ""
    function = raw_function.strip() if isinstance(raw_function, str) else ""
    defined_functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if not function.isidentifier():
        error = "元数据中的 function 不是有效的 Python 函数名"
    elif function not in defined_functions:
        error = f"未找到函数 {function}"
    else:
        error = ""
    return PluginInfo(
        id=plugin_id,
        folder=folder,
        name=name,
        description=description,
        function=function or "extract",
        builtin=builtin,
        path=absolute_path,
        available=not error,
        error=error,
        legacy=legacy,
    )


def discover_python_plugins(plugin_dir: Path) -> list[PluginInfo]:
    """Discover plugins from source only; plugin code is never imported here.

    Immediate child folders are the primary format.  Root-level ``.py`` files
    are reported as legacy entries so users can migrate them deliberately.
    Broken folders are included with ``available=False`` and an actionable
    error instead of disappearing from the settings page.
    """
    root = Path(plugin_dir).expanduser()
    if not root.is_dir():
        return []
    try:
        entries = sorted(root.iterdir(), key=lambda path: path.name.casefold())
    except OSError:
        return []

    plugins: list[PluginInfo] = []
    for entry in entries:
        if entry.name.startswith(".") or entry.name == "__pycache__":
            continue
        if entry.is_dir():
            plugins.append(_inspect_plugin(entry.name, entry / PLUGIN_FILENAME, legacy=False))
        elif entry.is_file() and entry.suffix.lower() == ".py" and entry.name != "__init__.py":
            plugins.append(_inspect_plugin(entry.name, entry, legacy=True))
    return plugins


def resolve_python_plugin(plugin_dir: Path, folder: str) -> Path:
    """Resolve one immediate child plugin folder (or legacy ``.py`` file)."""
    reference = folder.strip()
    if not reference or Path(reference).name != reference or reference in {".", ".."}:
        raise ExtractError("Python 插件必须是插件目录中的单个文件夹名")
    root = Path(plugin_dir).expanduser().resolve(strict=False)
    candidate = root / reference
    if candidate.is_dir():
        return candidate / PLUGIN_FILENAME
    if candidate.suffix.lower() == ".py":
        return candidate
    shipped = SHIPPED_PLUGIN_DIR / reference / PLUGIN_FILENAME
    if shipped.is_file():
        return shipped
    # Returning the canonical expected path gives the extractor a precise
    # missing-plugin error and also supports a folder created after settings
    # were saved.
    return candidate / PLUGIN_FILENAME


def seed_shipped_plugins(plugin_dir: Path) -> list[Path]:
    """Copy Semdex's shipped plugins into ``plugin_dir`` without overwriting.

    The returned paths are only those copied during this call.  If a user has
    already created a folder named ``ocr`` or ``asr``, the complete folder is
    left untouched.
    """
    destination = Path(plugin_dir).expanduser()
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError:
        return []
    copied: list[Path] = []
    if not SHIPPED_PLUGIN_DIR.is_dir():
        return copied
    for source in sorted(SHIPPED_PLUGIN_DIR.iterdir(), key=lambda path: path.name.casefold()):
        if not source.is_dir() or not (source / PLUGIN_FILENAME).is_file():
            continue
        target = destination / source.name
        if target.exists():
            continue
        try:
            shutil.copytree(
                source,
                target,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )
        except FileExistsError:
            # Another process may seed the same configuration concurrently.
            continue
        except OSError:
            # Discovery will report whatever is already present; an explicit
            # settings save performs stricter path validation.
            continue
        copied.append(target)
    return copied


class ScriptExtractor(Extractor):
    """Run a legacy executable and use its stdout as index text."""

    name = "script"

    def __init__(self, script: str):
        self.script = script

    def extract(self, path: Path, ctx: ExtractContext) -> str:
        try:
            proc = subprocess.run(
                [self.script, str(path)],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=SCRIPT_TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExtractError(f"脚本超时（>{SCRIPT_TIMEOUT}s）: {self.script}") from exc
        except OSError as exc:
            raise ExtractError(f"脚本无法执行: {exc}") from exc
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip()[-500:]
            raise ExtractError(f"脚本退出码 {proc.returncode}: {tail}")
        return proc.stdout


class PythonFunctionExtractor(Extractor):
    """Call a function from a folder plugin or backward-compatible ``.py``."""

    name = "python"
    _module_cache: dict[tuple[str, str], types.ModuleType] = {}
    _cache_lock = threading.Lock()

    def __init__(self, script: Path, function: str = "extract"):
        source = Path(script).expanduser()
        # A folder may be configured before it is seeded/created, so the
        # decision cannot depend only on current filesystem state.
        self.script = source if source.suffix.lower() == ".py" else source / PLUGIN_FILENAME
        self.function = function.strip()

    @classmethod
    def from_plugin(
        cls,
        plugin_dir: Path,
        folder: str,
        function: str = "extract",
    ) -> "PythonFunctionExtractor":
        return cls(resolve_python_plugin(plugin_dir, folder), function)

    def _load_module(self) -> types.ModuleType:
        if self.script.suffix.lower() != ".py" or not self.script.is_file():
            raise ExtractError(f"Python 插件不存在或缺少 {PLUGIN_FILENAME}: {self.script}")
        try:
            source_digest = hashlib.sha256(self.script.read_bytes()).hexdigest()
        except OSError as exc:
            raise ExtractError(f"无法读取 Python 插件: {exc}") from exc
        key = (str(self.script.resolve()), source_digest)
        cls = type(self)
        with cls._cache_lock:
            cached = cls._module_cache.get(key)
            if cached is not None:
                return cached
            module = self._import_module(key)
            # Drop stale revisions of this plugin while retaining loaded model
            # state for the current revision.
            cls._module_cache = {
                cached_key: value
                for cached_key, value in cls._module_cache.items()
                if cached_key[0] != key[0]
            }
            cls._module_cache[key] = module
            return module

    def _import_module(self, key: tuple[str, str]) -> types.ModuleType:
        digest = hashlib.sha256(f"{key[0]}:{key[1]}".encode()).hexdigest()[:20]
        in_folder = self.script.name == PLUGIN_FILENAME
        package_name = f"_semdex_plugin_{digest}"
        module_name = f"{package_name}.plugin" if in_folder else package_name
        package: types.ModuleType | None = None
        if in_folder:
            package = types.ModuleType(package_name)
            package.__path__ = [str(self.script.parent)]  # type: ignore[attr-defined]
            package.__package__ = package_name
            sys.modules[package_name] = package
        spec = importlib.util.spec_from_file_location(module_name, self.script)
        if spec is None or spec.loader is None:
            if package is not None:
                sys.modules.pop(package_name, None)
            raise ExtractError(f"无法加载 Python 插件: {self.script}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(module_name, None)
            if package is not None:
                sys.modules.pop(package_name, None)
            raise ExtractError(f"Python 插件导入失败: {exc}") from exc
        return module

    @staticmethod
    def _call(function: Callable[..., Any], path: Path, ctx: ExtractContext) -> Any:
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError) as exc:
            raise ExtractError(f"无法检查 Python 插件函数签名: {exc}") from exc
        try:
            signature.bind(path, ctx)
        except TypeError:
            try:
                signature.bind(path)
            except TypeError as exc:
                raise ExtractError("Python 插件函数必须接收 path 或 path, ctx") from exc
            return function(path)
        return function(path, ctx)

    def extract(self, path: Path, ctx: ExtractContext) -> str:
        if not self.function.isidentifier():
            raise ExtractError(f"Python 插件函数名无效: {self.function}")
        module = self._load_module()
        function = getattr(module, self.function, None)
        if not callable(function):
            raise ExtractError(f"Python 插件缺少可调用函数 {self.function}(path[, ctx])")
        try:
            value = self._call(function, path, ctx)
        except (
            ExtractError,
            ModelNotConfigured,
            ModelUnavailable,
            CapabilityNotConfigured,
            CapabilityUnavailable,
        ):
            raise
        except Exception as exc:
            raise ExtractError(f"Python 插件函数执行失败: {exc}") from exc
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)


__all__ = [
    "PLUGIN_FILENAME",
    "PluginInfo",
    "PythonFunctionExtractor",
    "ScriptExtractor",
    "discover_python_plugins",
    "resolve_python_plugin",
    "seed_shipped_plugins",
]
