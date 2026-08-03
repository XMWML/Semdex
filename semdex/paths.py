"""Project-local runtime paths and cache environment defaults.

Semdex is designed to be movable as one folder.  ``SEMDEX_HOME`` lets a
launcher pin that folder explicitly; a source checkout otherwise uses its
repository root.  External callers can still override individual settings
with ``SEMDEX_CONFIG`` or an explicit CLI ``--config`` path.
"""
from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the portable application folder without depending on CWD."""
    configured = os.environ.get("SEMDEX_HOME")
    if configured:
        return Path(configured).expanduser().resolve()

    source_root = Path(__file__).resolve().parent.parent
    if (source_root / "pyproject.toml").is_file():
        return source_root
    return Path.cwd().resolve()


def state_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / ".semdex"


def default_config_path() -> Path:
    return state_dir() / "config.toml"


def default_database_path() -> Path:
    return state_dir() / "index.db"


def default_temp_dir() -> Path:
    return state_dir() / "tmp"


def default_model_dir() -> Path:
    return state_dir() / "models"


def ensure_private_directory(path: Path) -> Path:
    """Create a Semdex-managed directory with owner-only permissions."""
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(path, 0o700)
    except OSError:
        # Windows does not expose POSIX mode semantics in the same way.
        pass
    return path


def configure_cache_environment() -> None:
    """Keep dependency and model caches in the portable application folder.

    ``uv`` reads its variable before Python starts, so launchers set it too.
    These defaults cover Python-side tools and preserve a user's explicit
    environment override.
    """
    root = project_root()
    state = state_dir(root)
    cache = root / ".uv-cache"
    hf_home = state / "models" / "huggingface"
    defaults = {
        "UV_CACHE_DIR": cache,
        "UV_PYTHON_INSTALL_DIR": root / ".uv-python",
        "PIP_CACHE_DIR": cache / "pip",
        "XDG_CACHE_HOME": state / "cache",
        "HF_HOME": hf_home,
        "HUGGINGFACE_HUB_CACHE": hf_home / "hub",
        "TRANSFORMERS_CACHE": hf_home / "transformers",
    }
    for name, path in defaults.items():
        os.environ.setdefault(name, str(path))
