"""Launch Semdex after putting uv and runtime data inside the project folder.

This module intentionally uses only the Python standard library.  It runs
before ``uv sync`` so uv's managed Python installation cannot escape to a
user-level cache on a first launch.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence


def application_root() -> Path:
    return Path(__file__).resolve().parent.parent


def portable_environment(
    root: Path | None = None,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment whose Semdex and uv state is rooted at ``root``."""
    project = (root or application_root()).resolve()
    state = project / ".semdex"
    hf_home = state / "models" / "huggingface"
    environment = dict(os.environ if base is None else base)
    environment.update({
        "SEMDEX_HOME": str(project),
        "SEMDEX_CONFIG": str(state / "config.toml"),
        "UV_CACHE_DIR": str(project / ".uv-cache"),
        "UV_PYTHON_INSTALL_DIR": str(project / ".uv-python"),
        "PIP_CACHE_DIR": str(project / ".uv-cache" / "pip"),
        "XDG_CACHE_HOME": str(state / "cache"),
        "HF_HOME": str(hf_home),
        "HUGGINGFACE_HUB_CACHE": str(hf_home / "hub"),
        "TRANSFORMERS_CACHE": str(hf_home / "transformers"),
        "TMPDIR": str(state / "tmp"),
        "TMP": str(state / "tmp"),
        "TEMP": str(state / "tmp"),
    })
    return environment


def ensure_portable_directories(environment: Mapping[str, str]) -> None:
    for name in (
        "UV_CACHE_DIR",
        "UV_PYTHON_INSTALL_DIR",
        "PIP_CACHE_DIR",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "TMPDIR",
    ):
        Path(environment[name]).mkdir(parents=True, exist_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用项目内目录启动 Semdex，避免 uv 和模型下载写入用户目录。"
    )
    ui = parser.add_mutually_exclusive_group()
    ui.add_argument("--ui", choices=("native", "web"), default="native", help="选择原生界面或 WebUI")
    ui.add_argument("--web", action="store_const", const="web", dest="ui", help="兼容旧版：启动 WebUI")
    ui.add_argument("--native", action="store_const", const="native", dest="ui", help="启动原生界面")
    parser.add_argument("--with-asr", action="store_true", help="同时安装 faster-whisper 可选依赖")
    parser.add_argument("--with-gguf", action="store_true", help="同时安装 GGUF 本地模型运行时")
    parser.add_argument("--with-mlx", action="store_true", help="同时安装 MLX 本地模型运行时（仅 macOS 可用）")
    parser.add_argument(
        "--with-local-models",
        action="store_true",
        help="同时安装可用的 GGUF 和 MLX 本地模型运行时",
    )
    parser.add_argument("--sync-only", action="store_true", help="只同步依赖，不启动界面")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = application_root()
    environment = portable_environment(root)
    ensure_portable_directories(environment)

    uv = shutil.which("uv", path=environment.get("PATH"))
    if uv is None:
        print("未找到 uv。请先安装 uv，然后再次运行 Start Semdex.py。", file=sys.stderr)
        return 1

    extras: list[str] = []
    if args.ui == "native":
        extras.append("gui")
    if args.with_asr:
        extras.append("asr")
    if args.with_gguf or args.with_local_models:
        extras.append("gguf")
    if args.with_mlx or args.with_local_models:
        extras.append("mlx")

    # Optional runtimes selected on an earlier launch must survive a later
    # normal launch without --with-*.  Exact sync would otherwise remove them.
    sync_command = [uv, "sync", "--inexact"]
    for extra in extras:
        sync_command.extend(["--extra", extra])
    if subprocess.run(sync_command, cwd=root, env=environment).returncode:
        return 1
    if args.sync_only:
        return 0

    # The explicit sync above has already installed the requested extras.  Do
    # not let uv perform a second sync with a narrower dependency selection.
    run_command = [uv, "run", "--no-sync"]
    run_command.extend(["semdex", "serve" if args.ui == "web" else "gui"])
    return subprocess.run(run_command, cwd=root, env=environment).returncode
