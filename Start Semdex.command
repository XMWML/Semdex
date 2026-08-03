#!/bin/zsh
# Double-click on macOS.  All Semdex-owned state stays beside this script.
set -euo pipefail

APP_ROOT="${0:A:h}"
cd "$APP_ROOT"

export SEMDEX_HOME="$APP_ROOT"
export SEMDEX_CONFIG="$APP_ROOT/.semdex/config.toml"
export UV_CACHE_DIR="$APP_ROOT/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$APP_ROOT/.uv-python"
export PIP_CACHE_DIR="$APP_ROOT/.uv-cache/pip"
export XDG_CACHE_HOME="$APP_ROOT/.semdex/cache"
export HF_HOME="$APP_ROOT/.semdex/models/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TMPDIR="$APP_ROOT/.semdex/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TMPDIR"

if ! command -v uv >/dev/null 2>&1; then
  osascript -e 'display alert "Semdex 无法启动" message "未找到 uv。请先安装 uv，然后再次双击此文件。" as critical'
  exit 1
fi

if ! uv sync --extra gui; then
  osascript -e 'display alert "Semdex 依赖安装失败" message "请检查网络、磁盘空间和 Python 环境后重试。" as critical'
  exit 1
fi

uv run --extra gui semdex gui
