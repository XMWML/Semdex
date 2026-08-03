#!/usr/bin/env sh
# Linux/macOS launcher. Use --ui web to open the WebUI instead of native UI.
set -eu

APP_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$APP_ROOT"

exec python3 "$APP_ROOT/Start Semdex.py" "$@"
