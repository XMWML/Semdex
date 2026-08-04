#!/usr/bin/env sh
# Start the native Semdex UI from a Linux/macOS terminal.
set -eu

APP_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$APP_ROOT"

exec python3 "$APP_ROOT/Start Semdex.py" "$@" --ui native
