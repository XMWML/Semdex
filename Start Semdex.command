#!/bin/zsh
# Double-click on macOS.  Use --ui web to open the WebUI instead.
set -euo pipefail

APP_ROOT="${0:A:h}"
cd "$APP_ROOT"

exec /usr/bin/env python3 "$APP_ROOT/Start Semdex.py" "$@"
