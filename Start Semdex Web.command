#!/bin/zsh
# Double-click on macOS to start Semdex WebUI in the default browser.
set -euo pipefail

APP_ROOT="${0:A:h}"
cd "$APP_ROOT"

exec /usr/bin/env python3 "$APP_ROOT/Start Semdex.py" "$@" --ui web
