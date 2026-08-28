#!/usr/bin/env bash
# 웹 UI 서버 기동:  ./run.sh   →  http://localhost:8765
set -euo pipefail
cd "$(dirname "$0")"
source ./_bootstrap.sh
PORT="${PORT:-8765}" exec ./venv/bin/python app.py
