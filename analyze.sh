#!/usr/bin/env bash
# UI 없이 분석:  ./analyze.sh              → input/ 폴더 안의 음원 전부
#                ./analyze.sh ~/Downloads/audio     → 그 폴더 안의 음원 전부
#                ./analyze.sh a.m4a b.m4a --skip 30 → 파일 지정 + 옵션
set -euo pipefail
cd "$(dirname "$0")"
source ./_bootstrap.sh
mkdir -p input
exec ./venv/bin/python cli.py "$@"
