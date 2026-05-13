#!/usr/bin/env bash
# Two-way rsync of results/ between local and remote-gpu.
# Usage: ./sync_results.sh push | pull | both
set -euo pipefail
DIR="results"
REMOTE_DIR="~/unifying-ptq/$DIR"
LOCAL_DIR="/home/ubuntu/unifying-ptq/$DIR"
mkdir -p "$LOCAL_DIR"
ssh remote-gpu "mkdir -p $REMOTE_DIR"
case "${1:-both}" in
  push) rsync -avz --partial "$LOCAL_DIR/" "remote-gpu:$REMOTE_DIR/" ;;
  pull) rsync -avz --partial "remote-gpu:$REMOTE_DIR/" "$LOCAL_DIR/" ;;
  both) rsync -avz --partial "$LOCAL_DIR/" "remote-gpu:$REMOTE_DIR/" && rsync -avz --partial "remote-gpu:$REMOTE_DIR/" "$LOCAL_DIR/" ;;
  *) echo "usage: $0 push|pull|both" >&2; exit 1 ;;
esac
