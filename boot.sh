#!/bin/bash
# =============================================================
# AI Gen API v2 — Container Start Command entrypoint
#
# Usage (RunPod template → Container Start Command):
#   bash -c "wget -qO /tmp/boot.sh https://raw.githubusercontent.com/cyrus688/ai-server/main/boot.sh && bash /tmp/boot.sh"
#
# Why this wrapper: RunPod's "Container Start Command" field replaces
# the image's CMD. runpod/comfyui:latest ships /start.sh as CMD — that's
# what starts SSH, JupyterLab, FileBrowser, and ComfyUI. If you only
# invoke setup.sh, those services never come up. This script runs
# /start.sh in the background, then fetches+runs setup.sh, then `wait`s
# on /start.sh so the container stays alive.
#
# Keeping the shell logic in a file (rather than an inlined `bash -c`
# string) avoids quoting / copy-paste errors in the template UI.
# =============================================================
set -u

log() { echo "[boot $(date '+%H:%M:%S')] $1"; }

SETUP_URL="https://raw.githubusercontent.com/cyrus688/ai-server/main/setup.sh"

log "running /start.sh in background (SSH / JupyterLab / ComfyUI)"
/start.sh &
START_PID=$!

# Give /start.sh a moment to bring DNS + venv online before we wget.
sleep 3

log "fetching setup.sh from $SETUP_URL"
if ! wget -qO /tmp/setup.sh "$SETUP_URL"; then
  log "ERROR: wget failed — check network. Container stays up so you can debug."
else
  if [ ! -s /tmp/setup.sh ]; then
    log "ERROR: downloaded setup.sh is empty — CDN or URL issue. Container stays up."
  else
    log "running setup.sh (first deploy: 3-10 min for model downloads)"
    bash /tmp/setup.sh || log "setup.sh exited non-zero (rc=$?) — container stays up for debugging"
  fi
fi

# The network volume keeps the last known-good API supervisor. Always ask it
# to start after the setup attempt so a temporary/retired setup URL cannot
# leave port 7860 offline after a pod restart. start_api.sh is flock-guarded,
# so this is harmless when setup.sh already launched it.
if [ -x /workspace/start_api.sh ]; then
  log "launching persistent API supervisor"
  setsid nohup bash /workspace/start_api.sh </dev/null >>/workspace/api_setup.log 2>&1 8>&- &
else
  log "WARN: /workspace/start_api.sh is unavailable; API supervisor not started"
fi

# Keep the container alive by waiting on /start.sh (ends in `sleep infinity`).
# Even if setup.sh failed, SSH/Jupyter/ComfyUI remain accessible so you can
# tail /workspace/api_setup.log and re-run setup.sh manually.
log "handing off to /start.sh (PID $START_PID) — container will stay alive"
wait "$START_PID"
