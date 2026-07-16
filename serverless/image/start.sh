#!/usr/bin/env bash
# =============================================================
# AI Gen API v2 — Image serverless worker entrypoint
#
# Responsibilities:
#   1. Link models from the network volume into ComfyUI's models dir.
#   2. Launch ComfyUI on 127.0.0.1:8188 in the background.
#   3. `exec` the handler so RunPod's SIGTERM reaches Python directly.
#
# The network volume is mounted at /runpod-volume. Models live under
# /runpod-volume/runpod-slim/ComfyUI/models — the same layout pod-mode uses,
# since both mount the same physical volume (just at different paths).
# =============================================================
set -eu

log() { echo "[start $(date '+%H:%M:%S')] $*"; }

COMFY_ROOT="${COMFY_ROOT:-/comfyui}"
VOLUME_MODELS="/runpod-volume/runpod-slim/ComfyUI/models"

# ─── 1. Link models from the network volume ───────────────────
if [ ! -d "$VOLUME_MODELS" ]; then
  log "FATAL: network volume models not found at $VOLUME_MODELS"
  log "  Make sure the endpoint is configured with a Network Volume that"
  log "  contains the FLUX.2 + LTX models (the same volume your pod uses)."
  ls -la /runpod-volume 2>/dev/null || log "  /runpod-volume does not exist"
  exit 1
fi

# Symlink each model subdir into ComfyUI's models tree. We replace any
# existing dirs (the base image ships empty ones) with symlinks so ComfyUI
# finds weights on the network volume without copying 27 GB into the
# container's writable layer.
mkdir -p "$COMFY_ROOT/models" "$VOLUME_MODELS/vae_approx"   # defensive — base image already provides this, but never assume

# Compliance filter dirs on the volume (idempotent). The blocklists start
# empty; admins manage them via the pod's /admin/blocklist and
# /admin/blocklist-logos endpoints (mounted on the same volume).
mkdir -p /runpod-volume/blocklist /runpod-volume/blocklist_logos \
         /runpod-volume/insightface_models /runpod-volume/clip_models
for sub in diffusion_models vae vae_approx text_encoders loras checkpoints latent_upscale_models; do
  src="$VOLUME_MODELS/$sub"
  dst="$COMFY_ROOT/models/$sub"
  if [ ! -d "$src" ]; then
    log "WARN: $src missing on volume — skipping (workflow may fail to load this category)"
    continue
  fi
  rm -rf "$dst"
  ln -s "$src" "$dst"
done
log "models linked from network volume"

# ─── 2. Boot ComfyUI in the background ────────────────────────
cd "$COMFY_ROOT"
log "launching ComfyUI on 127.0.0.1:8188"
python -u main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch \
  >/tmp/comfyui.log 2>&1 &
COMFY_PID=$!
log "ComfyUI pid=$COMFY_PID (logs: /tmp/comfyui.log)"

# Hand off — the handler blocks until /system_stats responds, then enters
# runpod.serverless.start(). exec replaces this shell so RunPod's SIGTERM
# goes straight to Python.
cd /app
exec python -u handler.py
