#!/usr/bin/env bash
# =============================================================
# AI Gen API v2 — Video serverless worker entrypoint
#
# Same shape as serverless/image/start.sh:
#   1. Link models from /runpod-volume/runpod-slim/ComfyUI/models
#   2. Ensure the output stage dir exists on the volume
#   3. Boot ComfyUI in the background, exec handler.py
# =============================================================
set -eu

log() { echo "[start $(date '+%H:%M:%S')] $*"; }

COMFY_ROOT="${COMFY_ROOT:-/comfyui}"
VOLUME_MODELS="/runpod-volume/runpod-slim/ComfyUI/models"
VOLUME_OUTPUTS="${VOLUME_OUTPUTS:-/runpod-volume/outputs}"

if [ ! -d "$VOLUME_MODELS" ]; then
  log "FATAL: network volume models not found at $VOLUME_MODELS"
  log "  Make sure the endpoint is configured with a Network Volume that"
  log "  contains the LTX-2.3 models (same volume the pod uses)."
  ls -la /runpod-volume 2>/dev/null || log "  /runpod-volume does not exist"
  exit 1
fi

mkdir -p "$VOLUME_OUTPUTS"
log "video outputs will stage to $VOLUME_OUTPUTS"

mkdir -p "$COMFY_ROOT/models" "$VOLUME_MODELS/vae_approx"   # defensive — base image already provides this
for sub in diffusion_models vae vae_approx text_encoders loras checkpoints latent_upscale_models; do
  src="$VOLUME_MODELS/$sub"
  dst="$COMFY_ROOT/models/$sub"
  if [ ! -d "$src" ]; then
    log "WARN: $src missing on volume — skipping"
    continue
  fi
  rm -rf "$dst"
  ln -s "$src" "$dst"
done
log "models linked from network volume"

cd "$COMFY_ROOT"
log "launching ComfyUI on 127.0.0.1:8188"
python -u main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch \
  >/tmp/comfyui.log 2>&1 &
COMFY_PID=$!
log "ComfyUI pid=$COMFY_PID (logs: /tmp/comfyui.log)"

cd /app
exec python -u handler.py
