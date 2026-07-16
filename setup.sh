#!/bin/bash
# =============================================================
# AI Gen API v2 — Setup
# FLUX.2 Klein 9B (face swap + text-to-image)
# LTX 2.3 22B (image-to-video, text-to-video, face-animate pipeline)
#
# Works with RunPod ComfyUI template (runpod/comfyui:latest)
# ComfyUI location: /workspace/runpod-slim/ComfyUI/
# Python venv: /workspace/runpod-slim/ComfyUI/.venv-cu128/
#
# Set as start command in template overrides:
#   bash -c "wget -qO /tmp/setup.sh https://raw.githubusercontent.com/cyrusjaysondev/ai-server/main/setup.sh && bash /tmp/setup.sh &"
#
# Required env var (set in RunPod template):
#   HF_TOKEN = your Hugging Face token
#     - needs access to: black-forest-labs/FLUX.2-klein-9B
#     - needs access to: Lightricks/LTX-2.3-fp8
#   Accept licenses at:
#     https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
#     https://huggingface.co/Lightricks/LTX-2.3-fp8
# =============================================================

LOG="/workspace/api_setup.log"
log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a $LOG; }

# ─────────────────────────────────────────────
# Single-instance lock. Prevents two setups colliding on aria2 partial
# files, duplicate /start.sh patches, or racing supervisor launches when
# e.g. an SSH session re-runs setup.sh while the template boot is still
# running. Exits 0 (no-op) if another setup is already in progress.
# ─────────────────────────────────────────────
exec 8>/var/lock/ai-gen-api-v2-setup.lock
if ! flock -n 8; then
  log "setup.sh: another setup already in progress — exiting"
  exit 0
fi

API_REPO="https://raw.githubusercontent.com/cyrusjaysondev/ai-server/main"

# ─────────────────────────────────────────────
# Bind :7860 IMMEDIATELY with an install-progress server so the proxy
# returns 503 + a useful JSON body during setup — not a silent Cloudflare
# 502. start_api.sh's STALE_PID logic will take the port cleanly when
# it's ready to bind uvicorn.
#
# Skip if :7860 is already held by the real API supervisor (pod restart
# with /start.sh hook) or if another status server is still alive.
# ─────────────────────────────────────────────
STATUS_PID_FILE=/var/run/ai-gen-api-v2-status.pid
if [ -f "$STATUS_PID_FILE" ]; then
  kill "$(cat "$STATUS_PID_FILE" 2>/dev/null)" 2>/dev/null || true
  rm -f "$STATUS_PID_FILE"
  sleep 0.5
fi

if pgrep -xf "bash /workspace/start_api.sh" >/dev/null 2>&1; then
  log "start_api.sh supervisor already running — skipping status server"
elif netstat -tln 2>/dev/null | grep -q ":7860 "; then
  log ":7860 already bound — skipping status server"
else
  cat > /tmp/ai-gen-api-v2-status.py <<'PYEOF'
import http.server, socketserver, json, os, subprocess, signal, sys
socketserver.ThreadingTCPServer.allow_reuse_address = True

def recent_log():
    try:
        return subprocess.check_output(
            ['tail', '-40', '/workspace/api_setup.log'],
            text=True, timeout=2
        ).splitlines()[-30:]
    except Exception:
        return []

class H(http.server.BaseHTTPRequestHandler):
    def _send(self, code, body):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        payload = {
            "status": "installing",
            "message": (
                "AI Gen API v2 is still setting up. First deploy downloads "
                "~72 GB of models (3-10 min on warm HF CDN). /health will "
                "return HTTP 200 once uvicorn is bound."
            ),
            "pod_id": os.environ.get('RUNPOD_POD_ID', 'unknown'),
            "hint": "tail -f /workspace/api_setup.log",
            "recent_log": recent_log(),
        }
        self._send(503, json.dumps(payload, indent=2).encode())

signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
try:
    with socketserver.ThreadingTCPServer(("0.0.0.0", 7860), H) as srv:
        srv.serve_forever()
except OSError:
    # port already taken (another instance or the real uvicorn grabbed it)
    sys.exit(0)
PYEOF
  # 8>&- closes the setup-lock FD so children don't inherit it (otherwise
  # the lock would persist for the lifetime of the daemon, blocking re-runs).
  setsid nohup python3 /tmp/ai-gen-api-v2-status.py </dev/null >/dev/null 2>&1 8>&- &
  echo $! > "$STATUS_PID_FILE"
  sleep 0.5
  if kill -0 "$(cat "$STATUS_PID_FILE")" 2>/dev/null; then
    log "Status server bound :7860 — /health returns 503 + install progress until API ready"
  else
    log "WARN: status server failed to start (port taken?) — continuing"
    rm -f "$STATUS_PID_FILE"
  fi
fi

# ─────────────────────────────────────────────
# HF Token (required for gated model downloads)
# ─────────────────────────────────────────────
TOKEN="${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}"
if [ -z "$TOKEN" ]; then
  log "ERROR: HF_TOKEN env var is not set."
  log "  Set it in your RunPod template environment variables."
  log "  Get a token at: https://huggingface.co/settings/tokens"
  log "  Then accept licenses at:"
  log "    https://huggingface.co/black-forest-labs/FLUX.2-klein-9B"
  log "    https://huggingface.co/Lightricks/LTX-2.3-fp8"
  exit 1
fi

# ─────────────────────────────────────────────
# Auto-detect ComfyUI location
# ─────────────────────────────────────────────
if [ -d "/workspace/runpod-slim/ComfyUI" ]; then
  COMFY_ROOT="/workspace/runpod-slim/ComfyUI"
elif [ -d "/workspace/ComfyUI" ]; then
  COMFY_ROOT="/workspace/ComfyUI"
else
  log "ERROR: ComfyUI not found. Searching..."
  COMFY_ROOT=$(find /workspace -name "main.py" -path "*/ComfyUI/*" -exec dirname {} \; 2>/dev/null | head -1)
  if [ -z "$COMFY_ROOT" ]; then
    log "ERROR: ComfyUI not found anywhere. Exiting."
    exit 1
  fi
fi

# ─────────────────────────────────────────────
# Auto-detect Python
# ─────────────────────────────────────────────
if [ -f "$COMFY_ROOT/.venv-cu128/bin/python" ]; then
  PYTHON="$COMFY_ROOT/.venv-cu128/bin/python"
  PIP="$COMFY_ROOT/.venv-cu128/bin/pip"
elif [ -f "/opt/venv/bin/python" ]; then
  PYTHON="/opt/venv/bin/python"
  PIP="/opt/venv/bin/pip"
else
  PYTHON=$(which python3)
  PIP=$(which pip3)
fi

MODELS="$COMFY_ROOT/models"
NODES="$COMFY_ROOT/custom_nodes"

log "=========================================="
log "AI Gen API v2 Setup Started"
log "Pod ID: $RUNPOD_POD_ID"
log "ComfyUI: $COMFY_ROOT"
log "Python: $PYTHON"
log "=========================================="

# ─────────────────────────────────────────────
# 1. Pip dependencies + aria2 (for parallel model downloads)
# ─────────────────────────────────────────────
log "[1/4] Installing pip dependencies + aria2..."
$PIP install -q fastapi uvicorn httpx websockets python-multipart pillow 2>&1 | tail -1
# Face-filter dependencies (compliance / blocklist enforcement).
# ~500MB total; the buffalo_l model itself (~280MB) is downloaded on first
# face_filter=true request and cached at $INSIGHTFACE_MODEL_ROOT on the volume.
$PIP install -q insightface onnxruntime-gpu 2>&1 | tail -1 || log "  WARN: insightface/onnxruntime-gpu install failed — face filter will return 503 if used"
# Logo/flag filter dependency (CLIP ViT-B/32). ~600MB on disk; the model
# weights (~150MB) download on first logo_filter=true request and cache at
# $CLIP_MODEL_ROOT on the volume.
$PIP install -q open_clip_torch 2>&1 | tail -1 || log "  WARN: open_clip_torch install failed — logo filter will return 503 if used"
# SageAttention package install only — DO NOT auto-enable via comfyui_args.txt.
# SageAttention 1.0.6 hangs the Gemma 12B prompt-enhancer's autoregressive
# token-generation path (the kernel is tuned for fixed-shape diffusion
# attention, not LLM decoding), which deadlocks every /ltx/* call before the
# KSampler ever runs. We leave the package installed so users who want sage
# for non-Gemma workflows can enable it manually in comfyui_args.txt.
$PIP install -q sageattention 2>&1 | tail -1 || log "  WARN: sageattention install failed — ComfyUI will fall back to PyTorch SDPA"

# Initialize the args file with just a header if it doesn't exist. We
# deliberately don't append --use-sage-attention or --fast here (see above).
ARGS_FILE="/workspace/runpod-slim/comfyui_args.txt"
[ -f "$ARGS_FILE" ] || echo "# Add your custom ComfyUI arguments here (one per line)" > "$ARGS_FILE"

if ! command -v aria2c >/dev/null 2>&1; then
  log "  Installing aria2..."
  apt-get update -qq 2>&1 | tail -1
  apt-get install -y -qq aria2 2>&1 | tail -1
fi

if ! command -v aria2c >/dev/null 2>&1; then
  log "  FATAL: aria2 install failed. Cannot do parallel downloads."
  exit 1
fi
log "  Done"

# ─────────────────────────────────────────────
# 2. Download all models in parallel via aria2
#    (Previously 6 serial wget phases → single parallel phase.
#     HF CDN routing + 16 connections/file yields 200–500 MB/s
#     vs ~1 MB/s serial wget. Full 72 GB in ~3–10 min, not hours.)
# ─────────────────────────────────────────────
log "[2/4] Downloading models (parallel, ~72 GB total)..."
mkdir -p "$MODELS/diffusion_models" "$MODELS/vae" "$MODELS/vae_approx" "$MODELS/text_encoders" \
         "$MODELS/loras" "$MODELS/checkpoints" "$MODELS/latent_upscale_models"

ARIA2_INPUT="/tmp/ai-gen-api-v2-downloads.txt"
cat > "$ARIA2_INPUT" <<EOF
https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/resolve/main/flux-2-klein-9b.safetensors
  dir=$MODELS/diffusion_models
  out=flux2-klein-9b.safetensors
https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/vae/flux2-vae.safetensors
  dir=$MODELS/vae
  out=flux2-vae.safetensors
https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors
  dir=$MODELS/text_encoders
  out=qwen_3_8b_fp8mixed.safetensors
https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/resolve/main/bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors
  dir=$MODELS/loras
  out=bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors
https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-dev-fp8.safetensors
  dir=$MODELS/checkpoints
  out=ltx-2.3-22b-dev-fp8.safetensors
https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-lora-384.safetensors
  dir=$MODELS/loras
  out=ltx-2.3-22b-distilled-lora-384.safetensors
https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors
  dir=$MODELS/loras
  out=gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors
https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors
  dir=$MODELS/text_encoders
  out=gemma_3_12B_it_fp4_mixed.safetensors
https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.0.safetensors
  dir=$MODELS/latent_upscale_models
  out=ltx-2.3-spatial-upscaler-x2-1.0.safetensors
EOF

# HF token is passed as an Authorization header for all requests.
# Public repos ignore it; gated repos (FLUX.2, LTX-2.3-fp8) require it.
# --continue=true skips fully-downloaded files and resumes partials,
# so rerunning this script after a network blip is a no-op for done files.
run_aria2() {
  aria2c \
    --input-file="$ARIA2_INPUT" \
    --header="Authorization: Bearer $TOKEN" \
    --max-connection-per-server=16 \
    --split=16 \
    --min-split-size=10M \
    --max-concurrent-downloads=3 \
    --continue=true \
    --allow-overwrite=true \
    --auto-file-renaming=false \
    --file-allocation=none \
    --console-log-level=warn \
    --summary-interval=30 \
    2>&1 | tee -a "$LOG" | grep -E "Download complete|error|FAILED" || true
  return ${PIPESTATUS[0]}
}

run_aria2
ARIA2_EXIT=$?
if [ $ARIA2_EXIT -ne 0 ]; then
  log "  WARN: aria2 exited with code $ARIA2_EXIT — will verify and retry."
fi

# Per-file verification. aria2 has been observed to exit 0 while silently
# leaving individual files missing or short (e.g., a gated URL transiently
# 401s, or the parent shell gets SIGTERM mid-download). Trusting the exit
# code alone causes "All N models downloaded" to print over a broken set.
# So we compare each file's on-disk size to HF's x-linked-size header and
# resume any that don't match before declaring success.
#
# local_path|HF_URL — kept in lockstep with $ARIA2_INPUT above.
EXPECTED_FILES="\
diffusion_models/flux2-klein-9b.safetensors|https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/resolve/main/flux-2-klein-9b.safetensors
vae/flux2-vae.safetensors|https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/vae/flux2-vae.safetensors
text_encoders/qwen_3_8b_fp8mixed.safetensors|https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors
loras/bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors|https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/resolve/main/bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors
checkpoints/ltx-2.3-22b-dev-fp8.safetensors|https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-dev-fp8.safetensors
loras/ltx-2.3-22b-distilled-lora-384.safetensors|https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-lora-384.safetensors
loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors|https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors
text_encoders/gemma_3_12B_it_fp4_mixed.safetensors|https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors
latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors|https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"

verify_models() {
  local bad=0
  local entry local_path url expected actual full
  while IFS='|' read -r local_path url; do
    [ -z "$local_path" ] && continue
    full="$MODELS/$local_path"
    expected=$(curl -sSI -H "Authorization: Bearer $TOKEN" "$url" 2>/dev/null \
      | tr -d '\r' | awk 'tolower($1)=="x-linked-size:"{print $2; exit}')
    actual=$(stat -c%s "$full" 2>/dev/null || echo 0)
    if [ -z "$expected" ]; then
      log "    WARN: could not fetch expected size for $local_path (HF unreachable?)"
      continue
    fi
    if [ "$expected" != "$actual" ]; then
      log "    INCOMPLETE: $local_path (have $actual, expected $expected)"
      bad=$((bad+1))
    fi
  done <<<"$EXPECTED_FILES"
  return $bad
}

log "  Verifying downloaded files against HF expected sizes..."
if ! verify_models; then
  log "  One or more files incomplete — resuming via aria2 (--continue)..."
  run_aria2 || true
  if ! verify_models; then
    log "  FATAL: model files still incomplete after retry. Check:"
    log "    - HF token has access to gated repos:"
    log "        black-forest-labs/FLUX.2-klein-9B"
    log "        Lightricks/LTX-2.3-fp8"
    log "    - Disk space ($(df -h "$MODELS" | awk 'NR==2{print $4 " free on " $6}'))"
    log "    - Network connectivity to huggingface.co"
    exit 1
  fi
fi

# Symlink so both filenames resolve (workflows.py references flux-2-klein-9b.safetensors).
# Use a RELATIVE link so it works regardless of mount point — pods see the
# volume at /workspace/, serverless workers see it at /runpod-volume/.
# An absolute link to /workspace/... would break inside serverless containers.
(cd "$MODELS/diffusion_models" && \
 ln -sfn flux2-klein-9b.safetensors flux-2-klein-9b.safetensors)

log "  All 9 models verified at expected sizes"

# ─────────────────────────────────────────────
# 2b. Brand assets (Metfone GenAI logo for the watermark_image option)
# ─────────────────────────────────────────────
# Tiny PNG kept on the volume so every pod + serverless worker can find it.
# watermark.py loads /workspace/assets/metfone_genai_watermark.png lazily and silently
# skips the image overlay if it's missing, so a network blip here is
# non-fatal.
ASSETS_DIR="/workspace/assets"
METFONE_LOGO="$ASSETS_DIR/metfone_genai_watermark.png"
METFONE_LOGO_URL="$API_REPO/assets/metfone_genai_watermark.png"
mkdir -p "$ASSETS_DIR"
if [ -s "$METFONE_LOGO" ]; then
  log "  Metfone GenAI logo already on volume ($(stat -c%s "$METFONE_LOGO" 2>/dev/null || stat -f%z "$METFONE_LOGO") bytes)"
else
  log "  Downloading Metfone GenAI logo..."
  if wget -qO "$METFONE_LOGO" "$METFONE_LOGO_URL"; then
    log "    Logo saved to $METFONE_LOGO"
  else
    log "    WARN: Metfone GenAI logo download failed — watermark_image will be a no-op until present"
    rm -f "$METFONE_LOGO"
  fi
fi

# Zodiac overlays (caption_icon option: gold sign glyph + gold divider)
# ─────────────────────────────────────────────
# 12 sign glyphs + one gold divider, kept on the volume for the horoscope
# caption design. watermark.py loads them lazily for `caption_icon` and
# silently degrades to text-only if any are missing, so a blip here is
# non-fatal. Fetched from the repo (same source of truth as the API code).
ZODIAC_DIR="$ASSETS_DIR/zodiac-overlays"
mkdir -p "$ZODIAC_DIR/icons"
if [ -s "$ZODIAC_DIR/divider-gold.png" ] && [ "$(ls -1 "$ZODIAC_DIR"/icons/*.png 2>/dev/null | wc -l)" -ge 12 ]; then
  log "  Zodiac overlays already on volume ($(ls -1 "$ZODIAC_DIR"/icons/*.png 2>/dev/null | wc -l) glyphs + divider)"
else
  log "  Downloading zodiac overlays (12 glyphs + divider)..."
  zod_ok=0
  for sign in aries taurus gemini cancer leo virgo libra scorpio sagittarius capricorn aquarius pisces; do
    if wget -qO "$ZODIAC_DIR/icons/$sign.png" "$API_REPO/assets/zodiac-overlays/icons/$sign.png" && [ -s "$ZODIAC_DIR/icons/$sign.png" ]; then
      zod_ok=$((zod_ok + 1))
    else
      rm -f "$ZODIAC_DIR/icons/$sign.png"
    fi
  done
  if wget -qO "$ZODIAC_DIR/divider-gold.png" "$API_REPO/assets/zodiac-overlays/divider-gold.png" && [ -s "$ZODIAC_DIR/divider-gold.png" ]; then
    log "    Zodiac overlays: $zod_ok/12 glyphs + divider saved"
  else
    rm -f "$ZODIAC_DIR/divider-gold.png"
    log "    WARN: divider download failed — caption_icon will show the glyph only ($zod_ok/12 glyphs)"
  fi
fi

# Background-music bed (background_music option on /ltx/i2v + /ltx/t2v)
# ─────────────────────────────────────────────
# A short royalty-free looping ambient track muxed under horoscope videos.
# main.py muxes it lazily and silently no-ops if it's missing, so a blip here
# is non-fatal. Fetched from the repo (same source of truth as the API code).
BGM_FILE="$ASSETS_DIR/horoscope_bgm.m4a"
if [ -s "$BGM_FILE" ]; then
  log "  Background music already on volume ($(stat -c%s "$BGM_FILE" 2>/dev/null || stat -f%z "$BGM_FILE") bytes)"
else
  log "  Downloading background-music bed..."
  if wget -qO "$BGM_FILE" "$API_REPO/assets/horoscope_bgm.m4a" && [ -s "$BGM_FILE" ]; then
    log "    Background music saved to $BGM_FILE"
  else
    rm -f "$BGM_FILE"
    log "    WARN: background-music download failed — background_music will be a no-op until present"
  fi
fi

# Background-music pool — one track is picked at random per video.
BGM_DIR="$ASSETS_DIR/bgm"
mkdir -p "$BGM_DIR"
if [ "$(ls -1 "$BGM_DIR"/*.mp3 2>/dev/null | wc -l)" -ge 5 ]; then
  log "  Background-music pool already on volume ($(ls -1 "$BGM_DIR"/* 2>/dev/null | wc -l) tracks)"
else
  log "  Downloading background-music pool..."
  bgm_ok=0
  for n in 1 2 3 4 5; do
    if wget -qO "$BGM_DIR/audio-$n.mp3" "$API_REPO/assets/bgm/audio-$n.mp3" && [ -s "$BGM_DIR/audio-$n.mp3" ]; then
      bgm_ok=$((bgm_ok + 1))
    else
      rm -f "$BGM_DIR/audio-$n.mp3"
    fi
  done
  log "    Background-music pool: $bgm_ok/5 tracks saved"
fi

# Caption font — EB Garamond (OFL, free for commercial use).
# ─────────────────────────────────────────────
# watermark.py renders the horoscope caption in this serif; it falls back to the
# bundled DejaVu Serif if this is missing, so a blip here is non-fatal.
FONTS_DIR="$ASSETS_DIR/fonts"
mkdir -p "$FONTS_DIR"
if [ -s "$FONTS_DIR/EBGaramond.ttf" ]; then
  log "  Caption font already on volume ($(stat -c%s "$FONTS_DIR/EBGaramond.ttf" 2>/dev/null || stat -f%z "$FONTS_DIR/EBGaramond.ttf") bytes)"
else
  log "  Downloading caption font (EB Garamond)..."
  if wget -qO "$FONTS_DIR/EBGaramond.ttf" "$API_REPO/assets/fonts/EBGaramond.ttf" && [ -s "$FONTS_DIR/EBGaramond.ttf" ]; then
    wget -qO "$FONTS_DIR/OFL.txt" "$API_REPO/assets/fonts/OFL.txt" 2>/dev/null || true
    log "    Caption font saved to $FONTS_DIR/EBGaramond.ttf"
  else
    rm -f "$FONTS_DIR/EBGaramond.ttf"
    log "    WARN: caption-font download failed — captions fall back to DejaVu Serif"
  fi
fi

# Khmer caption font — Noto Serif Khmer (OFL). EB Garamond has no Khmer glyphs, so
# watermark.py uses this for Khmer ('km') horoscope captions; Latin/Vietnamese
# stay on EB Garamond. Missing it only affects Khmer captions (they'd tofu).
if [ -s "$FONTS_DIR/NotoSerifKhmer.ttf" ]; then
  log "  Khmer caption font already on volume ($(stat -c%s "$FONTS_DIR/NotoSerifKhmer.ttf" 2>/dev/null || stat -f%z "$FONTS_DIR/NotoSerifKhmer.ttf") bytes)"
else
  log "  Downloading Khmer caption font (Noto Serif Khmer)..."
  if wget -qO "$FONTS_DIR/NotoSerifKhmer.ttf" "$API_REPO/assets/fonts/NotoSerifKhmer.ttf" && [ -s "$FONTS_DIR/NotoSerifKhmer.ttf" ]; then
    wget -qO "$FONTS_DIR/NotoSerifKhmer-OFL.txt" "$API_REPO/assets/fonts/NotoSerifKhmer-OFL.txt" 2>/dev/null || true
    log "    Khmer caption font saved to $FONTS_DIR/NotoSerifKhmer.ttf"
  else
    rm -f "$FONTS_DIR/NotoSerifKhmer.ttf"
    log "    WARN: Khmer caption-font download failed — Khmer captions will tofu"
  fi
fi

# ─────────────────────────────────────────────
# 3. Custom nodes: LanPaint (FLUX face swap) + ComfyUI-KJNodes (ColorMatch for i2v)
# KJNodes ships with runpod/comfyui:latest at the time of writing — this clone is
# a defensive fallback in case a future base image drops it.
# ─────────────────────────────────────────────
mkdir -p "$NODES"
LANPAINT_FRESH=0
if [ ! -d "$NODES/LanPaint" ]; then
  log "[3/4] Installing LanPaint custom node..."
  (
    cd "$NODES"
    git clone -q https://github.com/scraed/LanPaint
    if [ -f "LanPaint/requirements.txt" ]; then
      $PIP install -q -r LanPaint/requirements.txt 2>&1 | tail -1
    fi
  )
  LANPAINT_FRESH=1
  log "  LanPaint installed"
else
  log "[3/4] LanPaint already installed"
fi

if [ ! -d "$NODES/ComfyUI-KJNodes" ]; then
  log "  Installing ComfyUI-KJNodes (provides ColorMatch for i2v color correction)..."
  (
    cd "$NODES"
    git clone -q https://github.com/kijai/ComfyUI-KJNodes
    if [ -f "ComfyUI-KJNodes/requirements.txt" ]; then
      $PIP install -q -r ComfyUI-KJNodes/requirements.txt 2>&1 | tail -1
    fi
  )
  log "  ComfyUI-KJNodes installed"
else
  log "  ComfyUI-KJNodes already installed"
fi

# ComfyUI-VideoHelperSuite — provides VHS_LoadVideo / VHS_VideoCombine for
# the /ltx/motion endpoint. Reads a reference video into a frame tensor that
# the LTX VAE can encode into motion latents, then writes the LTX sampler's
# output back out as mp4. Without this the motion-control workflow can't
# load the reference. Pinned to upstream main; the pod auto-pulls on each
# boot via this block.
#
# Unlike KJNodes/LanPaint above we ALWAYS run the install step, even when
# the directory exists, because a previous boot may have partially cloned
# the repo or failed pip install — leaving an empty / broken dir that the
# `[ ! -d ]` guard would skip on the next boot, permanently blocking the
# motion endpoint. The git clone command short-circuits if the repo is
# already healthy; pip install is idempotent. Output is captured to a
# named log file so post-mortem inspection doesn't need ssh into the pod.
VHS_DIR="$NODES/ComfyUI-VideoHelperSuite"
VHS_LOG="/workspace/setup-vhs.log"
{
  echo "=== VHS install run: $(date -u +%FT%TZ) ==="
  if [ ! -d "$VHS_DIR/.git" ]; then
    # Either missing OR a partial clone (no .git). Wipe and re-clone.
    log "  Installing ComfyUI-VideoHelperSuite (clean clone)..."
    rm -rf "$VHS_DIR"
    (cd "$NODES" && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) || \
      log "  ⚠️  ComfyUI-VideoHelperSuite git clone FAILED — /ltx/motion will not work"
  else
    log "  ComfyUI-VideoHelperSuite: git pull (refresh)"
    (cd "$VHS_DIR" && git pull --ff-only) || log "  ⚠️  VHS git pull failed (continuing with existing checkout)"
  fi

  if [ -f "$VHS_DIR/requirements.txt" ]; then
    log "  Installing VHS requirements..."
    $PIP install -r "$VHS_DIR/requirements.txt" || \
      log "  ⚠️  VHS pip install FAILED — node will be present but may fail to load at ComfyUI startup"
  else
    log "  ⚠️  VHS requirements.txt missing — clone is corrupt"
  fi

  # Quick sanity check — does the __init__ exist? If not, the node will
  # silently fail to register and the pod-side error will look like
  # "Node 'VHS_LoadVideo' not found".
  if [ -f "$VHS_DIR/__init__.py" ]; then
    log "  ComfyUI-VideoHelperSuite ready ($(wc -l < "$VHS_DIR/__init__.py") lines in __init__.py)"
  else
    log "  ⚠️  VHS __init__.py missing — ComfyUI will not load VHS_LoadVideo"
  fi
} >>"$VHS_LOG" 2>&1
log "  (VHS install details: tail -200 $VHS_LOG)"

# (The conditional LanPaint-only ComfyUI relaunch that used to live here
# is now subsumed by the start_comfy.sh supervisor below: it unconditionally
# kills /start.sh's unsupervised ComfyUI and relaunches under flock'd
# auto-restart, so newly-installed custom nodes are always picked up.)

# ─────────────────────────────────────────────
# 4. Download API + create startup scripts
# ─────────────────────────────────────────────
log "[4/4] Setting up API..."
mkdir -p /workspace/api

# Always fetch latest main.py + workflows.py + safety.py from repo
wget -q -O /workspace/api/main.py "${API_REPO}/main.py"
if [ ! -s "/workspace/api/main.py" ]; then
  log "  ERROR: Failed to download main.py"
  exit 1
fi
wget -q -O /workspace/api/workflows.py "${API_REPO}/workflows.py"
if [ ! -s "/workspace/api/workflows.py" ]; then
  log "  ERROR: Failed to download workflows.py (shared with serverless workers)"
  exit 1
fi
wget -q -O /workspace/api/image_output.py "${API_REPO}/image_output.py"
if [ ! -s "/workspace/api/image_output.py" ]; then
  log "  ERROR: Failed to download image_output.py"
  exit 1
fi
wget -q -O /workspace/api/safety.py "${API_REPO}/safety.py"
if [ ! -s "/workspace/api/safety.py" ]; then
  log "  WARN: Failed to download safety.py — face_filter parameter will return 503 if used"
fi
wget -q -O /workspace/api/logo_safety.py "${API_REPO}/logo_safety.py"
if [ ! -s "/workspace/api/logo_safety.py" ]; then
  log "  WARN: Failed to download logo_safety.py — logo_filter parameter will return 503 if used"
fi
wget -q -O /workspace/api/watermark.py "${API_REPO}/watermark.py"
if [ ! -s "/workspace/api/watermark.py" ]; then
  log "  WARN: Failed to download watermark.py — watermark parameter will be a no-op"
fi
log "  main.py + workflows.py + image_output.py + safety.py + logo_safety.py + watermark.py downloaded (latest)"

# Create blocklist dirs so admins know where files land
mkdir -p /workspace/blocklist /workspace/blocklist_logos

# Save detected paths for start_api.sh and start_comfy.sh
cat > /workspace/api/config.env << CONFEOF
COMFY_ROOT=$COMFY_ROOT
PYTHON=$PYTHON
PIP=$PIP
API_REPO=$API_REPO
CONFEOF

# ─────────────────────────────────────────────
# Create the ComfyUI supervisor (mirrors start_api.sh: flock'd, auto-restart
# loop, parsed CLI args from comfyui_args.txt). RunPod's /start.sh launches
# ComfyUI as an unsupervised child — if it segfaults, OOMs, or is killed,
# /start.sh's `wait` returns and ComfyUI stays down until pod restart. The
# supervisor relaunches it within 5 seconds on any exit code.
# ─────────────────────────────────────────────
cat > /workspace/start_comfy.sh << 'COMFYEOF'
#!/bin/bash
# =============================================================
# AI Gen API v2 — supervisor for ComfyUI on :8188
#
# Safe to invoke multiple times: a flock guards the while-loop so
# a second invocation (e.g. setup.sh re-running on pod restart)
# exits immediately instead of racing the first supervisor.
#
# Reads extra ComfyUI CLI flags one-per-line from
# /workspace/runpod-slim/comfyui_args.txt (comments + blanks ignored).
# =============================================================
LOG_SETUP="/workspace/comfy_setup.log"
LOG_OUT="/workspace/comfyui.log"
ARGS_FILE="/workspace/runpod-slim/comfyui_args.txt"
PORT=8188
FIXED_ARGS=(--listen 0.0.0.0 --port "$PORT" --enable-cors-header)

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_SETUP"; }

# ─── single-instance guard ───
exec 9>/var/lock/ai-gen-comfy.lock
if ! flock -n 9; then
  log "start_comfy.sh: another supervisor already running — exiting"
  exit 0
fi

# Truncate old logs on restart (cap at last 500 lines each)
tail -500 "$LOG_SETUP" > "${LOG_SETUP}.tmp" 2>/dev/null && mv "${LOG_SETUP}.tmp" "$LOG_SETUP"
tail -500 "$LOG_OUT"   > "${LOG_OUT}.tmp"   2>/dev/null && mv "${LOG_OUT}.tmp"   "$LOG_OUT"

if [ ! -f /workspace/api/config.env ]; then
  log "ERROR: /workspace/api/config.env missing — setup.sh did not complete"
  exit 1
fi
source /workspace/api/config.env

if [ -z "$PYTHON" ] || [ -z "$COMFY_ROOT" ]; then
  log "ERROR: PYTHON or COMFY_ROOT not set in config.env"
  exit 1
fi
if [ ! -x "$PYTHON" ]; then
  log "ERROR: PYTHON ($PYTHON) not executable"
  exit 1
fi
if [ ! -f "$COMFY_ROOT/main.py" ]; then
  log "ERROR: $COMFY_ROOT/main.py not found"
  exit 1
fi

# Free :PORT if a stale ComfyUI is holding it (e.g. the one /start.sh
# launched on pod boot). Target by socket owner — safer than pkill -f
# against an argv pattern, which can accidentally match caller shells.
STALE_PID=$(netstat -tlnp 2>/dev/null | awk -v p=":$PORT\$" '$4 ~ p {split($7, a, "/"); print a[1]; exit}')
if [ -n "$STALE_PID" ] && [ "$STALE_PID" != "-" ]; then
  log "Freeing :$PORT (stale owner PID=$STALE_PID)"
  kill "$STALE_PID" 2>/dev/null || true
  for _ in 1 2 3 4 5; do kill -0 "$STALE_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$STALE_PID" 2>/dev/null || true
fi

# Parse extra args from comfyui_args.txt: strip comments + blank lines,
# word-split each remaining line into one or more args.
EXTRA_ARGS=()
if [ -f "$ARGS_FILE" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    trimmed="${line#"${line%%[![:space:]]*}"}"
    [ -z "$trimmed" ] && continue
    [[ "$trimmed" =~ ^# ]] && continue
    # shellcheck disable=SC2206
    args=($trimmed)
    EXTRA_ARGS+=("${args[@]}")
  done < "$ARGS_FILE"
fi
log "ComfyUI extra args: ${EXTRA_ARGS[*]:-(none)}"

cd "$COMFY_ROOT" || exit 1
log "Starting ComfyUI on port $PORT..."
while true; do
  "$PYTHON" main.py "${FIXED_ARGS[@]}" "${EXTRA_ARGS[@]}" >> "$LOG_OUT" 2>&1
  EXIT_CODE=$?
  log "ComfyUI exited with code $EXIT_CODE — restarting in 5s..."
  sleep 5
done
COMFYEOF

chmod +x /workspace/start_comfy.sh

# Launch the ComfyUI supervisor unless one's already running. The supervisor's
# STALE_PID step will adopt :8188 from /start.sh's unsupervised ComfyUI on the
# first launch; subsequent setup re-runs hit the flock and exit cleanly.
# 8>&- closes the setup-lock FD so the daemon doesn't inherit it.
if pgrep -xf "bash /workspace/start_comfy.sh" >/dev/null 2>&1; then
  log "  ComfyUI supervisor already running — leaving it alone"
else
  setsid nohup bash /workspace/start_comfy.sh </dev/null >>/workspace/api_setup.log 2>&1 8>&- &
  disown 2>/dev/null || true
  log "  ComfyUI supervisor launched (will adopt :8188 from /start.sh)"
fi

# Create startup script (runs on every pod start/restart)
cat > /workspace/start_api.sh << 'STARTEOF'
#!/bin/bash
# =============================================================
# AI Gen API v2 — supervisor for uvicorn on :7860
#
# Safe to invoke multiple times: a flock guards the while-loop so
# a second invocation (e.g. setup.sh re-running on pod restart)
# exits immediately instead of racing the first supervisor.
# =============================================================
LOG="/workspace/api_setup.log"
log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

# ─── single-instance guard ───
# Hold an exclusive lock for the lifetime of this process. If another
# supervisor is already running, exit 0 (not an error — it's a no-op).
exec 9>/var/lock/ai-gen-api-v2.lock
if ! flock -n 9; then
  log "start_api.sh: another supervisor already running — exiting"
  exit 0
fi

# Truncate old logs on restart (cap at last 500 lines each)
tail -500 "$LOG" > "${LOG}.tmp" 2>/dev/null && mv "${LOG}.tmp" "$LOG"
tail -500 /workspace/api.log > /workspace/api.log.tmp 2>/dev/null && mv /workspace/api.log.tmp /workspace/api.log

# Load detected Python/pip paths
if [ ! -f /workspace/api/config.env ]; then
  log "ERROR: /workspace/api/config.env missing — setup.sh did not complete"
  exit 1
fi
source /workspace/api/config.env

# Reinstall pip deps (can be lost on pod restart)
log "Installing pip deps..."
$PIP install -q fastapi uvicorn httpx websockets python-multipart pillow 2>&1 | tail -1

# Fetch latest main.py + workflows.py + safety.py + logo_safety.py + watermark.py from repo
# Defined as a function so the supervisor loop below can call it BEFORE
# every uvicorn restart. Without that, killing uvicorn (e.g. via the
# /admin/install-comfy-node pkill) only restarts the OLD code — to deploy
# fresh code you'd need a container reboot. Calling fetch_api_code inside
# the loop turns a uvicorn-only restart into a real code deploy.
fetch_api_code() {
  log "Fetching latest API code..."
  for f in main.py workflows.py image_output.py safety.py logo_safety.py watermark.py; do
    wget -q -O "/workspace/api/$f.new" "${API_REPO}/$f"
    if [ -s "/workspace/api/$f.new" ]; then
      mv "/workspace/api/$f.new" "/workspace/api/$f"
    else
      log "WARN: Failed to download $f — using existing version"
      rm -f "/workspace/api/$f.new"
    fi
  done
}

fetch_api_code

# Wait for ComfyUI to be ready
log "Waiting for ComfyUI..."
MAX_WAIT=600; WAITED=0
until curl -s http://localhost:8188/system_stats > /dev/null 2>&1; do
  sleep 3; WAITED=$((WAITED + 3))
  if [ $WAITED -ge $MAX_WAIT ]; then log "ERROR: ComfyUI did not start within 10 min"; exit 1; fi
done
log "ComfyUI ready after ${WAITED}s"

# Free :7860 if a stale uvicorn from a prior supervisor is holding it.
# Target by socket owner (netstat) — safer than pkill -f against an argv
# pattern, which can accidentally match caller shells whose cmdline
# happens to contain "uvicorn main:app" as a substring.
STALE_PID=$(netstat -tlnp 2>/dev/null | awk '$4 ~ /:7860$/ {split($7, a, "/"); print a[1]; exit}')
if [ -n "$STALE_PID" ]; then
  log "Freeing :7860 (stale owner PID=$STALE_PID)"
  kill "$STALE_PID" 2>/dev/null || true
  for _ in 1 2 3 4 5; do kill -0 "$STALE_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$STALE_PID" 2>/dev/null || true
fi

# Start API with auto-restart on crash.
# IMPORTANT: re-fetch latest code on EVERY restart, not just when this
# supervisor first starts. This is what lets `/admin/refresh-api-code` (and
# any pkill -9 -f uvicorn) deliver a fresh deploy without a container
# reboot. The wget is fast (~50ms × 5 files); the cost is negligible.
cd /workspace/api || exit 1
log "Starting API on port 7860..."
while true; do
  fetch_api_code
  $PYTHON -m uvicorn main:app --host 0.0.0.0 --port 7860 >> /workspace/api.log 2>&1
  EXIT_CODE=$?
  log "API exited with code $EXIT_CODE — restarting in 5s..."
  sleep 5
done
STARTEOF

chmod +x /workspace/start_api.sh

# ─────────────────────────────────────────────
# Patch /start.sh (idempotent) so pod RESTARTS also auto-launch the API.
#
# /start.sh is the image's CMD — RunPod runs it on every pod start.
# Without this hook, a restart would bring up ComfyUI but not the API,
# forcing manual `bash /workspace/start_api.sh` each time.
#
# The patch is injected on the container layer (/start.sh is not on the
# /workspace volume). It survives pod restarts but is lost on pod
# recreate/rebuild — setup.sh re-applies it on every run, so as long as
# setup.sh is in the template Start Command, the hook self-heals.
# ─────────────────────────────────────────────
patch_start_sh() {
  local f="/start.sh"
  [ -f "$f" ] || { log "  /start.sh not found — skipping restart hook"; return; }
  # The hook block is versioned (v2). If the file already contains the v2
  # marker, skip; if it contains the older v1 marker, we strip it and re-apply
  # v2 so we don't end up with stacked hooks.
  if grep -q "AI Gen API v2 auto-start hook v2" "$f"; then
    log "  /start.sh already has v2 restart hook"
    return
  fi
  # Insert hook right after ComfyUI is launched (line: `python main.py $FIXED_ARGS &`),
  # before the `wait $COMFY_PID` call. setsid + nohup + </dev/null fully detaches so
  # the supervisors survive SSH disconnect, shell exit, and /start.sh teardown.
  #
  # Use atomic rename (write-new-then-mv) so the currently-running /start.sh
  # (which may still be reading the script) isn't truncated mid-read. Processes
  # holding the old inode via their open fd continue reading the old content;
  # new invocations see the new file.
  python3 - "$f" <<'PYEOF'
import os, sys, re
p = sys.argv[1]
src = open(p).read()
# Strip any previously-injected hook block (v1 or v2) so re-runs don't stack.
src = re.sub(
    r'\n# === AI Gen API v2 auto-start hook[^\n]*===\n.*?# === end AI Gen API v2 auto-start hook ===\n',
    '\n',
    src,
    flags=re.S,
)
hook = '''
# === AI Gen API v2 auto-start hook v2 ===
# Launches the ComfyUI + API supervisors in detached sessions so they
# survive shell exit, SSH disconnect, and /start.sh teardown.
#   - start_comfy.sh adopts :8188 from /start.sh's unsupervised ComfyUI
#     and provides auto-restart on crash.
#   - start_api.sh waits for ComfyUI before binding :7860.
if [ -x /workspace/start_comfy.sh ]; then
    echo "AI Gen API v2: launching /workspace/start_comfy.sh"
    setsid nohup bash /workspace/start_comfy.sh </dev/null >>/workspace/api_setup.log 2>&1 &
fi
if [ -x /workspace/start_api.sh ]; then
    echo "AI Gen API v2: launching /workspace/start_api.sh"
    setsid nohup bash /workspace/start_api.sh </dev/null >>/workspace/api_setup.log 2>&1 &
fi
# === end AI Gen API v2 auto-start hook ===
'''
m = re.search(r'^(python main\.py \$FIXED_ARGS &\s*\nCOMFY_PID=\$!\s*\n)', src, re.M)
if not m:
    sys.exit("could not locate ComfyUI launch block in /start.sh")
out = src[:m.end()] + hook + src[m.end():]
tmp = p + ".new"
with open(tmp, "w") as fh:
    fh.write(out)
os.chmod(tmp, os.stat(p).st_mode)
os.rename(tmp, p)
PYEOF
  log "  /start.sh patched with v2 restart hook (supervisors for ComfyUI + API)"
}
patch_start_sh

# ─────────────────────────────────────────────
# Install ComfyUI bootstrap custom node — primary auto-recovery mechanism.
#
# Why: /start.sh patching above only survives the current container. On a pod
# restart RunPod recreates the container, so /start.sh is wiped back to its
# image-default state and our hook is gone. ComfyUI's custom_nodes/ directory
# lives on the /workspace network volume (persistent) and is imported by
# ComfyUI on every startup — so a side-effect-only custom node is the only
# place we can reliably hook "run something every time ComfyUI starts"
# without owning the image.
#
# The bootstrap module spawns /workspace/start_api.sh on import. The
# supervisor is flock-guarded, so duplicate launches are no-ops.
# ─────────────────────────────────────────────
install_bootstrap_node() {
  local NODE_DIR="$COMFY_ROOT/custom_nodes/ai_gen_api_bootstrap"
  mkdir -p "$NODE_DIR"
  cat > "$NODE_DIR/__init__.py" << 'BOOTEOF'
"""
ai-gen-api-v2 bootstrap — launches the API supervisor when ComfyUI starts.

This is a no-op ComfyUI custom node whose only purpose is the side-effect of
spawning /workspace/start_api.sh on import. ComfyUI imports every custom node
on startup, so this fires on every pod restart without needing /start.sh to
be patched (the image's /start.sh is wiped on each container recreation).

The supervisor itself is flock-guarded, so duplicate spawns are no-ops.
"""
import os
import subprocess

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_SUPERVISOR = "/workspace/start_api.sh"
_LOG = "/workspace/api_setup.log"
_TAG = "[ai-gen-api-bootstrap]"


def _launch_supervisor():
    if not os.path.exists(_SUPERVISOR):
        print(f"{_TAG} {_SUPERVISOR} not found — run setup.sh first", flush=True)
        return
    try:
        with open(_LOG, "a") as logf:
            subprocess.Popen(
                ["setsid", "nohup", "bash", _SUPERVISOR],
                stdin=subprocess.DEVNULL,
                stdout=logf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        print(f"{_TAG} launched {_SUPERVISOR} (flock-guarded; safe to retry)", flush=True)
    except Exception as e:
        print(f"{_TAG} failed to launch supervisor: {e}", flush=True)


_launch_supervisor()
BOOTEOF
  log "  bootstrap custom node installed at $NODE_DIR"
}
install_bootstrap_node

# ─────────────────────────────────────────────
# Start the API — but only if it isn't already healthy.
#
# On a pod restart, the /start.sh hook already launched start_api.sh in
# parallel with setup.sh. If that supervisor is up and the health probe
# passes, don't tear it down — a pointless relaunch would cause a ~10s
# outage where uvicorn isn't bound to :7860.
# Otherwise: kill any stale supervisor and start fresh.
# ─────────────────────────────────────────────
if pgrep -xf "bash /workspace/start_api.sh" >/dev/null 2>&1 && \
   curl -s -m 3 http://localhost:7860/health 2>/dev/null | grep -q '"status":"ok"'; then
  log "API supervisor already healthy — leaving it alone"
else
  # Use -xf (exact full-argv match) so we only kill processes whose argv
  # is literally "bash /workspace/start_api.sh" — never a caller shell
  # that merely mentions the string in its own command line.
  pkill -xf "bash /workspace/start_api.sh" 2>/dev/null || true
  sleep 1
  setsid nohup bash /workspace/start_api.sh </dev/null >>/workspace/api_setup.log 2>&1 8>&- &
  disown 2>/dev/null || true
  log "API supervisor launched"
fi

# ─────────────────────────────────────────────
# Video reaper — deletes videos older than RETENTION_DAYS (default 3)
# from both the pod-mode output dir and the serverless staging dir on
# the network volume. Without this, generated videos accumulate
# forever and eventually fill the volume.
# ─────────────────────────────────────────────
log "Installing video reaper..."
wget -q -O /workspace/cleanup.sh "${API_REPO}/cleanup.sh"
if [ ! -s "/workspace/cleanup.sh" ]; then
  log "  WARN: failed to download cleanup.sh — videos will not be auto-pruned"
else
  chmod +x /workspace/cleanup.sh

  # Supervisor: runs cleanup.sh once, then sleeps 24h, repeats. Flock'd so
  # re-running setup.sh doesn't spawn a second loop.
  cat > /workspace/start_cleanup.sh << 'CLEANEOF'
#!/bin/bash
# =============================================================
# AI Gen API v2 — daily reaper supervisor
# Runs /workspace/cleanup.sh once every 24h. Flock guarded.
# =============================================================
LOG="/workspace/cleanup.log"
exec 9>/var/lock/ai-gen-cleanup.lock
if ! flock -n 9; then
  echo "[$(date '+%H:%M:%S')] start_cleanup.sh: another reaper running — exiting" >> "$LOG"
  exit 0
fi
while true; do
  bash /workspace/cleanup.sh
  sleep 86400  # 24h
done
CLEANEOF
  chmod +x /workspace/start_cleanup.sh

  if pgrep -xf "bash /workspace/start_cleanup.sh" >/dev/null 2>&1; then
    log "  reaper supervisor already running — leaving it alone"
  else
    setsid nohup bash /workspace/start_cleanup.sh </dev/null >>/workspace/api_setup.log 2>&1 8>&- &
    disown 2>/dev/null || true
    log "  reaper supervisor launched (deletes videos older than 3 days, daily)"
  fi
fi

log "=========================================="
log "Setup Complete!"
log "  API docs: https://${RUNPOD_POD_ID}-7860.proxy.runpod.net/docs"
log "  Swagger:  https://${RUNPOD_POD_ID}-7860.proxy.runpod.net/docs"
log "  Health:   https://${RUNPOD_POD_ID}-7860.proxy.runpod.net/health"
log "  Setup log: tail -f /workspace/api_setup.log"
log "  API log:   tail -f /workspace/api.log"
log ""
log "  Pod restarts auto-launch the API via the ComfyUI bootstrap custom node."
log "  (Located at $COMFY_ROOT/custom_nodes/ai_gen_api_bootstrap/__init__.py)"
log "  Manual relaunch (if needed):"
log "    setsid nohup bash /workspace/start_api.sh </dev/null >>/workspace/api_setup.log 2>&1 &"
log "=========================================="
