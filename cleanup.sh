#!/bin/bash
# =============================================================
# AI Gen API v2 — Output reaper
#
# Deletes generated files older than RETENTION_DAYS from the network volume.
#
#   serverless outputs dir   — both VIDEOS and IMAGES
#       Path: /workspace/outputs/<job_id>/<filename>
#       Why both: serverless image handler returns a path on this volume
#       (just like the video handler), so images accumulate too.
#
#   pod-mode video output dir — VIDEOS only
#       Path: /workspace/runpod-slim/ComfyUI/output/video/<filename>
#       Why videos only: pod-mode images are referenced by URL in the
#       in-memory `jobs` dict and served via /image/{filename}. Deleting
#       them would break those URLs unexpectedly.
#
# Idempotent: re-running is safe. Invoked daily by start_cleanup.sh.
# =============================================================
set -u

RETENTION_DAYS="${RETENTION_DAYS:-3}"
LOG="/workspace/cleanup.log"

SERVERLESS_OUTPUTS="/workspace/outputs"
POD_VIDEO_DIR="/workspace/runpod-slim/ComfyUI/output/video"

VIDEO_EXTS=(mp4 webm mov gif)
IMAGE_EXTS=(png jpg jpeg webp)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"; }

# Cap log at last 200 lines so it doesn't grow unbounded
tail -200 "$LOG" > "${LOG}.tmp" 2>/dev/null && mv "${LOG}.tmp" "$LOG"

log "reaper start (RETENTION_DAYS=$RETENTION_DAYS)"

reap_dir() {
  local dir="$1"
  local label="$2"
  shift 2
  local -a extensions=("$@")

  [ -d "$dir" ] || { log "  $label: skip (dir doesn't exist yet)"; return; }

  # Build "(-iname *.ext1 -o -iname *.ext2 ...)" predicate
  local -a expr=()
  for ext in "${extensions[@]}"; do
    [ ${#expr[@]} -gt 0 ] && expr+=("-o")
    expr+=("-iname" "*.$ext")
  done

  local deleted
  deleted=$(find "$dir" -type f \( "${expr[@]}" \) \
    -mtime "+$RETENTION_DAYS" -print -delete 2>/dev/null | wc -l)

  local empty_dirs
  empty_dirs=$(find "$dir" -mindepth 1 -type d -empty -print -delete 2>/dev/null | wc -l)

  log "  $label: deleted $deleted file(s), $empty_dirs empty subdir(s)"
}

# Serverless: both videos AND images
reap_dir "$SERVERLESS_OUTPUTS" "serverless outputs (videos+images)" \
  "${VIDEO_EXTS[@]}" "${IMAGE_EXTS[@]}"

# Pod-mode: videos only (pod-mode images are served via /image/{filename}
# and tracked in the API's in-memory jobs dict; deleting them would break
# those URLs without invalidating the dict entry)
reap_dir "$POD_VIDEO_DIR" "pod-mode videos" "${VIDEO_EXTS[@]}"

log "reaper done"
