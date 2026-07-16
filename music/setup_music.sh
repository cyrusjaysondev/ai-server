#!/usr/bin/env bash
set -euo pipefail

MUSIC_ROOT="${MUSIC_ROOT:-/workspace/music}"
ACE_STEP_DIR="${ACE_STEP_DIR:-$MUSIC_ROOT/ACE-Step-1.5}"
LOG_FILE="${MUSIC_SETUP_LOG:-/workspace/music_setup.log}"

mkdir -p "$MUSIC_ROOT" "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

log "Music generator setup started"
log "MUSIC_ROOT=$MUSIC_ROOT"

export DEBIAN_FRONTEND="${DEBIAN_FRONTEND:-noninteractive}"
export PATH="$MUSIC_ROOT/.local/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export HF_HOME="${HF_HOME:-$MUSIC_ROOT/hf-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$MUSIC_ROOT/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$MUSIC_ROOT/uv-cache}"
export ACESTEP_TMPDIR="${ACESTEP_TMPDIR:-$MUSIC_ROOT/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$MUSIC_ROOT/triton-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$MUSIC_ROOT/torchinductor-cache}"
export ACESTEP_PROJECT_ROOT="$ACE_STEP_DIR"
export ACESTEP_CHECKPOINTS_DIR="${ACESTEP_CHECKPOINTS_DIR:-$ACE_STEP_DIR/checkpoints}"
export ACESTEP_DOWNLOAD_SOURCE="${ACESTEP_DOWNLOAD_SOURCE:-huggingface}"
export ACESTEP_CONFIG_PATH="${ACESTEP_CONFIG_PATH:-acestep-v15-turbo}"
export ACESTEP_LM_MODEL_PATH="${ACESTEP_LM_MODEL_PATH:-acestep-5Hz-lm-1.7B}"
export ACESTEP_INIT_LLM="${ACESTEP_INIT_LLM:-auto}"
export ACESTEP_API_HOST="${ACESTEP_API_HOST:-0.0.0.0}"
export ACESTEP_API_PORT="${ACESTEP_API_PORT:-8001}"
export ACESTEP_QUEUE_WORKERS="${ACESTEP_QUEUE_WORKERS:-1}"
export ACESTEP_API_WORKERS="${ACESTEP_API_WORKERS:-1}"
export ACESTEP_QUEUE_MAXSIZE="${ACESTEP_QUEUE_MAXSIZE:-32}"
export ACESTEP_NO_INIT="${ACESTEP_NO_INIT:-false}"

if [[ "${MUSIC_EAGER_LOAD:-true}" == "false" ]]; then
  export ACESTEP_NO_INIT=true
fi

mkdir -p \
  "$HF_HOME" \
  "$XDG_CACHE_HOME" \
  "$UV_CACHE_DIR" \
  "$ACESTEP_TMPDIR" \
  "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$ACESTEP_CHECKPOINTS_DIR"

install_os_packages() {
  local missing=()
  for command_name in git curl ffmpeg gcc make; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
      missing+=("$command_name")
    fi
  done

  if (( ${#missing[@]} == 0 )); then
    log "OS packages already available"
    return
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    log "apt-get not found and required tools are missing: ${missing[*]}"
    return 1
  fi

  log "Installing OS packages: git curl wget ca-certificates ffmpeg build-essential libsndfile1 libgl1 libglib2.0-0"
  apt-get update
  apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    ca-certificates \
    ffmpeg \
    build-essential \
    pkg-config \
    libsndfile1 \
    libgl1 \
    libglib2.0-0
}

install_uv() {
  if command -v uv >/dev/null 2>&1; then
    log "uv already installed: $(command -v uv)"
    return
  fi

  log "Installing uv into $MUSIC_ROOT/.local/bin"
  export UV_INSTALL_DIR="$MUSIC_ROOT/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$UV_INSTALL_DIR:$PATH"
  uv --version
}

sync_ace_step_repo() {
  local repo_url="${ACE_STEP_REPO:-https://github.com/ace-step/ACE-Step-1.5.git}"
  local repo_ref="${ACE_STEP_REF:-main}"

  if [[ -d "$ACE_STEP_DIR/.git" ]]; then
    log "ACE-Step repo already exists; updating with git pull --ff-only"
    git -C "$ACE_STEP_DIR" fetch origin "$repo_ref" --depth 1
    git -C "$ACE_STEP_DIR" checkout "$repo_ref" || git -C "$ACE_STEP_DIR" checkout FETCH_HEAD
    git -C "$ACE_STEP_DIR" pull --ff-only origin "$repo_ref" || true
  elif [[ -f "$ACE_STEP_DIR/pyproject.toml" && -d "$ACE_STEP_DIR/acestep" ]]; then
    log "Using existing ACE-Step source directory without git metadata"
  else
    if [[ -e "$ACE_STEP_DIR" ]]; then
      log "Removing incomplete ACE-Step directory at $ACE_STEP_DIR"
      rm -rf "$ACE_STEP_DIR"
    fi

    log "Cloning ACE-Step from $repo_url"
    git clone --depth 1 --branch "$repo_ref" "$repo_url" "$ACE_STEP_DIR" || {
      log "Branch/ref clone failed; cloning default branch and checking out $repo_ref"
      rm -rf "$ACE_STEP_DIR"
      git clone --depth 1 "$repo_url" "$ACE_STEP_DIR"
      git -C "$ACE_STEP_DIR" fetch origin "$repo_ref" --depth 1 || true
      git -C "$ACE_STEP_DIR" checkout "$repo_ref" || true
    }
  fi

  log "ACE-Step revision: $(git -C "$ACE_STEP_DIR" rev-parse --short HEAD)"
}

install_python_dependencies() {
  log "Installing Python 3.11 with uv"
  uv python install 3.11

  log "Installing ACE-Step Python dependencies"
  cd "$ACE_STEP_DIR"
  uv sync --python 3.11

  log "Python dependency check"
  uv run --no-sync python - <<'PY'
import sys
import torch
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
}

download_models() {
  if [[ "${MUSIC_PRELOAD_MODELS:-true}" != "true" ]]; then
    log "Skipping model pre-download because MUSIC_PRELOAD_MODELS is not true"
    return
  fi

  local download_args=(--dir "$ACESTEP_CHECKPOINTS_DIR")
  if [[ -n "${HF_TOKEN:-}" ]]; then
    download_args+=(--token "$HF_TOKEN")
  fi

  log "Downloading ACE-Step main model bundle into $ACESTEP_CHECKPOINTS_DIR"
  cd "$ACE_STEP_DIR"
  uv run --no-sync acestep-download "${download_args[@]}"

  if [[ -n "${MUSIC_EXTRA_MODELS:-}" ]]; then
    IFS=',' read -ra extra_models <<< "$MUSIC_EXTRA_MODELS"
    for model_name in "${extra_models[@]}"; do
      model_name="$(echo "$model_name" | xargs)"
      [[ -z "$model_name" ]] && continue
      log "Downloading extra ACE-Step model: $model_name"
      local extra_args=(--dir "$ACESTEP_CHECKPOINTS_DIR" --model "$model_name" --skip-main)
      if [[ -n "${HF_TOKEN:-}" ]]; then
        extra_args+=(--token "$HF_TOKEN")
      fi
      uv run --no-sync acestep-download "${extra_args[@]}"
    done
  fi
}

write_runtime_helpers() {
  log "Writing runtime helper scripts"

  cat > "$MUSIC_ROOT/run_music_api.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

MUSIC_ROOT="${MUSIC_ROOT:-/workspace/music}"
ACE_STEP_DIR="${ACE_STEP_DIR:-$MUSIC_ROOT/ACE-Step-1.5}"

export PATH="$MUSIC_ROOT/.local/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export HF_HOME="${HF_HOME:-$MUSIC_ROOT/hf-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$MUSIC_ROOT/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$MUSIC_ROOT/uv-cache}"
export ACESTEP_TMPDIR="${ACESTEP_TMPDIR:-$MUSIC_ROOT/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$MUSIC_ROOT/triton-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$MUSIC_ROOT/torchinductor-cache}"
export ACESTEP_PROJECT_ROOT="$ACE_STEP_DIR"
export ACESTEP_CHECKPOINTS_DIR="${ACESTEP_CHECKPOINTS_DIR:-$ACE_STEP_DIR/checkpoints}"
export ACESTEP_DOWNLOAD_SOURCE="${ACESTEP_DOWNLOAD_SOURCE:-huggingface}"
export ACESTEP_CONFIG_PATH="${ACESTEP_CONFIG_PATH:-acestep-v15-turbo}"
export ACESTEP_LM_MODEL_PATH="${ACESTEP_LM_MODEL_PATH:-acestep-5Hz-lm-1.7B}"
export ACESTEP_INIT_LLM="${ACESTEP_INIT_LLM:-auto}"
export ACESTEP_NO_INIT="${ACESTEP_NO_INIT:-false}"
export ACESTEP_API_HOST="${ACESTEP_API_HOST:-0.0.0.0}"
export ACESTEP_API_PORT="${ACESTEP_API_PORT:-8001}"
export ACESTEP_QUEUE_WORKERS="${ACESTEP_QUEUE_WORKERS:-1}"
export ACESTEP_API_WORKERS="${ACESTEP_API_WORKERS:-1}"
export ACESTEP_QUEUE_MAXSIZE="${ACESTEP_QUEUE_MAXSIZE:-32}"

if [[ -f "$MUSIC_ROOT/secrets.env" ]]; then
  set -a
  source "$MUSIC_ROOT/secrets.env"
  set +a
fi

mkdir -p "$ACESTEP_TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$ACESTEP_CHECKPOINTS_DIR"

cd "$ACE_STEP_DIR"
exec uv run --no-sync acestep-api \
  --host "$ACESTEP_API_HOST" \
  --port "$ACESTEP_API_PORT" \
  --download-source "$ACESTEP_DOWNLOAD_SOURCE"
EOF
  chmod +x "$MUSIC_ROOT/run_music_api.sh"
}

install_os_packages
install_uv
sync_ace_step_repo
install_python_dependencies
download_models
write_runtime_helpers

log "Music generator setup complete"
log "API helper: $MUSIC_ROOT/run_music_api.sh"
log "Logs: $LOG_FILE"

if [[ "${MUSIC_RUN_API:-false}" == "true" ]]; then
  log "Starting ACE-Step music API on ${ACESTEP_API_HOST}:${ACESTEP_API_PORT}"
  exec "$MUSIC_ROOT/run_music_api.sh"
fi
