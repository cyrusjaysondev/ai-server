# AI Gen API v2 - Music Generator RunPod Setup

Dedicated music worker setup for ACE-Step 1.5 on RunPod.

This does not change the CMS or the existing image/video worker. It creates a separate network volume and pod for music generation.

## Model Choice

Default model stack:

| Layer | Default | Why |
|-------|---------|-----|
| Music generator | ACE-Step 1.5 | Open-source, fast, full-song text-to-music |
| DiT model | `acestep-v15-turbo` | Fast first production default |
| LM model | `acestep-5Hz-lm-1.7B` | Good balance for prompt/lyrics planning |
| API | `acestep-api` | Upstream FastAPI worker with queue/status endpoints |

ACE-Step upstream:

- Repo: https://github.com/ace-step/ACE-Step-1.5
- License: MIT
- API docs: https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/API.md

## Dedicated Repo Files

| File | Purpose |
|------|---------|
| `music/setup_music.sh` | RunPod install/start script |
| `music/smoke_test_music.py` | Health + generation smoke test |
| `MUSIC_API.md` | Separate API docs for music generation |
| `MUSIC_SETUP.md` | This RunPod setup guide |

## RunPod Network Volume

Recommended volume:

| Setting | Value |
|---------|-------|
| Name | `ai-music-models` |
| Size | `200 GB` |
| Mount path | `/workspace` |

The volume stores:

```text
/workspace/music/
  ACE-Step-1.5/
    checkpoints/
  hf-cache/
  uv-cache/
  tmp/
  run_music_api.sh
/workspace/music_setup.log
```

## RunPod Pod

Recommended pod for first build:

| Setting | Value |
|---------|-------|
| GPU | RTX 5090, RTX 4090, A40, A100, or better |
| Container image | `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` |
| Volume | `ai-music-models` mounted at `/workspace` |
| HTTP port | `8001` |

Start command:

```bash
bash -lc 'export MUSIC_RUN_API=true MUSIC_EAGER_LOAD=true MUSIC_PRELOAD_MODELS=true; curl -fsSL https://raw.githubusercontent.com/cyrus688/ai-server/main/music/setup_music.sh | bash'
```

Environment variables:

| Variable | Required | Default | Notes |
|----------|----------|---------|-------|
| `HF_TOKEN` | No | empty | Use if HuggingFace rate limits downloads |
| `ACESTEP_API_KEY` | Recommended | empty | Enables Bearer auth |
| `ACESTEP_CONFIG_PATH` | No | `acestep-v15-turbo` | Fast default |
| `ACESTEP_LM_MODEL_PATH` | No | `acestep-5Hz-lm-1.7B` | Prompt/lyrics LM |
| `MUSIC_PRELOAD_MODELS` | No | `true` | Downloads models during setup |
| `MUSIC_EAGER_LOAD` | No | `true` | Loads models when API starts so first request is warm |
| `MUSIC_EXTRA_MODELS` | No | empty | Comma-separated optional model names |

## Logs

```bash
tail -f /workspace/music_setup.log
```

After setup, the API runs on:

```text
https://YOUR_POD_ID-8001.proxy.runpod.net
```

Health check:

```bash
curl https://YOUR_POD_ID-8001.proxy.runpod.net/health
```

With auth:

```bash
curl https://YOUR_POD_ID-8001.proxy.runpod.net/health \
  -H "Authorization: Bearer $ACESTEP_API_KEY"
```

## Smoke Test

From your local machine:

```bash
python3 music/smoke_test_music.py \
  --base-url https://YOUR_POD_ID-8001.proxy.runpod.net \
  --output music-smoke-test.mp3
```

With auth:

```bash
python3 music/smoke_test_music.py \
  --base-url https://YOUR_POD_ID-8001.proxy.runpod.net \
  --api-key "$ACESTEP_API_KEY" \
  --output music-smoke-test.mp3
```

The test confirms:

1. `/health` responds.
2. `/release_task` accepts a 10-second music job.
3. `/query_result` completes successfully.
4. The generated MP3 downloads.
5. The response includes displayable `lyrics`.

## Warm vs Cold Behavior

With `MUSIC_EAGER_LOAD=true`, the pod pays the load time during startup. The first user request should avoid model-loading delay after the API is ready.

If you set `MUSIC_EAGER_LOAD=false`, startup is faster but the first generation request will download/load models lazily and feel much slower.

## Future CMS Integration

Do not call this pod directly from public apps long-term. The intended production shape is:

```text
3rd-party app -> SuperCMS load balancer -> music pod/serverless worker -> ACE-Step API
```

For this pass, the music worker is isolated so we can validate generation, audio download, and lyrics output before wiring it into the CMS.
