# AI Gen API v2

API on RunPod for:
- **FLUX.2 Klein 9B** — text-to-image, head/face swap, multi-reference image editing (1–5 input images)
- **LTX 2.3 22B** — image-to-video, text-to-video, face-animate pipeline

Two ways to run it:
- **Pod mode** (this README, [SETUP.md](SETUP.md), [API.md](API.md)) — always-on
  FastAPI on `:7860`, models on `/workspace`. Pay per pod-second.
- **Serverless mode** ([SERVERLESS_SETUP.md](SERVERLESS_SETUP.md) for step-by-step
  deploy, [serverless/README.md](serverless/README.md) for the API reference) —
  split into an **image endpoint** (`t2i`, `flux/face-swap`, `flux/i2i`) and a **video
  endpoint** (`ltx/i2v`, `ltx/t2v`). Same models, mounted from the same network
  volume the pod uses. Pay per request, scale to zero.

---

## Prerequisites

You need both of these before the pod can install:

1. **HuggingFace token** with Read access — https://huggingface.co/settings/tokens
2. **Accept model licenses** (click "Agree and access repository" on each):
   - https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
   - https://huggingface.co/Lightricks/LTX-2.3-fp8

---

## Quick Deploy (new pod)

### 1. Create Template
- **runpod.io** → **Templates** → **+ New Template**
- **Container Image:** `runpod/comfyui:latest`
- **Volume:** `200 GB` mounted at `/workspace`
- **Expose Ports:** `7860, 8188, 8888`

### 2. Set Environment Variables
In the template, add:

```
HF_TOKEN = hf_your_token_here
```

> Setting `SETUP_SCRIPT_URL` as an env var alone does **not** trigger the installer — nothing in the stock image reads it. You'll run `setup.sh` manually once after the pod boots (step 4).

### 3. Leave the Container Start Command empty

The stock `runpod/comfyui:latest` image has `/start.sh` as its **ENTRYPOINT**.
RunPod's "Container Start Command" field becomes **args** to that entrypoint —
and `/start.sh` ignores its args. That means you cannot use the Start Command
field to auto-run setup on a brand-new pod with this image. Anything you put
there is silently dropped. Leave it empty.

For first deploys you'll run `setup.sh` once manually (step 4). After that,
a ComfyUI bootstrap custom node (installed by `setup.sh` on the
**/workspace** volume) auto-launches the API on every subsequent pod restart —
no manual step needed unless the volume itself is wiped.

### 4. Deploy, then run setup.sh once (first deploy only)

Click Deploy. SSH/Jupyter/ComfyUI come up via `/start.sh` but `:7860` stays
**502** until you run `setup.sh`. Open the Jupyter terminal on port 8888 (or
SSH in) and run these as **two separate commands** (press Enter after each):

```bash
wget -qO /tmp/setup.sh https://raw.githubusercontent.com/cyrusjaysondev/ai-gen-api-v2/main/setup.sh
```
```bash
bash /tmp/setup.sh
```

> **Why not one line with `&&`?** Long pasted lines sometimes wrap mid-command
> in the terminal and drop you into a nested shell. Two lines can't be split wrong.

`setup.sh` runs 4 steps:

1. **Install pip deps + aria2** (~5 s)
2. **Download all 9 models in parallel via aria2** (~72 GB; resumable, skips anything already on the volume)
3. **Install LanPaint custom node** (skipped if already present)
4. **Fetch `main.py`, install the ComfyUI bootstrap custom node, launch the API supervisor**

**First deploy: ~3–10 minutes** on a warm HuggingFace CDN (parallel aria2 at 200–500 MB/s beats serial wget by 30–50×).
**Subsequent pod restarts: automatic, ~10–20 s** — the bootstrap custom node on the volume re-launches the API; you don't run anything.

### 5. Monitor Progress
While `setup.sh` is running, tail the log:
```bash
tail -f /workspace/api_setup.log
```

Healthy states (live on `/health` via the proxy):
- **HTTP 503 + `{"status":"installing", …}`** — setup is running, status server is bound. Wait for 200.
- **HTTP 200 + `{"status":"ok", …}`** — API is ready.
- **HTTP 502** before you've run `setup.sh` is expected; after `setup.sh` reports "Setup Complete!", it should flip to 200 within a few seconds.

### 6. Verify
```
https://YOUR_POD_ID-7860.proxy.runpod.net/health
https://YOUR_POD_ID-7860.proxy.runpod.net/docs
```

---

## Scaling (spin up more pods)

Just repeat steps 1-3. Each new pod auto-configures:
- Models download once per volume, skip on restart
- Pod ID is read from `$RUNPOD_POD_ID` at runtime — no manual config needed
- API starts automatically after ComfyUI is ready

---

## API Endpoints

### Health Check
```bash
curl https://YOUR_POD_ID-7860.proxy.runpod.net/health
```

### Text to Image
```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/t2i \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful sunset over mountains, photorealistic, 4K",
    "width": 1024,
    "height": 1024
  }'
```

**Parameters:**
| Param | Default | Description |
|-------|---------|-------------|
| `prompt` | required | What to generate |
| `width` | 1024 | Image width |
| `height` | 1024 | Image height |
| `seed` | -1 (random) | Reproducibility seed |
| `steps` | 4 | Inference steps (4 is good for Klein) |
| `cfg` | 1.0 | CFG scale |
| `guidance` | 4.0 | FLUX guidance strength (2.0-6.0) |
| `watermark` | `null` | Optional text to overlay bottom-right of the output (e.g. `"AI"`). Null/empty = off. See [Output watermark](#output-watermark). |

### Head Swap (FLUX)
```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/flux/face-swap \
  -F "target_image=@body_template.png" \
  -F "face_image=@my_face.png"
```

**Parameters:**
| Param | Default | Description |
|-------|---------|-------------|
| `target_image` | required | Body/template image (head gets replaced) |
| `face_image` | required | Face photo (identity to transfer) |
| `seed` | -1 (random) | Reproducibility seed |
| `megapixels` | 2.0 | Output resolution (1.0-2.0) |
| `steps` | 4 | Inference steps |
| `cfg` | 1.0 | CFG scale |
| `guidance` | 4.0 | FLUX guidance (2.0-6.0) |
| `lora_strength` | 1.0 | Head swap LoRA strength (0.0-1.5) |
| `watermark` | `null` | Optional bottom-right text overlay (e.g. `"AI"`). See [Output watermark](#output-watermark). |

### Multi-reference Image Editing (FLUX)
Send 1 to 5 reference images plus a prompt. The prompt drives the edit; the
images supply style, identity, objects, composition cues. Output canvas
defaults to the first image's dimensions.

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/flux/i2i \
  -F "prompt=combine the subject from image 1 with the outfit from image 2" \
  -F "images=@subject.png" \
  -F "images=@outfit.png"
```

**Parameters:**
| Param | Default | Description |
|-------|---------|-------------|
| `prompt` | required | Edit instruction |
| `images` | required | 1 to 5 image files (repeat `-F "images=@..."`) |
| `seed` | -1 (random) | Reproducibility seed |
| `megapixels` | 2.0 | Resolution per reference image (0.5-4.0) |
| `width` / `height` | 0 / 0 | `0` = derive from first image |
| `steps` | 4 | Inference steps |
| `guidance` | 4.0 | FLUX guidance (2.0-6.0) |
| `lora_strength` | 0.0 | `0` = general edits, `0.5-1.0` = face-focused edits |
| `watermark` | `null` | Optional bottom-right text overlay (e.g. `"AI"`). See [Output watermark](#output-watermark). |

### Image to Video (LTX 2.3)
```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/ltx/i2v \
  -F "image=@my_photo.jpg" \
  -F "prompt=the person smiles and waves" \
  -F "preset=fast" \
  -F "aspect_ratio=9:16" \
  -F "length=97"
```

**Parameters** (most-used):
| Param | Default | Description |
|-------|---------|-------------|
| `image` | required | First-frame image |
| `prompt` | `""` | Motion / scene description (auto-enhanced via Gemma unless `enhance_prompt=false`) |
| `preset` | `fast` | `fast` (8 steps single-pass) or `quality` (8+3 steps two-pass). Both ~12s warm @544×960 — see [API.md](API.md) |
| `aspect_ratio` | `9:16` | `original` \| `16:9` \| `9:16` \| `1:1` \| `4:3` \| `3:4` \| `3:2` \| `2:3` \| `21:9` \| `9:21` |
| `length` | `121` | Frame count. 49≈2s, 97≈4s, 121≈5s @ 24 fps |
| `fps` | `24` | |
| `seed` | -1 (random) | Reproducibility seed |
| `audio` | `false` | Generate audio track (adds ~5-10s) |
| `enhance_prompt` | `true` | Disable to save 2-5s when you've written a detailed prompt |
| `watermark` | `null` | Optional bottom-right text overlay (e.g. `"AI"`). Re-encodes via ffmpeg (~1-3s for a 5s clip). See [Output watermark](#output-watermark). |

See [API.md](API.md) for `/ltx/t2v` (text-to-video) and `/face-animate` (face-swap + animate pipeline). Both accept the same `watermark` parameter.

### Check Job Status
```bash
# Poll until status is "completed"
curl https://YOUR_POD_ID-7860.proxy.runpod.net/status/JOB_ID
```

Response when completed:
```json
{
  "status": "completed",
  "url": "https://YOUR_POD_ID-7860.proxy.runpod.net/image/filename.png",
  "filename": "filename.png",
  "duration_seconds": 12.3
}
```

### List All Jobs
```bash
curl https://YOUR_POD_ID-7860.proxy.runpod.net/jobs
```

### Delete Jobs
```bash
# Delete one job + its file
curl -X DELETE https://YOUR_POD_ID-7860.proxy.runpod.net/jobs/JOB_ID

# Delete all completed jobs
curl -X DELETE https://YOUR_POD_ID-7860.proxy.runpod.net/jobs

# Delete ALL jobs (including queued/processing)
curl -X DELETE "https://YOUR_POD_ID-7860.proxy.runpod.net/jobs?completed_only=false"
```

---

## Typical Workflow

```bash
POD="https://YOUR_POD_ID-7860.proxy.runpod.net"

# 1. Submit a head swap job
JOB=$(curl -s -X POST "$POD/flux/face-swap" \
  -F "target_image=@template.png" \
  -F "face_image=@face.png" | jq -r '.job_id')

echo "Job: $JOB"

# 2. Poll until done
while true; do
  STATUS=$(curl -s "$POD/status/$JOB" | jq -r '.status')
  echo "Status: $STATUS"
  [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ] && break
  sleep 3
done

# 3. Download result
URL=$(curl -s "$POD/status/$JOB" | jq -r '.url')
curl -o result.png "$URL"
```

---

## Output watermark

Every generation endpoint (`/t2i`, `/flux/face-swap`, `/flux/i2i`, `/ltx/i2v`,
`/ltx/t2v`, `/face-animate`) accepts an optional `watermark` parameter that
overlays text on the bottom-right of the result.

- **Default:** off. Pass `null`, omit the field, or send `""` to skip.
- **Enable:** pass any non-empty string (e.g. `"AI"`, `"My Brand"`, `"© 2026"`).
- **Style:** bold white text with a black outline, sized ~4% of frame height.
- **Cost:**
  - Images: negligible (~50 ms Pillow draw).
  - Videos: re-encoded via ffmpeg `drawtext`. ~1–3 s for a 5 s clip at 544×960; audio is stream-copied so quality is preserved.
- **Failure mode:** if the watermark step errors (e.g. ffmpeg missing), the
  generation still completes and the response includes a `watermark_warning`
  field. The unwatermarked file is still returned.

Examples:

```bash
# Image with watermark
curl -X POST $POD/t2i \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a serene mountain lake","watermark":"AI"}'

# Video with watermark
curl -X POST $POD/ltx/i2v \
  -F "image=@photo.jpg" -F "prompt=person waves" \
  -F "watermark=AI"

# Custom text
curl -X POST $POD/flux/face-swap \
  -F "target_image=@body.png" -F "face_image=@face.png" \
  -F "watermark=Made by AI"
```

The same `watermark` field works on the serverless endpoints — pass it as
part of the `input` JSON.

---

## Troubleshooting

### API not starting
Open Jupyter terminal (port 8888) and run:
```bash
tail -20 /workspace/api.log
```

### Manual start (if auto-start fails)
The supervisor script handles everything (deps, fetch `main.py`, wait for ComfyUI,
launch uvicorn with auto-restart on crash). Launch it fully detached:

```bash
setsid nohup bash /workspace/start_api.sh </dev/null >>/workspace/api_setup.log 2>&1 &

# Verify
sleep 10 && curl http://localhost:7860/health
```

### API didn't come back after pod restart
Confirm the bootstrap custom node is on the volume (`setup.sh` installs it):
```bash
ls -la /workspace/runpod-slim/ComfyUI/custom_nodes/ai_gen_api_bootstrap/__init__.py
```
And check ComfyUI's log for the bootstrap line:
```bash
grep "ai-gen-api-bootstrap" /workspace/comfyui.log
# expect: [ai-gen-api-bootstrap] launched /workspace/start_api.sh (...)
```
If the file is missing (network volume was wiped or the pod was created with a
different volume), re-run setup:
```bash
wget -qO /tmp/setup.sh https://raw.githubusercontent.com/cyrusjaysondev/ai-gen-api-v2/main/setup.sh && bash /tmp/setup.sh
```
If the file is present but ComfyUI didn't log the line, ComfyUI may not have
loaded the node. Tail `/workspace/comfyui.log` while restarting the pod to see
the import sequence.

### main.py failed to download
```bash
wget -O /workspace/api/main.py \
  "https://raw.githubusercontent.com/cyrusjaysondev/ai-gen-api-v2/main/main.py"
```

### ComfyUI not starting
```bash
# Check GPU
nvidia-smi

# Restart ComfyUI (same command /start.sh uses)
cd /workspace/runpod-slim/ComfyUI && \
  nohup .venv-cu128/bin/python main.py --listen 0.0.0.0 --port 8188 \
    --enable-cors-header >> /workspace/comfyui.log 2>&1 & disown
```

---

## Models (~72 GB total)

| Model | Size | Purpose | HF token |
|-------|------|---------|----------|
| FLUX.2 Klein 9B UNET | 18 GB | Image gen + head swap | Yes (gated) |
| FLUX VAE | 321 MB | VAE decoder | No |
| Qwen 3 8B text encoder | 8.1 GB | FLUX text encoder | No |
| BFS Head Swap LoRA | 633 MB | Head swap LoRA | No |
| LTX-2.3 22B checkpoint | 27 GB | Video gen | Yes (gated) |
| LTX-2.3 distilled LoRA | 7.1 GB | LTX speed LoRA | No |
| Gemma abliterated LoRA | 599 MB | LTX prompt LoRA | No |
| Gemma 3 12B text encoder | 8.8 GB | LTX text encoder | No |
| LTX-2.3 spatial upscaler | 950 MB | 2× spatial upscaler | No |

## File Structure on Pod

```
/workspace/
  api/
    main.py              # FastAPI app
    config.env           # detected Python/pip/ComfyUI paths
  start_api.sh           # Supervisor: flock-guarded while-loop around uvicorn
  api.log                # API runtime log
  api_setup.log          # Setup + supervisor progress log
  runpod-slim/
    ComfyUI/             # the runpod/comfyui:latest image's ComfyUI tree
      .venv-cu128/       # Python venv (used by both ComfyUI and API)
      models/
        diffusion_models/  # FLUX UNET
        vae/               # FLUX VAE
        text_encoders/     # Qwen 3 8B, Gemma 12B
        loras/             # BFS, distilled LTX, Gemma abliterated
        checkpoints/       # LTX 2.3 22B
        latent_upscale_models/  # LTX spatial upscaler
      custom_nodes/
        LanPaint/                  # FLUX head swap nodes
        ai_gen_api_bootstrap/      # side-effect-only custom node:
          __init__.py              # spawns start_api.sh on ComfyUI import,
                                   # giving us a hook that survives pod
                                   # restarts (lives on the /workspace volume)

/start.sh                # container entrypoint (image-owned, not persistent).
                         # Wiped on every container recreation, so we can't
                         # rely on patching it. Auto-recovery happens via
                         # the ComfyUI bootstrap custom node instead.
```
# ai-server
# ai-server
# ai-server
