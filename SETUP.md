# AI Gen API v2 — RunPod Setup Guide

FLUX.2 Klein 9B · Face Swap + Text-to-Image + Multi-reference Image Editing
LTX 2.3 22B · Image-to-Video + Text-to-Video + Face-Animate Pipeline

---

## Prerequisites

Before starting, you need:

1. **HuggingFace account** — https://huggingface.co
2. **HuggingFace token** — https://huggingface.co/settings/tokens (create a token with **Read** access)
3. **Accept model licenses:**
   - https://huggingface.co/black-forest-labs/FLUX.2-klein-9B — click **Agree and access repository**
   - https://huggingface.co/Lightricks/LTX-2.3-fp8 — click **Agree and access repository**

---

## Step 1 — Create a RunPod Template

1. Go to **runpod.io** → **Templates** → **+ New Template**
2. Set the following:
   - **Template Name:** `AI Gen API v2`
   - **Container Image:** `runpod/comfyui:latest`
   - **Volume:** `200 GB` mounted at `/workspace`
   - **Expose Ports:** `7860, 8188, 8888`
3. Under **Environment Variables**, add:
   ```
   HF_TOKEN = hf_your_token_here

   # (optional) protect the admin blocklist API with a Bearer token:
   # ADMIN_TOKEN = <generate a long random string>
   ```
   `HF_TOKEN` is required (model downloads).

   `ADMIN_TOKEN` is **optional**. By default (unset), the
   `/admin/blocklist` and `/admin/blocklist-logos` endpoints are open
   for easy access during development. Set `ADMIN_TOKEN` to require
   `Authorization: Bearer <token>` on every admin call — recommended
   before sharing the pod URL or going to production.
   Generate one with: `python -c "import secrets; print(secrets.token_hex(32))"`
4. Under **Container Start Command**, paste this **exactly** (single line):
   ```bash
   bash -c "wget -qO /tmp/boot.sh https://raw.githubusercontent.com/cyrusjaysondev/ai-server/main/boot.sh && bash /tmp/boot.sh"
   ```

   > ⚠️ If you leave this field **empty**, the pod will only run the image's default
   > entrypoint (`/start.sh`) — SSH/Jupyter/ComfyUI come up, but the API never installs
   > and `:7860` stays 502. The Container Start Command is required.

   **What `boot.sh` does:** `runpod/comfyui:latest` ships `/start.sh` as its entrypoint
   (starts SSH, JupyterLab, FileBrowser, and ComfyUI). A Container Start Command replaces
   that entrypoint, so `boot.sh` re-invokes `/start.sh` in the background, fetches and
   runs `setup.sh`, and then `wait`s on `/start.sh` (which ends in `sleep infinity`) so
   the container stays alive. Keeping the logic in a file avoids quoting errors that
   happen when long inline shell strings are pasted into UI fields.
5. Save the template.

---

## Step 2 — Deploy a Pod

1. Go to **Pods** → **+ Deploy**
2. Select GPU: **RTX 5090** (32 GB VRAM) — recommended
3. Select your **AI Gen API v2** template
4. Click **Deploy**

---

## Step 3 — Monitor Setup Progress

The setup runs automatically as part of the Container Start Command. Open the Jupyter
terminal (port 8888) — or SSH in — and tail the log:

```bash
tail -f /workspace/api_setup.log
```

You should see output like:
```
[HH:MM:SS] ==========================================
[HH:MM:SS] AI Gen API v2 Setup Started
[HH:MM:SS] Pod ID: xxxxxxxxxxxxxxxx
[HH:MM:SS] ComfyUI: /workspace/runpod-slim/ComfyUI
[HH:MM:SS] Python: /workspace/runpod-slim/ComfyUI/.venv-cu128/bin/python
[HH:MM:SS] ==========================================
[HH:MM:SS] [1/4] Installing pip dependencies + aria2...
[HH:MM:SS]   Done
[HH:MM:SS] [2/4] Downloading models (parallel, ~72 GB total)...
[HH:MM:SS]   Download complete: /workspace/runpod-slim/ComfyUI/models/...
[HH:MM:SS]   Verifying downloaded files against HF expected sizes...
[HH:MM:SS]   All 9 models verified at expected sizes
[HH:MM:SS] [3/4] Installing LanPaint custom node...
[HH:MM:SS] [4/4] Setting up API...
[HH:MM:SS]   main.py downloaded (latest)
[HH:MM:SS]   /start.sh patched with API restart hook
[HH:MM:SS] API supervisor launched
[HH:MM:SS] Setup Complete!
```

**First deploy: ~3–10 minutes** on a warm HuggingFace CDN. The 9 model files are
pulled in parallel via aria2 (16 connections/file), which typically saturates the
pod's network link.
**Subsequent deploys / pod restarts: ~10–20 seconds.** Models are cached on the
`/workspace` volume; aria2 just verifies sizes and skips. `setup.sh` is idempotent
and skips every step whose work is already done.

---

## Step 3b — Manual fallback (if auto-setup didn't trigger)

If a couple of minutes pass and `/workspace/api_setup.log` shows **no new `AI Gen API v2 Setup Started` line at the current time**, the template's Container Start Command didn't fire — most commonly because the field is empty or was set to an earlier, broken form. Signs:

- `curl https://<pod>-7860.proxy.runpod.net/health` returns **502** (not 503).
- ComfyUI on `:8188` works, but `:7860` silent.
- `tail -f /workspace/api_setup.log` is frozen on old timestamps.

**Run `setup.sh` manually** — SSH in (or open a Jupyter terminal) and paste these as **two separate commands**, pressing Enter after each:

```bash
wget -qO /tmp/setup.sh https://raw.githubusercontent.com/cyrusjaysondev/ai-server/main/setup.sh
```

```bash
bash /tmp/setup.sh
```

> **Why two lines and not one `&&`-chained line?** Some terminals wrap long commands at inconvenient points when you paste them, which splits `&& bash` from its argument and leaves you dropped into a nested interactive shell. Two separate commands can't be split that way.

Setup finishes in ~15–20 s on a volume with cached models (first-ever deploy: 3–10 min while it downloads 72 GB). `setup.sh` also patches `/start.sh` with a restart hook, so **subsequent pod restarts of this same pod** will auto-start the API without you running anything — even without a template fix. The template Start Command only matters for **fresh pods** (new container, new or reused volume).

Then go fix the template so your next fresh pod is truly hands-off — see [the template troubleshooting block](#api-didnt-come-back-after-pod-restart).

---

## Step 4 — Verify the API is Running

Once setup completes, check the health endpoint:

```bash
curl https://YOUR_POD_ID-7860.proxy.runpod.net/health
```

Expected response:
```json
{"status": "ok", "pod_id": "YOUR_POD_ID"}
```

Or open the interactive docs in your browser:
```
https://YOUR_POD_ID-7860.proxy.runpod.net/docs
```

---

## Step 5 — Test the Endpoints

### Text to Video (fast, no audio)

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/ltx/t2v \
  -F "prompt=a sunset over the ocean, cinematic, slow motion" \
  -F "preset=fast"
```

### Image to Video (fast)

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/ltx/i2v \
  -F "image=@my_photo.jpg" \
  -F "prompt=camera slowly zooms in" \
  -F "preset=fast"
```

> **Default output is 9:16 vertical (544×960).** Pass `-F "aspect_ratio=16:9" -F "width=960" -F "height=544"` to get horizontal output, or `-F "aspect_ratio=original"` to match the input image.

> **`enhance_prompt` (default `true`)** runs the Gemma 12B text encoder to rewrite your prompt before sampling — helps short prompts, costs ~2–5s + VRAM per request. Add `-F "enhance_prompt=false"` to skip it when you've written a detailed prompt yourself.

### Image to Video (quality, with audio)

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/ltx/i2v \
  -F "image=@my_photo.jpg" \
  -F "prompt=birds chirping, gentle breeze" \
  -F "preset=quality" \
  -F "audio=true"
```

### Text to Image

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/t2i \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a beautiful sunset over mountains, photorealistic, 4K"}'
```

### Face Swap

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/flux/face-swap \
  -F "target_image=@body_photo.jpg" \
  -F "face_image=@face_photo.jpg"
```

### Face Swap + Animate (Pipeline)

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/face-animate \
  -F "target_image=@body_photo.jpg" \
  -F "face_image=@face_photo.jpg" \
  -F "animate_prompt=person smiles and looks at the camera" \
  -F "preset=fast"
```

All endpoints return a `job_id`. Poll for results:

```bash
curl https://YOUR_POD_ID-7860.proxy.runpod.net/status/JOB_ID
```

---

## GPU Acceleration (enabled by default)

`setup.sh` installs [SageAttention](https://github.com/thu-ml/SageAttention) into the ComfyUI venv and writes the following flags into `/workspace/runpod-slim/comfyui_args.txt` (read by `/start.sh` on every ComfyUI launch):

- `--use-sage-attention` — replaces PyTorch's SDPA with SageAttention's Triton-based attention kernels (~20–40% faster on diffusion transformers).
- `--fast fp16_accumulation fp8_matrix_mult cublas_ops` — enables FP16 accumulators, FP8 matmul on Blackwell sm_120 (RTX 5090+), and cuBLAS op selection.

You should see this in `/workspace/comfyui.log` on boot:

```
Enabled fp16 accumulation.
Using sage attention
```

To disable any of these (e.g. for debugging), comment out the relevant line in `/workspace/runpod-slim/comfyui_args.txt` and restart ComfyUI (or the pod).

---

## Speed & Quality Presets

All LTX video endpoints (`/ltx/i2v`, `/ltx/t2v`, `/face-animate`) support `preset` and `audio` parameters.

| Preset | Steps | Pipeline | Speed (4s @544×960, warm) | Use case |
|--------|-------|----------|---------------------------|----------|
| `fast` (default) | 8 | Single pass at target resolution | ~12s | General use, fastest single-pass pipeline |
| `quality` | 8 + 3 | Two-pass: half-res sampling → 2× spatial upscale → high-res refine | ~12s | Same wall-time as `fast`, slightly sharper detail |

Both presets use the official LTX-2.3 distilled inference profile (sigmas and LoRA strength taken from Lightricks' reference workflows). The 8-step warmup-cluster sigma schedule is what the distilled LoRA was trained against — using fewer steps causes motion/anatomy artifacts and identity drift.

### Speed tips

- **Lower resolution = faster.** Wall time scales roughly with pixel count.
- **Shorter clips = faster.** `length=49` (2s) is about half the wall time of `length=97` (4s).
- **`audio=false` (default) saves ~5-10s** by skipping the audio VAE entirely
- **`enhance_prompt=false` saves 2–5s** on `/ltx/i2v` by skipping the Gemma 12B prompt rewrite. Recommended when you've written a detailed prompt yourself; leave the default `true` for short / generic prompts where the rewrite materially improves output.
- **First request after pod start is slower** (~30-60s extra) because models load into VRAM. All subsequent requests reuse cached models.
- **Warm benchmarks (either preset, since they're equivalent in speed):**

| Resolution | 2s video (`length=49`) | 4s video (`length=97`) | 5s video (`length=121`) |
|------------|------------------------|------------------------|--------------------------|
| 544×960 | ~7s | ~12s | ~14s |
| 768×1344 | ~12s | ~22s | ~27s |

### Audio control

| Parameter | Effect |
|-----------|--------|
| `audio=false` (default) | Video only — faster generation |
| `audio=true` | Generates audio track with the video |

---

## Pod Restart Behavior

**You don't have to do anything on pod restart** — the API comes back automatically.

`setup.sh` patches the image's `/start.sh` with an idempotent hook that launches
`/workspace/start_api.sh` (detached via `setsid nohup`) right after ComfyUI boots.
So the boot chain on every pod restart is:

1. RunPod runs the template Start Command → invokes `/start.sh`
2. `/start.sh` starts SSH, Jupyter, FileBrowser, and ComfyUI
3. Patched hook in `/start.sh` launches `start_api.sh` in a detached session
4. `start_api.sh` fetches the latest `main.py`, waits for ComfyUI, then starts uvicorn on :7860
5. `start_api.sh` auto-restarts uvicorn if it crashes; `flock` prevents duplicate supervisors

The patch lives on the container layer (not `/workspace`), so it survives pod
**restart** but is lost on pod **recreate/rebuild**. That's fine: the template Start
Command always re-runs `setup.sh`, which re-applies the patch. Self-healing.

### Deploying API updates

1. Push changes to `main.py` on GitHub
2. Restart the pod — `start_api.sh` fetches the latest `main.py` automatically
3. No SSH, no manual redeploy step

### Manually relaunching (only if you skipped the restart)

```bash
setsid nohup bash /workspace/start_api.sh </dev/null >>/workspace/api_setup.log 2>&1 &
```

Plain `bash /workspace/start_api.sh &` also works from an interactive shell, but will
die if the shell exits. `setsid nohup … </dev/null &` fully detaches.

---

## Scaling to Multiple Pods

1. Create your template once (Step 1 above)
2. Deploy as many pods as you need — each runs the same setup
3. Push API updates to the repo — all pods pick up changes on restart
4. Models are cached on each pod's volume — only first deploy downloads ~72 GB

### Recommended scaling setup

| Component | Setting |
|-----------|---------|
| Template | `AI Gen API v2` with start command |
| GPU | RTX 5090 (32 GB) or A100 (80 GB) |
| Volume | 200 GB (persistent across restarts) |
| Ports | 7860 (API), 8188 (ComfyUI), 8888 (Jupyter) |
| Start command | `bash -c "wget -qO /tmp/boot.sh https://raw.githubusercontent.com/cyrusjaysondev/ai-server/main/boot.sh && bash /tmp/boot.sh"` |

---

## Logs

| Log file | Contents |
|----------|----------|
| `/workspace/api_setup.log` | Setup progress, model downloads, API start |
| `/workspace/api.log` | Live API request/response logs |

```bash
# Watch setup progress
tail -f /workspace/api_setup.log

# Watch API logs live
tail -f /workspace/api.log
```

Logs are automatically truncated to the last 500 lines on each restart to prevent disk bloat.

---

## Troubleshooting

**Setup stuck / no progress**
```bash
tail -50 /workspace/api_setup.log
```

**API not responding on port 7860**
```bash
# Check if the supervisor + uvicorn are alive
pgrep -xaf "bash /workspace/start_api.sh"
netstat -tlnp 2>/dev/null | grep :7860

# Check API logs
tail -50 /workspace/api.log

# Manually relaunch the supervisor (fully detached — survives shell exit)
setsid nohup bash /workspace/start_api.sh </dev/null >>/workspace/api_setup.log 2>&1 &
```

**Job stuck in `processing`**
First job after pod start is slow — models load into VRAM (~3–5 min). Subsequent jobs are fast.

**`HF_TOKEN` error in setup log**
Set `HF_TOKEN` in your RunPod template environment variables and redeploy.
Make sure you have accepted both model licenses (see Prerequisites above).

**LTX checkpoint download failed**
Ensure you've accepted the license at https://huggingface.co/Lightricks/LTX-2.3-fp8

**API crashed and not restarting**
The `start_api.sh` has an auto-restart loop. Check if the supervisor is running:
```bash
ps aux | grep -v grep | grep start_api
# If not running:
setsid nohup bash /workspace/start_api.sh </dev/null >>/workspace/api_setup.log 2>&1 &
```

**API didn't come back after pod restart**
Confirm the `/start.sh` hook is in place (it should be, after `setup.sh` ran once):
```bash
grep -c "AI Gen API v2 auto-start" /start.sh    # should print 2
```
If the hook is missing, re-run setup to re-apply it (two separate commands):
```bash
wget -qO /tmp/setup.sh https://raw.githubusercontent.com/cyrusjaysondev/ai-server/main/setup.sh
```
```bash
bash /tmp/setup.sh
```

**`:7860` returns 502 on a brand-new pod (auto-setup never ran)**
This happens when your template's **Container Start Command** is empty or wrong, so
only the image's default `/start.sh` ran. ComfyUI (`:8188`) and Jupyter (`:8888`) will
be up but `:7860` stays 502 — and `/workspace/api_setup.log` will have no new entries
at the current time.

Quick check on the pod:
```bash
# If the log stops at an old timestamp, setup.sh never fired
tail -5 /workspace/api_setup.log && date -u
```

Two fixes:

1. **Unblock this pod right now** — follow [Step 3b](#step-3b--manual-fallback-if-auto-setup-didnt-trigger) to run setup.sh manually. Takes ~15 s if the volume has cached models.

2. **Fix the template for future pods** — open runpod.io → Templates → edit your
   template → **Container Start Command**. Paste this single line (copy exactly, no
   wrapping):

   ```
   bash -c "wget -qO /tmp/boot.sh https://raw.githubusercontent.com/cyrusjaysondev/ai-server/main/boot.sh && bash /tmp/boot.sh"
   ```
   Save the template. Next fresh pod deploys hands-off.

---

## Models Installed

| Model | Size | Source | Token required |
|-------|------|--------|----------------|
| FLUX.2 Klein 9B UNET | 18 GB | black-forest-labs/FLUX.2-klein-9B | Yes (gated) |
| FLUX VAE | 321 MB | Comfy-Org/flux2-klein-9B | No |
| Qwen 3 8B text encoder | 8.1 GB | Comfy-Org/flux2-klein-9B | No |
| BFS Head Swap LoRA | 633 MB | Alissonerdx/BFS-Best-Face-Swap | No |
| LTX-2.3 22B checkpoint | 27 GB | Lightricks/LTX-2.3-fp8 | Yes (gated) |
| LTX-2.3 distilled LoRA | 7.1 GB | Lightricks/LTX-2.3 | No |
| Gemma abliterated LoRA | 599 MB | Comfy-Org/ltx-2 | No |
| Gemma 3 12B text encoder | 8.8 GB | Comfy-Org/ltx-2 | No |
| LTX-2.3 spatial upscaler | 950 MB | Lightricks/LTX-2.3 | No |
| **Total** | **~72 GB** | | |

> Recommended volume size: **200 GB** to have room for generated outputs.

---

## Architecture

```
RunPod Pod
├── ComfyUI (port 8188) — model inference engine
│   ├── models/checkpoints/     — LTX 2.3, FLUX Klein
│   ├── models/loras/           — distilled LoRA, BFS, Gemma
│   ├── models/text_encoders/   — Gemma 12B, Qwen 8B
│   └── custom_nodes/LanPaint/  — face swap node
│
├── FastAPI (port 7860) — REST API layer
│   ├── /workspace/api/main.py  — all endpoints, workflow builder
│   └── /workspace/api/config.env
│
├── /workspace/start_api.sh     — auto-start + auto-restart script
└── /workspace/api_setup.log    — setup + runtime logs
```

Requests flow: **Client → FastAPI (7860) → ComfyUI (8188) → GPU → output video/image**
