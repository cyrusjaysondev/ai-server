# AI Gen API v2 — Serverless Setup Guide

Step-by-step deployment of the two RunPod serverless endpoints:

- **Image endpoint** (`t2i`, `flux/face-swap`, `flux/i2i`) — FLUX.2 Klein 9B
- **Video endpoint** (`ltx/i2v`, `ltx/t2v`) — LTX-2.3 22B

`face-animate` is **client-orchestrated**: call image first, feed the result
into video. Keeping them split avoids loading 70 GB of weights into one
worker.

For the **API request/response shape**, see [`serverless/README.md`](serverless/README.md).
For pod deployment instead, see [`SETUP.md`](SETUP.md).

---

## Prerequisites

You need all of these before starting:

1. **A RunPod network volume** with the FLUX.2 + LTX models on it, at:
   ```
   <volume>/runpod-slim/ComfyUI/models/
   ```
   The pod template's `setup.sh` already populates the volume in this exact
   layout — if you have a working pod, you're done. To populate from scratch
   without a pod, mount the volume in any RunPod GPU pod and run
   `bash setup.sh` from this repo with `HF_TOKEN` set.

2. **Note the volume's region.** RunPod serverless can only attach to
   network volumes in the same region. In runpod.io → **Storage**, find
   your volume's region (e.g. `EU-RO-1`, `US-CA-2`, `AP-JP-1`). You'll
   need this in Step 3.

   **Verify on this pod:**
   ```bash
   df /workspace | tail -1 | awk '{print $1}'
   # Example output: mfs#ap-jp-1.runpod.net:9421
   #                       ^^^^^ that's your region — AP-JP-1
   ```

3. **A Docker registry account** (Docker Hub free tier works).
   - **Public images** (recommended for simplicity): no extra setup on
     RunPod's side.
   - **Private images**: you'll need to add registry credentials to the
     endpoint config in Step 3 (runpod.io endpoint → **Container** →
     **Container Registry Credentials**).

4. **A build machine with Docker.** Options:
   - Your laptop (fastest if you have decent upload bandwidth).
   - A RunPod **CPU pod** (cheap; same region as the volume helps push
     speed but isn't required since you push to Docker Hub, not the volume).
   - **Not the GPU pod** you use for inference — Docker-in-Docker on RunPod
     GPU pods is awkward and slow.

5. **A RunPod API key** for invoking endpoints (Settings → API Keys).

---

## Step 1 — Verify models are on the network volume

Easiest: SSH into a pod that has the volume mounted at `/workspace`, then:

```bash
cd /workspace/runpod-slim/ComfyUI/models
ls -lh diffusion_models/flux2-klein-9b.safetensors \
       vae/flux2-vae.safetensors \
       text_encoders/qwen_3_8b_fp8mixed.safetensors \
       loras/bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors \
       checkpoints/ltx-2.3-22b-dev-fp8.safetensors \
       loras/ltx-2.3-22b-distilled-lora-384.safetensors \
       loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors \
       text_encoders/gemma_3_12B_it_fp4_mixed.safetensors \
       latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors
```

All 9 files should exist with non-zero sizes. Expected totals: image set
~27 GB, video set ~44 GB.

If anything is missing, the pod's `setup.sh` has a verify-and-resume step:

```bash
bash setup.sh    # safe to re-run; skips files that match HF sizes
```

---

## Step 2 — Build and push the Docker images

On your build machine (laptop or CPU pod):

```bash
git clone https://github.com/cyrusjaysondev/ai-server.git
cd ai-gen-api-v2
```

Log in to your Docker registry (substitute your username):

```bash
docker login            # or: docker login ghcr.io
```

Build and push the **image** worker (build context must be the repo root
so `workflows.py` is in scope):

```bash
docker build -f serverless/image/Dockerfile -t <user>/ai-gen-image:latest .
docker push   <user>/ai-gen-image:latest
```

Build and push the **video** worker:

```bash
docker build -f serverless/video/Dockerfile -t <user>/ai-gen-video:latest .
docker push   <user>/ai-gen-video:latest
```

Each image is ~6 GB (mostly the `runpod/worker-comfyui` base; no model
weights). First build downloads the base layer; second build reuses it.

> **Note:** The base image is `runpod/worker-comfyui:5.4.1-base`. If you
> want to pin a different version, edit `FROM` in both Dockerfiles. The
> base must ship **ComfyUI ≥ 0.18** so the built-in LTX 2.3 nodes
> (`LTXVImgToVideoInplace`, `LTXAVTextEncoderLoader`, etc.) and
> `EmptyFlux2LatentImage` are present. If a cold worker logs
> `KeyError: 'LTXVImgToVideoInplace'` or similar, the base image's
> ComfyUI is too old — bump the tag.

### Custom nodes baked into the images

- **Image worker:** `LanPaint` (required by `flux/face-swap`).
- **Video worker:** `ComfyUI-KJNodes` (required by `ltx/i2v` for
  `ColorMatch` + `ResizeImageMaskNode` + `ResizeImagesByLongerEdge`).
  `ltx/t2v` doesn't use these but `i2v` is unconditional.

If you push fixes to the handlers but the workflow files reference a
**new** custom node, you must update the Dockerfile to install it and
rebuild the image — workers don't pull nodes at runtime.

---

## Step 3 — Create the Image serverless endpoint

1. Go to **runpod.io** → **Serverless** → **+ New Endpoint**.
2. Fill in:
   - **Endpoint Name:** `ai-gen-image`
   - **Container Image:** `<user>/ai-gen-image:latest`
   - **Container Disk:** `20 GB`
   - **GPU:** at least **24 GB VRAM** (A5000, A6000, L4, or L40S all work).
     FLUX.2 Klein in fp8 needs ~14 GB for weights plus headroom for
     activations.
3. **Workers:**
   - Min: `0` (scale to zero — pay nothing when idle)
   - Max: `3` (or however much concurrency you need)
   - Idle Timeout: `5 seconds` (kill cold workers fast; raise if you want
     warmer workers to handle bursts)
4. **Network Volume:** select your volume.
   **Region must match the volume's region** from prerequisites step 2.
   If your volume's region isn't listed, RunPod doesn't have serverless
   capacity there yet — pick a different volume region or contact RunPod.
5. **Environment Variables:** none required. Optional:
   - `COMFYUI_URL` — override if you change ComfyUI's listen port.
   - `FACE_FILTER_THRESHOLD` — cosine similarity threshold for the face
     blocklist (default `0.6`, higher = stricter, fewer matches).
   - `LOGO_FILTER_THRESHOLD` — cosine similarity threshold for the
     logo/flag blocklist (default `0.85`).
6. Click **Deploy**. RunPod pulls the image (~2 min first time), and
   the endpoint goes **Ready** when a worker has booted.

---

## Step 4 — Create the Video serverless endpoint

Same as Step 3 with these differences:

- **Endpoint Name:** `ai-gen-video`
- **Container Image:** `<user>/ai-gen-video:latest`
- **GPU:** **48 GB+ VRAM strongly recommended** (A6000, L40S, A100 40GB,
  or H100). LTX-2.3 22B in fp8 (~24 GB) plus Gemma 12B encoder (~9 GB)
  plus activations is tight on 24 GB cards and may OOM on long clips.
- **Container Disk:** `20 GB`
- **Environment Variables:**
  - `VOLUME_OUTPUTS` (optional) — where to stage generated videos on the
    volume. Defaults to `/runpod-volume/outputs`. Override only if you
    want a different path.

---

## Step 5 — Verify both endpoints

Grab your endpoint IDs from the runpod.io serverless dashboard and set
shell variables:

```bash
export RUNPOD_TOKEN=<your-api-key>
export IMG_ID=<image-endpoint-id>
export VID_ID=<video-endpoint-id>
```

### Test the image endpoint

```bash
curl -X POST "https://api.runpod.ai/v2/$IMG_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "endpoint": "t2i",
      "prompt": "a serene mountain lake at sunrise, photorealistic, 4K",
      "width": 1024, "height": 1024, "seed": 42
    }
  }' | tee response.json | jq '.status, .output.filename, .output.duration_seconds'
```

First call is the cold start — expect **60–90 s** while ComfyUI boots and
loads FLUX into VRAM. Subsequent calls on the warm worker: 8–15 s.

The response returns a **path** on the network volume (not base64), so
images and videos use the same retrieval flow. Example response:

```json
{
  "output": {
    "image_path": "/runpod-volume/outputs/<job_id>/t2i_42_00001_.png",
    "filename":   "t2i_42_00001_.png",
    "size_bytes": 423104,
    "seed": 42,
    "duration_seconds": 12.3
  }
}
```

See **"Downloading outputs"** below for the three ways to fetch the
file. The reaper deletes everything in `/runpod-volume/outputs/` older
than 3 days, so save anything you need to keep.

### Test the video endpoint

Video requests take 1–3 minutes — use `/run` (async) and poll:

```bash
# Submit
JOB=$(curl -sX POST "https://api.runpod.ai/v2/$VID_ID/run" \
  -H "Authorization: Bearer $RUNPOD_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "endpoint": "ltx/t2v",
      "prompt": "drone footage flying low over a misty forest at dawn, cinematic",
      "preset": "fast",
      "aspect_ratio": "16:9"
    }
  }' | jq -r .id)
echo "job=$JOB"

# Poll
while true; do
  STATUS=$(curl -s "https://api.runpod.ai/v2/$VID_ID/status/$JOB" \
    -H "Authorization: Bearer $RUNPOD_TOKEN" | jq -r .status)
  echo "status=$STATUS"
  [ "$STATUS" = "COMPLETED" ] || [ "$STATUS" = "FAILED" ] && break
  sleep 10
done

# Inspect final result
curl -s "https://api.runpod.ai/v2/$VID_ID/status/$JOB" \
  -H "Authorization: Bearer $RUNPOD_TOKEN" | jq .output
```

Output shape:
```json
{
  "video_path": "/runpod-volume/outputs/<job_id>/ltx_t2v_42_00001_.mp4",
  "filename":   "ltx_t2v_42_00001_.mp4",
  "size_bytes": 18472104,
  "seed":       42,
  "duration_seconds": 87.4
}
```

To **read the file**, see "Downloading video outputs" below.

---

## Downloading outputs

Both endpoints write to `/runpod-volume/outputs/<job_id>/<filename>` and
return that path. Three ways to fetch the file:

### Option A — From any pod with the volume mounted

```bash
# In a pod with the same network volume attached at /workspace:
cp /workspace/outputs/<job_id>/*.png ./   # or *.mp4 for video
```

### Option B — RunPod S3-compatible API (recommended for external clients)

RunPod exposes every network volume via an S3 API. From any machine:

```bash
# One-time: install aws-cli or use any S3-compatible tool
aws configure                                # use your RunPod S3 keys
# endpoint URL: https://s3api-<region>.runpod.io

aws --endpoint-url https://s3api-ap-jp-1.runpod.io s3 \
    cp s3://<volume-id>/outputs/<job_id>/<filename> ./
```

Get your S3 keys at runpod.io → **Settings → S3 API**. Use the region
that matches your volume.

### Option C — Use a small "downloader" pod

For one-off downloads, spin up a CPU pod with the volume mounted, then
`scp` the file out. Cheapest for the occasional check; not for production.

> ⚠️ **Files older than 3 days are auto-deleted** by the reaper. Save
> anything you need to keep before that.

---

## Optional — Keep workers warm

By default workers scale to zero. The cost: every cold call pays a 60–90 s
(image) or 2–3 min (video) startup penalty.

To eliminate cold starts at the cost of paying for idle GPU time:

- runpod.io → your endpoint → **Edit**
- Set **Min Workers** to `1` (or more)
- Save

Warm workers stay loaded with weights in VRAM and respond in seconds. Use
this for production traffic; leave at `0` for development.

---

## Force-refresh code on the running pod

If you push a fix to GitHub `main` and want the pod-mode API to pick it up
without a full pod restart, just kill uvicorn — the supervisor re-fetches
`main.py` + `workflows.py` from GitHub and restarts within ~5 seconds:

```bash
pkill -f "uvicorn main:app"
# Wait ~10s, then verify:
curl http://localhost:7860/health
```

Serverless workers always pull the image you pushed — to deploy new
handler code, rebuild and push the Docker image, then in runpod.io →
endpoint → **Workers** → terminate the active workers. The next request
spawns fresh workers that pull the new image.

---

## Managing the compliance blocklists

**The admin API for the face + logo blocklists lives on the pod, NOT on
the serverless workers.** Serverless workers come and go (scale to zero,
restart on demand) — they're a bad surface for write operations. The pod
is always-on, has the same network volume mounted, and is the natural
single-writer.

The flow is:

```
  Your CMS  ─POST /admin/blocklist────►  POD (https://<POD>-7860...)
                                          │
                                          │ writes face image to
                                          │ /workspace/blocklist/ on the volume
                                          ▼
                                  Network Volume
                                          ▲
                                          │ serverless workers see the
                                          │ same files at /runpod-volume/
                                          │ blocklist/ and hot-reload them
                                          │ on every face_filter check
                            ┌─────────────┴──────────────┐
                  Image Worker A           Image Worker B
                  (enforces filter)        (enforces filter)
```

This means:

- **Your CMS only talks to the pod** for blocklist management. There's no
  serverless admin endpoint.
- **Serverless workers enforce the filter** when you pass
  `face_filter=true` or `logo_filter=true` on a generation request.
- **Updates propagate automatically.** Add a face via the pod's
  `POST /admin/blocklist` → the next serverless request that runs the
  filter picks it up (hot-reload via volume-mtime scan).
- **The pod must be running** to manage the blocklist. If you spin the
  pod down, you can still mount the volume on any RunPod pod or use the
  RunPod S3 API to add/remove files manually under
  `<volume>/blocklist/` and `<volume>/blocklist_logos/`.

See [`API.md`](API.md) → "Admin API (blocklist management)" for the
endpoint details (`/admin/blocklist`, `/admin/blocklist-logos`, list /
upload / delete / preview).

**Auth:** by default the pod's admin API is **open** (no auth required).
Set `ADMIN_TOKEN` as a pod env var to require
`Authorization: Bearer <token>` on every admin call — recommended before
sharing the pod URL.

---

## Troubleshooting

### Worker logs show: `FATAL: network volume models not found at /runpod-volume/runpod-slim/ComfyUI/models`

The endpoint isn't getting the network volume, OR the endpoint's region
doesn't match the volume's region.

- runpod.io → endpoint → **Edit** → **Network Volume** → select your
  volume → **Save**.
- **Region match is mandatory.** If your volume is in `AP-JP-1` and the
  endpoint is in `US-CA-2`, RunPod won't even let you attach (the volume
  won't appear in the dropdown). Recreate the endpoint in the volume's
  region.
- Restart the worker (or just send a new request — the next cold start
  picks up the new config).

### Worker logs show: `ComfyUI not ready after 300s`

ComfyUI is failing to boot. Most common causes:

1. **A model file on the volume is incomplete or zero-byte.** Mount the
   volume on a pod and re-run `bash setup.sh` — it verifies each file
   against HuggingFace's expected size and resumes anything short.
2. **GPU too small.** For the video worker, this often means OOM during
   weight load. Increase GPU size in the endpoint config.

### Handler returns `{"error": "unknown endpoint 'foo'"}`

`input.endpoint` must be exactly one of:
- Image worker: `t2i`, `flux/face-swap`, or `flux/i2i`
- Video worker: `ltx/i2v` or `ltx/t2v`

No leading slash. Case-sensitive.

### Cold starts are unbearable

Three knobs, in order of impact:

1. Set **Min Workers ≥ 1** (eliminates cold start entirely; you pay for
   idle GPU).
2. Use **Flashboot** in the endpoint settings (RunPod's snapshot-restore;
   knocks ~30–50% off cold start without paying full idle cost).
3. Move to a faster GPU — model load time is bandwidth-bound, so an A100
   loads FLUX faster than an A5000.

### Image build is slow / disk full

The base layer `runpod/worker-comfyui:5.4.1-base` is ~5 GB. If you're
rebuilding repeatedly, use BuildKit cache:

```bash
DOCKER_BUILDKIT=1 docker build \
  --cache-from <user>/ai-gen-image:latest \
  -f serverless/image/Dockerfile \
  -t <user>/ai-gen-image:latest .
```

---

## What gets billed

- **GPU time on the endpoint** — billed per second a worker is running,
  whether processing or idle. Scale to zero (`Min Workers: 0`) to pay
  nothing between requests.
- **Network volume storage** — billed monthly per GB, regardless of
  endpoint state. Same volume the pod uses, so no double-billing.
- **Docker Hub egress** — free for public images; paid for private images
  past Docker Hub's free quota.

Pod-mode and serverless-mode are **independent**. You can run both
simultaneously off the same volume — the pod always-on for development,
the endpoints for production bursts.
