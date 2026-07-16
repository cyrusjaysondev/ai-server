# Serverless Deployment — API Reference

> **For the step-by-step deploy guide, see [`../SERVERLESS_SETUP.md`](../SERVERLESS_SETUP.md).**
> This file is the API reference: endpoint inputs/outputs and example calls.

**Both endpoints write outputs to the network volume** at
`/runpod-volume/outputs/<job_id>/<filename>` and return the **path** in the
response (not base64). Images and videos use the same convention — clients
fetch the file via the RunPod S3 API or from any pod that has the same
volume mounted. The reaper (`cleanup.sh`) deletes everything in this dir
older than 3 days.

Two RunPod serverless endpoints share the same network volume that the pod uses:

| Endpoint | Workflows | Image size | Models loaded from volume |
|---|---|---|---|
| **Image** (`serverless/image/`) | `t2i`, `flux/face-swap`, `flux/i2i` | ~6 GB (no weights baked in) | FLUX.2 Klein 9B + VAE + Qwen 3 + BFS LoRA |
| **Video** (`serverless/video/`) | `ltx/i2v`, `ltx/t2v` | ~6 GB | LTX-2.3 22B + distilled LoRA + Gemma 12B + Gemma LoRA + upscaler |

`face-animate` is **client-orchestrated**: call image first, then feed the
returned image into video's `ltx/i2v`. Keeping the two endpoints physically
separate avoids loading 70 GB of weights into one worker.

## Prerequisites

1. A RunPod network volume containing your models at:
   ```
   /runpod-volume/runpod-slim/ComfyUI/models/
       diffusion_models/flux2-klein-9b.safetensors        (image)
       vae/flux2-vae.safetensors                          (image)
       text_encoders/qwen_3_8b_fp8mixed.safetensors       (image)
       loras/bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors  (image)
       checkpoints/ltx-2.3-22b-dev-fp8.safetensors        (video)
       loras/ltx-2.3-22b-distilled-lora-384.safetensors   (video)
       loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors (video)
       text_encoders/gemma_3_12B_it_fp4_mixed.safetensors (video)
       latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors (video)
   ```
   The pod template's `setup.sh` already downloads everything to this layout
   on the volume — if you have a working pod, the models are in the right
   place.
2. A Docker Hub (or other registry) account to push images to.
3. RunPod account with serverless access.

## Build & push images

From the repo root (build context must be the root so `workflows.py` is in scope):

```bash
# Image worker (~6 GB, mostly ComfyUI runtime)
docker build -f serverless/image/Dockerfile -t <user>/ai-gen-image:latest .
docker push   <user>/ai-gen-image:latest

# Video worker
docker build -f serverless/video/Dockerfile -t <user>/ai-gen-video:latest .
docker push   <user>/ai-gen-video:latest
```

Builds on the same machine reuse the `runpod/worker-comfyui:5.4.1-base` layer
across both, so the second build is fast.

## Create the endpoints

In **runpod.io → Serverless → New Endpoint** for each worker:

1. **Container Image:** `<user>/ai-gen-image:latest` (or `:ai-gen-video:latest`)
2. **GPU:** image worker — 24 GB VRAM minimum (A5000/A6000/L4 ok).
   Video worker — **48 GB+ recommended** (A6000/L40S/A100). LTX 22B in fp8
   plus Gemma 12B encoder is tight on 24 GB.
3. **Workers:** Min 0 (scale to zero), Max as needed.
4. **Container Disk:** 20 GB (image is ~6 GB, plus ComfyUI runtime swap).
5. **Network Volume:** select the same volume your pod uses.
   RunPod mounts it at `/runpod-volume` automatically — `start.sh` reads
   models from there.
6. **Environment Variables (optional):**
   - `VOLUME_OUTPUTS` (video only) — where to write generated videos.
     Defaults to `/runpod-volume/outputs`.

No env vars needed for HF tokens at runtime — models are already on the
volume.

## Invoke

### Image / t2i

```bash
ENDPOINT=https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync
TOKEN=<RUNPOD_API_TOKEN>

curl -X POST "$ENDPOINT" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "endpoint": "t2i",
      "prompt": "a beautiful sunset over mountains, photorealistic, 4K",
      "width": 1024,
      "height": 1024,
      "seed": 42
    }
  }'
```

Response (`runsync` blocks; use `/run` for async):
```json
{
  "id": "<job_id>",
  "status": "COMPLETED",
  "output": {
    "image_path": "/runpod-volume/outputs/<job_id>/t2i_42_00001_.png",
    "filename":   "t2i_42_00001_.png",
    "size_bytes": 423104,
    "seed": 42,
    "duration_seconds": 12.3
  }
}
```

`image_path` is the file on the network volume. Fetch it (see
"Downloading outputs" below for the three options).

### Image / flux/face-swap

```bash
TARGET_B64=$(base64 -w0 body.png)
FACE_B64=$(base64 -w0 face.png)

curl -X POST "$ENDPOINT" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": {
      \"endpoint\": \"flux/face-swap\",
      \"target_image_b64\": \"$TARGET_B64\",
      \"face_image_b64\": \"$FACE_B64\",
      \"aspect_ratio\": \"original\",
      \"megapixels\": 2.0
    }
  }"
```

### Image / flux/i2i (multi-reference editing, 1–5 images)

Send 1 to 5 base64-encoded images plus an edit prompt. Output canvas
defaults to the first image's rescaled dimensions; override with
`width`/`height` if you need a fixed size. Each input image accepts
either raw base64 or a `data:image/...;base64,...` data URI.

```bash
A=$(base64 -w0 subject.png)
B=$(base64 -w0 outfit.png)
C=$(base64 -w0 setting.png)

curl -X POST "$ENDPOINT" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": {
      \"endpoint\": \"flux/i2i\",
      \"images_b64\": [\"$A\", \"$B\", \"$C\"],
      \"prompt\": \"the subject from image 1 wearing the outfit from image 2 in the setting of image 3, photorealistic\",
      \"megapixels\": 2.0,
      \"seed\": 42
    }
  }"
```

Response (same shape as t2i + a `ref_count` field):
```json
{
  "output": {
    "image_path": "/runpod-volume/outputs/<job_id>/flux_i2i_42_00001_.png",
    "filename":   "flux_i2i_42_00001_.png",
    "size_bytes": 412104,
    "seed": 42,
    "ref_count": 3,
    "duration_seconds": 14.7
  }
}
```

| Param | Default | Notes |
|---|---|---|
| `images_b64` | required | List of 1–5 base64 strings (data URI prefix ok) |
| `prompt` | required | The edit instruction |
| `seed` | -1 | Reproducibility seed |
| `megapixels` | 2.0 | Resolution per reference image |
| `width` / `height` | 0 / 0 | `0` = derive from first image |
| `steps` | 4 | Inference steps |
| `cfg` | 1.0 | CFG scale |
| `guidance` | 4.0 | FLUX guidance (2.0–6.0) |
| `lora_strength` | 0.0 | `0` = general edits, `0.5–1.0` = face/head-focused |
| `face_filter` | false | Reject if any input image matches a face on `/runpod-volume/blocklist/`. |
| `logo_filter` | false | Reject if any input image matches a logo/flag on `/runpod-volume/blocklist_logos/`. |

### Compliance filters (face + logo, independent)

Both `flux/face-swap` and `flux/i2i` accept **two independent** compliance
toggles: `face_filter` and `logo_filter`. They run in sequence on every
input image. Both default to `false`. The blocklists are managed via the
pod's `/admin/blocklist` and `/admin/blocklist-logos` admin APIs (see
top-level `API.md`); serverless workers share the same network volume and
hot-reload changes on every request.

Block response (`filter` field tells you which check fired):
```json
{
  "error": "blocked",
  "filter": "face",                       // or "logo"
  "reason": "target_image_b64 matches blocked face",
  "matched_identity": "celebrity_x",      // present when filter="face"
  "score": 0.87,
  "image_index": 0
}
```

```json
{
  "error": "blocked",
  "filter": "logo",
  "reason": "images_b64[0] matches blocked logo/flag",
  "matched_logo": "apple_logo",            // present when filter="logo"
  "score": 0.91,
  "image_index": 0
}
```

Every `face_filter=false` or `logo_filter=false` is appended to
`/runpod-volume/face_filter_bypass.log` for compliance audit.

**Note on coverage:** the face filter (InsightFace) is precise on clear
faces; the logo filter (CLIP whole-image) catches dominant logos but
**may miss small logos in a corner** of a larger photo. Crop your
blocklist entries tightly on the target for best CLIP recall.

### Video / ltx/i2v

```bash
ENDPOINT=https://api.runpod.ai/v2/<VIDEO_ENDPOINT_ID>/run
IMG_B64=$(base64 -w0 input.png)

curl -X POST "$ENDPOINT" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": {
      \"endpoint\": \"ltx/i2v\",
      \"image_b64\": \"$IMG_B64\",
      \"prompt\": \"slow cinematic zoom, gentle breeze\",
      \"preset\": \"fast\",
      \"aspect_ratio\": \"9:16\"
    }
  }"
```

Use `/run` (async) for video — generations routinely take 1-3 minutes. Poll
`/status/<job_id>` for completion. The output is **not** base64; instead:

```json
{
  "output": {
    "video_path": "/runpod-volume/outputs/<job_id>/ltx_i2v_42_00001_.mp4",
    "filename": "ltx_i2v_42_00001_.mp4",
    "size_bytes": 18472104,
    "seed": 42,
    "duration_seconds": 87.4
  }
}
```

The `video_path` is inside the network volume — read it from any pod or
serverless worker that has the same volume mounted, or use the RunPod
S3-compatible API (`https://s3api-<region>.runpod.io`) to download it from
outside RunPod.

### Video / ltx/t2v

Same as i2v but without `image_b64`:

```json
{
  "input": {
    "endpoint": "ltx/t2v",
    "prompt": "drone footage flying over a forest at dawn, cinematic",
    "preset": "fast",
    "aspect_ratio": "16:9"
  }
}
```

## Face-animate (client-orchestrated)

The image endpoint returns a path, not bytes. To chain into video/i2v we
need the bytes — read them via the RunPod S3 API (any client) or directly
off the volume if the orchestrator runs on a pod that mounts it.

```python
import base64, requests, boto3

def call(endpoint_id, payload, token):
    r = requests.post(f"https://api.runpod.ai/v2/{endpoint_id}/runsync",
                      json={"input": payload},
                      headers={"Authorization": f"Bearer {token}"})
    return r.json()["output"]

# 1. Face-swap on image endpoint — returns a path
swap = call(IMAGE_ENDPOINT, {
    "endpoint": "flux/face-swap",
    "target_image_b64": base64.b64encode(open("body.png", "rb").read()).decode(),
    "face_image_b64":   base64.b64encode(open("face.png", "rb").read()).decode(),
    "aspect_ratio": "9:16",
}, TOKEN)

# 2. Fetch the swapped image bytes — pick ONE based on where this script runs:
#    (a) Same network volume mounted locally: just read the file
#        swap_bytes = open(swap["image_path"], "rb").read()
#    (b) Anywhere with RunPod S3 creds:
s3 = boto3.client("s3", endpoint_url=f"https://s3api-{VOLUME_REGION}.runpod.io",
                  aws_access_key_id=RUNPOD_S3_KEY, aws_secret_access_key=RUNPOD_S3_SECRET)
# Strip the leading "/runpod-volume/" to get the S3 key
key = swap["image_path"].removeprefix("/runpod-volume/")
swap_bytes = s3.get_object(Bucket=VOLUME_ID, Key=key)["Body"].read()

# 3. Animate the swapped image on video endpoint
anim = call(VIDEO_ENDPOINT, {
    "endpoint": "ltx/i2v",
    "image_b64": base64.b64encode(swap_bytes).decode(),
    "prompt": "subject smiles and turns head slowly, soft cinematic light",
    "aspect_ratio": "9:16",
    "preset": "fast",
}, TOKEN)

print("video at:", anim["video_path"])
```

## Cold start expectations

- **Image worker:** ~30 s ComfyUI boot + ~30 s loading FLUX 9B into VRAM on
  first request → first response in ~60-90 s. Subsequent requests on the
  same warm worker: 8-15 s.
- **Video worker:** ~30 s boot + ~60-90 s loading LTX 22B + Gemma 12B on
  first request → first response in 2-3 min. Subsequent on warm: 60-180 s
  depending on preset.

Set **active workers** (min > 0) to keep workers warm and skip the model
load on each request — at the cost of paying for idle GPU time.

## Troubleshooting

**`FATAL: network volume models not found at /runpod-volume/...`** — the
endpoint is missing the Network Volume attachment. Edit the endpoint → add
the volume → restart workers.

**`ComfyUI not ready after 300s`** — check the worker logs. The model files
on the volume may be incomplete; try mounting the volume on a pod and
running `setup.sh` to re-verify sizes.

**`unknown endpoint 'foo'`** — `input.endpoint` must be exactly one of
`t2i`, `flux/face-swap`, `flux/i2i` (image worker) or `ltx/i2v`, `ltx/t2v`
(video worker). No leading slash.
