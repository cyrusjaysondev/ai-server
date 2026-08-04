# AI Gen API v2 — API Reference

Base URL: `https://YOUR_POD_ID-7860.proxy.runpod.net`

Interactive docs (Swagger UI): `https://YOUR_POD_ID-7860.proxy.runpod.net/docs`

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/t2i` | Text to image (FLUX.2 Klein 9B) |
| POST | `/flux/face-swap` | Head / face swap (FLUX.2 Klein 9B) |
| POST | `/flux/multi-face-swap` | One- or two-person face swap (FLUX.2 Klein 9B) |
| POST | `/flux/i2i` | Multi-reference image editing — 1 to 5 input images (FLUX.2 Klein 9B) |
| GET | `/admin/blocklist` | List blocked face identities (admin auth) |
| POST | `/admin/blocklist` | Upload a face to block |
| DELETE | `/admin/blocklist/{identity}` | Remove a blocked face |
| GET | `/admin/blocklist/{identity}/image` | Preview a blocked face image |
| GET | `/admin/blocklist-logos` | List blocked logos/flags (admin auth) |
| POST | `/admin/blocklist-logos` | Upload a logo/flag to block |
| DELETE | `/admin/blocklist-logos/{identity}` | Remove a blocked logo/flag |
| GET | `/admin/blocklist-logos/{identity}/image` | Preview a blocked logo image |
| POST | `/ltx/i2v` | Image to video (LTX 2.3) |
| POST | `/ltx/motion` | Copy a reference video's body motion onto a character image |
| POST | `/ltx/t2v` | Text to video (LTX 2.3) |
| POST | `/face-animate` | Face swap + animate pipeline |
| GET | `/ltx/presets` | List available speed/quality presets |
| GET | `/status/{job_id}` | Poll job status |
| GET | `/jobs` | List all jobs |
| GET | `/queue` | Active queue |
| POST | `/jobs/{job_id}/retry` | Retry failed job |
| DELETE | `/jobs/{job_id}` | Delete job + file |
| DELETE | `/jobs` | Bulk delete jobs |
| GET | `/image/{filename}` | Download image |
| GET | `/video/{filename}` | Download video |
| GET | `/videos` | List all videos |

---

## Speed & Quality Presets

All LTX video endpoints (`/ltx/i2v`, `/ltx/t2v`, `/face-animate`) support a `preset` parameter. Both presets use the official LTX-2.3 distilled inference profile (sigmas + LoRA strength taken from Lightricks' reference workflows).

| Preset | Mode | Steps | LoRA Strength | Speed (4s @544×960, warm) | Best for |
|--------|------|-------|---------------|-----------------------------|----------|
| `fast` | Single pass at target resolution | 8 | 0.5 | **~12s** | Default. Quick iteration. |
| `quality` | Two-pass: 8 steps at half-res → 2× spatial upscale → 3 refine steps at full-res | 8 + 3 | 0.5 | **~12s** | Slightly sharper detail. Same wall-time as `fast` because the bulk of compute happens at half-res. |

### How fast preset works

- **Single pass** at full target resolution
- **8 denoising steps** using distilled LoRA (the 8-step warmup-cluster schedule the LoRA was trained against)
- **No audio** by default (skip audio VAE load/encode/decode entirely)

### How quality preset works

- **Two-pass pipeline**: 8 low-res steps → spatial upscale → 3 refine steps at full-res
- The 2× spatial upscale uses Lightricks' `ltx-2.3-spatial-upscaler-x2-1.0.safetensors`
- LoRA strength 0.5 across both passes — matches reference

### Speed by length (fast preset, 544×960, warm GPU)

| `length` | Duration | Approx wall time |
|----------|----------|------------------|
| 49 | ~2s | ~7s |
| 97 | ~4s | ~12s |
| 121 | ~5s | ~14s |
| 161 | ~6.7s | ~18s |

> **First request after pod start takes ~30-60s longer** as ComfyUI loads the 27 GB LTX checkpoint, 8.8 GB Gemma text encoder, and 7 GB distilled LoRA into VRAM. Subsequent requests reuse the cached models.

### Additional speed knobs

| Knob | Effect |
|------|--------|
| `length=49` | Roughly halves wall time (49-frame ≈ 2s clip) |
| `enhance_prompt=false` | Skips the Gemma prompt-rewrite pass, saves 2-5s. Use when you already wrote a detailed prompt. |
| Smaller `width`/`height` | LTX scales roughly with pixel count. `384×640` is ~2× faster than `544×960`. |
| `audio=false` | Default; skips audio encode/decode (saves ~5-10s) |

### Audio control

All LTX video endpoints accept `audio` (bool, default `false`).

- `audio=false` — video only, faster (skips audio VAE entirely)
- `audio=true` — generates audio track with the video (adds ~5-10s overhead)

```bash
# Without audio (default, faster)
curl -X POST .../ltx/t2v -F "prompt=cat on beach" -F "preset=fast"

# With audio
curl -X POST .../ltx/t2v -F "prompt=cat on beach" -F "preset=fast" -F "audio=true"
```

---

## Health Check

```bash
curl https://YOUR_POD_ID-7860.proxy.runpod.net/health
```

**Response**
```json
{
  "status": "ok",
  "pod_id": "771ykso2hagd1l"
}
```

---

## GET /ltx/presets — List Presets

```bash
curl https://YOUR_POD_ID-7860.proxy.runpod.net/ltx/presets
```

**Response**
```json
{
  "presets": {
    "fast": { "mode": "single_pass", "steps": 8, "lora_strength": 0.5 },
    "quality": { "mode": "two_pass", "low_res_steps": 8, "high_res_steps": 3, "lora_strength": 0.5 }
  },
  "default": "fast",
  "endpoints": ["/ltx/i2v", "/ltx/t2v", "/face-animate"]
}
```

---

## Watermarks

Every generation endpoint accepts two optional watermark parameters. They
are independent and can stack — set both and you get a logo with text
beside it.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `watermark` | string \| null | `null` | Short text drawn at the bottom-right in bold white with a black outline. `null` / empty = off. Example: `"AI"`. |
| `watermark_image` | bool | `false` | Composite the **Metfone GenAI logo** at the bottom-right. The PNG lives on the network volume at `/workspace/assets/metfone_genai_watermark.png` (fetched by `setup.sh`). |

Images stamp in-place via Pillow. Videos re-encode through `libx264` /
`drawtext` / `overlay` filters with the audio stream-copied — typically
adds ~1–3 s for a 5 s clip. A missing logo file is logged and silently
skipped so a network blip during setup never fails a generation.

If a watermark step fails, the job still completes successfully — the
result will include a `watermark_warning` field on the `/status/{job_id}`
response.

---

## Captions

A styled **lower-third caption** for horoscope-style content (the "sign of the
day" text). Same fixed design on images and videos, so a face-swap still and the
animated video it feeds into stay visually consistent.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `caption` | string \| null | `null` | The caption text (e.g. the day's horoscope). Word-wrapped and centered in bold white with a heavy black outline, anchored in the lower third. `null` / empty = no caption. |
| `caption_icon` | string \| null | `null` | A zodiac sign — one of `aries, taurus, gemini, cancer, leo, virgo, libra, scorpio, sagittarius, capricorn, aquarius, pisces`. When set alongside `caption`, a gold sign glyph + a gold divider are stacked **above** the text. Any unrecognised value is ignored (text renders without the glyph). |

**Layout.** From top to bottom in the lower third: optional gold zodiac glyph →
optional gold divider → the centered caption text. The glyph and divider only
appear together, and only when `caption_icon` resolves to a known sign.

**Images vs videos.** On images the caption is baked in-place. On videos it
**fades in ~1 s after the start** (so the opening frame is clean) and stays for
the rest of the clip; the video re-encodes once via ffmpeg.

**Endpoints.** `caption` / `caption_icon` are accepted on `POST /t2i`,
`POST /flux/face-swap`, `POST /flux/i2i`, `POST /ltx/i2v`, and `POST /ltx/t2v`.
They stack with `watermark` / `watermark_image`.

**Assets.** The 12 glyphs + the divider live on the network volume at
`/workspace/assets/zodiac-overlays/` (`icons/<sign>.png` + `divider-gold.png`),
fetched by `setup.sh` from the repo on boot. A missing glyph degrades gracefully
to text-only — it never fails the job.

---

## POST /t2i — Text to Image

Generate an image from a text prompt using FLUX.2 Klein 9B.

### Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | **required** | What to generate |
| `width` | int | `1024` | Output width in pixels |
| `height` | int | `1024` | Output height in pixels |
| `seed` | int | `-1` (random) | Set for reproducible results |
| `steps` | int | `4` | Inference steps (4 is ideal for FLUX Klein) |
| `cfg` | float | `1.0` | CFG scale |
| `guidance` | float | `4.0` | FLUX guidance strength (2.0 – 6.0) |
| `watermark` | string \| null | `null` | See [Watermarks](#watermarks). |
| `watermark_image` | bool | `false` | See [Watermarks](#watermarks). |

### Example

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/t2i \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a woman in a red dress standing in Times Square, photorealistic, 4K",
    "width": 1024,
    "height": 1024,
    "steps": 4,
    "guidance": 4.0
  }'
```

**Response**
```json
{
  "job_id": "a1b2c3d4-...",
  "status": "queued",
  "model": "flux2-klein-9b",
  "poll_url": "https://YOUR_POD_ID-7860.proxy.runpod.net/status/a1b2c3d4-..."
}
```

---

---

## POST /flux/face-swap — Head / Face Swap

Replace the head in a target image with a face from a source image using FLUX.2 Klein 9B + BFS LoRA.

### Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target_image` | file | **required** | Body/template photo — head gets replaced |
| `face_image` | file | **required** | Face photo — identity to transfer |
| `aspect_ratio` | string | `original` | Output aspect ratio (see options below) |
| `megapixels` | float | `2.0` | Total output resolution in megapixels (0.5 – 4.0) |
| `seed` | int | `-1` (random) | Set for reproducible results |
| `steps` | int | `4` | Inference steps |
| `cfg` | float | `1.0` | CFG scale |
| `guidance` | float | `4.0` | FLUX guidance strength (2.0 – 6.0) |
| `lora_strength` | float | `1.0` | BFS LoRA strength (0.5 – 1.0) |
| `require_detectable_face` | bool | `false` | When true, reject `face_image` with `422 image_quality_insufficient` unless InsightFace detects a clear face. |
| `watermark` | string \| null | `null` | See [Watermarks](#watermarks). |
| `watermark_image` | bool | `false` | See [Watermarks](#watermarks). |

### Aspect Ratio Options

| Value | Ratio | Use case |
|-------|-------|----------|
| `original` | Input image AR | Preserve source composition (default) |
| `1:1` | Square | Social media posts |
| `4:3` / `3:4` | Standard | Photos, presentations |
| `16:9` / `9:16` | Wide / Vertical | YouTube / Instagram |
| `3:2` / `2:3` | Classic | DSLR landscape / portrait |
| `21:9` / `9:21` | Cinematic | Ultra-wide / ultra-tall |

### Example

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/flux/face-swap \
  -F "target_image=@body_photo.jpg" \
  -F "face_image=@face_photo.jpg" \
  -F "aspect_ratio=9:16" \
  -F "megapixels=2.0"
```

---

## POST /flux/multi-face-swap — One- or Two-Person Face Swap

Personalize a group/couple template with one or two user face photos. The
template is image 1 internally; repeated `face_images` uploads become images
2–3. Each upload maps to the same-position entry in `target_face_indices`.

Target slot `0` is the first person under `face_order`; slot `1` is the second.
Either slot can be used alone. The delivered image is composited over the
original template so unselected people and all pixels outside selected head
regions remain unchanged. To use the same identity for both people, upload the
same photo twice.

### Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target_image` | file | **required** | Template containing the people to personalize |
| `face_images` | file[] | **required** | One or two files; repeat the multipart field in target mapping order |
| `face_order` | string | `left-to-right` | `left-to-right`, `right-to-left`, `top-to-bottom`, `bottom-to-top`, or `largest-first` |
| `target_face_indices` | comma-separated ints | upload order (`0` or `0,1`) | One distinct `0` or `1` per `face_images` upload; use `1` to replace only the second person |
| `prompt` | string \| null | `null` | Optional template-specific instruction appended after the protected mapping/preservation prompt; max 2,000 characters |
| `aspect_ratio` | string | `original` | Output aspect ratio; same options as `/flux/face-swap` |
| `megapixels` | float | `2.0` | Total output resolution in megapixels (0.5–4.0) |
| `seed` | int | `-1` (random) | Set for reproducible results |
| `steps` | int | `4` | Inference steps |
| `cfg` | float | `1.0` | CFG scale |
| `guidance` | float | `4.0` | FLUX guidance strength |
| `lora_strength` | float | `1.0` | BFS head-swap LoRA strength |
| `require_detectable_face` | bool | `true` | Every `face_images` file must contain a clear human face |
| `face_filter` | bool | `true` | Reject blocked identities in inputs and output |
| `logo_filter` | bool | `true` | Reject blocked logos/flags in inputs and output |
| `watermark` | string \| null | `null` | Optional text watermark |
| `watermark_image` | bool | `false` | Optionally apply the configured logo watermark |

### Two-person example

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/flux/multi-face-swap \
  -F "target_image=@couple_template.jpg" \
  -F "face_images=@left_person.jpg" \
  -F "face_images=@right_person.jpg" \
  -F "face_order=left-to-right" \
  -F "target_face_indices=0,1" \
  -F "aspect_ratio=9:16"
```

### One-person example

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/flux/multi-face-swap \
  -F "target_image=@couple_template.jpg" \
  -F "face_images=@replacement.jpg" \
  -F "face_order=left-to-right" \
  -F "target_face_indices=1"
```

The second example replaces only the second (rightmost) person and preserves
the first person exactly.

---

## POST /flux/i2i — Multi-reference Image Editing

Edit / compose with 1 to 5 reference images. All inputs are chained as FLUX.2
reference latents on top of the prompt's conditioning — the prompt drives
the edit, the images supply style, identity, objects, composition cues.

Output canvas dimensions default to the **first image's rescaled size**, so
you can use the first image as the "edit target" and the rest as references.
Override explicitly with `width` and `height` if you want a fixed canvas.

> **New: composition modes for prompt-less callers** — set
> [`composition_mode`](#composition-modes) and you can leave `prompt`
> empty. The server fills in a mode-specific template prompt and a
> recommended `lora_strength`. Useful when the caller can't (or shouldn't)
> craft a prompt themselves — frontends that only let the user upload a
> photo + pick a library scene, for example.

### Parameters (multipart/form-data)

| Param | Default | Description |
|-------|---------|-------------|
| `images` | required | 1 to 5 image files. First image's dimensions are used as the canvas unless `width`/`height` are set. For `composition_mode=scene_blend`, the **scene image** becomes the canvas (server-side reorder — see [scene_image_index](#composition-modes)). |
| `prompt` | `""` (now optional) | The edit instruction. Optional when `composition_mode` is set — server fills in a mode template. |
| `composition_mode` | `"none"` | `none` \| `auto` \| `scene_blend` \| `outfit_swap` \| `style_transfer`. See [Composition modes](#composition-modes). |
| `quality_preset` | `"none"` | `none` \| `fast` (4 steps) \| `balanced` (8 steps) \| `high` (12 steps). When set, overrides `steps`. |
| `scene_image_index` | `-1` | Only used with `composition_mode=scene_blend`. Which input image is the scene/canvas. `-1` = last (matches "user uploads first, library scene last" UI flow). |
| `seed` | -1 (random) | Reproducibility seed |
| `megapixels` | 2.0 | Resolution per reference image (0.5–4.0) |
| `width` | 0 | Output width — `0` means "derive from first image" |
| `height` | 0 | Output height — `0` means "derive from first image" |
| `steps` | 4 | Inference steps. Overridden by `quality_preset` when one is selected. |
| `cfg` | 1.0 | CFG scale |
| `guidance` | 4.0 | FLUX guidance strength (2.0–6.0) |
| `lora_strength` | `-1` (= mode default) | `-1` (default) → server picks based on `composition_mode` (0 for none/auto, 0.5 for scene_blend, 0.7 for outfit_swap). Pass `0`–`1.5` to override. |
| `require_detectable_face` | false | When true, validate the first image. In `scene_blend`, validate every non-scene image and exclude the scene/template image. |
| `watermark` | null | See [Watermarks](#watermarks). |
| `watermark_image` | false | See [Watermarks](#watermarks). |

### Composition modes

Mode templates are applied **only when the caller leaves `prompt` empty**.
If you provide a `prompt`, the caller wins — the mode just supplies the
LoRA strength default. Likewise `lora_strength` honors any explicit value
≥ 0; only the sentinel `-1` falls back to the mode default.

| Mode | When to use | Template prompt | Default `lora_strength` | Image role layout |
|------|-------------|-----------------|------------------------|--------------------|
| `none` *(default)* | Existing callers. No template applied; behaves exactly like before. | — | (caller's value, or 0) | unchanged |
| `auto` | "I have a few photos, just blend them sensibly." | *"high quality detailed composition of the reference images, photorealistic, sharp, natural lighting"* | 0 | all equal |
| **`scene_blend`** | User uploads subject photo(s); picks a scene from a library. The most common no-prompt case. | *"the subject(s) from the reference images placed naturally in the scene shown in the first image, matched lighting, integrated shadows, photorealistic, sharp focus, detailed environment"* | 0.5 | **scene = canvas** (server reorders so the image at `scene_image_index` is at slot 0). All other images are subjects placed into it. |
| `outfit_swap` | Person + outfit reference. | *"the person from the first image wearing the outfit shown in the second image, full body, photorealistic, natural lighting, detailed fabric texture"* | 0.7 | image[0] = person, image[1] = outfit |
| `style_transfer` | Image 1 painted in image 2's style. | *"the first image reimagined in the artistic style of the second image, preserving composition and subject"* | 0.0 | image[0] = content, image[1] = style ref |

### Example — classic prompted call (unchanged)

```bash
curl -X POST "$POD/flux/i2i" \
  -F "prompt=combine the subject from image 1 with the outfit from image 2 in the setting of image 3" \
  -F "images=@subject.png" \
  -F "images=@outfit.png" \
  -F "images=@setting.png" \
  -F "megapixels=2.0" \
  -F "seed=42"
```

### Example — prompt-less, scene from library

User uploads a photo of themselves and a photo of their car, then picks
"Beach" from the library. The frontend POSTs all three with the library
scene last; the server places `beach_scene.png` at canvas slot 0.

```bash
curl -X POST "$POD/flux/i2i" \
  -F "images=@user_photo.png" \
  -F "images=@user_car.png" \
  -F "images=@beach_scene.png" \
  -F "composition_mode=scene_blend" \
  -F "quality_preset=balanced"
# prompt left empty; lora_strength defaults to 0.5; scene_image_index defaults to -1 (=last).
```

### Response

```json
{
  "job_id": "...",
  "status": "queued",
  "model": "flux2-klein-9b",
  "ref_count": 3,
  "composition_mode": "scene_blend",
  "resolved": {
    "prompt_used": "the subject(s) from the reference images placed naturally in the scene shown in the first image…",
    "lora_strength": 0.5,
    "steps": 8
  },
  "poll_url": "https://YOUR_POD_ID-7860.proxy.runpod.net/status/..."
}
```

The new `resolved` block lets the caller verify what the server actually
decided — useful for prompt-less flows where the caller never wrote a
prompt themselves.

Poll `/status/{job_id}` for the result URL, just like the other endpoints.

---

## POST /ltx/i2v — Image to Video

Generate a video from an input image using LTX 2.3 (22B).

### Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | file | **required** | Source image — first frame of the video |
| `prompt` | string | `""` | Description of the desired motion/scene |
| `negative_prompt` | string | *see below* | What to avoid in the output |
| `preset` | string | `fast` | `fast` (8 steps single-pass, ~10–15s @544×960) or `quality` (8+3 steps two-pass with 2× spatial upscale, ~40–60s) |
| `audio` | bool | `false` | Generate audio track (`true` adds ~5-10s overhead) |
| `aspect_ratio` | string | `9:16` | Output aspect ratio — see table below. **When set, `height` is ignored** and derived from `width`. Use `original` to honor the explicit `width`/`height`. |
| `width` | int | `544` | Output width in pixels. With `aspect_ratio=9:16`, `width=544` → 544×960 (fast preset spec); `width=720` → 720×1280 (quality preset spec). |
| `height` | int | `960` | Output height in pixels. **Ignored unless `aspect_ratio=original`.** |
| `length` | int | `121` | Number of frames (121 = ~5 sec at 24fps) |
| `fps` | int | `24` | Frames per second |
| `seed` | int | `-1` (random) | Set for reproducible results |
| `enhance_prompt` | bool | `true` | Rewrite the prompt via Gemma 12B using the input image as context (adds 2-5s + VRAM). Recommended ON for short prompts (`"make her run"`); OFF when you've already written a detailed scene description. |
| `inplace_strength` | float | `0.7` | How tightly each frame's latent is pinned to the input image. `0.7` is the reference distilled value (best identity, weakest motion). **Lower it for action prompts:** `0.5` ≈ moderate motion, `0.4` ≈ strong motion (some identity drift), `0.3` ≈ near-t2v. Range `0.3`–`1.0`. Two-pass refine tracks this. |
| `require_detectable_face` | bool | `false` | When true, reject `image` with `422 image_quality_insufficient` unless InsightFace detects a clear face. |
| `watermark` | string \| null | `null` | See [Watermarks](#watermarks). Video re-encodes via ffmpeg (~1-3s for a 5s clip). |
| `watermark_image` | bool | `false` | See [Watermarks](#watermarks). |

Default negative prompt: `"low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly"`

### Why your video isn't moving

If the subject barely moves despite a clear action prompt, two things are usually fighting you:

1. **Wrong resolution.** Setting `width=1280 aspect_ratio=9:16` renders **1280×2272** (height is recomputed from width). At that size the input image dominates every frame. Use `width=544` (fast) or `width=720` (quality) for 9:16.
2. **`inplace_strength` too high.** The default `0.7` matches Lightricks' reference profile and prioritizes identity. For motion-heavy prompts, lower it to `0.5` or `0.4`. The model has cfg=1.0 hardwired (mandatory for the distilled LoRA), so the prompt cannot push hard — `inplace_strength` is the real motion knob.

### Aspect Ratio Options

All dimensions are snapped to multiples of 32. When an aspect ratio is set, **height is computed from width** — pick `width` from the table to land on the spec'd resolution.

| Value | `width` for fast | `width` for quality | Use case |
|-------|------------------|---------------------|----------|
| `original` | n/a (uses input dims) | n/a | Preserve source composition |
| `9:16` | `544` → 544×960 | `720` → 720×1280 | **Instagram / TikTok Reels (default)** |
| `16:9` | `960` → 960×544 | `1280` → 1280×720 | YouTube, landscape video |
| `1:1` | `768` → 768×768 | `1024` → 1024×1024 | Social media posts |
| `4:3` / `3:4` | `768` / `576` | `1024` / `768` | Standard / portrait |
| `3:2` / `2:3` | `864` / `576` | `1152` / `768` | Classic landscape / portrait |
| `21:9` / `9:21` | `1120` / `480` | `1280` / `544` | Cinematic ultra-wide / tall |

### Frame length guide

| `length` | Duration (24fps) |
|----------|-----------------|
| 49 | ~2 sec |
| 73 | ~3 sec |
| 97 | ~4 sec |
| 121 | ~5 sec |
| 161 | ~6.7 sec |
| 257 | ~10 sec |

### Examples

```bash
# 9:16 reel — recommended fast preview (5s, ~10-15s on a warm GPU)
curl -X POST .../ltx/i2v \
  -F "image=@my_photo.jpg" \
  -F "prompt=she walks forward, hair moving in the wind" \
  -F "preset=fast" \
  -F "aspect_ratio=9:16" \
  -F "width=544" \
  -F "length=121" \
  -F "fps=24" \
  -F "seed=-1"

# 9:16 action prompt — needs lower inplace_strength so the subject can actually move
curl -X POST .../ltx/i2v \
  -F "image=@my_photo.jpg" \
  -F "prompt=she runs across the frame" \
  -F "preset=fast" \
  -F "aspect_ratio=9:16" \
  -F "width=544" \
  -F "length=121" \
  -F "inplace_strength=0.45" \
  -F "enhance_prompt=true"

# 9:16 final-quality (two-pass with upscale, ~40-60s)
curl -X POST .../ltx/i2v \
  -F "image=@my_photo.jpg" \
  -F "prompt=camera slowly orbits around her, cinematic lighting" \
  -F "preset=quality" \
  -F "aspect_ratio=9:16" \
  -F "width=720" \
  -F "length=121"

# Subtle motion (default inplace_strength is fine for this)
curl -X POST .../ltx/i2v \
  -F "image=@my_photo.jpg" \
  -F "prompt=subtle head turn, eyes blink" \
  -F "preset=fast" \
  -F "aspect_ratio=9:16" \
  -F "width=544" \
  -F "length=49" \
  -F "seed=42"

# 16:9 with audio
curl -X POST .../ltx/i2v \
  -F "image=@my_photo.jpg" \
  -F "prompt=camera slowly zooms in, birds chirping" \
  -F "preset=quality" \
  -F "audio=true" \
  -F "aspect_ratio=16:9" \
  -F "width=1280"
```

---

## POST /ltx/motion — Motion Control

Copy the body motion from a reference video onto a separate character image.
The endpoint extracts a DWPose skeleton from the video and uses LTX 2.3
Union-Control IC-LoRA to render the character following that pose. The
reference person's face, clothing, and background are not used as appearance
inputs. Identity lock is always applied: the character image remains the sole
source for the subject's face, apparent gender, age, body proportions, hair,
skin tone, and clothing.

### Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `reference_video` | file | **required** | Video containing the body motion to copy |
| `image` | file | **required** | Character image to animate; a clear full-body image works best |
| `prompt` | string | `""` | Action description; the server combines it with an image-aware description of the uploaded character |
| `negative_prompt` | string | *server default* | Artifacts and traits to avoid |
| `preset` | string | `quality` | `quality` uses a half-resolution 8-step pass, learned 2× upscale, and 3-step full-resolution refine for better eyes, faces, and hands. `fast` keeps the single-stage 8-step path. |
| `aspect_ratio` | string | `9:16` | Output aspect ratio |
| `width` | int | `544` | Requested width; dimensions snap to IC-LoRA-safe multiples of 64 |
| `height` | int | `960` | Used with `aspect_ratio=original`; otherwise the ratio determines it |
| `length` | int | `121` | Fallback frame budget used only when `match_reference_duration=false` |
| `fps` | int | `24` | Accepted for compatibility; motion processing and output use 30 fps |
| `match_reference_duration` | bool | `true` | Match the output to the reference duration instead of forcing a five-second result |
| `max_duration_seconds` | float | `15` | Maximum reference duration to render (1–15 seconds) |
| `seed` | int | `-1` | Random when `-1`; set a value for repeatability |
| `audio` | bool | `false` | Mux the reference video's original audio onto the result |
| `enhance_prompt` | bool | `true` | Accepted for compatibility; image-aware identity enhancement is always enabled for motion control |
| `inplace_strength` | float | `1.0` | Character identity-anchor strength. The server enforces an exact `1.0` lock and derives an image-aware appearance prompt to prevent identity and apparent-gender drift. |
| `motion_strength` | float | `1.0` | DWPose motion-guide strength (0–1) |
| `face_filter` | bool | `true` | Reject a character image matching a blocked identity |
| `require_detectable_face` | bool | `false` | Require a detectable face in the character image |

The endpoint follows the reference duration up to 15 seconds. It generates
long references in GPU-safe four-second segments, shares the boundary frame
between adjacent segments, and joins them into one continuous result. After
every sample, the server runs the required `LTXVCropGuides` node before VAE
decode, so IC-LoRA padding is not exposed as noisy frames. No percentage-based
post-generation trim is applied. Status polling reports overall segment-aware
progress and a whole-job ETA; a full 15-second render on the current 22B GPU
typically takes about 8–12 minutes, while shorter templates complete sooner.
Quality mode also disables reference facial landmarks and raises DWPose
hand/body tracking resolution. The learned upscaled latent carries the original
character identity into the refinement pass without a conflicting second anchor.

### Submit, poll, and download

```bash
# 1. Submit the motion job.
RESPONSE=$(curl -sS -X POST "$POD/ltx/motion" \
  -F "reference_video=@dance-reference.mp4" \
  -F "image=@character.png" \
  -F "prompt=the woman performs the reference dance, full body" \
  -F "aspect_ratio=9:16" \
  -F "width=544" \
  -F "height=960" \
  -F "match_reference_duration=true" \
  -F "max_duration_seconds=15" \
  -F "preset=quality" \
  -F "audio=true" \
  -F "inplace_strength=1.0" \
  -F "motion_strength=1.0")

echo "$RESPONSE" | python3 -m json.tool
POLL_URL=$(echo "$RESPONSE" | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["poll_url"])')

# 2. Poll until completed or failed.
while true; do
  STATUS=$(curl -sS "$POLL_URL")
  STATE=$(echo "$STATUS" | python3 -c \
    'import json,sys; print(json.load(sys.stdin)["status"])')
  echo "$STATE"
  if [ "$STATE" = "completed" ] || [ "$STATE" = "failed" ]; then
    echo "$STATUS" | python3 -m json.tool
    break
  fi
  sleep 5
done

# 3. Download the public URL returned on completion.
VIDEO_URL=$(echo "$STATUS" | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["url"])')
curl -L "$VIDEO_URL" -o motion-result.mp4
```

Queued submissions return `job_id`, `status`, `poll_url`, and `cancel_url`.
Completed status responses include `url`, `filename`, `thumbnail_url`, and
generation `duration_seconds`. Cancel an active job with
`DELETE /jobs/{job_id}/cancel`.

---

## POST /ltx/t2v — Text to Video

Generate a video from a text prompt using LTX 2.3. No input image required.

### Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | **required** | Description of the video to generate |
| `negative_prompt` | string | *see below* | What to avoid in the output |
| `preset` | string | `fast` | `fast` (8 steps single-pass) or `quality` (8+3 steps two-pass with 2× spatial upscale). Both ~12s on warm GPU at 544×960. |
| `audio` | bool | `false` | Generate audio track (`true` adds ~5-10s overhead) |
| `aspect_ratio` | string | `16:9` | Output aspect ratio |
| `width` | int | `1280` | Output width in pixels |
| `height` | int | `720` | Output height in pixels |
| `length` | int | `121` | Number of frames (121 = ~5 sec at 24fps) |
| `fps` | int | `24` | Frames per second |
| `seed` | int | `-1` (random) | Set for reproducible results |
| `watermark` | string \| null | `null` | See [Watermarks](#watermarks). Video re-encodes via ffmpeg (~1-3s for a 5s clip). |
| `watermark_image` | bool | `false` | See [Watermarks](#watermarks). |

### Examples

```bash
# Fast text-to-video
curl -X POST .../ltx/t2v \
  -F "prompt=a golden retriever running through a meadow"

# Quality with audio, cinematic widescreen
curl -X POST .../ltx/t2v \
  -F "prompt=waves crashing on rocks, cinematic slow motion" \
  -F "preset=quality" \
  -F "audio=true" \
  -F "aspect_ratio=21:9"

# Vertical for social media
curl -X POST .../ltx/t2v \
  -F "prompt=a person dancing in a neon-lit room" \
  -F "aspect_ratio=9:16" \
  -F "length=121"
```

---

## POST /face-animate — Face Swap + Animate (Pipeline)

Two-step pipeline: replaces the head/face in a template image (FLUX.2 Klein 9B), then animates the result into a video (LTX 2.3).

### Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target_image` | file | **required** | Template/body photo — head gets replaced |
| `face_image` | file | **required** | User's face photo — identity to transfer |
| `animate_prompt` | string | **required** | Describes the motion/scene for the video |
| `swap_prompt` | string | `""` | Prompt for the face swap step (uses smart default if empty) |
| `negative_prompt` | string | *see below* | What to avoid in the video |
| `preset` | string | `fast` | `fast` (8 steps single-pass) or `quality` (8+3 steps two-pass with 2× spatial upscale) |
| `audio` | bool | `false` | Generate audio track with the video |
| `aspect_ratio` | string | `16:9` | Output video aspect ratio |
| `width` | int | `1280` | Output width in pixels |
| `height` | int | `720` | Output height |
| `length_seconds` | float | `5.0` | Video duration in seconds |
| `fps` | int | `24` | Frames per second |
| `seed` | int | `-1` (random) | Set for reproducible results |
| `megapixels` | float | `2.0` | Face swap resolution in megapixels (0.5–4.0) |
| `lora_strength` | float | `1.0` | BFS LoRA strength for face swap (0.5–1.0) |
| `swap_steps` | int | `4` | Face swap inference steps |
| `swap_guidance` | float | `4.0` | Face swap guidance strength |
| `watermark` | string \| null | `null` | See [Watermarks](#watermarks). Applied to the final video. |
| `watermark_image` | bool | `false` | See [Watermarks](#watermarks). Applied to the final video. |

### How It Works

```
face_image + target_image
        |
  [Step 1] FLUX.2 Klein 9B face swap
        |
  swapped image
        |
  [Step 2] LTX 2.3 image-to-video animation
        |
  output video (.mp4)
```

Poll `/status/{job_id}` — the `step` field shows current progress: `face_swap` or `animating`.

### Examples

```bash
# Fast face-animate
curl -X POST .../face-animate \
  -F "target_image=@template_body.jpg" \
  -F "face_image=@user_face.jpg" \
  -F "animate_prompt=the person smiles and looks at the camera"

# Quality with audio, Instagram Reel
curl -X POST .../face-animate \
  -F "target_image=@template_body.jpg" \
  -F "face_image=@user_face.jpg" \
  -F "animate_prompt=person walks confidently forward" \
  -F "preset=quality" \
  -F "audio=true" \
  -F "aspect_ratio=9:16" \
  -F "length_seconds=8"
```

---

## GET /status/{job_id} — Poll Job Status

```bash
curl https://YOUR_POD_ID-7860.proxy.runpod.net/status/{job_id}
```

### Possible responses

**Queued**
```json
{ "status": "queued", "created_at": "2026-03-28T06:00:00Z" }
```

**Processing**
```json
{ "status": "processing", "created_at": "...", "started_at": "..." }
```

**Completed (video)**
```json
{
  "status": "completed",
  "url": "https://YOUR_POD_ID-7860.proxy.runpod.net/video/ltx_t2v_42_00001_.mp4",
  "thumbnail_url": "https://YOUR_POD_ID-7860.proxy.runpod.net/image/ltx_t2v_42_00001__thumb.jpg",
  "filename": "ltx_t2v_42_00001_.mp4",
  "completed_at": "2026-03-28T06:00:40Z",
  "duration_seconds": 36.1
}
```

> `thumbnail_url` is present only on video outputs (`/ltx/i2v`, `/ltx/t2v`,
> `/ltx/motion`, `/face-animate`, and future video workflows). It's the first
> frame as a JPG, served from the same domain.
> If a watermark was applied, the thumbnail reflects it.

**Completed (image)**
```json
{
  "status": "completed",
  "url": "https://YOUR_POD_ID-7860.proxy.runpod.net/image/t2i_42_00001_.png",
  "filename": "t2i_42_00001_.png",
  "completed_at": "2026-03-28T06:00:40Z",
  "duration_seconds": 1.2
}
```

**Failed**
```json
{ "status": "failed", "error": "error message here", "failed_at": "..." }
```

---

## Job Management

### GET /jobs — List all jobs

```bash
curl .../jobs
```

### GET /queue — Active queue

```bash
curl .../queue
```

### POST /jobs/{job_id}/retry — Retry failed job

```bash
curl -X POST .../jobs/{job_id}/retry
```

### DELETE /jobs/{job_id} — Delete job + output file

```bash
curl -X DELETE .../jobs/{job_id}
```

### DELETE /jobs — Bulk delete completed jobs

```bash
curl -X DELETE .../jobs
# Delete all (including queued/processing):
curl -X DELETE ".../jobs?completed_only=false"
```

---

## File Access

### GET /image/{filename}

```bash
curl -O .../image/flux_swap_42_00001_.png
```

### GET /video/{filename}

```bash
curl -O .../video/ltx_t2v_42_00001_.mp4
```

### GET /videos — List all videos

```bash
curl .../videos
```

---

## Typical Generation Times (warm model)

| Operation | fast preset | quality preset |
|-----------|------------|----------------|
| Text to image (1024×1024) | ~10-15s | N/A |
| Face swap (2MP) | ~20-30s | N/A |
| T2V / I2V (544×960, 4s) | ~12s | ~12s |
| T2V / I2V (544×960, 5s) | ~14s | ~14s |
| T2V / I2V (768×1344, 5s) | ~22s | ~24s |
| Motion control (544×960, length 121) | ~5–6 min | Fixed IC-LoRA workflow |
| Face animate (544×960, 4s) | ~35s (swap + video) | ~35s |

> **Both presets land at roughly the same wall time** because `quality` does most of its work at half-resolution (8 steps at ~128k pixels → 2× spatial upscale → 3 refine steps at full-res). Use `quality` when you want sharper detail at no real speed cost; use `fast` for the simpler single-pass pipeline.

> Cold start (first job after pod start) takes ~30-60s extra to load ~50 GB of models into VRAM. All subsequent jobs use cached models.

> Adding `audio=true` adds ~5-10s overhead to any video generation.

---

## Polling Pattern

```bash
# 1. Submit
JOB_ID=$(curl -s -X POST .../ltx/t2v \
  -F "prompt=a sunset over the ocean" \
  -F "preset=fast" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['job_id'])")

echo "Job: $JOB_ID"

# 2. Poll every 5 seconds
while true; do
  STATUS=$(curl -s .../status/$JOB_ID)
  STATE=$(echo $STATUS | python3 -c "import json,sys; print(json.load(sys.stdin)['status'])")
  echo "Status: $STATE"
  if [ "$STATE" = "completed" ] || [ "$STATE" = "failed" ]; then
    echo $STATUS | python3 -m json.tool
    break
  fi
  sleep 5
done
```

---

## Compliance Filters

Face-presence validation is opt-in and separate from identity blocking. Set
`require_detectable_face=true` on `/flux/face-swap`, `/flux/i2i`, `/ltx/i2v`,
`/ltx/motion`, or `/face-animate` to reject user images that contain no
significant detectable face. It defaults to `false` for backward compatibility.

`/flux/face-swap` and `/flux/i2i` accept two **independent** compliance toggles:

| Parameter | Default | Detector | Blocklist dir | What it catches |
|---|---|---|---|---|
| `face_filter` | `false` | InsightFace `buffalo_l` (face recognition) | `/workspace/blocklist/` | Specific human faces (politicians, celebrities, banned individuals) |
| `logo_filter` | `false` | CLIP ViT-B/32 (whole-image semantic) | `/workspace/blocklist_logos/` | Logos, flags, symbols, propaganda imagery — anything that's the **main subject** of the input |

Set either or both to `true` per request. They run in sequence; the first
blocked input fails the whole request with `400`.

### Limits

- **Face filter** is precise (~99% recall on clear faces above threshold).
- **Logo filter** is whole-image — it catches "this image is mostly the Apple logo"
  but **may miss tiny logos in corners** of larger photos. For tight detection
  of small logos, that's a v2 feature (SIFT keypoint matching).

### Face Filter

### Request

```bash
curl -X POST "$POD/flux/i2i" \
  -F "prompt=stylize as a watercolor" \
  -F "images=@person.png" \
  -F "face_filter=true"
```

### Block response

```json
{
  "detail": {
    "error": "blocked",
    "reason": "images[0] matches blocked identity",
    "matched_identity": "tom_hanks",
    "score": 0.87,
    "image_index": 0
  }
}
```

`score` is cosine similarity vs the closest Blocked Faces reference. The
default threshold is `0.68`, deliberately precision-first so weak resemblance
to a listed person is allowed. Override it via `FACE_FILTER_THRESHOLD` only
after testing against labelled same-person and different-person examples.

### Logo / flag filter

```bash
curl -X POST "$POD/flux/i2i" \
  -F "prompt=stylize" \
  -F "images=@input.png" \
  -F "logo_filter=true"
```

Block response:
```json
{
  "detail": {
    "error": "blocked",
    "filter": "logo",
    "reason": "images[0] matches blocked logo/flag",
    "matched_logo": "apple_logo",
    "score": 0.91,
    "image_index": 0
  }
}
```

Threshold defaults to `0.85` (override via `LOGO_FILTER_THRESHOLD` env var).

### Both filters at once

```bash
curl -X POST "$POD/flux/face-swap" \
  -F "target_image=@body.png" -F "face_image=@face.png" \
  -F "face_filter=true" \
  -F "logo_filter=true"
```

The response's `filter` field (`"face"` or `"logo"`) tells you which check
fired. Face filter runs first.

### Bypass audit

Every `face_filter=false` and `logo_filter=false` call is appended to
`/workspace/face_filter_bypass.log` with timestamp, endpoint, job_id, and
which filter was bypassed.

---

## Admin API (blocklist management)

Two parallel sets of admin endpoints — one for faces, one for logos/flags.
Same shape, same hot-reload, same optional auth.

### Auth

Auth is **optional and off by default** so the admin API is easy to access
during development. Behavior is controlled by the `ADMIN_TOKEN` env var:

- **`ADMIN_TOKEN` unset (default):** admin endpoints are open — no auth
  required. Anyone with the pod URL can manage the blocklist. Fine for
  dev / private pods.
- **`ADMIN_TOKEN` set:** every admin call must include
  `Authorization: Bearer <token>`. `401` without header, `403` with wrong
  token. Recommended before going to production / sharing the pod URL.

Switch between modes by setting/unsetting the env var on the RunPod template
and restarting the pod — no code change needed.

### Faces — `/admin/blocklist`

The blocklist is stored on the network volume at `/workspace/blocklist/`,
one image per identity. It's shared with serverless workers (mounted at
`/runpod-volume/blocklist/`) and hot-reloaded on every face-filter check —
uploads and deletes take effect on the next request.

> The examples below show the open-mode (no `ADMIN_TOKEN`). If you set
> `ADMIN_TOKEN` on the pod, add `-H "Authorization: Bearer $ADMIN_TOKEN"`
> to every call.

### POST /admin/blocklist — Upload a face

```bash
curl -X POST "$POD/admin/blocklist" \
  -F "identity=tom_hanks" \
  -F "image=@hanks.png" \
  -F "overwrite=false"
```

**Accepted input:** PNG / JPG / JPEG / WEBP (anything Pillow can decode).
EXIF orientation is honored, so phone photos rotate correctly.

**Auto-normalize on upload:** every accepted image is downscaled so the
longer edge is ≤ `BLOCKLIST_MAX_EDGE` (default `1024` px, env-overridable)
and re-encoded as PNG before storage. Callers do **not** need to resize or
re-format client-side — upload the raw photo. The stored filename is
always `<identity>.png` regardless of input format. Identity match accuracy
is unaffected: InsightFace recognizes at 112x112 internally, so anything
above ~256 px on the face is identical to the full-resolution input.

Validation runs on the normalized image: the upload must contain **exactly
one detectable face**. Identity must match `[A-Za-z0-9_-]{1,64}` — no
spaces or path separators.

Returns:
```json
{
  "status": "added",       // or "replaced" if overwrite=true and existed
  "identity": "tom_hanks",
  "filename": "tom_hanks.png",
  "size_bytes": 87012,      // post-normalize PNG size, not the upload size
  "blocklist_count": 12
}
```

Errors:
- `400` — undecodable image, no face detected, multiple faces, or invalid identity name
- `409` — identity already exists (use `overwrite=true` to replace)
- `503` — face filter / image normalizer unavailable (e.g. `safety` module or Pillow not installed)

### GET /admin/blocklist — List

```bash
curl "$POD/admin/blocklist"
```

```json
{
  "count": 2,
  "blocklist": [
    {"identity": "tom_hanks", "filename": "tom_hanks.png",
     "size_bytes": 87012, "added_at": "2026-05-14T07:30:00+00:00"},
    {"identity": "celebrity_42", "filename": "celebrity_42.png",
     "size_bytes": 102488, "added_at": "2026-05-14T07:35:00+00:00"}
  ]
}
```

> Existing entries uploaded before the auto-normalize change may still have
> `.jpg` / `.jpeg` / `.webp` extensions — those keep working, but any new
> upload (or `overwrite=true` replace) re-saves as `.png`.

### DELETE /admin/blocklist/{identity} — Remove

```bash
curl -X DELETE "$POD/admin/blocklist/tom_hanks" \
 
```

```json
{"status": "deleted", "identity": "tom_hanks",
 "filename": "tom_hanks.png", "blocklist_count": 1}
```

### GET /admin/blocklist/{identity}/image — Preview

Returns the stored face image as raw bytes (for CMS preview).

```bash
curl "$POD/admin/blocklist/tom_hanks/image" \
  -o tom_hanks.png
```

### Logos / flags — `/admin/blocklist-logos`

Same shape as the face endpoints, different storage (`/workspace/blocklist_logos/`)
and different validation (no face-detection prereq — any valid image is accepted).

```bash
# Upload
curl -X POST "$POD/admin/blocklist-logos" \
  -F "identity=apple_logo" \
  -F "image=@apple.png"

# List
curl "$POD/admin/blocklist-logos"

# Delete
curl -X DELETE "$POD/admin/blocklist-logos/apple_logo" \
 

# Preview
curl "$POD/admin/blocklist-logos/apple_logo/image" \
  -o apple.png
```

Response shape mirrors `/admin/blocklist`. The list response groups blocked
logos by their `identity` (filename stem) — the same name returned in
`matched_logo` on a block.

**Tip:** tight crops give best CLIP discrimination. A blocklist image that
fills the frame with the logo/flag scores ~0.9+ against itself; if the
logo is small in the corner of your blocklist image, CLIP will embed the
background's content instead and miss real-world matches.

---

## Error Responses

All errors return standard HTTP status codes with a JSON body.

```json
{ "detail": "error message here" }
```

| Code | Meaning |
|------|---------|
| `400` | Bad request (missing field, invalid preset/aspect_ratio, etc.) |
| `404` | Job or file not found |
| `422` | Validation error (wrong type for a parameter) |
| `500` | Internal server error |
