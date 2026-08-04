import asyncio
import re
import subprocess
import uuid, httpx, os
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
try:
    from image_output import optimize_image_file
except ImportError:
    optimize_image_file = None

from workflows import (
    ASPECT_RATIOS,
    LTX_ASPECT_RATIOS,
    LTX_DEFAULT_NEGATIVE,
    LTX_PRESETS,
    MOTION_CHUNK_FRAMES,
    MOTION_FPS,
    MOTION_MAX_DURATION_SECONDS,
    MULTI_FACE_SWAP_ORDERS,
    build_flux_i2i_workflow,
    build_flux_multi_face_swap_workflow,
    build_t2i_workflow,
    build_ltx_i2v_workflow,
    build_ltx_lipdub_workflow,
    build_ltx_motion_workflow,
    build_ltx_motion_workflow_no_vhs,
    build_ltx_t2v_workflow,
    compute_dimensions,
    compute_ltx_dimensions,
    crop_to_aspect,
    duration_to_ltx_frames,
    get_flux_face_swap_workflow,
    ltx_base_nodes,
    normalize_target_face_indices,
    preserve_selected_faces,
    snap_ltx_frame_count,
    split_ltx_frame_count,
)

# Compliance face filter (loaded lazily on first face_filter=true request).
# Module exists even if insightface is uninstalled — it'll raise a clear
# RuntimeError when actually invoked, never at import time.
try:
    import safety as face_safety
except ImportError:
    face_safety = None

# Compliance logo/flag filter — CLIP-based, separate blocklist dir.
try:
    import logo_safety
except ImportError:
    logo_safety = None

# Optional output watermark — defaults to off; callers pass `watermark="AI"`
# (or any short string) to overlay it on the result.
try:
    import watermark
except ImportError:
    watermark = None

app = FastAPI(title="AI Gen API v2")

API_VERSION = "2.3.1"

# Open CORS so browser-based admin UIs (super-cms-vn /ai-pods + /blocked-faces)
# can call /admin/blocklist directly across the multi-pod registry. We
# previously routed everything through the face-swap-proxy edge function,
# but the proxy was timing out for admin paths and routing through it adds
# a hop for what's already an admin-only operation. With CORS on the pod
# itself, the CMS can fan out to every registered pod URL in parallel.
#
# allow_origins=["*"] is acceptable because:
#   - /admin/* endpoints will require ADMIN_API_TOKEN when set (future)
#   - All other endpoints are already meant to be reachable from app browsers
#   - RunPod's pod hostnames aren't truly secret but aren't published either
# If we add auth later we can lock origins down.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

COMFYUI_URL = "http://127.0.0.1:8188"
POD_ID = os.environ.get("RUNPOD_POD_ID", "RUNPOD_POD_ID_PLACEHOLDER")
BASE_URL = f"https://{POD_ID}-7860.proxy.runpod.net"
COMFY_JOB_TIMEOUT_SECONDS = int(os.environ.get("COMFY_JOB_TIMEOUT_SECONDS", "900"))

# Allow local/staging smoke tests to point at a writable ComfyUI tree while
# preserving the existing RunPod auto-detection and production default.
_configured_comfy_root = os.environ.get("COMFY_ROOT", "").strip()
COMFY_ROOT = Path(_configured_comfy_root) if _configured_comfy_root else None
if COMFY_ROOT is None:
    for _p in ["/workspace/runpod-slim/ComfyUI", "/workspace/ComfyUI"]:
        if Path(_p).exists():
            COMFY_ROOT = Path(_p)
            break
if COMFY_ROOT is None:
    COMFY_ROOT = Path("/workspace/ComfyUI")

OUTPUT_DIR = COMFY_ROOT / "output"
INPUT_DIR = COMFY_ROOT / "input"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional background-music beds muxed under videos when background_music=true.
# BGM_DIR holds a pool of tracks (one is picked at random per request); BGM_PATH
# is a single-file fallback. Both live on the network volume (fetched by setup.sh).
BGM_DIR = Path("/workspace/assets/bgm")
BGM_PATH = Path("/workspace/assets/horoscope_bgm.m4a")
_BGM_EXTS = {".mp3", ".m4a", ".aac", ".wav", ".ogg", ".opus", ".flac"}


def _pick_background_track():
    """Pick a random track from BGM_DIR (falling back to the single BGM_PATH).
    Returns a Path or None. Uses `secrets` (OS entropy) NOT the global `random`
    module — the generation pipeline seeds `random` for reproducible renders,
    which would otherwise make this pick deterministic (always the same track)."""
    try:
        import secrets
        if BGM_DIR.is_dir():
            tracks = sorted(p for p in BGM_DIR.iterdir() if p.suffix.lower() in _BGM_EXTS and p.is_file())
            if tracks:
                return secrets.choice(tracks)
    except Exception:
        pass
    return BGM_PATH if BGM_PATH.exists() else None

# In-memory job store
jobs = {}

# Dedicated video pods should execute one heavyweight LTX workflow at a time.
# Setting this to 0 keeps the historical unlimited-queue behavior. The CMS
# load balancer normally diverts excess work before it reaches the pod, while
# this pod-side guard closes the small race between simultaneous submissions.
MAX_ACTIVE_VIDEO_JOBS = max(0, int(os.environ.get("MAX_ACTIVE_VIDEO_JOBS", "0")))
AI_GEN_ROLE = os.environ.get("AI_GEN_ROLE", "general").strip().lower() or "general"
VIDEO_PROGRESS_ESTIMATE_SECONDS = max(
    20,
    int(os.environ.get("VIDEO_PROGRESS_ESTIMATE_SECONDS", "55")),
)


def _active_video_job_count() -> int:
    return sum(
        1
        for job in jobs.values()
        if job.get("workload") == "video" and job.get("status") in {"queued", "processing"}
    )


def _reserve_video_job(job_id: str, cleanup_paths: list[str] | None = None) -> None:
    active = _active_video_job_count()
    if MAX_ACTIVE_VIDEO_JOBS and active >= MAX_ACTIVE_VIDEO_JOBS:
        for path in cleanup_paths or []:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass
        raise HTTPException(
            429,
            detail={
                "error": "video_capacity_reached",
                "message": "Dedicated video pod is at its concurrency limit",
                "active_jobs": active,
                "max_concurrency": MAX_ACTIVE_VIDEO_JOBS,
                "retry_backend": "serverless",
            },
        )
    jobs[job_id] = {
        "status": "queued",
        "workload": "video",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "progress": 5,
        "stage": "queued",
        "progress_message": "Waiting for the dedicated video GPU...",
        "eta_seconds": VIDEO_PROGRESS_ESTIMATE_SECONDS,
    }


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".gif"}


def _extract_video_thumbnail(video_path: Path) -> Path | None:
    """Grab the first frame of `video_path` as a JPG sibling under
    OUTPUT_DIR/images/. Returns the saved thumbnail path, or None on failure.

    Always uses the very first frame — predictable for /ltx/i2v (the input
    image) and fast (~50 ms with libx264-decoded mp4). If you need a
    cinematic mid-frame later, add an `-ss` offset.

    Saved name: `{video_stem}_thumb.jpg`. Lands under OUTPUT_DIR/images/ so
    the existing GET /image/{filename} handler picks it up without route
    changes.
    """
    if not video_path.exists():
        return None
    images_dir = OUTPUT_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    thumb = images_dir / f"{video_path.stem}_thumb.jpg"
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(video_path),
                "-frames:v", "1",
                "-q:v", "3",          # 1=best, 31=worst — 3 ≈ visually lossless JPG
                "-update", "1",
                str(thumb),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0 and thumb.exists() and thumb.stat().st_size > 0:
            return thumb
        # ffmpeg succeeded but produced nothing — log enough to debug without
        # spamming on every "no video" job.
        print(
            f"[thumbnail] ffmpeg rc={proc.returncode} for {video_path.name}: "
            f"{(proc.stderr or '').strip()[-200:]}"
        )
    except Exception as exc:  # noqa: BLE001 — never crash a completed job
        print(f"[thumbnail] exception on {video_path.name}: {exc}")
    return None


# ─────────────────────────────────────────────
# Core job runner
# ─────────────────────────────────────────────

def _mux_background_music(video_path: Path, music_path: Path) -> tuple[bool, str]:
    """Mux a looping background-music bed under `video_path` (which has no audio).
    The track is looped to cover the clip, trimmed to its length, with a short
    fade in/out so the video's loop seam isn't a hard click. Video is stream-copied
    (no re-encode → fast). Returns (changed, message); never raises."""
    import subprocess, json as _json
    if not music_path.exists():
        return False, "music track missing"
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(video_path)],
            capture_output=True, text=True, timeout=15,
        )
        dur = float(_json.loads(probe.stdout)["format"]["duration"])
    except Exception:
        dur = 5.0
    fade_st = max(0.0, dur - 0.6)
    tmp = video_path.with_suffix(".bgm.mp4")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path), "-stream_loop", "-1", "-i", str(music_path),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-af", f"afade=t=in:st=0:d=0.2,afade=t=out:st={fade_st:.2f}:d=0.6",
        "-shortest", "-movflags", "+faststart", str(tmp),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception as e:
        return False, f"ffmpeg error: {e}"
    if r.returncode != 0 or not tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass
        return False, (r.stderr or "ffmpeg failed")[:200]
    try:
        tmp.replace(video_path)
    except Exception as e:
        return False, f"replace failed: {e}"
    return True, f"music muxed ({dur:.2f}s)"


def _mux_reference_audio(video_path: Path, audio_source: Path) -> tuple[bool, str]:
    """Replace the audio track on `video_path` with the audio from
    `audio_source`. Returns (changed, message). `changed` is True only if
    the file on disk was actually rewritten with audio.

    Kling-style behavior: if the source audio is shorter than the output
    video, loop it (-stream_loop -1). If longer, trim to the video's
    duration (-shortest). If the source has no audio stream at all, this
    is a no-op (caller's `audio=true` is silently honored as best-effort).
    """
    import subprocess
    if not audio_source.exists():
        return False, "audio source missing"

    # Probe — ffprobe is faster than letting ffmpeg discover mid-encode.
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0",
             str(audio_source)],
            capture_output=True, timeout=15,
        )
    except Exception as e:
        return False, f"ffprobe error: {e}"
    if probe.returncode != 0 or b"audio" not in probe.stdout:
        return False, "reference has no audio track"

    # Write to a sibling temp then atomic replace — never half-mutate the
    # output file while the URL is already advertised. The temp file MUST
    # preserve the original extension: ffmpeg infers the container from
    # the path extension and "<name>.mp4.muxing" gave it nothing to read,
    # failing with "Error initializing the muxer ... Invalid argument".
    # "<stem>.tmp<suffix>" keeps the .mp4/.webm visible to autodetect.
    tmp_out = video_path.with_name(f"{video_path.stem}.tmp{video_path.suffix}")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-stream_loop", "-1", "-i", str(audio_source),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(tmp_out),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=120)
    except Exception as e:
        tmp_out.unlink(missing_ok=True)
        return False, f"ffmpeg error: {e}"
    if res.returncode != 0 or not tmp_out.exists() or tmp_out.stat().st_size == 0:
        tmp_out.unlink(missing_ok=True)
        err = (res.stderr or b"").decode(errors="replace")[-300:]
        return False, f"ffmpeg failed: {err}"
    os.replace(str(tmp_out), str(video_path))
    return True, "ok"


class JobCancelled(Exception):
    """Internal signal used when a user cancels an active API job."""


async def _wait_for_comfy_prompt(
    prompt_id: str,
    *,
    job_id: str | None = None,
    timeout_seconds: int = COMFY_JOB_TIMEOUT_SECONDS,
) -> dict:
    """Poll ComfyUI history until a prompt completes or fails.

    Polling avoids the websocket race where a very fast prompt can finish
    before the listener connects, leaving the API waiting forever.
    """

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    started_monotonic = loop.time()

    async with httpx.AsyncClient(timeout=30.0) as client:
        while loop.time() < deadline:
            if job_id and jobs.get(job_id, {}).get("status") == "cancelled":
                raise JobCancelled(job_id)

            if job_id and jobs.get(job_id, {}).get("workload") == "video":
                elapsed = max(0, int(loop.time() - started_monotonic))
                if elapsed < 10:
                    stage = "loading_models"
                    message = "Loading the video model..."
                elif elapsed < max(35, VIDEO_PROGRESS_ESTIMATE_SECONDS - 12):
                    stage = "sampling"
                    message = "Rendering video frames..."
                else:
                    stage = "decoding"
                    message = "Decoding the rendered frames..."
                progress = min(
                    90,
                    15 + int((elapsed / VIDEO_PROGRESS_ESTIMATE_SECONDS) * 75),
                )
                jobs[job_id] = {
                    **jobs[job_id],
                    "progress": progress,
                    "stage": stage,
                    "progress_message": message,
                    "eta_seconds": max(0, VIDEO_PROGRESS_ESTIMATE_SECONDS - elapsed),
                }

            response = await client.get(f"{COMFYUI_URL}/history/{prompt_id}")
            response.raise_for_status()
            job_data = response.json().get(prompt_id, {})
            status = job_data.get("status", {})
            if status.get("completed") or status.get("status_str") in {"success", "error"}:
                return job_data
            await asyncio.sleep(0.5)

    raise TimeoutError(
        f"ComfyUI prompt {prompt_id} did not finish within {timeout_seconds} seconds"
    )


async def _comfy_run_get_image(workflow: dict):
    """Submit a workflow to ComfyUI, wait for completion, return the first output
    image Path (or None). Self-contained + never raises — used by the optional
    face-refine pass so a failure there can't break the base job."""
    try:
        client_id = str(uuid.uuid4())
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow, "client_id": client_id})
            if resp.status_code != 200:
                print(f"[face-refine] comfy submit failed: {resp.text[:200]}")
                return None
            prompt_id = resp.json()["prompt_id"]
        job_data = await _wait_for_comfy_prompt(prompt_id)
        if job_data.get("status", {}).get("status_str") == "error":
            print(f"[face-refine] comfy reported execution error")
            return None
        outputs = job_data.get("outputs", {})
        for node_output in outputs.values():
            if "images" in node_output and node_output["images"]:
                item = node_output["images"][0]
                sub = item.get("subfolder", "")
                p = OUTPUT_DIR / sub / item["filename"] if sub else OUTPUT_DIR / item["filename"]
                if p.exists():
                    return p
        return None
    except Exception as e:
        print(f"[face-refine] comfy run error: {e}")
        return None


async def _refine_face_inplace(image_path, face_filename: str, *, job_id: str,
                               megapixels: float = 2.0, steps: int = 8, cfg: float = 1.0,
                               guidance: float = 4.0, lora_strength: float = 1.0) -> bool:
    """Second-pass face detailer. Detect the largest face in `image_path`, crop a
    padded region around it, re-run the FLUX head-swap on JUST that crop at high
    resolution (so the face is rendered with far more pixels than its small
    in-frame size allowed), then composite the sharp face back with a feathered
    mask. Overwrites `image_path` on success.

    Fail-open by design: any problem (no face, comfy error, etc.) leaves the
    original swap untouched and returns False — the base result still ships."""
    from pathlib import Path as _Path
    image_path = _Path(image_path)
    try:
        from PIL import Image, ImageDraw, ImageFilter
        if face_safety is None or not hasattr(face_safety, "get_largest_face_bbox"):
            print(f"[{job_id}] face-refine: detector unavailable, skipping")
            return False
        bbox = face_safety.get_largest_face_bbox(image_path.read_bytes())
        if not bbox:
            print(f"[{job_id}] face-refine: no face detected, skipping")
            return False
        img = Image.open(image_path).convert("RGB")
        W, H = img.size
        x1, y1, x2, y2 = bbox
        fw, fh = max(1, x2 - x1), max(1, y2 - y1)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        # Pad generously so the crop includes hair/jaw/neck context for a clean swap.
        side = max(fw, fh) * 2.2
        half = side / 2.0
        cx1, cy1 = max(0, int(cx - half)), max(0, int(cy - half))
        cx2, cy2 = min(W, int(cx + half)), min(H, int(cy + half))
        crop = img.crop((cx1, cy1, cx2, cy2))
        crop_w, crop_h = crop.size
        if crop_w < 24 or crop_h < 24:
            return False
        crop_name = f"refine_crop_{uuid.uuid4().hex}.png"
        crop_path = INPUT_DIR / crop_name
        crop.save(crop_path)
        # Re-swap the crop. megapixels>=1.5 means ImageScaleToTotalPixels renders the
        # (now face-dominated) crop at ~1.5-2 MP → a high-detail face.
        refine_seed = uuid.uuid4().int % 2**32
        wf = get_flux_face_swap_workflow(
            crop_name, face_filename, refine_seed,
            megapixels=max(1.5, float(megapixels)), steps=max(4, int(steps)),
            cfg=cfg, guidance=guidance, lora_strength=lora_strength,
        )
        refined_path = await _comfy_run_get_image(wf)
        try:
            crop_path.unlink()
        except Exception:
            pass
        if not refined_path or not refined_path.exists():
            print(f"[{job_id}] face-refine: re-swap produced no output, keeping original")
            return False
        refined = Image.open(refined_path).convert("RGB").resize((crop_w, crop_h), Image.LANCZOS)
        # Feathered ellipse over the face region (crop coords) — blends only the
        # face, leaving the original body/background/seam untouched.
        mask = Image.new("L", (crop_w, crop_h), 0)
        fx1, fy1 = (x1 - cx1), (y1 - cy1)
        fx2, fy2 = (x2 - cx1), (y2 - cy1)
        ex, ey = (fx2 - fx1) * 0.22, (fy2 - fy1) * 0.30
        ImageDraw.Draw(mask).ellipse([fx1 - ex, fy1 - ey, fx2 + ex, fy2 + ey], fill=255)
        feather = max(6, int(side * 0.06))
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
        img.paste(refined, (cx1, cy1), mask)
        img.save(image_path)
        try:
            refined_path.unlink()
        except Exception:
            pass
        print(f"[{job_id}] face-refine: applied (face {fw}x{fh}px re-rendered at ~{max(1.5, float(megapixels))}MP)")
        return True
    except Exception as e:
        print(f"[{job_id}] face-refine failed (keeping original): {e}")
        return False


async def _preserve_body_inplace(image_path, template_path: str, *, job_id: str) -> bool:
    """Make the swap change ONLY the head: composite the swapped head onto the
    ORIGINAL template, so the body, clothing, pose, lighting and background are
    the template's exact pixels (the base swap regenerates the whole frame from
    references, which drifts). Detect the head in the swap, mask it (head + hair,
    feathered at the neck/hairline), and overlay it on the template.

    Fail-open: any problem keeps the full swap output untouched."""
    from pathlib import Path as _Path
    image_path = _Path(image_path)
    try:
        from PIL import Image, ImageDraw, ImageFilter
        if face_safety is None or not hasattr(face_safety, "get_largest_face_bbox"):
            return False
        swap = Image.open(image_path).convert("RGB")
        W, H = swap.size
        tmpl = Image.open(template_path).convert("RGB")
        # The swap output is spatially aligned with the (scaled) template — same
        # composition/pose — so scaling the template to the swap size lines them up.
        if tmpl.size != (W, H):
            tmpl = tmpl.resize((W, H), Image.LANCZOS)
        bbox = face_safety.get_largest_face_bbox(image_path.read_bytes())
        if not bbox:
            print(f"[{job_id}] preserve-body: no face detected, keeping full swap")
            return False
        x1, y1, x2, y2 = bbox
        fw, fh = max(1, x2 - x1), max(1, y2 - y1)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        # Head ellipse: generous so it encloses hair + jaw (and the template's
        # original head, since they're aligned) — up for hair, down to the neck.
        hx1, hx2 = cx - fw * 1.25, cx + fw * 1.25
        hy1, hy2 = cy - fh * 1.65, cy + fh * 1.35
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).ellipse([hx1, hy1, hx2, hy2], fill=255)
        feather = max(8, int(max(fw, fh) * 0.18))
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
        out = tmpl.copy()
        out.paste(swap, (0, 0), mask)
        out.save(image_path)
        print(f"[{job_id}] preserve-body: head composited onto template (body/bg from template)")
        return True
    except Exception as e:
        print(f"[{job_id}] preserve-body failed (keeping full swap): {e}")
        return False


async def _preserve_selected_faces_inplace(
    image_path,
    template_path: str,
    *,
    job_id: str,
    face_order: str,
    target_face_indices: list[int],
) -> bool:
    """Keep every unselected person and all non-head pixels exactly unchanged."""
    if face_safety is None or not hasattr(face_safety, "get_face_bboxes"):
        print(f"[{job_id}] preserve-selected-faces: detector unavailable")
        return False
    try:
        preserved, message = preserve_selected_faces(
            image_path,
            template_path,
            face_order=face_order,
            target_face_indices=target_face_indices,
            detect_face_bboxes=face_safety.get_face_bboxes,
        )
        print(f"[{job_id}] preserve-selected-faces: {message}")
        return preserved
    except Exception as exc:
        print(f"[{job_id}] preserve-selected-faces failed: {exc}")
        return False


async def run_job(job_id: str, workflow: dict, cleanup_paths: list = None,
                  watermark_text: str | None = None,
                  watermark_image: bool = False,
                  audio_source_path: str | None = None,
                  *,
                  output_face_filter: bool = False,
                  output_logo_filter: bool = False,
                  output_endpoint: str = "/unknown",
                  caption: str | None = None,
                  caption_icon: str | None = None,
                  caption_fade: bool = True,
                  background_music: bool = False,
                  refine_face: bool = False,
                  refine_face_filename: str | None = None,
                  refine_megapixels: float = 2.0,
                  refine_steps: int = 8,
                  refine_cfg: float = 1.0,
                  refine_guidance: float = 4.0,
                  refine_lora: float = 1.0,
                  preserve_body: bool = False,
                  preserve_body_template: str | None = None,
                  preserve_face_selection: bool = False,
                  preserve_face_template: str | None = None,
                  preserve_face_order: str = "left-to-right",
                  preserve_face_indices: list[int] | None = None):
    """Generic ComfyUI job runner.

    ── output_face_filter / output_logo_filter ──
    When true, the generated image (NOT video — videos skip output scan
    for now) is run through face_safety.check_image / logo_safety.check_image
    BEFORE watermarking / thumbnail / URL exposure. Catches blocked
    identities or logos that ended up in the OUTPUT despite passing the
    INPUT check — necessary for:
      - /t2i: no input image at all, so input check is meaningless
      - /flux/face-swap: face-swap can introduce a blocklist identity in
        the OUTPUT even when neither input matched (e.g. prompt-only
        identity adjustment in some workflows)
      - /flux/i2i: same — i2i can morph an input face toward a blocked
        identity if the prompt suggests it

    `output_endpoint` is the originating endpoint path; logged for audit
    so admins can see "X /t2i jobs blocked at output stage today".

    A blocked output is deleted from disk and the job is marked failed
    with status="blocked" so the client gets a clear error instead of
    a generic completion."""
    jobs[job_id] = {
        **jobs.get(job_id, {}),
        "status": "processing",
        "workflow": workflow,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "progress": 15 if jobs.get(job_id, {}).get("workload") == "video" else 10,
        "stage": "loading_models" if jobs.get(job_id, {}).get("workload") == "video" else "processing",
        "progress_message": "Loading the video model..." if jobs.get(job_id, {}).get("workload") == "video" else "Processing...",
    }
    try:
        client_id = str(uuid.uuid4())
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow, "client_id": client_id})
            if resp.status_code != 200:
                jobs[job_id] = {**jobs[job_id], "status": "failed", "error": resp.text}
                return
            prompt_id = resp.json()["prompt_id"]
        jobs[job_id] = {**jobs[job_id], "prompt_id": prompt_id}

        job_data = await _wait_for_comfy_prompt(prompt_id, job_id=job_id)
        status = job_data.get("status", {}).get("status_str", "")
        if status == "error":
            messages = job_data.get("status", {}).get("messages", [])
            for m in messages:
                if m[0] == "execution_error":
                    jobs[job_id] = {**jobs[job_id], "status": "failed", "error": m[1].get("exception_message")}
                    return
            jobs[job_id] = {**jobs[job_id], "status": "failed", "error": "ComfyUI execution failed"}
            return
        outputs = job_data.get("outputs", {})

        for node_output in outputs.values():
            for key in ["videos", "gifs", "images"]:
                if key in node_output:
                    item = node_output[key][0]
                    filename = item["filename"]
                    subfolder = item.get("subfolder", "")
                    path = OUTPUT_DIR / subfolder / filename if subfolder else OUTPUT_DIR / filename
                    if path.exists():
                        ext = Path(filename).suffix.lower()
                        is_image_output = ext in [".png", ".jpg", ".jpeg", ".webp"]

                        if ext in _VIDEO_EXTS:
                            jobs[job_id] = {
                                **jobs[job_id],
                                "progress": 93,
                                "stage": "encoding",
                                "progress_message": "Encoding the final video...",
                                "eta_seconds": 5,
                            }

                        # ── OPTIONAL 2nd-PASS FACE REFINE (opt-in) ───
                        # Re-render the (small, soft) swapped face at high res
                        # and composite it back — fixes soft faces in full-body
                        # templates without reframing them. Runs BEFORE the
                        # output filters + watermark/caption so they all act on
                        # the refined image. Fail-open: errors keep the base swap.
                        if refine_face and is_image_output and refine_face_filename:
                            try:
                                await _refine_face_inplace(
                                    path, refine_face_filename, job_id=job_id,
                                    megapixels=refine_megapixels, steps=refine_steps,
                                    cfg=refine_cfg, guidance=refine_guidance, lora_strength=refine_lora,
                                )
                            except Exception as _re:
                                print(f"[{job_id}] face-refine call raised (ignored): {_re}")

                        # ── OPTIONAL HEAD-ONLY: keep template body/bg ───
                        # Composite the (refined) head onto the ORIGINAL template
                        # so only the head changes — body, clothing, pose and
                        # background stay the template's exact pixels. Runs after
                        # refine so the composited head is already sharp.
                        if preserve_body and is_image_output and preserve_body_template:
                            try:
                                await _preserve_body_inplace(path, preserve_body_template, job_id=job_id)
                            except Exception as _pb:
                                print(f"[{job_id}] preserve-body call raised (ignored): {_pb}")

                        # FLUX regenerates the full frame. For group templates,
                        # deliver only explicitly selected heads over the exact
                        # original template pixels.
                        if preserve_face_selection and is_image_output and preserve_face_template:
                            selected_indices = preserve_face_indices or [0]
                            preserved = await _preserve_selected_faces_inplace(
                                path,
                                preserve_face_template,
                                job_id=job_id,
                                face_order=preserve_face_order,
                                target_face_indices=selected_indices,
                            )
                            if not preserved:
                                path.unlink(missing_ok=True)
                                jobs[job_id] = {
                                    **jobs[job_id],
                                    "status": "failed",
                                    "error": "Could not isolate the selected template face. Please try another template.",
                                    "failed_at": datetime.now(timezone.utc).isoformat(),
                                }
                                return

                        # ── OUTPUT-SIDE FACE FILTER ──────────────────
                        # Scan the generated image against the blocklist
                        # BEFORE watermarking / thumbnail / URL exposure.
                        # This is the last line of defense against:
                        #   • /t2i prompts that produce a blocked identity
                        #     (no input image, so no input check possible)
                        #   • face-swap workflows that introduce a blocked
                        #     identity in the output via prompt drift
                        #   • i2i edits that morph an input face toward a
                        #     blocked identity
                        # On block: delete the file (so even a leaked URL
                        # can't fetch it), then mark the job failed with
                        # status="blocked" so the client sees the rejection.
                        # Skipped for videos — output frame extraction +
                        # per-frame scanning is a separate (future) effort.
                        if output_face_filter and is_image_output and face_safety is not None:
                            try:
                                output_bytes = path.read_bytes()
                                face_result = face_safety.check_image(output_bytes)
                                if face_result.blocked:
                                    print(f"[{job_id}] OUTPUT BLOCKED by face filter "
                                          f"(endpoint={output_endpoint}, "
                                          f"identity={face_result.matched_identity}, "
                                          f"score={face_result.score:.4f})")
                                    try:
                                        path.unlink()
                                    except Exception as del_err:
                                        print(f"[{job_id}] could not delete blocked output {path}: {del_err}")
                                    jobs[job_id] = {
                                        **jobs[job_id],
                                        "status": "failed",
                                        "blocked": True,
                                        "error": "blocked",
                                        "filter": "face",
                                        "reason": f"output image matches blocked face identity",
                                        "matched_identity": face_result.matched_identity,
                                        "score": round(face_result.score, 4),
                                        "endpoint": output_endpoint,
                                        "failed_at": datetime.now(timezone.utc).isoformat(),
                                    }
                                    return
                            except RuntimeError as filter_err:
                                # Filter unavailable — be loud about it but
                                # don't block the job. Fail-open is correct
                                # here because failing-closed would brick
                                # ALL generation on a filter init bug.
                                print(f"[{job_id}] WARN: output face filter unavailable: {filter_err}")
                            except Exception as filter_err:
                                print(f"[{job_id}] WARN: output face filter raised: {filter_err}")

                        # ── OUTPUT-SIDE LOGO FILTER ──────────────────
                        # Same idea for the logo/flag blocklist (CLIP-based).
                        # Catches "draw the [forbidden] flag" prompts on /t2i.
                        if output_logo_filter and is_image_output and logo_safety is not None:
                            try:
                                output_bytes = path.read_bytes()
                                logo_result = logo_safety.check_image(output_bytes)
                                if logo_result.blocked:
                                    print(f"[{job_id}] OUTPUT BLOCKED by logo filter "
                                          f"(endpoint={output_endpoint}, "
                                          f"logo={logo_result.matched_logo}, "
                                          f"score={logo_result.score:.4f})")
                                    try:
                                        path.unlink()
                                    except Exception as del_err:
                                        print(f"[{job_id}] could not delete blocked output {path}: {del_err}")
                                    jobs[job_id] = {
                                        **jobs[job_id],
                                        "status": "failed",
                                        "blocked": True,
                                        "error": "blocked",
                                        "filter": "logo",
                                        "reason": f"output image matches blocked logo/flag",
                                        "matched_logo": logo_result.matched_logo,
                                        "score": round(logo_result.score, 4),
                                        "endpoint": output_endpoint,
                                        "failed_at": datetime.now(timezone.utc).isoformat(),
                                    }
                                    return
                            except RuntimeError as filter_err:
                                print(f"[{job_id}] WARN: output logo filter unavailable: {filter_err}")
                            except Exception as filter_err:
                                print(f"[{job_id}] WARN: output logo filter raised: {filter_err}")

                        # Optional reference-audio mux — only used by
                        # /ltx/motion right now. Runs BEFORE watermark so
                        # the (re-encoded) watermark video carries the
                        # muxed audio too. Non-fatal: if mux fails we
                        # still return the silent video.
                        audio_mux_warning: str | None = None
                        if audio_source_path and ext in _VIDEO_EXTS:
                            try:
                                ok, msg = await asyncio.to_thread(
                                    _mux_reference_audio, path, Path(audio_source_path),
                                )
                                if not ok:
                                    audio_mux_warning = msg
                                    print(f"[{job_id}] audio mux skipped: {msg}")
                            except Exception as mux_err:
                                audio_mux_warning = str(mux_err)
                                print(f"[{job_id}] audio mux raised: {mux_err}")

                        # Optional watermark — text and/or logo. Both run in
                        # place. Failures don't nuke the job; the
                        # unwatermarked file is still valid output.
                        wm_warnings: list[str] = []
                        if watermark_text and watermark is not None:
                            try:
                                watermark.apply(path, watermark_text)
                            except Exception as wm_err:
                                wm_warnings.append(f"text: {wm_err}")
                        if watermark_image and watermark is not None:
                            try:
                                watermark.apply_logo(path)
                            except Exception as wm_err:
                                wm_warnings.append(f"image: {wm_err}")
                        # Optional styled caption (e.g. horoscope-of-the-day).
                        # Applied last so it sits above the logo. Body-only
                        # lower-third design; videos fade it in. Failure is
                        # non-fatal — the un-captioned file is still valid.
                        if caption and watermark is not None:
                            try:
                                watermark.apply_caption(path, caption, fade_in=caption_fade, icon_sign=caption_icon)
                            except Exception as wm_err:
                                wm_warnings.append(f"caption: {wm_err}")
                        # Optional background-music bed (video outputs only) — a
                        # random track from the pool. Stream-copies the video
                        # (no re-encode) + adds the looped/trimmed audio track.
                        if background_music and ext in _VIDEO_EXTS:
                            try:
                                _bgm_track = _pick_background_track()
                                if _bgm_track is None:
                                    print(f"[{job_id}] bgm skipped: no track available")
                                else:
                                    ok, bgm_msg = await asyncio.to_thread(_mux_background_music, path, _bgm_track)
                                    print(f"[{job_id}] bgm [{_bgm_track.name}]: {bgm_msg}")
                            except Exception as bgm_err:
                                print(f"[{job_id}] bgm mux raised: {bgm_err}")

                        image_delivery = None
                        if is_image_output and optimize_image_file is not None:
                            try:
                                optimized = await asyncio.to_thread(optimize_image_file, path)
                                path = optimized.path
                                filename = path.name
                                ext = path.suffix.lower()
                                image_delivery = {
                                    "original_bytes": optimized.original_bytes,
                                    "output_bytes": optimized.output_bytes,
                                    "width": optimized.width,
                                    "height": optimized.height,
                                    "quality": optimized.quality,
                                }
                                print(
                                    f"[{job_id}] image delivery: "
                                    f"{optimized.original_bytes // 1024} KB -> "
                                    f"{optimized.output_bytes // 1024} KB "
                                    f"({optimized.width}x{optimized.height})"
                                )
                            except Exception as image_err:
                                print(f"[{job_id}] image optimization skipped: {image_err}")

                        if jobs.get(job_id, {}).get("status") == "cancelled":
                            path.unlink(missing_ok=True)
                            return

                        url = (
                            f"{BASE_URL}/image/{filename}"
                            if is_image_output
                            else f"{BASE_URL}/video/{filename}"
                        )

                        # For video outputs, snap a thumbnail (first frame).
                        # Runs AFTER watermarks so the thumbnail reflects the
                        # final, stamped video. Failure is non-fatal — the
                        # video itself is still returned.
                        thumbnail_url = None
                        if ext in _VIDEO_EXTS:
                            thumb = _extract_video_thumbnail(path)
                            if thumb is not None:
                                thumbnail_url = f"{BASE_URL}/image/{thumb.name}"
                        completed_at = datetime.now(timezone.utc)
                        created_at_str = jobs[job_id].get("created_at")
                        duration_seconds = None
                        if created_at_str:
                            started = datetime.fromisoformat(created_at_str)
                            duration_seconds = round((completed_at - started).total_seconds(), 1)
                        # IMPORTANT: build the completed dict last so we can
                        # fold the watermark warning into it. Replacing
                        # jobs[job_id] without this carry would silently
                        # swallow the warning.
                        completed = {
                            "status": "completed",
                            "url": url,
                            "filename": filename,
                            "completed_at": completed_at.isoformat(),
                            "duration_seconds": duration_seconds,
                            "progress": 100,
                            "stage": "completed",
                            "progress_message": "Video ready" if ext in _VIDEO_EXTS else "Image ready",
                            "eta_seconds": 0,
                        }
                        if thumbnail_url:
                            completed["thumbnail_url"] = thumbnail_url
                        if image_delivery:
                            completed["image_delivery"] = image_delivery
                        if wm_warnings:
                            completed["watermark_warning"] = " | ".join(wm_warnings)
                        if audio_mux_warning:
                            completed["audio_warning"] = audio_mux_warning
                        if jobs.get(job_id, {}).get("status") != "cancelled":
                            jobs[job_id] = completed
                        return

        jobs[job_id] = {**jobs[job_id], "status": "failed", "error": "No output found"}
    except JobCancelled:
        jobs[job_id] = {**jobs.get(job_id, {}), "status": "cancelled"}
    except Exception as e:
        jobs[job_id] = {**jobs[job_id], "status": "failed", "error": str(e), "failed_at": datetime.now(timezone.utc).isoformat()}
    finally:
        if cleanup_paths:
            for p in cleanup_paths:
                Path(p).unlink(missing_ok=True)


def _probe_video_duration_seconds(video_path: Path) -> float:
    """Return the media duration reported by ffprobe, or raise ValueError."""
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nokey=1:noprint_wrappers=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if probe.returncode != 0:
        raise ValueError((probe.stderr or "ffprobe failed").strip())
    try:
        duration = float(probe.stdout.strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("reference video has no readable duration") from exc
    if duration <= 0:
        raise ValueError("reference video duration must be greater than zero")
    return duration


def _concat_motion_chunks(chunk_paths: list[Path], output_path: Path) -> None:
    """Join clean motion chunks while removing duplicated boundary frames."""
    if len(chunk_paths) < 2:
        raise ValueError("at least two chunks are required for concatenation")

    command = ["ffmpeg", "-y", "-loglevel", "error"]
    for chunk_path in chunk_paths:
        command.extend(["-i", str(chunk_path)])

    filters: list[str] = []
    labels: list[str] = []
    for index in range(len(chunk_paths)):
        label = f"v{index}"
        trim = "" if index == 0 else "trim=start_frame=1,"
        filters.append(f"[{index}:v]{trim}setpts=PTS-STARTPTS[{label}]")
        labels.append(f"[{label}]")
    filters.append(f"{''.join(labels)}concat=n={len(labels)}:v=1:a=0[outv]")

    command.extend([
        "-filter_complex", ";".join(filters),
        "-map", "[outv]",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "19",
        "-movflags", "+faststart",
        str(output_path),
    ])
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"could not join motion segments: {(result.stderr or 'ffmpeg failed')[-500:]}")


def _extract_motion_continuity_frame(video_path: Path, image_path: Path) -> bool:
    """Extract the last decoded frame of a chunk for the next chunk's anchor."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-sseof", "-1", "-i", str(video_path),
            "-vf", "reverse",
            "-frames:v", "1",
            str(image_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.returncode == 0 and image_path.exists() and image_path.stat().st_size > 0


async def run_motion_control_job(
    job_id: str,
    chunk_specs: list[dict],
    character_image_filename: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int,
    inplace_strength: float,
    motion_strength: float,
    cleanup_paths: list[str],
    *,
    audio_source_path: str | None,
    watermark_text: str | None,
    watermark_image: bool,
    reference_duration_seconds: float,
    target_frame_count: int,
) -> None:
    """Generate GPU-safe motion chunks, preserve continuity, and join them."""
    chunk_outputs: list[Path] = []
    continuity_inputs: list[Path] = []
    final_path: Path | None = None
    started_at = datetime.now(timezone.utc)
    jobs[job_id] = {
        **jobs.get(job_id, {}),
        "status": "processing",
        "started_at": started_at.isoformat(),
        "stage": "generating_motion",
        "progress": 10,
        "progress_message": f"Generating motion segment 1 of {len(chunk_specs)}...",
    }

    try:
        current_character_filename = character_image_filename
        for index, spec in enumerate(chunk_specs):
            if jobs.get(job_id, {}).get("status") == "cancelled":
                raise JobCancelled(job_id)

            jobs[job_id] = {
                **jobs[job_id],
                "stage": "generating_motion",
                "segment": index + 1,
                "segments": len(chunk_specs),
                "progress": min(85, 10 + int(index / len(chunk_specs) * 75)),
                "progress_message": f"Generating motion segment {index + 1} of {len(chunk_specs)}...",
            }
            workflow_args = dict(
                character_image_filename=current_character_filename,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                length=spec["length"],
                fps=MOTION_FPS,
                seed=(seed + index) % 2**32,
                preset="fast",
                audio=False,
                enhance_prompt=False,
                inplace_strength=inplace_strength,
                motion_strength=motion_strength,
            )
            if spec.get("frame_filenames"):
                workflow = build_ltx_motion_workflow_no_vhs(
                    reference_frame_filenames=spec["frame_filenames"],
                    **workflow_args,
                )
            else:
                workflow = build_ltx_motion_workflow(
                    reference_video_filename=spec["video_filename"],
                    **workflow_args,
                )

            _, output_path_str = await _submit_and_wait_comfyui(workflow, job_id)
            output_path = Path(output_path_str)
            chunk_outputs.append(output_path)

            if index + 1 < len(chunk_specs):
                continuity_filename = f"ltx_motion_continuity_{uuid.uuid4().hex}.png"
                continuity_path = INPUT_DIR / continuity_filename
                extracted = await asyncio.to_thread(
                    _extract_motion_continuity_frame, output_path, continuity_path,
                )
                if extracted:
                    continuity_inputs.append(continuity_path)
                    current_character_filename = continuity_filename
                else:
                    continuity_path.unlink(missing_ok=True)
                    # Fail open to the original identity image. The generated
                    # chunks are still usable and the join remains frame-exact.
                    current_character_filename = character_image_filename

        jobs[job_id] = {
            **jobs[job_id],
            "stage": "joining_segments",
            "progress": 90,
            "progress_message": "Joining motion segments...",
        }
        if len(chunk_outputs) == 1:
            final_path = chunk_outputs[0]
        else:
            final_path = OUTPUT_DIR / f"ltx_motion_full_{uuid.uuid4().hex}.mp4"
            await asyncio.to_thread(_concat_motion_chunks, chunk_outputs, final_path)

        audio_warning = None
        if audio_source_path:
            ok, message = await asyncio.to_thread(
                _mux_reference_audio, final_path, Path(audio_source_path),
            )
            if not ok:
                audio_warning = message

        watermark_warnings: list[str] = []
        if watermark_text and watermark is not None:
            try:
                watermark.apply(final_path, watermark_text)
            except Exception as exc:
                watermark_warnings.append(f"text: {exc}")
        if watermark_image and watermark is not None:
            try:
                watermark.apply_logo(final_path)
            except Exception as exc:
                watermark_warnings.append(f"image: {exc}")

        if jobs.get(job_id, {}).get("status") == "cancelled":
            raise JobCancelled(job_id)

        media_duration = await asyncio.to_thread(_probe_video_duration_seconds, final_path)
        thumbnail = await asyncio.to_thread(_extract_video_thumbnail, final_path)
        completed_at = datetime.now(timezone.utc)
        completed = {
            "status": "completed",
            "url": f"{BASE_URL}/video/{final_path.name}",
            "filename": final_path.name,
            "completed_at": completed_at.isoformat(),
            "duration_seconds": round((completed_at - started_at).total_seconds(), 1),
            "media_duration_seconds": round(media_duration, 3),
            "reference_duration_seconds": round(reference_duration_seconds, 3),
            "frames": target_frame_count,
            "fps": MOTION_FPS,
            "segments": len(chunk_specs),
            "progress": 100,
            "stage": "completed",
            "progress_message": "Video ready",
            "eta_seconds": 0,
        }
        if thumbnail is not None:
            completed["thumbnail_url"] = f"{BASE_URL}/image/{thumbnail.name}"
        if audio_warning:
            completed["audio_warning"] = audio_warning
        if watermark_warnings:
            completed["watermark_warning"] = " | ".join(watermark_warnings)
        jobs[job_id] = completed
    except JobCancelled:
        jobs[job_id] = {**jobs.get(job_id, {}), "status": "cancelled"}
        if final_path is not None:
            final_path.unlink(missing_ok=True)
    except Exception as exc:
        jobs[job_id] = {
            **jobs.get(job_id, {}),
            "status": "failed",
            "error": str(exc),
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        if final_path is not None:
            final_path.unlink(missing_ok=True)
    finally:
        for path in cleanup_paths:
            Path(path).unlink(missing_ok=True)
        for path in continuity_inputs:
            path.unlink(missing_ok=True)
        for path in chunk_outputs:
            if final_path is None or path != final_path:
                path.unlink(missing_ok=True)


# ─────────────────────────────────────────────
# Health & job management
# ─────────────────────────────────────────────


def _job_links(job_id: str) -> dict:
    return {
        "poll_url": f"{BASE_URL}/status/{job_id}",
        "cancel_url": f"{BASE_URL}/jobs/{job_id}/cancel",
    }


WORKFLOW_CATALOG = [
    {
        "id": "text-to-image",
        "label": "FLUX text to image",
        "endpoint": "/t2i",
        "media": "image",
        "inputs": [],
    },
    {
        "id": "face-swap",
        "label": "FLUX face swap",
        "endpoint": "/flux/face-swap",
        "media": "image",
        "inputs": ["target.png", "face.png"],
    },
    {
        "id": "multi-face-swap",
        "label": "FLUX multi-person face swap",
        "endpoint": "/flux/multi-face-swap",
        "media": "image",
        "inputs": ["target.png", "face-1.png", "face-2.png (optional)"],
    },
    {
        "id": "image-to-image",
        "label": "FLUX image editing",
        "endpoint": "/flux/i2i",
        "media": "image",
        "inputs": ["reference.png"],
    },
    {
        "id": "image-to-video",
        "label": "LTX image to video",
        "endpoint": "/ltx/i2v",
        "media": "video",
        "inputs": ["input.png"],
    },
    {
        "id": "text-to-video",
        "label": "LTX text to video",
        "endpoint": "/ltx/t2v",
        "media": "video",
        "inputs": [],
    },
    {
        "id": "personalized-video",
        "label": "Face swap then animate",
        "endpoint": "/face-animate",
        "media": "video",
        "inputs": ["target.png", "face.png"],
        "stages": ["face-swap", "image-to-video"],
    },
]


def _sample_workflow(workflow_id: str) -> dict:
    seed = 12345
    if workflow_id == "text-to-image":
        return build_t2i_workflow(
            "cinematic portrait, natural light", 1024, 1024, seed,
            steps=4, cfg=1.0, guidance=4.0,
        )
    if workflow_id == "face-swap":
        return get_flux_face_swap_workflow(
            "target.png", "face.png", seed,
            megapixels=1.0, steps=4, cfg=1.0, guidance=4.0,
            lora_strength=1.0,
        )
    if workflow_id == "multi-face-swap":
        return build_flux_multi_face_swap_workflow(
            "target.png", ["face-1.png", "face-2.png"], seed,
            face_order="left-to-right", megapixels=1.0, steps=4,
            cfg=1.0, guidance=4.0, lora_strength=1.0,
            target_face_indices=[0, 1],
        )
    if workflow_id == "image-to-image":
        return build_flux_i2i_workflow(
            ["reference.png"], "cinematic photo edit", seed,
            megapixels=1.0, steps=4, cfg=1.0, guidance=4.0,
        )
    if workflow_id == "image-to-video":
        return build_ltx_i2v_workflow(
            "input.png", "subtle natural movement", LTX_DEFAULT_NEGATIVE,
            544, 960, 121, 24, seed,
            preset="fast", audio=False, enhance_prompt=True,
        )
    if workflow_id == "text-to-video":
        return build_ltx_t2v_workflow(
            "cinematic scene with gentle camera movement", LTX_DEFAULT_NEGATIVE,
            768, 432, 121, 24, seed, preset="fast", audio=False,
        )
    if workflow_id == "personalized-video":
        return {
            "stages": [
                {
                    "id": "face-swap",
                    "workflow": _sample_workflow("face-swap"),
                    "output": "personalized.png",
                },
                {
                    "id": "image-to-video",
                    "workflow": build_ltx_i2v_workflow(
                        "personalized.png", "subtle natural movement",
                        LTX_DEFAULT_NEGATIVE, 544, 960, 121, 24, seed,
                        preset="fast", audio=False, enhance_prompt=True,
                    ),
                },
            ]
        }
    raise KeyError(workflow_id)


@app.get("/health")
async def health():
    active = sum(1 for job in jobs.values() if job.get("status") in {"queued", "processing"})
    active_video = _active_video_job_count()
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{COMFYUI_URL}/system_stats")
            response.raise_for_status()
        comfy_status = "ready"
        status_code = 200
    except Exception:
        comfy_status = "unavailable"
        status_code = 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if status_code == 200 else "degraded",
            "version": API_VERSION,
            "pod_id": POD_ID,
            "role": AI_GEN_ROLE,
            "comfyui": comfy_status,
            "active_jobs": active,
            "active_video_jobs": active_video,
            "max_video_concurrency": MAX_ACTIVE_VIDEO_JOBS,
            "video_capacity_available": MAX_ACTIVE_VIDEO_JOBS == 0 or active_video < MAX_ACTIVE_VIDEO_JOBS,
        },
    )


@app.get("/workflows")
async def list_workflows():
    return {
        "format": "comfyui-api",
        "items": WORKFLOW_CATALOG,
        "hint": "Open /workflows/<id> and load the returned workflow in ComfyUI.",
    }


@app.get("/workflows/{workflow_id}")
async def get_workflow_export(workflow_id: str, download: bool = False):
    definition = next((item for item in WORKFLOW_CATALOG if item["id"] == workflow_id), None)
    if definition is None:
        raise HTTPException(404, f"Unknown workflow '{workflow_id}'")
    body = {
        "format": "comfyui-api",
        "definition": definition,
        "workflow": _sample_workflow(workflow_id),
    }
    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{workflow_id}.json"'
    return JSONResponse(content=body, headers=headers)

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    # Workflow graphs and ComfyUI prompt IDs are internal implementation
    # details. Keeping them out of every poll response cuts response size and
    # avoids exposing the complete node graph to third-party applications.
    return {
        key: value
        for key, value in jobs[job_id].items()
        if key not in {"workflow", "prompt_id"}
    }

@app.get("/jobs")
async def get_all_jobs():
    return {
        "total": len(jobs),
        "summary": {s: sum(1 for j in jobs.values() if j.get("status") == s) for s in ["queued", "processing", "completed", "failed"]},
        "jobs": [{"job_id": jid, **info} for jid, info in jobs.items()]
    }

@app.get("/queue")
async def get_queue():
    active = {jid: info for jid, info in jobs.items() if info.get("status") in ["queued", "processing"]}
    return {"count": len(active), "jobs": [{"job_id": jid, "status": info["status"]} for jid, info in active.items()]}

def _delete_output_files(filename: str) -> int:
    """Remove the primary output file plus its `_thumb.jpg` sibling, if any.
    Returns the number of files actually deleted (0–2)."""
    deleted = 0
    for path in [OUTPUT_DIR / "video" / filename, OUTPUT_DIR / "images" / filename, OUTPUT_DIR / filename]:
        if path.exists():
            path.unlink()
            deleted += 1
    thumb = OUTPUT_DIR / "images" / f"{Path(filename).stem}_thumb.jpg"
    if thumb.exists():
        thumb.unlink()
        deleted += 1
    return deleted


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    filename = job.get("filename")
    result = {"job_id": job_id, "deleted": True}
    if filename and _delete_output_files(filename):
        result["file_deleted"] = filename
    del jobs[job_id]
    return result

@app.delete("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job.get("status") == "completed":
        raise HTTPException(400, "Job already completed")
    if job.get("status") == "failed":
        raise HTTPException(400, "Job already failed")
    prompt_id = job.get("prompt_id")
    if prompt_id:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(f"{COMFYUI_URL}/queue", json={"delete": [prompt_id]})
                if job.get("status") == "processing":
                    await client.post(f"{COMFYUI_URL}/interrupt")
        except Exception as cancel_error:
            print(f"[{job_id}] ComfyUI cancel warning: {cancel_error}")
    jobs[job_id] = {
        **job,
        "status": "cancelled",
        "cancelled_at": datetime.now(timezone.utc).isoformat(),
    }
    return {"job_id": job_id, "status": "cancelled"}

@app.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str, background_tasks: BackgroundTasks):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job.get("status") not in ["failed", "cancelled"]:
        raise HTTPException(400, f"Can only retry failed/cancelled jobs. Current: {job.get('status')}")
    if "workflow" not in job:
        raise HTTPException(400, "No workflow stored — submit a new request")
    new_job_id = str(uuid.uuid4())
    jobs[new_job_id] = {"status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    background_tasks.add_task(run_job, new_job_id, job["workflow"])
    return {
        "new_job_id": new_job_id,
        "original_job_id": job_id,
        "status": "queued",
        **_job_links(new_job_id),
    }

@app.delete("/jobs")
async def delete_all_jobs(completed_only: bool = True):
    deleted_jobs = deleted_files = 0
    for job_id in list(jobs.keys()):
        job = jobs[job_id]
        if completed_only and job.get("status") != "completed":
            continue
        filename = job.get("filename")
        if filename:
            deleted_files += _delete_output_files(filename)
        del jobs[job_id]
        deleted_jobs += 1
    return {"deleted_jobs": deleted_jobs, "deleted_files": deleted_files}


# ─────────────────────────────────────────────
# File serving
# ─────────────────────────────────────────────

def _safe_output_relpath(file_path: str) -> Path:
    rel = Path(file_path)
    if rel.is_absolute() or any(part in ("", ".", "..") for part in rel.parts):
        raise HTTPException(400, "Invalid file path")
    return rel


def _safe_output_candidate(root: Path, rel: Path) -> Path:
    root_resolved = root.resolve()
    candidate = (root / rel).resolve()
    if not candidate.is_relative_to(root_resolved):
        raise HTTPException(400, "Invalid file path")
    return candidate


@app.get("/image/{file_path:path}")
async def serve_image(file_path: str):
    rel = _safe_output_relpath(file_path)
    for path in [_safe_output_candidate(OUTPUT_DIR / "images", rel), _safe_output_candidate(OUTPUT_DIR, rel)]:
        if path.exists():
            # Pick a sensible content-type from the extension so .jpg
            # thumbnails don't get served as image/png and broken in some
            # clients (Safari is strict about this).
            ext = path.suffix.lower()
            mt = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".gif": "image/gif",
            }.get(ext, "image/png")
            return FileResponse(str(path), media_type=mt, filename=rel.name)
    raise HTTPException(404, f"Image not found: {file_path}")

@app.get("/video/{file_path:path}")
async def serve_video(file_path: str):
    rel = _safe_output_relpath(file_path)
    for path in [_safe_output_candidate(OUTPUT_DIR / "video", rel), _safe_output_candidate(OUTPUT_DIR, rel)]:
        if path.exists():
            return FileResponse(str(path), media_type="video/mp4", filename=rel.name)
    raise HTTPException(404, f"Not found: {file_path}")

@app.delete("/video/{file_path:path}")
async def delete_video(file_path: str):
    rel = _safe_output_relpath(file_path)
    for path in [_safe_output_candidate(OUTPUT_DIR / "video", rel), _safe_output_candidate(OUTPUT_DIR, rel)]:
        if path.exists():
            path.unlink()
            # Drop the thumbnail sibling too (best-effort).
            thumb = OUTPUT_DIR / "images" / f"{rel.stem}_thumb.jpg"
            thumb.unlink(missing_ok=True)
            for job_id, info in list(jobs.items()):
                if info.get("filename") in {file_path, rel.name}:
                    del jobs[job_id]
            return {"status": "deleted", "filename": file_path}
    raise HTTPException(404, f"File not found: {file_path}")

@app.get("/videos")
async def list_videos():
    video_dir = OUTPUT_DIR / "video"
    if not video_dir.exists():
        return {"total": 0, "videos": []}
    videos = []
    images_dir = OUTPUT_DIR / "images"
    for f in sorted(video_dir.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True):
        stat = f.stat()
        entry = {
            "filename": f.name,
            "size_mb": round(stat.st_size / 1024 / 1024, 2),
            "url": f"{BASE_URL}/video/{f.name}",
            "created_at": stat.st_mtime,
        }
        thumb = images_dir / f"{f.stem}_thumb.jpg"
        if thumb.exists():
            entry["thumbnail_url"] = f"{BASE_URL}/image/{thumb.name}"
        videos.append(entry)
    return {"total": len(videos), "videos": videos}


# ─────────────────────────────────────────────
# Text to Image (FLUX.2 Klein 9B)
# ─────────────────────────────────────────────

class T2IRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    seed: int = -1
    steps: int = 4
    cfg: float = 1.0
    guidance: float = 4.0
    watermark: str | None = None  # e.g. "AI" — overlay at bottom-right; null/empty = off
    watermark_image: bool = False  # composite the Metfone GenAI logo at bottom-right
    # Output-side face filter — applied AFTER generation. /t2i has no input
    # image so this is the only way a blocked-identity prompt can be caught.
    # Defaults ON (safe default, matching /flux/face-swap & /flux/i2i): callers
    # may pass face_filter=false to skip (logged to
    # /workspace/face_filter_bypass.log). The proxies now FORWARD the caller's
    # value rather than forcing it, so this default is what protects callers
    # that omit the flag.
    face_filter: bool = True
    # Output-side logo filter — same reasoning as face_filter but for
    # blocked logos/flags. Catches "draw the [logo] flag" prompts. Defaults ON.
    logo_filter: bool = True
    # Optional styled lower-third caption (e.g. horoscope text). Null/empty = off.
    caption: str | None = None
    # Optional zodiac sign — adds a gold glyph + divider above the caption.
    caption_icon: str | None = None

@app.post("/t2i")
async def text_to_image(req: T2IRequest, background_tasks: BackgroundTasks):
    seed = req.seed if req.seed != -1 else uuid.uuid4().int % 2**32
    workflow = build_t2i_workflow(
        prompt=req.prompt, width=req.width, height=req.height, seed=seed,
        steps=req.steps, cfg=req.cfg, guidance=req.guidance,
    )
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    # Pass face_filter / logo_filter down to run_job so it scans the OUTPUT
    # before exposing the result URL. /t2i has no input image, so this is
    # the ONLY safety check that runs for this endpoint.
    if not req.face_filter and face_safety is not None:
        face_safety.log_bypass(job_id, "/t2i", note="face_filter=false (output check skipped)")
    if not req.logo_filter and face_safety is not None:
        face_safety.log_bypass(job_id, "/t2i", note="logo_filter=false (output check skipped)")
    background_tasks.add_task(
        run_job, job_id, workflow, None, req.watermark, req.watermark_image,
        output_face_filter=req.face_filter, output_logo_filter=req.logo_filter,
        output_endpoint="/t2i", caption=req.caption, caption_icon=req.caption_icon,
    )
    return {"job_id": job_id, "status": "queued", "model": "flux2-klein-9b", **_job_links(job_id)}


# ─────────────────────────────────────────────
# Compliance helper — checks N input images against the blocklist.
# Raises HTTPException(400) on the first blocked image. Returns silently
# if the filter is disabled or no images match.
#
# `face_filter=False` is recorded to /workspace/face_filter_bypass.log for
# audit purposes — anyone calling these endpoints with face_filter=false
# leaves a trail.
# ─────────────────────────────────────────────

def _apply_face_filter(endpoint: str, job_id: str, face_filter: bool,
                       images_with_names: list) -> None:
    """images_with_names: list of (bytes, label) pairs. label is used in the error."""
    if not face_filter:
        if face_safety is not None:
            face_safety.log_bypass(job_id, endpoint, note=f"face_filter=false, {len(images_with_names)} images")
        return
    if face_safety is None:
        raise HTTPException(503, "face filter requested but `safety` module unavailable (insightface not installed)")
    for idx, (img_bytes, label) in enumerate(images_with_names):
        try:
            result = face_safety.check_image(img_bytes)
        except RuntimeError as e:
            raise HTTPException(503, f"face filter unavailable: {e}")
        if result.blocked:
            raise HTTPException(400, {
                "error": "blocked",
                "filter": "face",
                "reason": f"{label} matches blocked face identity",
                "matched_identity": result.matched_identity,
                "score": round(result.score, 4),
                "image_index": idx,
            })


def _require_detectable_face(endpoint: str, enabled: bool,
                             images_with_names: list) -> None:
    """Reject opted-in user images when InsightFace finds no clear face."""
    if not enabled:
        return
    if face_safety is None:
        raise HTTPException(503, detail={
            "error": "server_busy",
            "error_code": "server_overload",
            "reason": "Face validation is temporarily unavailable.",
        })
    for image_index, (img_bytes, label) in enumerate(images_with_names):
        try:
            face_count = face_safety.detect_face_count(img_bytes)
        except RuntimeError:
            raise HTTPException(503, detail={
                "error": "server_busy",
                "error_code": "server_overload",
                "reason": "Face validation is temporarily unavailable.",
            })
        if face_count < 1:
            raise HTTPException(422, detail={
                "error": "image_quality",
                "error_code": "image_quality_insufficient",
                "reason": f"{label} does not contain a clearly detectable face.",
                "image_index": image_index,
            })


def _apply_logo_filter(endpoint: str, job_id: str, logo_filter: bool,
                       images_with_names: list) -> None:
    """Parallel to _apply_face_filter but for the logo/flag blocklist (CLIP-based)."""
    if not logo_filter:
        # Reuse the face-filter bypass log so admins have one audit trail
        if face_safety is not None:
            face_safety.log_bypass(job_id, endpoint, note=f"logo_filter=false, {len(images_with_names)} images")
        return
    if logo_safety is None:
        raise HTTPException(503, "logo filter requested but `logo_safety` module unavailable (open_clip_torch not installed)")
    for idx, (img_bytes, label) in enumerate(images_with_names):
        try:
            result = logo_safety.check_image(img_bytes)
        except RuntimeError as e:
            raise HTTPException(503, f"logo filter unavailable: {e}")
        if result.blocked:
            raise HTTPException(400, {
                "error": "blocked",
                "filter": "logo",
                "reason": f"{label} matches blocked logo/flag",
                "matched_logo": result.matched_logo,
                "score": round(result.score, 4),
                "image_index": idx,
            })


# ─────────────────────────────────────────────
# FLUX.2 Klein 9B Head/Face Swap — builders live in workflows.py
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# LTX-2.3 — presets & shared helpers live in workflows.py
# ─────────────────────────────────────────────


@app.get("/ltx/presets")
async def get_ltx_presets():
    info = {}
    for k, v in LTX_PRESETS.items():
        if v["two_pass"]:
            info[k] = {"mode": "two_pass", "low_res_steps": v["low_res_sigmas"].count(","), "high_res_steps": v["high_res_sigmas"].count(","), "lora_strength": v["lora_strength"]}
        else:
            info[k] = {"mode": "single_pass", "steps": v["sigmas"].count(","), "lora_strength": v["lora_strength"]}
    return {"presets": info, "default": "fast", "endpoints": ["/ltx/i2v", "/ltx/t2v", "/face-animate"]}


# ─────────────────────────────────────────────
# LTX-2.3 Image to Video
# ─────────────────────────────────────────────

@app.post("/ltx/i2v")
async def ltx_image_to_video(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="Input image to animate"),
    prompt: str = Form("", description="What should happen in the video"),
    negative_prompt: str = Form(LTX_DEFAULT_NEGATIVE),
    preset: str = Form("fast", description="Speed/quality preset: realtime (4 steps), fast (8 steps), or quality (8+3 steps with 2× spatial upscale)"),
    aspect_ratio: str = Form("9:16", description="Output aspect ratio: original | 16:9 | 9:16 | 1:1 | 4:3 | 3:4 | 3:2 | 2:3 | 21:9 | 9:21. When set, height is derived from width — see below."),
    width: int = Form(544, description="Output width in pixels. When aspect_ratio is set, height is COMPUTED from this and `height` is ignored. For 9:16 use 544 (→544×960, fast) or 720 (→720×1280, quality)."),
    height: int = Form(960, description="Output height in pixels. IGNORED when aspect_ratio is set (only used with aspect_ratio=original)."),
    length: int = Form(121, description="Number of frames — 97 (~4s), 121 (~5s), 161 (~6.7s)"),
    fps: int = Form(24, description="Frames per second"),
    seed: int = Form(-1),
    audio: bool = Form(False, description="Generate audio track with the video (adds overhead)"),
    enhance_prompt: bool = Form(True, description="Rewrite prompt via Gemma 12B using the input image as context (adds 2-5s + VRAM). Recommended ON for short prompts (e.g. 'make her run'); OFF when you've already written a detailed scene description."),
    inplace_strength: float = Form(0.7, ge=0.3, le=1.0, description="How tightly each frame is pinned to the input image. 0.7 = reference distilled value (good identity, weak motion). Lower it for action prompts: 0.5 ≈ moderate motion, 0.4 ≈ strong motion (some identity drift), 0.3 ≈ near-t2v. Two-pass refine tracks this (= min(1.0, x+0.3))."),
    watermark: str | None = Form(None, description="Optional text to overlay at the bottom-right of the output (e.g. 'AI'). Null/empty = no watermark. Video re-encodes via ffmpeg (~1-3s for a 5s clip)."),
    watermark_image: bool = Form(False, description="Composite the Metfone GenAI logo (loaded once from /workspace/assets/metfone_genai_watermark.png) at the bottom-right. Stacks with `watermark` if both are set."),
    caption: str | None = Form(None, description="Optional styled lower-third caption (e.g. the horoscope of the day). Word-wrapped, centered white text with a heavy black outline; videos fade it in ~1s after the start. Same fixed design on images and videos. Null/empty = no caption."),
    caption_icon: str | None = Form(None, description="Optional zodiac sign for the caption (aries|taurus|gemini|cancer|leo|virgo|libra|scorpio|sagittarius|capricorn|aquarius|pisces). When set alongside `caption`, a gold zodiac glyph + divider are stacked above the text. Ignored if not a recognised sign."),
    caption_fade: bool = Form(True, description="Video only: when true (default) the caption fades in ~1s after the start; set false to show it from the very first frame. No effect on images (their caption is always immediate)."),
    background_music: bool = Form(False, description="Video only: mux a looping royalty-free background-music bed (/workspace/assets/horoscope_bgm.m4a) under the clip, trimmed to length with a soft fade. No effect on images."),
    face_filter: bool = Form(True, description="Reject the input image when it matches a blocked face identity. Enabled by default."),
    require_detectable_face: bool = Form(False, description="Opt-in input validation. When true, reject the uploaded image unless at least one clear face is detectable. Default false."),
):
    if preset not in LTX_PRESETS:
        raise HTTPException(400, f"Invalid preset '{preset}'. Valid: {', '.join(LTX_PRESETS)}")
    if aspect_ratio != "original" and aspect_ratio not in LTX_ASPECT_RATIOS:
        raise HTTPException(400, f"Invalid aspect_ratio. Valid: original, {', '.join(LTX_ASPECT_RATIOS)}")

    seed = seed if seed != -1 else uuid.uuid4().int % 2**32
    width, height = compute_ltx_dimensions(width, height, aspect_ratio)

    img_bytes = await image.read()
    job_id = str(uuid.uuid4())
    _require_detectable_face(
        "/ltx/i2v", require_detectable_face, [(img_bytes, "image")],
    )
    _apply_face_filter("/ltx/i2v", job_id, face_filter, [(img_bytes, "image")])
    img_filename = f"ltx_i2v_{uuid.uuid4().hex}.png"
    img_path = str(INPUT_DIR / img_filename)
    Path(img_path).write_bytes(img_bytes)

    workflow = build_ltx_i2v_workflow(
        image_filename=img_filename, prompt=prompt, negative_prompt=negative_prompt,
        width=width, height=height, length=length, fps=fps, seed=seed,
        preset=preset, audio=audio, enhance_prompt=enhance_prompt,
        inplace_strength=inplace_strength,
    )

    _reserve_video_job(job_id, [img_path])
    background_tasks.add_task(run_job, job_id, workflow, [img_path], watermark, watermark_image, caption=caption, caption_icon=caption_icon, caption_fade=caption_fade, background_music=background_music)
    return {"job_id": job_id, "status": "queued", "model": "ltx-2.3-22b", **_job_links(job_id)}


# ─────────────────────────────────────────────
# LTX-2.3 Text to Video
# ─────────────────────────────────────────────

@app.post("/ltx/t2v")
async def ltx_text_to_video(
    background_tasks: BackgroundTasks,
    prompt: str = Form(..., description="What should appear/happen in the video"),
    negative_prompt: str = Form(LTX_DEFAULT_NEGATIVE),
    preset: str = Form("fast", description="Speed/quality preset: realtime (4 steps), fast (8 steps), or quality (8+3 steps with 2× spatial upscale)"),
    aspect_ratio: str = Form("16:9", description="Output aspect ratio: 16:9 | 9:16 | 1:1 | 4:3 | 3:4 | 3:2 | 2:3 | 21:9 | 9:21"),
    width: int = Form(1280, description="Output width in pixels (height auto-computed from aspect_ratio)"),
    height: int = Form(720, description="Output height in pixels (ignored if aspect_ratio set, default used for 'original')"),
    length: int = Form(121, description="Number of frames — 97 (~4s), 121 (~5s), 161 (~6.7s)"),
    fps: int = Form(24, description="Frames per second"),
    seed: int = Form(-1),
    audio: bool = Form(False, description="Generate audio track with the video (adds overhead)"),
    watermark: str | None = Form(None, description="Optional text to overlay at the bottom-right of the output (e.g. 'AI'). Null/empty = no watermark. Video re-encodes via ffmpeg (~1-3s for a 5s clip)."),
    watermark_image: bool = Form(False, description="Composite the Metfone GenAI logo (loaded once from /workspace/assets/metfone_genai_watermark.png) at the bottom-right. Stacks with `watermark` if both are set."),
    caption: str | None = Form(None, description="Optional styled lower-third caption (e.g. the horoscope of the day). Word-wrapped, centered white text with a heavy black outline; videos fade it in ~1s after the start. Same fixed design on images and videos. Null/empty = no caption."),
    caption_icon: str | None = Form(None, description="Optional zodiac sign for the caption (aries|taurus|gemini|cancer|leo|virgo|libra|scorpio|sagittarius|capricorn|aquarius|pisces). When set alongside `caption`, a gold zodiac glyph + divider are stacked above the text. Ignored if not a recognised sign."),
    caption_fade: bool = Form(True, description="Video only: when true (default) the caption fades in ~1s after the start; set false to show it from the very first frame. No effect on images (their caption is always immediate)."),
    background_music: bool = Form(False, description="Video only: mux a looping royalty-free background-music bed (/workspace/assets/horoscope_bgm.m4a) under the clip, trimmed to length with a soft fade. No effect on images."),
):
    if preset not in LTX_PRESETS:
        raise HTTPException(400, f"Invalid preset '{preset}'. Valid: {', '.join(LTX_PRESETS)}")
    if aspect_ratio not in LTX_ASPECT_RATIOS and aspect_ratio != "original":
        raise HTTPException(400, f"Invalid aspect_ratio. Valid: {', '.join(LTX_ASPECT_RATIOS)}")

    seed = seed if seed != -1 else uuid.uuid4().int % 2**32
    width, height = compute_ltx_dimensions(width, height, aspect_ratio)

    workflow = build_ltx_t2v_workflow(
        prompt=prompt, negative_prompt=negative_prompt,
        width=width, height=height, length=length, fps=fps, seed=seed,
        preset=preset, audio=audio,
    )

    job_id = str(uuid.uuid4())
    _reserve_video_job(job_id)
    background_tasks.add_task(run_job, job_id, workflow, None, watermark, watermark_image, caption=caption, caption_icon=caption_icon, caption_fade=caption_fade, background_music=background_music)
    return {"job_id": job_id, "status": "queued", "model": "ltx-2.3-22b", **_job_links(job_id)}


# ─────────────────────────────────────────────
# LTX-2.3 Motion Control (Kling-style)
#
# Take a character image + a reference video of motion (dance, gesture,
# action) and produce a new video where the character does what the
# reference does. DWPose extracts only the reference person's skeleton;
# LTX Union-Control IC-LoRA applies that motion to the character image
# without copying the reference person's face or clothes.
#
# Reference videos are matched automatically up to 15 seconds. Longer
# timelines are generated as GPU-safe four-second passes and joined with
# shared boundary frames so the final duration follows the source instead
# of silently falling back to a five-second sample.
# ─────────────────────────────────────────────

@app.post("/ltx/motion")
async def ltx_motion_control(
    background_tasks: BackgroundTasks,
    reference_video: UploadFile = File(..., description="Reference video whose motion the character should mimic. The output matches its duration up to 15 seconds by default."),
    image: UploadFile = File(..., description="Character image — identity / appearance source. Same role as /ltx/i2v's image."),
    prompt: str = Form("", description="Free-form description of the character and action. The motion workflow uses it directly; enhance_prompt is accepted for API compatibility but ignored."),
    negative_prompt: str = Form(LTX_DEFAULT_NEGATIVE),
    preset: str = Form("fast", description="Accepted for API compatibility. Motion control currently uses the fixed 8-step distilled IC-LoRA workflow."),
    aspect_ratio: str = Form("9:16", description="Output aspect ratio: original | 16:9 | 9:16 | 1:1 | 4:3 | 3:4 | 3:2 | 2:3 | 21:9 | 9:21"),
    width: int = Form(544, description="Output width — height is derived from aspect_ratio. For 9:16 dance refs the 544×960 fast / 720×1280 quality presets are tuned for clean motion."),
    height: int = Form(960, description="Only used when aspect_ratio=original."),
    length: int = Form(121, description="Fallback frame count when match_reference_duration=false. It is snapped to LTX's required 8n+1 format."),
    fps: int = Form(24, description="Accepted for API compatibility. Motion control renders at 30 fps."),
    match_reference_duration: bool = Form(True, description="Match the output to the uploaded motion video's duration. Enabled by default so template videos are not cut to five seconds."),
    max_duration_seconds: float = Form(MOTION_MAX_DURATION_SECONDS, ge=1.0, le=MOTION_MAX_DURATION_SECONDS, description="Maximum source duration to render. The production limit is 15 seconds."),
    seed: int = Form(-1),
    audio: bool = Form(False, description="Carry the reference video's original audio track onto the output (Kling-style). If the reference is shorter than the output, audio loops to fill. If the reference has no audio, this is a silent no-op. We do NOT use LTX's audio synthesis path here — the reference audio is muxed via ffmpeg post-generation."),
    enhance_prompt: bool = Form(True, description="Accepted for API compatibility; currently ignored by the IC-LoRA motion workflow."),
    inplace_strength: float = Form(0.5, ge=0.0, le=1.0, description="Character-image identity anchor. 1.0 locks appearance most strongly; 0.5 balances identity and motion."),
    motion_strength: float = Form(1.0, ge=0.0, le=1.0, description="DWPose IC-LoRA guide strength. 1.0 follows the reference motion most closely."),
    watermark: str | None = Form(None, description="Optional text overlay at bottom-right. Stripped by Supabase proxies in prod."),
    watermark_image: bool = Form(False, description="Composite the Metfone GenAI logo at the bottom-right."),
    face_filter: bool = Form(True, description="Reject the character image when it matches a blocked face identity. Enabled by default."),
    require_detectable_face: bool = Form(False, description="Opt-in input validation. When true, reject the character image unless at least one clear face is detectable. Default false."),
):
    """Kling-style motion control via LTX 2.3.

    Pipeline:
      1. Save uploaded reference video + character image to ComfyUI input dir.
      2. Read the source duration and normalize it to a fixed 30 fps IC-LoRA
         timeline (up to 15 seconds by default).
      3. VHS_LoadVideo + DWPose turn the reference into pose-only frames.
         Union-Control IC-LoRA combines that pose guide with the separate
         character-image identity anchor, then LTX renders the new subject.
      4. Crop IC-LoRA guide tokens after every sample, as required by the
         official graph, so padded guide latents never decode as noisy frames.
      5. Generate long references in four-second chunks, carry the previous
         clean last frame into the next pass, and join shared boundaries.
      6. Enqueue as a background job; client polls /status/<job_id>.
      7. (If audio=True) After ComfyUI returns the silent output video,
         ffmpeg-mux the ORIGINAL reference's audio onto it — looping the
         audio with -stream_loop -1 if the source is shorter than the
         output, trimming with -shortest. The LTX audio-synthesis path is
         NOT used here — Kling-style carry-over of the source audio is
         what users expect from a motion-control endpoint.
    """
    if preset not in LTX_PRESETS:
        raise HTTPException(400, f"Invalid preset '{preset}'. Valid: {', '.join(LTX_PRESETS)}")
    if aspect_ratio != "original" and aspect_ratio not in LTX_ASPECT_RATIOS:
        raise HTTPException(400, f"Invalid aspect_ratio. Valid: original, {', '.join(LTX_ASPECT_RATIOS)}")

    seed = seed if seed != -1 else uuid.uuid4().int % 2**32
    width, height = compute_ltx_dimensions(width, height, aspect_ratio)

    # Persist the character image into ComfyUI's input dir under a stable
    # name — same pattern as /ltx/i2v. The cleanup list at the end ensures
    # both this and the normalized video get deleted after the job runs.
    img_bytes = await image.read()
    job_id = str(uuid.uuid4())
    _require_detectable_face(
        "/ltx/motion", require_detectable_face, [(img_bytes, "image")],
    )
    _apply_face_filter("/ltx/motion", job_id, face_filter, [(img_bytes, "image")])
    img_filename = f"ltx_motion_img_{uuid.uuid4().hex}.png"
    img_path = str(INPUT_DIR / img_filename)
    Path(img_path).write_bytes(img_bytes)

    # Reference video — save the raw upload, then ffmpeg-normalize into the
    # canvas / fps / length the workflow expects. The intermediate raw file
    # is dropped after normalize completes; only the normalized clip is fed
    # to ComfyUI. ffmpeg is preinstalled by setup.sh.
    raw_video_bytes = await reference_video.read()
    # Defensive cap: anything beyond ~100MB is almost certainly someone
    # uploading a 4K phone clip we can't process inside Supabase's edge-
    # function body limit anyway. Reject early so the pod doesn't churn
    # ffmpeg on it for 60s only to fail downstream.
    REF_VIDEO_MAX_BYTES = 100 * 1024 * 1024
    if len(raw_video_bytes) > REF_VIDEO_MAX_BYTES:
        Path(img_path).unlink(missing_ok=True)
        raise HTTPException(
            413,
            f"reference video too large: {len(raw_video_bytes) // (1024*1024)} MB > 100 MB. "
            f"Trim to 15s or less and downscale to <1080p before upload.",
        )
    raw_video_ext = (reference_video.filename or "").lower().rsplit(".", 1)[-1] or "mp4"
    raw_video_path = str(INPUT_DIR / f"ltx_motion_ref_raw_{uuid.uuid4().hex}.{raw_video_ext}")
    Path(raw_video_path).write_bytes(raw_video_bytes)

    raw_video_file = Path(raw_video_path)
    try:
        reference_duration = await asyncio.to_thread(
            _probe_video_duration_seconds, raw_video_file,
        )
    except (ValueError, subprocess.SubprocessError) as exc:
        raw_video_file.unlink(missing_ok=True)
        Path(img_path).unlink(missing_ok=True)
        raise HTTPException(400, f"could not read reference video duration: {exc}")

    if match_reference_duration:
        target_frame_count = duration_to_ltx_frames(
            reference_duration,
            fps=MOTION_FPS,
            max_duration_seconds=max_duration_seconds,
        )
    else:
        max_frames = duration_to_ltx_frames(
            max_duration_seconds,
            fps=MOTION_FPS,
            max_duration_seconds=max_duration_seconds,
        )
        target_frame_count = min(snap_ltx_frame_count(length), max_frames)
    chunk_lengths = split_ltx_frame_count(
        target_frame_count,
        max_chunk_frames=MOTION_CHUNK_FRAMES,
    )

    # VideoHelperSuite is the fast loader. If it is unavailable, normalized
    # chunks are expanded into stock LoadImage/ImageBatch nodes; that fallback
    # now uses the exact same DWPose + IC-LoRA + crop graph.
    use_vhs = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            object_info = await client.get(f"{COMFYUI_URL}/object_info")
            use_vhs = (
                object_info.status_code == 200
                and "VHS_LoadVideo" in object_info.json()
            )
    except Exception as exc:
        print(f"[ltx/motion] object_info probe failed: {exc} — using frame extraction")

    cleanup_paths: list[str] = [img_path]
    chunk_specs: list[dict] = []
    created_paths: list[Path] = []
    elapsed_intervals = 0

    try:
        for index, chunk_length in enumerate(chunk_lengths):
            chunk_filename = f"ltx_motion_ref_{uuid.uuid4().hex}.mp4"
            chunk_path = INPUT_DIR / chunk_filename
            start_seconds = elapsed_intervals / MOTION_FPS
            normalize_command = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-stream_loop", "-1",
                "-ss", f"{start_seconds:.6f}",
                "-i", raw_video_path,
                "-vf", (
                    f"scale=w={width}:h={height}:force_original_aspect_ratio=decrease,"
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                    f"fps={MOTION_FPS}"
                ),
                "-frames:v", str(chunk_length),
                "-an",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "veryfast",
                str(chunk_path),
            ]
            normalized = await asyncio.to_thread(
                subprocess.run,
                normalize_command,
                capture_output=True,
                timeout=120,
            )
            if normalized.returncode != 0 or not chunk_path.exists():
                stderr = (normalized.stderr or b"").decode(errors="replace")[-500:]
                raise ValueError(stderr or "ffmpeg produced no normalized motion segment")
            created_paths.append(chunk_path)

            spec = {"length": chunk_length}
            if use_vhs:
                spec["video_filename"] = chunk_filename
                cleanup_paths.append(str(chunk_path))
            else:
                frame_id = uuid.uuid4().hex[:8]
                frame_pattern = INPUT_DIR / f"ltx_motion_frame_{frame_id}_%04d.png"
                extracted = await asyncio.to_thread(
                    subprocess.run,
                    [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-i", str(chunk_path),
                        "-vsync", "0",
                        str(frame_pattern),
                    ],
                    capture_output=True,
                    timeout=120,
                )
                if extracted.returncode != 0:
                    stderr = (extracted.stderr or b"").decode(errors="replace")[-500:]
                    raise ValueError(stderr or "could not extract normalized motion frames")
                frame_files = sorted(
                    INPUT_DIR.glob(f"ltx_motion_frame_{frame_id}_*.png")
                )
                if len(frame_files) != chunk_length:
                    raise ValueError(
                        f"motion segment {index + 1} produced {len(frame_files)} "
                        f"frames; expected {chunk_length}"
                    )
                spec["frame_filenames"] = [path.name for path in frame_files]
                cleanup_paths.extend(str(path) for path in frame_files)
                created_paths.extend(frame_files)
                chunk_path.unlink(missing_ok=True)

            chunk_specs.append(spec)
            elapsed_intervals += chunk_length - 1
    except subprocess.TimeoutExpired:
        for path in created_paths:
            path.unlink(missing_ok=True)
        raw_video_file.unlink(missing_ok=True)
        Path(img_path).unlink(missing_ok=True)
        raise HTTPException(
            408,
            "reference video normalization timed out — try a lower-resolution upload",
        )
    except (ValueError, OSError) as exc:
        for path in created_paths:
            path.unlink(missing_ok=True)
        raw_video_file.unlink(missing_ok=True)
        Path(img_path).unlink(missing_ok=True)
        raise HTTPException(400, f"could not normalize reference video: {exc}")

    if audio:
        cleanup_paths.append(raw_video_path)
        audio_source_path = raw_video_path
    else:
        raw_video_file.unlink(missing_ok=True)
        audio_source_path = None

    _reserve_video_job(job_id, cleanup_paths)
    background_tasks.add_task(
        run_motion_control_job,
        job_id,
        chunk_specs,
        img_filename,
        prompt,
        negative_prompt,
        width,
        height,
        seed,
        inplace_strength,
        motion_strength,
        cleanup_paths,
        audio_source_path=audio_source_path,
        watermark_text=watermark,
        watermark_image=watermark_image,
        reference_duration_seconds=reference_duration,
        target_frame_count=target_frame_count,
    )
    return {
        "job_id": job_id,
        "status": "queued",
        "model": "ltx-2.3-22b",
        **_job_links(job_id),
        "workflow_path": "vhs" if use_vhs else "frame-extract",
        "reference_duration_seconds": round(reference_duration, 3),
        "target_duration_seconds": round(target_frame_count / MOTION_FPS, 3),
        "target_frames": target_frame_count,
        "fps": MOTION_FPS,
        "segments": len(chunk_specs),
        "audio_source": "reference" if audio else "none",
        "note": (
            "Output duration follows the reference video up to 15 seconds. "
            "IC-LoRA guide padding is cropped before decode; no destructive "
            "post-generation trim is applied."
        ),
    }


# ─────────────────────────────────────────────
# /ltx/lipdub — supported Lightricks LipDub IC-LoRA workflow
# Direct port of LTX-2.3_ICLoRA_Lipdub_Two_Stage_Distilled.json.
# Inputs: reference video (speaker) + new dialogue text.
# Output: same speaker saying the new dialogue with synced lips +
# generated voice matching the source speaker's tone.
# ─────────────────────────────────────────────
@app.post("/ltx/lipdub")
async def ltx_lipdub(
    background_tasks: BackgroundTasks,
    reference_video: UploadFile = File(..., description="Source speaker video. Audio in this file is used as the voice reference (the output speaker will sound like them). Length determines output length — trim to the segment you want re-dubbed before upload."),
    prompt: str = Form(..., description="The NEW dialogue text. Include translated words directly (the model does NOT translate). Use native script (Cyrillic for Russian, Chinese for Chinese, etc). Match the LENGTH of the original dialogue for best results: too long → words skipped, too short → unnatural pauses."),
    negative_prompt: str = Form(LTX_DEFAULT_NEGATIVE),
    seed: int = Form(-1),
    reference_strength: float = Form(1.0, ge=0.0, le=2.0, description="LipDub IC-LoRA strength. 1.0 = Lightricks default (recommended). Lower if you want looser lip-sync. >1.0 increases adherence at the cost of identity blur."),
):
    """LTX 2.3 lip dubbing — re-sync a speaker's lips + voice to new dialogue.

    Pipeline:
      1. Save the uploaded reference video to ComfyUI input dir.
      2. Build the LipDub two-stage workflow (low-res sample → 2x
         upsample → high-res refine). The LipDub IC-LoRA was trained
         with reference_downscale_factor=1, so unlike Union-Control
         it doesn't have the temporal halving bug — output is fully
         conditioned end-to-end.
      3. Enqueue as a background job; client polls /status/<job_id>.

    The output video has the source speaker's appearance + voice
    timbre, but the lip movements and audio match the new prompt.
    """
    if not prompt.strip():
        raise HTTPException(400, "prompt is required (the new dialogue text)")

    # Save the reference video into ComfyUI's input dir so the
    # LoadVideo node can find it.
    raw_video_bytes = await reference_video.read()
    REF_VIDEO_MAX_BYTES = 100 * 1024 * 1024
    if len(raw_video_bytes) > REF_VIDEO_MAX_BYTES:
        raise HTTPException(
            413,
            f"reference video too large: {len(raw_video_bytes) // (1024*1024)} MB > 100 MB.",
        )
    raw_video_ext = (reference_video.filename or "").lower().rsplit(".", 1)[-1] or "mp4"
    ref_video_filename = f"ltx_lipdub_{uuid.uuid4().hex}.{raw_video_ext}"
    ref_video_path = str(INPUT_DIR / ref_video_filename)
    Path(ref_video_path).write_bytes(raw_video_bytes)

    # Seed handling — same convention as other LTX endpoints.
    if seed < 0:
        import random as _random
        seed = _random.randint(0, 2**32 - 1)

    workflow = build_ltx_lipdub_workflow(
        reference_video_filename=ref_video_filename,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        reference_strength=reference_strength,
    )

    job_id = str(uuid.uuid4())
    _reserve_video_job(job_id, [ref_video_path])
    background_tasks.add_task(
        run_job, job_id, workflow, [ref_video_path], None, False,
    )
    return {
        "job_id": job_id,
        "status": "queued",
        "model": "ltx-2.3-22b + LipDub IC-LoRA",
        **_job_links(job_id),
        "note": "Two-stage workflow (960×544 sample + 1920×1088 refine). Typical runtime ~2-4 min depending on source length.",
    }


# ─────────────────────────────────────────────
# Face Swap + Animate Pipeline
# ─────────────────────────────────────────────

async def _submit_and_wait_comfyui(workflow: dict, job_id: str | None = None) -> tuple[str, str]:
    """Submit a workflow to ComfyUI, wait for completion, return (filename, full_path)."""
    client_id = str(uuid.uuid4())
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow, "client_id": client_id})
        if resp.status_code != 200:
            raise RuntimeError(f"ComfyUI rejected workflow: {resp.text}")
        prompt_id = resp.json()["prompt_id"]
    if job_id:
        jobs[job_id] = {**jobs.get(job_id, {}), "prompt_id": prompt_id}

    job_data = await _wait_for_comfy_prompt(prompt_id, job_id=job_id)

    status = job_data.get("status", {}).get("status_str", "")
    if status == "error":
        for m in job_data.get("status", {}).get("messages", []):
            if m[0] == "execution_error":
                raise RuntimeError(m[1].get("exception_message", "ComfyUI execution error"))
        raise RuntimeError("ComfyUI execution error")

    for node_output in job_data.get("outputs", {}).values():
        for key in ["images", "videos", "gifs"]:
            if key in node_output:
                item = node_output[key][0]
                filename = item["filename"]
                subfolder = item.get("subfolder", "")
                path = OUTPUT_DIR / subfolder / filename if subfolder else OUTPUT_DIR / filename
                if path.exists():
                    return filename, str(path)

    raise RuntimeError("No output file found in ComfyUI history")


async def run_face_animate_pipeline(
    job_id: str,
    face_swap_workflow: dict,
    animate_prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    length: int,
    fps: int,
    seed: int,
    swap_cleanup_paths: list,
    preset: str = "fast",
    audio: bool = False,
    watermark_text: str | None = None,
    watermark_image: bool = False,
):
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()
    ltx_img_path = None

    try:
        # ── Step 1: Face swap ──
        jobs[job_id]["step"] = "face_swap"
        swap_filename, swap_img_path = await _submit_and_wait_comfyui(face_swap_workflow, job_id)

        # ── Step 2: Animate the swapped image ──
        jobs[job_id]["step"] = "animating"

        img_bytes = Path(swap_img_path).read_bytes()
        ltx_input_filename = f"face_animate_{uuid.uuid4().hex}.png"
        ltx_img_path = str(INPUT_DIR / ltx_input_filename)
        Path(ltx_img_path).write_bytes(img_bytes)

        two_pass = LTX_PRESETS[preset]["two_pass"]

        img_nodes = {
            "269": {"class_type": "LoadImage", "inputs": {"image": ltx_input_filename}},
            "238": {"class_type": "ResizeImageMaskNode", "inputs": {
                "input": ["269", 0], "resize_type": "scale dimensions",
                "resize_type.width": width, "resize_type.height": height,
                "resize_type.crop": "center", "scale_method": "lanczos"
            }},
            "235": {"class_type": "ResizeImagesByLongerEdge", "inputs": {"images": ["238", 0], "longer_edge": 1536}},
            "248": {"class_type": "LTXVPreprocess",           "inputs": {"image": ["235", 0], "img_compression": 18}},
            "274": {"class_type": "TextGenerateLTX2Prompt", "inputs": {
                "clip": ["272", 1], "image": ["269", 0], "prompt": animate_prompt,
                "max_length": 256, "sampling_mode": "on",
                "sampling_mode.temperature": 0.7, "sampling_mode.top_k": 64,
                "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.05,
                "sampling_mode.repetition_penalty": 1.05, "sampling_mode.seed": seed
            }},
            "249": {"class_type": "LTXVImgToVideoInplace", "inputs": {
                "vae": ["236", 2], "image": ["248", 0], "latent": ["228", 0],
                "strength": 0.7 if two_pass else 1.0, "bypass": False
            }},
            "240": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["243", 0], "text": ["274", 0]}},
        }

        if two_pass:
            img_nodes["230"] = {"class_type": "LTXVImgToVideoInplace", "inputs": {
                "vae": ["236", 2], "image": ["248", 0], "latent": ["253", 0], "strength": 1.0, "bypass": False
            }}
            high_res_src = ["230", 0]
        else:
            high_res_src = None

        ltx_workflow = ltx_base_nodes(
            animate_prompt, negative_prompt, width, height, length, fps, seed,
            low_res_video_src=["249", 0], high_res_video_src=high_res_src, prefix="face_animate", preset=preset, audio=audio
        )
        ltx_workflow.update(img_nodes)

        video_filename, video_full_path = await _submit_and_wait_comfyui(ltx_workflow, job_id)

        ext = Path(video_filename).suffix.lower()
        url = f"{BASE_URL}/video/{video_filename}" if ext not in [".png", ".jpg", ".jpeg", ".webp"] else f"{BASE_URL}/image/{video_filename}"

        # Optional watermarks — applied to the final video, not the
        # intermediate swap. Text + logo can both be set; they stack.
        watermark_warnings: list[str] = []
        if watermark_text and watermark is not None:
            try:
                watermark.apply(video_full_path, watermark_text)
            except Exception as wm_err:
                watermark_warnings.append(f"text: {wm_err}")
        if watermark_image and watermark is not None:
            try:
                watermark.apply_logo(video_full_path)
            except Exception as wm_err:
                watermark_warnings.append(f"image: {wm_err}")
        watermark_warning = " | ".join(watermark_warnings) if watermark_warnings else None

        # Thumbnail of the final (post-watermark) video. Skip for the rare
        # case where the workflow produced an image instead of a video.
        thumbnail_url = None
        if ext in _VIDEO_EXTS:
            thumb = _extract_video_thumbnail(Path(video_full_path))
            if thumb is not None:
                thumbnail_url = f"{BASE_URL}/image/{thumb.name}"

        completed_at = datetime.now(timezone.utc)
        created_at_str = jobs[job_id].get("created_at")
        duration_seconds = round((completed_at - datetime.fromisoformat(created_at_str)).total_seconds(), 1) if created_at_str else None

        result = {
            "status": "completed",
            "url": url,
            "filename": video_filename,
            "swap_filename": swap_filename,
            "completed_at": completed_at.isoformat(),
            "duration_seconds": duration_seconds,
        }
        if thumbnail_url:
            result["thumbnail_url"] = thumbnail_url
        if watermark_warning:
            result["watermark_warning"] = watermark_warning
        if jobs.get(job_id, {}).get("status") != "cancelled":
            jobs[job_id] = result

    except JobCancelled:
        jobs[job_id] = {**jobs.get(job_id, {}), "status": "cancelled"}
    except Exception as e:
        jobs[job_id] = {**jobs[job_id], "status": "failed", "error": str(e), "failed_at": datetime.now(timezone.utc).isoformat()}
    finally:
        for p in (swap_cleanup_paths or []):
            Path(p).unlink(missing_ok=True)
        if ltx_img_path:
            Path(ltx_img_path).unlink(missing_ok=True)


@app.post("/face-animate")
async def face_animate(
    background_tasks: BackgroundTasks,
    target_image: UploadFile = File(..., description="Template/body photo — head gets replaced"),
    face_image: UploadFile = File(..., description="User's face photo — identity to transfer"),
    animate_prompt: str = Form(..., description="Describes the motion/scene for the video"),
    swap_prompt: str = Form("", description="Prompt for the face swap step (uses smart default if empty)"),
    negative_prompt: str = Form(LTX_DEFAULT_NEGATIVE),
    preset: str = Form("fast", description="Speed/quality preset for video: realtime (4 steps), fast (8 steps), or quality (8+3 steps with 2× spatial upscale)"),
    aspect_ratio: str = Form("16:9", description="Output video aspect ratio: 16:9 | 9:16 | 1:1 | 4:3 | 3:4 | 3:2 | 2:3 | 21:9 | 9:21 | original"),
    width: int = Form(1280, description="Output width in pixels (height auto-derived from aspect_ratio)"),
    height: int = Form(720, description="Output height — used only when aspect_ratio=original"),
    length_seconds: float = Form(5.0, description="Video duration in seconds"),
    fps: int = Form(24, description="Frames per second"),
    seed: int = Form(-1),
    megapixels: float = Form(2.0, description="Face swap resolution in megapixels (0.5–4.0)"),
    lora_strength: float = Form(1.0, description="BFS LoRA strength for face swap (0.5–1.0)"),
    swap_steps: int = Form(4),
    swap_guidance: float = Form(4.0),
    audio: bool = Form(False, description="Generate audio track with the video (adds overhead)"),
    watermark: str | None = Form(None, description="Optional text to overlay at the bottom-right of the final video (e.g. 'AI'). Null/empty = no watermark. Re-encodes via ffmpeg (~1-3s for a 5s clip)."),
    watermark_image: bool = Form(False, description="Composite the Metfone GenAI logo (loaded once from /workspace/assets/metfone_genai_watermark.png) at the bottom-right of the final video. Stacks with `watermark` if both are set."),
    require_detectable_face: bool = Form(False, description="Opt-in input validation. When true, reject the user's face image unless at least one clear face is detectable. Default false."),
):
    if preset not in LTX_PRESETS:
        raise HTTPException(400, f"Invalid preset '{preset}'. Valid: {', '.join(LTX_PRESETS)}")
    if aspect_ratio != "original" and aspect_ratio not in LTX_ASPECT_RATIOS:
        raise HTTPException(400, f"Invalid aspect_ratio. Valid: original, {', '.join(LTX_ASPECT_RATIOS)}")

    seed = seed if seed != -1 else uuid.uuid4().int % 2**32
    width, height = compute_ltx_dimensions(width, height, aspect_ratio)
    length = max(25, round(length_seconds * fps))

    target_bytes = await target_image.read()
    face_bytes = await face_image.read()
    _require_detectable_face(
        "/face-animate", require_detectable_face, [(face_bytes, "face_image")],
    )

    # Pre-crop target image to match output aspect ratio for face swap
    if aspect_ratio != "original":
        w_r, h_r = LTX_ASPECT_RATIOS[aspect_ratio]
        swap_w, swap_h = compute_dimensions(w_r, h_r, megapixels)
        target_bytes = crop_to_aspect(target_bytes, swap_w, swap_h)

    target_filename = f"fa_target_{uuid.uuid4().hex}.png"
    face_filename = f"fa_face_{uuid.uuid4().hex}.png"
    target_path = str(INPUT_DIR / target_filename)
    face_path = str(INPUT_DIR / face_filename)
    Path(target_path).write_bytes(target_bytes)
    Path(face_path).write_bytes(face_bytes)

    face_swap_workflow = get_flux_face_swap_workflow(
        target_filename, face_filename, seed,
        prompt=swap_prompt or None,
        megapixels=megapixels, steps=swap_steps, cfg=1.0,
        guidance=swap_guidance, lora_strength=lora_strength,
    )

    job_id = str(uuid.uuid4())
    _reserve_video_job(job_id, [target_path, face_path])
    background_tasks.add_task(
        run_face_animate_pipeline,
        job_id, face_swap_workflow, animate_prompt, negative_prompt,
        width, height, length, fps, seed,
        [target_path, face_path], preset, audio, watermark, watermark_image,
    )
    return {
        "job_id": job_id,
        "status": "queued",
        "model": "flux2-klein-9b + ltx-2.3-22b",
        "pipeline": ["face_swap", "image_to_video"],
        **_job_links(job_id),
    }


@app.post("/flux/face-swap")
async def flux_face_swap(
    background_tasks: BackgroundTasks,
    target_image: UploadFile = File(..., description="Base/template image — body stays, head gets replaced"),
    face_image: UploadFile = File(..., description="Source face — identity to transfer"),
    seed: int = Form(-1),
    megapixels: float = Form(2.0, description="Total output resolution in megapixels (0.5–4.0)"),
    aspect_ratio: str = Form("original", description="Output aspect ratio: original | 1:1 | 16:9 | 9:16 | 4:3 | 3:4 | 3:2 | 2:3 | 21:9 | 9:21"),
    steps: int = Form(4),
    cfg: float = Form(1.0),
    guidance: float = Form(4.0),
    lora_strength: float = Form(1.0),
    face_filter: bool = Form(True, description="Reject the request if either input image matches a face in /workspace/blocklist/. ON by default — clients must explicitly pass face_filter=false to skip (and the proxies/edge functions always force True so this default only matters for direct pod callers)."),
    logo_filter: bool = Form(True, description="Reject the request if either input image matches a logo/flag in /workspace/blocklist_logos/. ON by default — same defense-in-depth rationale as face_filter."),
    watermark: str | None = Form(None, description="Optional text to overlay at the bottom-right of the output (e.g. 'AI'). Null/empty = no watermark."),
    watermark_image: bool = Form(False, description="Composite the Metfone GenAI logo (loaded once from /workspace/assets/metfone_genai_watermark.png) at the bottom-right. Stacks with `watermark` if both are set."),
    caption: str | None = Form(None, description="Optional styled lower-third caption (e.g. the horoscope of the day). Word-wrapped, centered white text with a heavy black outline; videos fade it in ~1s after the start. Same fixed design on images and videos. Null/empty = no caption."),
    caption_icon: str | None = Form(None, description="Optional zodiac sign for the caption (aries|taurus|gemini|cancer|leo|virgo|libra|scorpio|sagittarius|capricorn|aquarius|pisces). When set alongside `caption`, a gold zodiac glyph + divider are stacked above the text. Ignored if not a recognised sign."),
    caption_fade: bool = Form(True, description="Video only: when true (default) the caption fades in ~1s after the start; set false to show it from the very first frame. No effect on images (their caption is always immediate)."),
    background_music: bool = Form(False, description="Video only: mux a looping royalty-free background-music bed (/workspace/assets/horoscope_bgm.m4a) under the clip, trimmed to length with a soft fade. No effect on images."),
    refine_face: bool = Form(False, description="Opt-in 2nd-pass face detailer. After the swap, detect the largest face, re-render it at high resolution, and composite it back — fixes soft/low-detail faces in full-body or wide templates where the face is small in frame. Adds ~30-50s. Default false (no behaviour change for existing callers)."),
    preserve_body: bool = Form(False, description="Opt-in head-only mode. Composite the swapped head onto the ORIGINAL template so the body, clothing, pose, lighting and background stay the template's EXACT pixels — only the head changes (the base swap regenerates the whole frame, which drifts). Default false."),
    require_detectable_face: bool = Form(False, description="Opt-in input validation. When true, reject the user's face image unless at least one clear face is detectable. Default false."),
):
    seed = seed if seed != -1 else uuid.uuid4().int % 2**32

    # Validate aspect_ratio
    if aspect_ratio != "original" and aspect_ratio not in ASPECT_RATIOS:
        raise HTTPException(400, f"Invalid aspect_ratio '{aspect_ratio}'. Valid values: original, {', '.join(ASPECT_RATIOS)}")

    target_bytes = await target_image.read()
    face_bytes = await face_image.read()
    _require_detectable_face(
        "/flux/face-swap", require_detectable_face, [(face_bytes, "face_image")],
    )

    # Compliance filters — must run before any heavy work, before writing to disk
    job_id = str(uuid.uuid4())
    inputs = [(target_bytes, "target_image"), (face_bytes, "face_image")]
    _apply_face_filter("flux/face-swap", job_id, face_filter, inputs)
    _apply_logo_filter("flux/face-swap", job_id, logo_filter, inputs)

    # If aspect ratio is specified, crop target image to that ratio before sending to ComfyUI.
    # The workflow's ImageScaleToTotalPixels + GetImageSize will then produce output at that AR.
    if aspect_ratio != "original":
        w_ratio, h_ratio = ASPECT_RATIOS[aspect_ratio]
        target_w, target_h = compute_dimensions(w_ratio, h_ratio, megapixels)
        target_bytes = crop_to_aspect(target_bytes, target_w, target_h)

    target_filename = f"flux_target_{uuid.uuid4().hex}.png"
    face_filename = f"flux_face_{uuid.uuid4().hex}.png"
    target_path = str(INPUT_DIR / target_filename)
    face_path = str(INPUT_DIR / face_filename)
    Path(target_path).write_bytes(target_bytes)
    Path(face_path).write_bytes(face_bytes)

    workflow = get_flux_face_swap_workflow(target_filename, face_filename, seed, megapixels=megapixels, steps=steps, cfg=cfg, guidance=guidance, lora_strength=lora_strength)

    jobs[job_id] = {"status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    # Output-side filter runs AFTER the swap completes — catches the case
    # where neither input matched but the swapped output ended up looking
    # like a blocked identity (e.g. LoRA drift in face-swap mode).
    background_tasks.add_task(
        run_job, job_id, workflow, [target_path, face_path], watermark, watermark_image,
        output_face_filter=face_filter, output_logo_filter=logo_filter,
        output_endpoint="/flux/face-swap", caption=caption, caption_icon=caption_icon, caption_fade=caption_fade, background_music=background_music,
        refine_face=refine_face, refine_face_filename=face_filename,
        refine_megapixels=megapixels, refine_steps=steps, refine_cfg=cfg,
        refine_guidance=guidance, refine_lora=lora_strength,
        preserve_body=preserve_body, preserve_body_template=target_path,
    )
    return {"job_id": job_id, "status": "queued", "model": "flux2-klein-9b", **_job_links(job_id)}


# ─────────────────────────────────────────────
# FLUX.2 Klein 9B Multi-Person Face Swap
# One template + one or two independently mapped user identities.
# ─────────────────────────────────────────────

@app.post("/flux/multi-face-swap")
async def flux_multi_face_swap(
    background_tasks: BackgroundTasks,
    target_image: UploadFile = File(..., description="Base/template image containing the people to personalize"),
    face_images: list[UploadFile] = File(..., description="One or two source face photos, repeated in mapping order"),
    face_order: str = Form("left-to-right", description="Target-person ordering: left-to-right | right-to-left | top-to-bottom | bottom-to-top | largest-first"),
    target_face_indices: str = Form("", description="Comma-separated zero-based target slots aligned with face_images, e.g. 1 for the second person or 0,1 for both"),
    prompt: str | None = Form(None, description="Optional template-specific instruction appended to the protected identity-mapping prompt"),
    seed: int = Form(-1),
    megapixels: float = Form(2.0, description="Total output resolution in megapixels (0.5–4.0)"),
    aspect_ratio: str = Form("original", description="Output aspect ratio: original | 1:1 | 16:9 | 9:16 | 4:3 | 3:4 | 3:2 | 2:3 | 21:9 | 9:21"),
    steps: int = Form(4),
    cfg: float = Form(1.0),
    guidance: float = Form(4.0),
    lora_strength: float = Form(1.0),
    face_filter: bool = Form(True, description="Reject blocked identities in user uploads and the generated output"),
    logo_filter: bool = Form(True, description="Reject blocked logos/flags in inputs and the generated output"),
    watermark: str | None = Form(None, description="Optional text watermark for the generated image"),
    watermark_image: bool = Form(False, description="Optionally add the configured logo watermark"),
    require_detectable_face: bool = Form(True, description="Reject each face_images upload unless it contains a clear human face"),
):
    if len(face_images) not in (1, 2):
        raise HTTPException(422, detail={
            "error": "invalid_face_count",
            "reason": f"face_images must contain 1 or 2 files; received {len(face_images)}.",
            "received": len(face_images),
            "minimum": 1,
            "maximum": 2,
        })
    if face_order not in MULTI_FACE_SWAP_ORDERS:
        raise HTTPException(400, detail={
            "error": "invalid_face_order",
            "reason": f"Invalid face_order '{face_order}'.",
            "valid_values": list(MULTI_FACE_SWAP_ORDERS),
        })
    try:
        parsed_target_indices = normalize_target_face_indices(
            target_face_indices,
            len(face_images),
        )
    except ValueError as exc:
        raise HTTPException(422, detail={
            "error": "invalid_target_face_indices",
            "reason": str(exc),
        }) from exc
    if aspect_ratio != "original" and aspect_ratio not in ASPECT_RATIOS:
        raise HTTPException(
            400,
            f"Invalid aspect_ratio '{aspect_ratio}'. Valid values: original, {', '.join(ASPECT_RATIOS)}",
        )
    if not 0.5 <= megapixels <= 4.0:
        raise HTTPException(400, "megapixels must be between 0.5 and 4.0")
    if prompt is not None and len(prompt) > 2000:
        raise HTTPException(400, "prompt must be 2000 characters or fewer")

    target_bytes = await target_image.read()
    face_bytes_list = [await image.read() for image in face_images]
    if not target_bytes:
        raise HTTPException(422, detail={
            "error": "empty_upload",
            "reason": "target_image is empty.",
        })
    for index, face_bytes in enumerate(face_bytes_list):
        if not face_bytes:
            raise HTTPException(422, detail={
                "error": "empty_upload",
                "reason": f"face_images[{index}] is empty.",
                "image_index": index,
            })

    job_id = str(uuid.uuid4())
    face_inputs = [
        (face_bytes, f"face_images[{index}]")
        for index, face_bytes in enumerate(face_bytes_list)
    ]
    all_inputs = [(target_bytes, "target_image"), *face_inputs]
    _require_detectable_face(
        "/flux/multi-face-swap",
        require_detectable_face,
        face_inputs,
    )
    _apply_face_filter(
        "/flux/multi-face-swap",
        job_id,
        face_filter,
        all_inputs,
    )
    _apply_logo_filter(
        "/flux/multi-face-swap",
        job_id,
        logo_filter,
        all_inputs,
    )

    if aspect_ratio != "original":
        w_ratio, h_ratio = ASPECT_RATIOS[aspect_ratio]
        target_w, target_h = compute_dimensions(w_ratio, h_ratio, megapixels)
        target_bytes = crop_to_aspect(target_bytes, target_w, target_h)

    seed = seed if seed != -1 else uuid.uuid4().int % 2**32
    target_filename = f"flux_multi_target_{uuid.uuid4().hex}.png"
    face_filenames = [
        f"flux_multi_face_{uuid.uuid4().hex}_{index}.png"
        for index in range(len(face_bytes_list))
    ]
    target_path = INPUT_DIR / target_filename
    face_paths = [INPUT_DIR / filename for filename in face_filenames]
    target_path.write_bytes(target_bytes)
    for path, face_bytes in zip(face_paths, face_bytes_list):
        path.write_bytes(face_bytes)

    workflow = build_flux_multi_face_swap_workflow(
        target_filename,
        face_filenames,
        seed,
        face_order=face_order,
        prompt=prompt,
        megapixels=megapixels,
        steps=steps,
        cfg=cfg,
        guidance=guidance,
        lora_strength=lora_strength,
        target_face_indices=parsed_target_indices,
    )

    jobs[job_id] = {
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "face_count": len(face_images),
        "face_order": face_order,
        "target_face_indices": parsed_target_indices,
    }
    background_tasks.add_task(
        run_job,
        job_id,
        workflow,
        [str(target_path), *[str(path) for path in face_paths]],
        watermark,
        watermark_image,
        output_face_filter=face_filter,
        output_logo_filter=logo_filter,
        output_endpoint="/flux/multi-face-swap",
        preserve_face_selection=True,
        preserve_face_template=str(target_path),
        preserve_face_order=face_order,
        preserve_face_indices=parsed_target_indices,
    )
    return {
        "job_id": job_id,
        "status": "queued",
        "model": "flux2-klein-9b",
        "face_count": len(face_images),
        "face_order": face_order,
        "target_face_indices": parsed_target_indices,
        **_job_links(job_id),
    }


# ─────────────────────────────────────────────
# FLUX.2 Klein 9B Image-to-Image (multi-reference editing)
# Up to 5 reference images — each one feeds a ReferenceLatent chained
# onto the prompt's conditioning. Output dimensions default to the first
# image's (rescaled) size, or override via width/height.
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# /flux/i2i composition modes (additive — see API.md "Composition modes")
#
# Each mode is a pre-baked prompt template + a recommended lora_strength
# tuned for that use case. When `composition_mode` is left at the default
# `"none"`, none of this fires — the existing /flux/i2i behavior is
# preserved bit-for-bit (caller's prompt is required, caller's
# lora_strength wins).
#
# When set, the mode supplies a template prompt + LoRA default the caller
# would have had to write themselves. The caller's `prompt` and
# `lora_strength`, if explicitly provided, always win — modes only fill
# in blanks.
# ─────────────────────────────────────────────

_I2I_MODE_PROMPTS: dict[str, str] = {
    "auto": (
        "high quality detailed composition of the reference images, "
        "photorealistic, sharp, natural lighting"
    ),
    "scene_blend": (
        "the subject(s) from the reference images placed naturally in the "
        "scene shown in the first image, matched lighting, integrated "
        "shadows, photorealistic, sharp focus, detailed environment"
    ),
    "outfit_swap": (
        "the person from the first image wearing the outfit shown in the "
        "second image, full body, photorealistic, natural lighting, "
        "detailed fabric texture"
    ),
    "style_transfer": (
        "the first image reimagined in the artistic style of the second "
        "image, preserving composition and subject"
    ),
}

# Default lora_strength per mode. Only applied when the caller didn't pass
# an explicit value (sentinel: lora_strength = -1).
_I2I_MODE_LORA: dict[str, float] = {
    "auto": 0.0,
    "scene_blend": 0.5,
    "outfit_swap": 0.7,
    "style_transfer": 0.0,
}

_I2I_QUALITY_PRESET_STEPS: dict[str, int] = {
    "fast": 4,
    "balanced": 8,
    "high": 12,
}


def _resolve_i2i_config(
    *,
    composition_mode: str,
    prompt: str,
    lora_strength: float,
    steps: int,
    quality_preset: str,
    scene_image_index: int,
    n_images: int,
) -> tuple[str, float, int, list[int]]:
    """Translate the public knobs into the final (prompt, lora, steps,
    image_order) tuple the workflow builder consumes.

    `composition_mode = "none"` (the default) means: no mode logic — return
    the caller's values verbatim, leave image order untouched. This is
    what lets us add the feature without changing existing callers.

    For any other mode:
    * If `prompt` is empty, substitute the mode's template prompt.
    * If `lora_strength < 0` (sentinel), substitute the mode's default.
    * If `quality_preset` is one of fast/balanced/high, use its step
      count, overriding the `steps` argument.
    * For `scene_blend` with 2+ images, reorder so the scene image (last
      by default, or the explicit `scene_image_index`) becomes the FLUX
      canvas (index 0).
    """
    # Auto mode if caller asked for no template but also didn't send a prompt.
    effective_mode = composition_mode
    if effective_mode == "none" and not prompt.strip():
        effective_mode = "auto"

    # Prompt: caller wins, then template, then auto fallback.
    if prompt.strip():
        final_prompt = prompt
    else:
        final_prompt = _I2I_MODE_PROMPTS.get(effective_mode) or _I2I_MODE_PROMPTS["auto"]

    # LoRA: explicit (>=0) wins, else mode default, else 0.
    if lora_strength >= 0:
        final_lora = lora_strength
    else:
        final_lora = _I2I_MODE_LORA.get(effective_mode, 0.0)

    # Steps: preset wins when one is selected, else caller's `steps`.
    if quality_preset in _I2I_QUALITY_PRESET_STEPS:
        final_steps = _I2I_QUALITY_PRESET_STEPS[quality_preset]
    else:
        final_steps = steps

    # Image order: only scene_blend reshuffles, and only when we have
    # something to reshuffle. The scene image becomes the canvas (index 0).
    indices = list(range(n_images))
    if effective_mode == "scene_blend" and n_images >= 2:
        # `-1` (the default) means "last image", which matches how the
        # frontend uploads user photos first and the library scene last.
        scene_idx = scene_image_index if scene_image_index >= 0 else n_images - 1
        scene_idx = max(0, min(scene_idx, n_images - 1))
        if scene_idx != 0:
            indices = [scene_idx] + [i for i in indices if i != scene_idx]

    return final_prompt, final_lora, final_steps, indices


@app.post("/flux/i2i")
async def flux_image_to_image(
    background_tasks: BackgroundTasks,
    images: list[UploadFile] = File(..., description="1 to 5 reference images. The first one's dimensions (after rescale) are used as the output canvas unless width/height are set."),
    prompt: str = Form("", description="What to do — edit instructions. Optional when `composition_mode` is set (server fills in a mode-specific template)."),
    seed: int = Form(-1),
    megapixels: float = Form(2.0, description="Resolution per reference image in megapixels (0.5–4.0)"),
    width: int = Form(0, description="Output width — 0 (default) means: derive from the first image"),
    height: int = Form(0, description="Output height — 0 (default) means: derive from the first image"),
    steps: int = Form(4, description="Inference steps. 4 is fine for FLUX Klein. Overridden by `quality_preset` when set."),
    cfg: float = Form(1.0),
    guidance: float = Form(4.0),
    lora_strength: float = Form(-1.0, description="Apply the head-swap LoRA. 0 = off (general edits). 0.5–1.0 for face/head-focused edits. -1 (default) = use the mode's recommended value (0 for `none`/`auto`, 0.5 for `scene_blend`, 0.7 for `outfit_swap`).", ge=-1.0, le=1.5),
    composition_mode: str = Form("none", description="Pre-baked prompt + LoRA preset for prompt-less callers. `none` (default) = no template, behaves like before. `auto` | `scene_blend` | `outfit_swap` | `style_transfer` = use that mode's template. See API.md → Composition modes."),
    quality_preset: str = Form("none", description="`none` (default) = use `steps` directly. `fast` = 4 steps, `balanced` = 8 steps, `high` = 12 steps. Overrides `steps` when set."),
    scene_image_index: int = Form(-1, description="For `composition_mode=scene_blend` only: which input image is the scene/canvas. -1 (default) = last image, which matches the typical 'user uploads first, library scene last' UI flow. Ignored for other modes.", ge=-1, le=4),
    face_filter: bool = Form(True, description="Reject if any input image matches a face in /workspace/blocklist/. ON by default — clients must explicitly pass false to skip. Proxies/edge functions always force True so this default only matters for direct pod callers."),
    logo_filter: bool = Form(True, description="Reject if any input image matches a logo/flag in /workspace/blocklist_logos/. ON by default — same defense-in-depth rationale as face_filter."),
    watermark: str | None = Form(None, description="Optional text to overlay at the bottom-right of the output (e.g. 'AI'). Null/empty = no watermark."),
    watermark_image: bool = Form(False, description="Composite the Metfone GenAI logo (loaded once from /workspace/assets/metfone_genai_watermark.png) at the bottom-right. Stacks with `watermark` if both are set."),
    caption: str | None = Form(None, description="Optional styled lower-third caption (e.g. the horoscope of the day). Word-wrapped, centered white text with a heavy black outline; videos fade it in ~1s after the start. Same fixed design on images and videos. Null/empty = no caption."),
    caption_icon: str | None = Form(None, description="Optional zodiac sign for the caption (aries|taurus|gemini|cancer|leo|virgo|libra|scorpio|sagittarius|capricorn|aquarius|pisces). When set alongside `caption`, a gold zodiac glyph + divider are stacked above the text. Ignored if not a recognised sign."),
    caption_fade: bool = Form(True, description="Video only: when true (default) the caption fades in ~1s after the start; set false to show it from the very first frame. No effect on images (their caption is always immediate)."),
    background_music: bool = Form(False, description="Video only: mux a looping royalty-free background-music bed (/workspace/assets/horoscope_bgm.m4a) under the clip, trimmed to length with a soft fade. No effect on images."),
    require_detectable_face: bool = Form(False, description="Opt-in input validation. For scene_blend, all non-scene input images must contain a clear face; for other modes, the first image must. Default false."),
):
    if not 1 <= len(images) <= 5:
        raise HTTPException(400, f"images must be 1–5 files, got {len(images)}")

    # Validate mode-ish inputs early so a typo doesn't silently behave as
    # `none`/`steps` (which would mask the bug for the caller).
    valid_modes = {"none", *_I2I_MODE_PROMPTS}
    if composition_mode not in valid_modes:
        raise HTTPException(
            400,
            f"composition_mode must be one of {sorted(valid_modes)}, got {composition_mode!r}",
        )
    valid_presets = {"none", *_I2I_QUALITY_PRESET_STEPS}
    if quality_preset not in valid_presets:
        raise HTTPException(
            400,
            f"quality_preset must be one of {sorted(valid_presets)}, got {quality_preset!r}",
        )

    seed = seed if seed != -1 else uuid.uuid4().int % 2**32

    final_prompt, final_lora, final_steps, image_order = _resolve_i2i_config(
        composition_mode=composition_mode,
        prompt=prompt,
        lora_strength=lora_strength,
        steps=steps,
        quality_preset=quality_preset,
        scene_image_index=scene_image_index,
        n_images=len(images),
    )

    image_bytes_list: list[bytes] = []
    for up in images:
        image_bytes_list.append(await up.read())

    # Validate against the original upload order. In scene_blend mode the
    # scene image is a background/template and is intentionally excluded.
    if composition_mode == "scene_blend":
        scene_idx = len(image_bytes_list) - 1 if scene_image_index == -1 else scene_image_index
        face_inputs = [
            (img_bytes, f"images[{idx}]")
            for idx, img_bytes in enumerate(image_bytes_list)
            if idx != scene_idx
        ]
    else:
        face_inputs = [(image_bytes_list[0], "images[0]")]
    _require_detectable_face("/flux/i2i", require_detectable_face, face_inputs)

    # Apply the mode-driven image reorder (scene_blend only; identity
    # otherwise). The blocklist filters and the workflow both see the
    # post-reorder list so logging / canvas selection stay consistent.
    image_bytes_list = [image_bytes_list[i] for i in image_order]

    job_id = str(uuid.uuid4())
    labeled = [(b, f"images[{i}]") for i, b in enumerate(image_bytes_list)]
    _apply_face_filter("flux/i2i", job_id, face_filter, labeled)
    _apply_logo_filter("flux/i2i", job_id, logo_filter, labeled)

    # Save uploads to ComfyUI's input dir (only after filter passes)
    input_filenames: list[str] = []
    cleanup_paths: list[str] = []
    for idx, img_bytes in enumerate(image_bytes_list):
        fn = f"flux_i2i_{uuid.uuid4().hex}_{idx}.png"
        p = str(INPUT_DIR / fn)
        Path(p).write_bytes(img_bytes)
        input_filenames.append(fn)
        cleanup_paths.append(p)

    workflow = build_flux_i2i_workflow(
        input_filenames, final_prompt, seed,
        megapixels=megapixels,
        output_width=width, output_height=height,
        steps=final_steps, cfg=cfg, guidance=guidance,
        lora_strength=final_lora,
    )

    jobs[job_id] = {"status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    # Output-side filter runs AFTER the edit completes — catches the case
    # where the prompt morphs an input face toward a blocked identity even
    # though the unedited input didn't match.
    background_tasks.add_task(
        run_job, job_id, workflow, cleanup_paths, watermark, watermark_image,
        output_face_filter=face_filter, output_logo_filter=logo_filter,
        output_endpoint="/flux/i2i", caption=caption, caption_icon=caption_icon, caption_fade=caption_fade, background_music=background_music,
    )
    return {
        "job_id": job_id,
        "status": "queued",
        "model": "flux2-klein-9b",
        "ref_count": len(images),
        "composition_mode": composition_mode,
        "resolved": {
            # Surfaced so callers can confirm what the server decided when
            # they passed prompt-less / mode-only requests.
            "prompt_used": final_prompt[:120] + ("…" if len(final_prompt) > 120 else ""),
            "lora_strength": final_lora,
            "steps": final_steps,
        },
        **_job_links(job_id),
    }


# ─────────────────────────────────────────────
# Admin API — manage the face-filter blocklist
#
# The blocklist lives on the network volume at /workspace/blocklist/ — one
# image per blocked identity, filename (minus extension) is the identity
# name returned in block responses. Hot-reloaded by safety.py on every
# face-filter check, so changes take effect immediately for the pod AND
# for any serverless workers mounted on the same volume.
#
# Auth: every admin endpoint requires `Authorization: Bearer <ADMIN_TOKEN>`.
# ADMIN_TOKEN is read from the env at request time, so rotating it doesn't
# require a restart. If ADMIN_TOKEN is unset, all admin endpoints return
# 503 — this is a feature (no accidental open admin).
# ─────────────────────────────────────────────

BLOCKLIST_DIR = Path(os.environ.get("BLOCKLIST_DIR", "/workspace/blocklist"))
ALLOWED_BLOCKLIST_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
IDENTITY_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def _require_admin(authorization):
    """Admin auth is optional. Behavior depends on the ADMIN_TOKEN env var:
      - ADMIN_TOKEN unset     → admin endpoints are OPEN (no auth required).
                                Convenient for dev / when the pod URL isn't shared.
      - ADMIN_TOKEN set       → caller MUST send `Authorization: Bearer <token>`.
                                401 without header, 403 with wrong token.
    Set the env var on the RunPod template when going to production."""
    token = os.environ.get("ADMIN_TOKEN")
    if not token:
        return  # open mode — no token configured
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing Authorization: Bearer <ADMIN_TOKEN> header")
    if authorization[len("Bearer "):] != token:
        raise HTTPException(403, "invalid admin token")


def _validate_identity(identity: str) -> None:
    if not IDENTITY_NAME_PATTERN.match(identity):
        raise HTTPException(400, "identity must match [A-Za-z0-9_-]{1,64} — no spaces, no path separators")


def _find_existing_blocklist_file(identity: str):
    for ext in ALLOWED_BLOCKLIST_EXTS:
        p = BLOCKLIST_DIR / f"{identity}{ext}"
        if p.exists():
            return p
    return None


def _blocklist_count() -> int:
    if not BLOCKLIST_DIR.is_dir():
        return 0
    return sum(1 for p in BLOCKLIST_DIR.iterdir()
               if p.is_file() and p.suffix.lower() in ALLOWED_BLOCKLIST_EXTS)


def _detect_comfy_root() -> str:
    """Mirror setup.sh's COMFY_ROOT detection so admin endpoints look at
    the same custom_nodes/ that setup.sh + start_comfy.sh use. Order:
      1. COMFY_ROOT env var (when start_api.sh exported it)
      2. /workspace/api/config.env — setup.sh writes COMFY_ROOT here
      3. Known image locations (/workspace/runpod-slim/ComfyUI,
         /workspace/ComfyUI, /ComfyUI, /app/ComfyUI, /root/ComfyUI)
      4. Filesystem scan for any */ComfyUI/main.py under /workspace and /
    """
    import os
    env = os.environ.get("COMFY_ROOT")
    if env and Path(env).is_dir():
        return env

    # setup.sh writes its detected path here — most reliable source.
    cfg_path = Path("/workspace/api/config.env")
    if cfg_path.is_file():
        try:
            for line in cfg_path.read_text().splitlines():
                if line.startswith("COMFY_ROOT="):
                    candidate = line.split("=", 1)[1].strip().strip("'\"")
                    if candidate and Path(candidate).is_dir():
                        return candidate
        except Exception:
            pass

    for candidate in (
        "/workspace/runpod-slim/ComfyUI",
        "/workspace/ComfyUI",
        "/ComfyUI",
        "/app/ComfyUI",
        "/root/ComfyUI",
    ):
        if Path(candidate).is_dir():
            return candidate

    # Broader filesystem scan — caps depth so we don't recurse into models/.
    for root in ("/workspace", "/"):
        try:
            for p in Path(root).glob("**/ComfyUI/main.py"):
                # main.py at <X>/ComfyUI/main.py means X/ComfyUI is the root.
                return str(p.parent)
        except Exception:
            continue

    return "/workspace/ComfyUI"  # last-resort fallback so callers see a path


@app.get("/admin/comfy-status")
async def admin_comfy_status(authorization: str = Header(default=None)):
    """Read-only introspection: which custom_nodes dirs are present, and
    which node class types is the running ComfyUI actually exposing?

    Use this when a workflow fails with "Node 'X' not found" — compare
    the file-system snapshot (what setup.sh produced) to the loaded-node
    snapshot (what ComfyUI sees). A node directory that exists on disk
    but is missing from object_info means the load failed silently —
    usually a Python import error in the custom node's __init__.py
    (missing pip dep is the common culprit). The trailing tail of
    /workspace/setup-vhs.log surfaces the install-step trace for VHS
    specifically, which has been the recurring offender.
    """
    _require_admin(authorization)

    comfy_root = _detect_comfy_root()
    nodes_dir = Path(comfy_root) / "custom_nodes"
    nodes_on_disk: list[dict] = []
    if nodes_dir.is_dir():
        for entry in sorted(nodes_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            nodes_on_disk.append({
                "name": entry.name,
                "has_init": (entry / "__init__.py").is_file(),
                "has_requirements": (entry / "requirements.txt").is_file(),
                "has_git": (entry / ".git").is_dir(),
            })

    loaded_nodes: list[str] = []
    object_info_error: str | None = None
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{COMFYUI_URL}/object_info")
            if resp.status_code == 200:
                loaded_nodes = sorted(resp.json().keys())
            else:
                object_info_error = f"ComfyUI /object_info → HTTP {resp.status_code}"
    except Exception as e:
        object_info_error = f"could not reach ComfyUI: {e}"

    # Surface just the names we care about so the admin doesn't have to grep
    # through 800 stock node types. Add to this list as we depend on more
    # custom nodes.
    interesting = ["VHS_LoadVideo", "VHS_VideoCombine", "LTXVImgToVideoInplace",
                   "LTXVPreprocess", "ColorMatch", "LatentMultiply",
                   "VAEEncode", "VAEDecode", "LoadImage"]
    interesting_status = {name: name in set(loaded_nodes) for name in interesting}

    vhs_log_tail: list[str] = []
    vhs_log = Path("/workspace/setup-vhs.log")
    if vhs_log.is_file():
        try:
            with vhs_log.open() as f:
                lines = f.readlines()
            vhs_log_tail = [ln.rstrip("\n") for ln in lines[-30:]]
        except Exception:
            pass

    return {
        "comfy_root": comfy_root,
        "comfy_object_info_error": object_info_error,
        "custom_nodes_on_disk": nodes_on_disk,
        "loaded_node_count": len(loaded_nodes),
        "key_nodes_loaded": interesting_status,
        "vhs_install_log_tail": vhs_log_tail,
    }


@app.post("/admin/install-comfy-node")
async def admin_install_comfy_node(
    repo: str = Form(..., description="GitHub slug like 'Kosinkadink/ComfyUI-VideoHelperSuite' OR full https:// URL."),
    restart_comfyui: bool = Form(True, description="After install, kill ComfyUI's python so the supervisor relaunches it and picks up the new node. Set false to skip the restart (you'll need to reload manually before the node is usable)."),
    authorization: str = Header(default=None),
):
    """Install a ComfyUI custom node at runtime without a pod restart.

    Pattern: git clone (or pull) into /workspace/ComfyUI/custom_nodes,
    pip install the node's requirements.txt if present, optionally SIGKILL
    ComfyUI's process (start_comfy.sh's supervisor relaunches within 5s
    with the new node loaded). Returns the install trace.

    Use for one-off custom-node additions when setup.sh's idempotent
    install block didn't fire (network blip on boot, partial clone, etc.).
    Long-term: every node we depend on should be listed in setup.sh too,
    so a fresh pod boot works without this manual step.
    """
    _require_admin(authorization)
    import subprocess
    import sys

    repo_slug = repo.strip()
    if not repo_slug:
        raise HTTPException(400, "repo is required")
    if "://" not in repo_slug:
        # Allow "owner/name" shorthand.
        repo_slug = f"https://github.com/{repo_slug}"
    repo_name = repo_slug.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")

    comfy_root = _detect_comfy_root()
    nodes_dir = Path(comfy_root) / "custom_nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    target_dir = nodes_dir / repo_name
    trace: list[str] = []

    def _run(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
        try:
            res = subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=180)
            output = (res.stdout + res.stderr).decode(errors="replace")
            trace.append(f"$ {' '.join(cmd)}\n{output[-1000:]}\n--- exit {res.returncode} ---")
            return res.returncode, output
        except subprocess.TimeoutExpired:
            trace.append(f"$ {' '.join(cmd)}\n  TIMEOUT (>180s)")
            return 124, ""

    # Clone or pull. If the dir exists but has no .git, treat as corrupt
    # and re-clone so the failure mode is recoverable from here without
    # another endpoint.
    if not (target_dir / ".git").is_dir():
        if target_dir.exists():
            trace.append(f"removing corrupt {target_dir}")
            subprocess.run(["rm", "-rf", str(target_dir)], check=False)
        rc, _ = _run(["git", "clone", repo_slug, str(target_dir)])
        if rc != 0:
            raise HTTPException(500, f"git clone failed (see trace): {trace[-1] if trace else ''}")
    else:
        _run(["git", "pull", "--ff-only"], cwd=str(target_dir))

    # Pip install requirements if any.
    req = target_dir / "requirements.txt"
    if req.is_file():
        _run([sys.executable, "-m", "pip", "install", "-r", str(req)])
    else:
        trace.append("(no requirements.txt)")

    restarted = False
    if restart_comfyui:
        # ComfyUI runs under start_comfy.sh's flock supervisor as e.g.
        #   /workspace/runpod-slim/ComfyUI/.venv-cu128/bin/python main.py --listen ...
        # The cmdline contains "main.py" but NOT "ComfyUI/main.py" (cwd'd
        # into ComfyUI before exec), so the original pattern matched
        # nothing. Walk a list of broad-to-specific patterns; first match
        # wins. The supervisor relaunches within ~5s with the new node
        # registered.
        # Restart by killing the :8188 port owner — same logic
        # start_comfy.sh's STALE_PID check uses. Avoids the brittle
        # argv-regex approach that quietly failed across all four
        # patterns even though ComfyUI was running.
        try:
            ns = subprocess.run(["netstat", "-tlnp"], capture_output=True, timeout=10)
            comfy_pid: str | None = None
            for line in (ns.stdout or b"").decode(errors="replace").splitlines():
                if ":8188" not in line:
                    continue
                tail = line.split()[-1] if line.split() else ""
                if "/" in tail:
                    pid_part = tail.split("/")[0]
                    if pid_part.isdigit():
                        comfy_pid = pid_part
                        break
            if comfy_pid:
                rc, _ = _run(["kill", "-9", comfy_pid])
                trace.append(f"kill -9 {comfy_pid} (ComfyUI :8188 owner) → exit {rc}")
                if rc == 0:
                    restarted = True
            else:
                trace.append("netstat -tlnp found no :8188 listener — "
                             "ComfyUI may already be down")
        except Exception as e:
            trace.append(f"netstat-based kill failed: {e}")

        # Fallback if netstat is missing: literal-substring pkill
        # against the exact argv start_comfy.sh emits.
        if not restarted:
            rc, _ = _run([
                "pkill", "-9", "-f",
                "main.py --listen 0.0.0.0 --port 8188",
            ])
            trace.append(
                f"pkill -9 -f 'main.py --listen 0.0.0.0 --port 8188' → exit {rc}"
            )
            if rc == 0:
                restarted = True

    return {
        "ok": True,
        "repo": repo_slug,
        "target_dir": str(target_dir),
        "has_init_py": (target_dir / "__init__.py").is_file(),
        "comfyui_restart_signaled": restarted,
        "note": (
            "ComfyUI restart takes ~30s. Poll GET /admin/comfy-status until "
            "key_nodes_loaded reflects the new node before submitting jobs."
        ),
        "trace": trace,
    }


@app.post("/admin/refresh-api-code")
async def admin_refresh_api_code(authorization: str = Header(default=None)):
    """Wget the latest main.py / workflows.py / safety.py / etc. from
    the API_REPO env var into /workspace/api/, then kill uvicorn so the
    start_api.sh supervisor re-launches it with the fresh code.

    Requires the supervisor's `fetch_api_code()` shell function — i.e.
    setup.sh must have been re-run since the wget-in-loop change landed.
    On older containers this endpoint may return ok but the supervisor
    won't actually re-fetch (because the in-memory start_api.sh still has
    the wget BEFORE the while loop).

    Use case: deploy a Python-file change to the pod without a container
    restart. Couple seconds of unavailability while uvicorn cycles, no
    GPU release, no risk of capacity loss.
    """
    _require_admin(authorization)
    import os
    import subprocess

    api_repo = os.environ.get("API_REPO", "https://raw.githubusercontent.com/cyrus688/ai-server/main")
    api_dir = Path("/workspace/api")
    api_dir.mkdir(parents=True, exist_ok=True)

    fetched: list[dict] = []
    for filename in (
        "main.py",
        "workflows.py",
        "face_targeting.py",
        "image_output.py",
        "safety.py",
        "logo_safety.py",
        "watermark.py",
    ):
        url = f"{api_repo}/{filename}"
        target = api_dir / filename
        tmp = api_dir / f"{filename}.new"
        try:
            res = subprocess.run(
                ["wget", "-q", "-O", str(tmp), url],
                capture_output=True, timeout=30,
            )
            if res.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
                tmp.replace(target)
                fetched.append({"file": filename, "ok": True, "size": target.stat().st_size})
            else:
                tmp.unlink(missing_ok=True)
                fetched.append({"file": filename, "ok": False, "reason": "empty or wget failed"})
        except Exception as e:
            fetched.append({"file": filename, "ok": False, "reason": str(e)})

    # Kill uvicorn — the supervisor restarts it within ~5s with the
    # freshly fetched code. We target the port owner via netstat (mirrors
    # start_api.sh's own stale-PID logic) — more reliable than pkill on
    # an argv pattern.
    restarted = False
    try:
        netstat = subprocess.run(["netstat", "-tlnp"], capture_output=True, timeout=10).stdout.decode()
        for line in netstat.splitlines():
            if ":7860" in line:
                # last column: "PID/program"
                pid_field = line.split()[-1]
                pid = pid_field.split("/")[0]
                if pid.isdigit():
                    subprocess.run(["kill", "-9", pid], capture_output=True, timeout=5)
                    restarted = True
                    break
    except Exception as e:
        return {
            "ok": False,
            "files": fetched,
            "uvicorn_kill_error": str(e),
            "note": "Files updated but uvicorn restart failed — kill it manually with: pkill -9 -f 'uvicorn main:app'",
        }

    return {
        "ok": True,
        "files": fetched,
        "uvicorn_killed": restarted,
        "note": "uvicorn restarts in ~5s via start_api.sh supervisor. The connection that called this endpoint dies — that's expected.",
    }


@app.get("/admin/comfy-objects")
async def admin_comfy_objects(
    filter: str = "",
    show_inputs: bool = False,
    authorization: str = Header(default=None),
):
    """Return the names (or full schema) of every node type ComfyUI's
    currently exposing via /object_info. Useful when /admin/comfy-status's
    hardcoded `key_nodes_loaded` list doesn't cover what you need to find.

    Query params:
      filter      Case-insensitive substring filter on node names.
      show_inputs If true, return each node's input/output schema too
                  (heavyweight — only for the specific node you're
                  investigating). False returns just the name list.
    """
    _require_admin(authorization)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{COMFYUI_URL}/object_info")
        if resp.status_code != 200:
            raise HTTPException(503, f"ComfyUI /object_info returned {resp.status_code}")
        data = resp.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503, f"could not reach ComfyUI: {e}")

    names = sorted(data.keys())
    if filter:
        f_lower = filter.lower()
        names = [n for n in names if f_lower in n.lower()]

    if not show_inputs:
        return {"count": len(names), "filter": filter, "names": names}

    return {
        "count": len(names),
        "filter": filter,
        "nodes": {n: data[n] for n in names},
    }


@app.post("/admin/restart-comfyui")
async def admin_restart_comfyui(authorization: str = Header(default=None)):
    """Kill ComfyUI so start_comfy.sh's supervisor relaunches it.

    Use after dropping a custom node into custom_nodes/ out-of-band (or
    when /admin/install-comfy-node's restart step couldn't find the
    process). Same pkill-by-pattern logic as install-comfy-node, just
    standalone so we don't have to re-install to trigger a reload.
    """
    _require_admin(authorization)
    import subprocess

    # Find ComfyUI by port owner (mirrors start_comfy.sh's STALE_PID
    # logic). pkill -f against argv patterns was flaky: start_comfy.sh
    # launches via `"$PYTHON" main.py ...` and the resolved argv[0] in
    # /proc shaped my regexes off, so all four patterns returned exit 1
    # while a perfectly good ComfyUI process was running. Port-owner is
    # the one signal that's always right.
    trace: list[str] = []
    matched: str | None = None
    killed_pid: str | None = None
    try:
        netstat = subprocess.run(
            ["netstat", "-tlnp"], capture_output=True, timeout=10,
        )
        ns_out = (netstat.stdout or b"").decode(errors="replace")
        for line in ns_out.splitlines():
            if ":8188 " not in line and ":8188\t" not in line and not line.rstrip().endswith(":8188"):
                # be permissive about whitespace + look for both LISTEN
                # rows and bound rows ending in :8188
                if ":8188" not in line:
                    continue
            tail = line.split()[-1] if line.split() else ""
            if "/" in tail:
                pid_part = tail.split("/")[0]
                if pid_part.isdigit():
                    killed_pid = pid_part
                    break
        if killed_pid:
            res = subprocess.run(["kill", "-9", killed_pid], capture_output=True, timeout=10)
            trace.append(f"kill -9 {killed_pid} (port :8188 owner) → exit {res.returncode}")
            if res.returncode == 0:
                matched = f"port:8188:pid={killed_pid}"
        else:
            trace.append("netstat -tlnp showed no owner for :8188")
    except Exception as e:
        trace.append(f"netstat path error: {e}")

    # Fallback for environments without netstat: pattern-match
    # against the literal `main.py --listen 0.0.0.0 --port 8188`
    # string that start_comfy.sh always uses. The :8188 makes it
    # unambiguous (won't match the FastAPI uvicorn on :7860).
    if matched is None:
        try:
            res = subprocess.run(
                ["pkill", "-9", "-f", "main.py --listen 0.0.0.0 --port 8188"],
                capture_output=True, timeout=10,
            )
            trace.append(f"pkill -9 -f 'main.py --listen 0.0.0.0 --port 8188' → exit {res.returncode}")
            if res.returncode == 0:
                matched = "pattern:start_comfy_argv"
        except Exception as e:
            trace.append(f"pkill fallback error: {e}")

    return {
        "ok": matched is not None,
        "matched_pattern": matched,
        "note": "Wait ~30s and poll /admin/comfy-status to verify the new node count.",
        "trace": trace,
    }


@app.post("/admin/reload-filter")
async def admin_reload_filter(authorization: str = Header(default=None)):
    """Force `safety._build_filter()` to re-run without a uvicorn restart.

    Use after changing FACE_FILTER_THRESHOLD / FACE_DETECTOR_THRESHOLD /
    FACE_MIN_AREA_RATIO env vars on the pod, or after a manual blocklist
    mutation that bypassed the /admin/blocklist endpoints (rsync, scp,
    etc.). The InsightFace model itself stays loaded — only the cached
    `_FILTER` dict and the blocklist embeddings are rebuilt, so this is
    fast (~50ms + ~10ms per blocklist entry).
    """
    _require_admin(authorization)
    if face_safety is None:
        raise HTTPException(503, "face filter module unavailable")
    return face_safety.force_reload_filter()


@app.post("/admin/test-face-filter")
async def admin_test_face_filter(
    image: UploadFile = File(..., description="Image to test against the loaded blocklist."),
    top_n: int = Form(5, ge=1, le=50, description="How many top-scoring identities to return."),
    authorization: str = Header(default=None),
):
    """Diagnose why a specific image is or isn't being blocked.

    Runs the same detect-then-match pipeline that /flux endpoints invoke
    when face_filter=true, but returns the full picture:
      - detected face count (significant + raw)
      - per-identity scores (cosine similarity, sorted descending)
      - the blocked/not-blocked verdict at the current threshold
      - any skipped blocklist files that would have been candidates

    Use this when a face you thought would block sails through. The
    response tells you exactly which knob is responsible: low detection?
    score below threshold? target identity in the skipped list? Etc.
    """
    _require_admin(authorization)
    if face_safety is None:
        raise HTTPException(503, "face filter module unavailable")

    import io
    import numpy as np
    from PIL import Image as PILImage

    raw = await image.read()
    if not raw:
        raise HTTPException(400, "image is empty")

    # Force a reload to ensure we're testing against the current state.
    face_safety._maybe_reload()
    filt = face_safety._FILTER
    if filt is None:
        raise HTTPException(503, f"filter not initialized: {face_safety._FILTER_INIT_ERROR}")

    try:
        pil = PILImage.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"could not decode image: {e}")
    # CRITICAL: InsightFace expects BGR (OpenCV convention). Feeding RGB
    # gives the model swapped R↔B channels, which dramatically lowers
    # detection recall on borderline images (archival B&W, faded photos).
    # See _detect_with_fallbacks in safety.py for the full explanation.
    arr = np.array(pil)[:, :, ::-1].copy()  # RGB → BGR
    img_h, img_w = arr.shape[:2]
    img_area = img_h * img_w

    app_ = filt["app"]
    blocklist = filt["blocklist"]
    threshold = filt["threshold"]

    # Two-pass detection so the admin can see BOTH:
    #   1. raw_faces: what the detector finds on the original image with no
    #      preprocessing — this is what the OLD test endpoint reported, kept
    #      for backward compatibility with admin debug habits.
    #   2. recovered_faces: what `check_image` actually runs in production,
    #      including the 10-variant preprocessing fallback chain (CLAHE,
    #      gamma, upscale, center-crops). This is the source of truth for
    #      "would this image be blocked during a real face-swap?".
    # The `blocked` verdict below uses the recovered set + the query-side
    # area threshold (0.5%) — same as the real check_image path.
    # Wrap detection in a try/except so any internal exception (cv2 missing,
    # weird image mode, numpy edge case) surfaces in the JSON response with a
    # full traceback instead of dying as a generic 500 with no detail. This is
    # an admin-only diagnostic endpoint — the verbosity is intentional.
    import traceback as _tb
    try:
        raw_faces = app_.get(arr)
    except Exception as e:
        return {
            "error": "raw detection failed",
            "exception": f"{type(e).__name__}: {e}",
            "traceback": _tb.format_exc().splitlines()[-15:],
            "image_size": [img_w, img_h],
        }
    try:
        recovered_faces, fallback_used, (detect_h, detect_w) = face_safety._detect_with_fallbacks(
            app_, pil, np=np, label_for_log="admin_test",
        )
    except Exception as e:
        return {
            "error": "fallback chain detection failed",
            "exception": f"{type(e).__name__}: {e}",
            "traceback": _tb.format_exc().splitlines()[-15:],
            "raw_face_count": len(raw_faces),
            "image_size": [img_w, img_h],
        }
    detect_area = max(1, detect_h * detect_w)
    # Use the QUERY threshold (0.5%) for the verdict — that's what check_image
    # uses. The MIN_FACE_AREA_RATIO (3%) is upload-side strict and would give
    # false-negative verdicts here (e.g. a Hun Sen photo with a small face
    # would say "not blocked" via that path even though check_image would
    # absolutely catch it at query time).
    significant_faces = [
        f for f in recovered_faces
        if (max(0.0, f.bbox[2] - f.bbox[0]) * max(0.0, f.bbox[3] - f.bbox[1])
            / detect_area) >= face_safety.MIN_FACE_AREA_RATIO_QUERY
    ]

    # For each detected face, compute scores against ALL blocklist
    # identities and pick the best per-face score. Then aggregate the
    # overall best across faces.
    per_face: list[dict] = []
    overall_best: tuple[float, str | None] = (-1.0, None)
    for fi, face in enumerate(significant_faces):
        emb = face.normed_embedding
        scored: list[tuple[str, float]] = []
        for identity, ref_emb in blocklist.items():
            scored.append((identity, float(np.dot(emb, ref_emb))))
        scored.sort(key=lambda t: t[1], reverse=True)
        if scored and scored[0][1] > overall_best[0]:
            overall_best = (scored[0][1], scored[0][0])
        bbox = face.bbox
        per_face.append({
            "face_index": fi,
            "bbox": [float(x) for x in bbox],
            "area_ratio": round(
                (max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]) / max(1, img_area)), 4
            ),
            "top_scores": [{"identity": i, "score": round(s, 4)} for i, s in scored[:top_n]],
        })

    return {
        "image_size": [img_w, img_h],
        "raw_face_count": len(raw_faces),
        "recovered_face_count": len(recovered_faces),
        "fallback_used": fallback_used or "none (raw detection passed)",
        "significant_face_count": len(significant_faces),
        "min_face_area_ratio_query": face_safety.MIN_FACE_AREA_RATIO_QUERY,
        "match_threshold": threshold,
        "blocked": overall_best[0] > threshold,
        "best_match": {"identity": overall_best[1], "score": round(overall_best[0], 4)} if overall_best[1] else None,
        "per_face": per_face,
        "blocklist_size": len(blocklist),
        "skipped_count": len(filt.get("skipped_files", [])),
        "hint": (
            "recovered_face_count=0 → detector failed even with all 10 preprocessing variants. The face is genuinely undetectable by SCRFD — try a sharper, larger, front-facing crop."
            if not recovered_faces
            else "significant_face_count=0 → face detected but too small (< min_face_area_ratio_query). At query time we accept faces down to 0.5% of image area."
            if not significant_faces
            else f"best score {round(overall_best[0], 4)} < threshold {threshold} → either no match in the loaded blocklist OR the target identity is in skipped_files (call /admin/reload-filter to see)."
            if not (overall_best[0] > threshold)
            else "blocked correctly."
        ),
    }


@app.get("/admin/blocklist")
async def admin_list_blocklist(authorization: str = Header(default=None)):
    """List all identities currently on the blocklist.

    Includes a `loaded` flag per entry — true if the face filter
    successfully embedded the face from this image, false if the file
    is on disk but detection failed and the file is effectively a no-op
    at filter time. The loader (`safety._build_filter`) tries multiple
    fallbacks (autocontrast, sharpen, 2× upscale, 4× upscale, center
    crop) before giving up; anything in the `skipped` list means even
    those fallbacks couldn't find a face and the entry won't actually
    block uploads.

    Admins should re-upload skipped entries with clearer crops.
    """
    _require_admin(authorization)
    BLOCKLIST_DIR.mkdir(parents=True, exist_ok=True)

    # Pull the skipped-files set from the in-memory filter state. If
    # the filter hasn't been initialized yet (e.g. cold pod) we treat
    # everything as "unknown" — better than lying.
    skipped_set: set[str] = set()
    filter_initialized = False
    if face_safety is not None:
        try:
            status = face_safety.get_status()
            skipped_set = set(status.get("skipped_files", []))
            filter_initialized = True
        except Exception as e:
            print(f"[admin_list_blocklist] could not fetch filter status: {e}")

    entries = []
    loaded_count = 0
    skipped_count = 0
    for p in sorted(BLOCKLIST_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in ALLOWED_BLOCKLIST_EXTS:
            is_skipped = p.name in skipped_set
            entries.append({
                "identity": p.stem,
                "filename": p.name,
                "size_bytes": p.stat().st_size,
                "added_at": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(),
                "loaded": (not is_skipped) if filter_initialized else None,
            })
            if is_skipped:
                skipped_count += 1
            else:
                loaded_count += 1
    return {
        "count": len(entries),
        "loaded_count": loaded_count if filter_initialized else None,
        "skipped_count": skipped_count if filter_initialized else None,
        "filter_initialized": filter_initialized,
        "blocklist": entries,
    }


@app.post("/admin/blocklist")
async def admin_upload_blocklist(
    image: UploadFile = File(..., description="Face image of the identity to block."),
    identity: str = Form(..., description="Stable identifier (used to delete later, and returned in block responses). [A-Za-z0-9_-]{1,64}"),
    overwrite: bool = Form(True, description="If true (default), replace an existing entry with the same identity. Set false to error 409 instead."),
    authorization: str = Header(default=None),
):
    """Add a face to the blocklist.

    The upload is accepted regardless of whether the face detector can
    find a face — the admin has eyeballed the image and knows what they
    intend to block; rejecting valid admin intent because SCRFD flaked is
    worse UX than accepting an entry that may turn out to be non-functional.
    The face filter loader (`_build_filter` in safety.py) does its own
    detection pass when it reads `/workspace/blocklist/` and warns + skips
    any file it can't embed, so undetectable entries fail closed (they
    won't crash the filter, they just won't block anything).

    The response includes `face_count` so callers can surface a soft
    warning when detection found 0 or 2+ faces — useful for the CMS
    badge but not a hard error.
    """
    _require_admin(authorization)
    _validate_identity(identity)
    BLOCKLIST_DIR.mkdir(parents=True, exist_ok=True)

    existing = _find_existing_blocklist_file(identity)
    if existing and not overwrite:
        raise HTTPException(409, f"identity '{identity}' already on blocklist as {existing.name}. Pass overwrite=true to replace.")

    raw_bytes = await image.read()

    if face_safety is None:
        raise HTTPException(503, "face filter module unavailable — cannot normalize the uploaded image")

    # Normalize first: EXIF-rotate + downscale to BLOCKLIST_MAX_EDGE + re-encode
    # as PNG. Lets the caller upload phone photos / 4K crops / odd formats
    # without hitting body-size or storage issues, and gives detection a
    # consistent input.
    try:
        norm_bytes, ext = face_safety.normalize_blocklist_image(raw_bytes)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(503, f"image normalizer unavailable: {e}")

    # Detection is now informational, not a gate. We still run it so the
    # response can include a soft warning, but neither 0 nor 2+ faces
    # rejects the upload.
    face_count = -1
    try:
        face_count = face_safety.detect_face_count(norm_bytes)
    except Exception as e:
        print(f"[admin_upload_blocklist] detector raised, treating as unknown: {e}")

    if existing:
        existing.unlink()

    target = BLOCKLIST_DIR / f"{identity}{ext}"
    target.write_bytes(norm_bytes)

    warning = None
    if face_count == 0:
        warning = "no face detected by the model — entry stored but may not actually filter generations until you re-upload with a clearer crop"
    elif face_count > 1:
        warning = f"detected {face_count} faces — the filter will pick the largest for matching"

    return {
        "status": "replaced" if existing else "added",
        "identity": identity,
        "filename": target.name,
        "size_bytes": target.stat().st_size,
        "blocklist_count": _blocklist_count(),
        "face_count": face_count,
        "warning": warning,
    }


@app.delete("/admin/blocklist")
async def admin_clear_blocklist(
    confirm: bool = False,
    authorization: str = Header(default=None),
):
    """DESTRUCTIVE — delete every face image in /workspace/blocklist.

    Used by the CMS "delete all blocked faces" flow when an admin wants
    a clean slate to re-upload from scratch (e.g. after tuning the
    detection chain and wanting to verify which photos load now).

    Requires `?confirm=true` so a stray `curl -X DELETE` against the
    admin URL doesn't nuke production. Returns the count of deleted
    files plus the freshly-reloaded filter status so the caller can
    confirm the wipe took effect.

    Reloads the in-memory filter immediately so check_image() sees the
    empty blocklist on the very next request (no waiting for the
    mtime-based auto-reload).

    Does NOT touch the logo blocklist (/workspace/blocklist_logos) —
    that has its own DELETE endpoint family. Each blocklist is wiped
    independently so admins can clean one without losing the other.
    """
    _require_admin(authorization)
    if not confirm:
        raise HTTPException(400, {
            "error": "destructive operation",
            "hint": "pass ?confirm=true to proceed — this deletes ALL files in /workspace/blocklist",
            "current_count": _blocklist_count(),
        })

    BLOCKLIST_DIR.mkdir(parents=True, exist_ok=True)
    deleted: list[str] = []
    errors: list[dict] = []
    for p in sorted(BLOCKLIST_DIR.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in ALLOWED_BLOCKLIST_EXTS:
            continue
        try:
            p.unlink()
            deleted.append(p.name)
        except Exception as e:
            errors.append({"file": p.name, "error": str(e)})
            print(f"[admin_clear_blocklist] could not delete {p}: {e}")

    # Reload the face filter so check_image() sees the wipe immediately.
    # Best-effort — if the filter module is unavailable (insightface not
    # installed) the directory wipe still succeeded and the next reload
    # cycle will catch up.
    reload_result: dict = {"ok": False, "reason": "face_safety module unavailable"}
    if face_safety is not None:
        try:
            reload_result = face_safety.force_reload_filter()
        except Exception as e:
            reload_result = {"ok": False, "error": str(e)}

    return {
        "status": "cleared",
        "deleted_count": len(deleted),
        "deleted": deleted,
        "errors": errors,
        "filter_reload": reload_result,
    }


@app.delete("/admin/blocklist/{identity}")
async def admin_delete_blocklist(
    identity: str,
    authorization: str = Header(default=None),
):
    """Remove a face from the blocklist."""
    _require_admin(authorization)
    _validate_identity(identity)
    existing = _find_existing_blocklist_file(identity)
    if not existing:
        raise HTTPException(404, f"identity '{identity}' is not on the blocklist")
    existing.unlink()
    return {
        "status": "deleted",
        "identity": identity,
        "filename": existing.name,
        "blocklist_count": _blocklist_count(),
    }


@app.get("/admin/blocklist/{identity}/image")
async def admin_get_blocklist_image(
    identity: str,
    authorization: str = Header(default=None),
):
    """Download the stored face image for a blocked identity (for CMS preview)."""
    _require_admin(authorization)
    _validate_identity(identity)
    existing = _find_existing_blocklist_file(identity)
    if not existing:
        raise HTTPException(404, f"identity '{identity}' is not on the blocklist")
    return FileResponse(str(existing), filename=existing.name)


# ─────────────────────────────────────────────
# Admin API — manage the LOGO/FLAG blocklist (CLIP-based)
#
# Parallel to /admin/blocklist (faces). Stored at /workspace/blocklist_logos/.
# Hot-reloaded on every logo-filter check.
# ─────────────────────────────────────────────

BLOCKLIST_LOGOS_DIR = Path(os.environ.get("BLOCKLIST_LOGOS_DIR", "/workspace/blocklist_logos"))


def _find_existing_logo_file(identity: str):
    for ext in ALLOWED_BLOCKLIST_EXTS:
        p = BLOCKLIST_LOGOS_DIR / f"{identity}{ext}"
        if p.exists():
            return p
    return None


def _logo_blocklist_count() -> int:
    if not BLOCKLIST_LOGOS_DIR.is_dir():
        return 0
    return sum(1 for p in BLOCKLIST_LOGOS_DIR.iterdir()
               if p.is_file() and p.suffix.lower() in ALLOWED_BLOCKLIST_EXTS)


@app.get("/admin/blocklist-logos")
async def admin_list_logos(authorization: str = Header(default=None)):
    """List all logos/flags on the blocklist."""
    _require_admin(authorization)
    BLOCKLIST_LOGOS_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for p in sorted(BLOCKLIST_LOGOS_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in ALLOWED_BLOCKLIST_EXTS:
            entries.append({
                "identity": p.stem,
                "filename": p.name,
                "size_bytes": p.stat().st_size,
                "added_at": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(),
            })
    return {"count": len(entries), "blocklist": entries}


@app.post("/admin/blocklist-logos")
async def admin_upload_logo(
    image: UploadFile = File(..., description="Logo / flag / symbol image. Should be cropped tight on the subject for best CLIP discrimination."),
    identity: str = Form(..., description="Stable identifier — e.g. 'apple_logo' or 'flag_xx'. Returned in block responses."),
    overwrite: bool = Form(False, description="If true, replace an existing entry with the same identity"),
    authorization: str = Header(default=None),
):
    """Add a logo/flag to the blocklist. Unlike faces, no face-detection
    prerequisite — but the file must be a valid image."""
    _require_admin(authorization)
    _validate_identity(identity)
    BLOCKLIST_LOGOS_DIR.mkdir(parents=True, exist_ok=True)

    existing = _find_existing_logo_file(identity)
    if existing and not overwrite:
        raise HTTPException(409, f"logo '{identity}' already on blocklist as {existing.name}. Pass overwrite=true to replace.")

    img_bytes = await image.read()

    if logo_safety is None:
        raise HTTPException(503, "logo filter module unavailable — cannot validate the uploaded image")
    err = logo_safety.validate_uploadable(img_bytes)
    if err:
        raise HTTPException(400, err)

    ext_from_filename = Path(image.filename or "").suffix.lower()
    ext = ext_from_filename if ext_from_filename in ALLOWED_BLOCKLIST_EXTS else ".png"

    if existing:
        existing.unlink()

    target = BLOCKLIST_LOGOS_DIR / f"{identity}{ext}"
    target.write_bytes(img_bytes)

    return {
        "status": "replaced" if existing else "added",
        "identity": identity,
        "filename": target.name,
        "size_bytes": target.stat().st_size,
        "blocklist_count": _logo_blocklist_count(),
    }


@app.delete("/admin/blocklist-logos/{identity}")
async def admin_delete_logo(
    identity: str,
    authorization: str = Header(default=None),
):
    """Remove a logo/flag from the blocklist."""
    _require_admin(authorization)
    _validate_identity(identity)
    existing = _find_existing_logo_file(identity)
    if not existing:
        raise HTTPException(404, f"logo '{identity}' is not on the blocklist")
    existing.unlink()
    return {
        "status": "deleted",
        "identity": identity,
        "filename": existing.name,
        "blocklist_count": _logo_blocklist_count(),
    }


@app.get("/admin/blocklist-logos/{identity}/image")
async def admin_get_logo_image(
    identity: str,
    authorization: str = Header(default=None),
):
    """Download the stored logo image for CMS preview."""
    _require_admin(authorization)
    _validate_identity(identity)
    existing = _find_existing_logo_file(identity)
    if not existing:
        raise HTTPException(404, f"logo '{identity}' is not on the blocklist")
    return FileResponse(str(existing), filename=existing.name)
