"""
RunPod serverless handler — Image worker (FLUX.2 Klein 9B).

Endpoints supported via `event["input"]["endpoint"]`:
  - "t2i"            text-to-image
  - "flux/face-swap" face / head swap (2 reference images: target body + face)
  - "flux/multi-face-swap" group face swap (target + one or two faces)
  - "flux/i2i"       multi-reference image editing (1–5 reference images + prompt)

Models read from the network volume at /runpod-volume/runpod-slim/ComfyUI/models
(linked into ComfyUI's models dir by start.sh before the handler boots).
Generated images are staged to /runpod-volume/outputs/<job_id>/<filename> so
the reaper can clean them up after RETENTION_DAYS and so clients can fetch
them via the same RunPod S3 API used for videos.

Input (t2i):
  {
    "input": {
      "endpoint": "t2i",
      "prompt": "...",
      "width": 1024, "height": 1024,
      "seed": -1, "steps": 4, "cfg": 1.0, "guidance": 4.0
    }
  }

Input (face-swap — base64 with or without `data:image/...;base64,` prefix):
  {
    "input": {
      "endpoint": "flux/face-swap",
      "target_image_b64": "iVBOR...",
      "face_image_b64":   "iVBOR...",
      "aspect_ratio": "original",
      "megapixels": 2.0,
      "seed": -1, "steps": 4, "cfg": 1.0, "guidance": 4.0,
      "lora_strength": 1.0
    }
  }

Output (success):
  {
    "image_path": "/runpod-volume/outputs/<job_id>/t2i_42_00001_.png",
    "filename":   "t2i_42_00001_.png",
    "size_bytes": 423104,
    "seed": 42,
    "duration_seconds": 12.3
  }
"""

import asyncio
import base64
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import httpx
import runpod

# Make repo-root /app/workflows.py + /app/safety.py importable
sys.path.insert(0, "/app")
from workflows import (
    ASPECT_RATIOS,
    MULTI_FACE_SWAP_ORDERS,
    build_flux_i2i_workflow,
    build_flux_multi_face_swap_workflow,
    build_t2i_workflow,
    compute_dimensions,
    crop_to_aspect,
    get_flux_face_swap_workflow,
    normalize_target_face_indices,
    preserve_selected_faces,
)
from image_output import optimize_image_file
try:
    import safety as face_safety
except ImportError:
    face_safety = None
try:
    import logo_safety
except ImportError:
    logo_safety = None
try:
    import watermark
except ImportError:
    watermark = None


COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", "/comfyui"))
OUTPUT_DIR = COMFY_ROOT / "output"
INPUT_DIR = COMFY_ROOT / "input"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Staging dir on the network volume — clients fetch results from here via
# the RunPod S3 API or any pod with the same volume mounted. The reaper
# (cleanup.sh) deletes contents older than RETENTION_DAYS.
VOLUME_OUTPUTS = Path(os.environ.get("VOLUME_OUTPUTS", "/runpod-volume/outputs"))
VOLUME_OUTPUTS.mkdir(parents=True, exist_ok=True)


def _stage_output_to_volume(filename: str, src: Path, job_id: str) -> Path:
    dest_dir = VOLUME_OUTPUTS / job_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    shutil.copy2(src, dest)
    return dest


def _apply_watermark(path: Path, text, logo=False):
    """Apply requested text/logo marks without letting either fail the job."""
    if watermark is None:
        return None
    warnings = []
    if text:
        try:
            watermark.apply(path, text)
        except Exception as exc:
            warnings.append(f"text: {exc}")
    if logo:
        try:
            watermark.apply_logo(path)
        except Exception as exc:
            warnings.append(f"image: {exc}")
    return " | ".join(warnings) or None


def _prepare_delivery_image(path: Path) -> tuple[Path, dict]:
    optimized = optimize_image_file(path)
    return optimized.path, {
        "original_bytes": optimized.original_bytes,
        "output_bytes": optimized.output_bytes,
        "width": optimized.width,
        "height": optimized.height,
        "quality": optimized.quality,
    }


# ─────────────────────────────────────────────
# Compliance / face filter — checks N input images against the blocklist
# on the network volume. Raises FaceFilterBlocked on the first match;
# the handler catches it and returns a structured error response.
# ─────────────────────────────────────────────

class FilterBlocked(Exception):
    """Either filter (face or logo) matched. The handler converts this to
    a structured error response."""
    def __init__(self, filter_name: str, matched: str, score: float,
                 image_index: int, label: str):
        self.filter_name = filter_name   # "face" or "logo"
        self.matched = matched
        self.score = score
        self.image_index = image_index
        self.label = label
        super().__init__(f"{label} matches blocked {filter_name} '{matched}'")


class DetectableFaceRequired(Exception):
    """An opted-in user image did not contain a significant face."""
    def __init__(self, label: str, image_index: int):
        self.label = label
        self.image_index = image_index
        super().__init__(f"{label} does not contain a clearly detectable face")


class FaceValidationUnavailable(Exception):
    """InsightFace could not initialize for an opted-in validation."""


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _require_detectable_face(enabled: bool, images_with_names: list) -> None:
    if not enabled:
        return
    if face_safety is None:
        raise FaceValidationUnavailable("face validation module is unavailable")
    for image_index, (img_bytes, label) in enumerate(images_with_names):
        try:
            face_count = face_safety.detect_face_count(img_bytes)
        except RuntimeError as exc:
            raise FaceValidationUnavailable(str(exc)) from exc
        if face_count < 1:
            raise DetectableFaceRequired(label, image_index)


def _apply_face_filter(endpoint: str, job_id: str, face_filter: bool,
                       images_with_names: list) -> None:
    if not face_filter:
        if face_safety is not None:
            face_safety.log_bypass(job_id, endpoint, note=f"face_filter=false, {len(images_with_names)} images")
        return
    if face_safety is None:
        raise RuntimeError("face filter requested but the `safety` module is not installed in this image")
    for idx, (img_bytes, label) in enumerate(images_with_names):
        result = face_safety.check_image(img_bytes)
        if result.blocked:
            raise FilterBlocked("face", result.matched_identity, result.score, idx, label)


def _apply_logo_filter(endpoint: str, job_id: str, logo_filter: bool,
                       images_with_names: list) -> None:
    if not logo_filter:
        if face_safety is not None:
            face_safety.log_bypass(job_id, endpoint, note=f"logo_filter=false, {len(images_with_names)} images")
        return
    if logo_safety is None:
        raise RuntimeError("logo filter requested but `logo_safety` (open_clip_torch) is not installed in this image")
    for idx, (img_bytes, label) in enumerate(images_with_names):
        result = logo_safety.check_image(img_bytes)
        if result.blocked:
            raise FilterBlocked("logo", result.matched_logo, result.score, idx, label)


# ─────────────────────────────────────────────
# Input helpers
# ─────────────────────────────────────────────

def _decode_image_b64(b64: str, field_name: str) -> bytes:
    """Decode a base64 image, accepting both raw and data-URI (`data:image/png;base64,…`) forms.
    Raises ValueError with the field name if the input is malformed."""
    if not isinstance(b64, str):
        raise ValueError(f"'{field_name}' must be a base64 string, got {type(b64).__name__}")
    # Strip data URI prefix if present — common in browser clients
    # (canvas.toDataURL, FileReader.readAsDataURL).
    if b64.startswith("data:") and "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        return base64.b64decode(b64, validate=False)
    except Exception as e:
        raise ValueError(f"'{field_name}' is not valid base64: {e}") from e


def _encode_result_b64(path: Path, max_edge: int = 2048, quality: int = 90) -> str:
    """Read a rendered output image and return it base64-encoded as JPEG.

    Used for INLINE delivery (the `return_b64` input flag). A serverless worker
    has no persistent HTTP server to serve `/image/<file>` like the pod does, so
    the load balancer asks for the pixels in the job result and hands the
    browser a `data:` URL. We re-encode to JPEG (and downscale very large
    renders) so the job output stays comfortably under RunPod's response-size
    cap — a raw 2 MP PNG base64 would risk truncation."""
    if path.suffix.lower() in {".jpg", ".jpeg"} and path.stat().st_size <= 200 * 1024:
        return base64.b64encode(path.read_bytes()).decode("ascii")

    from io import BytesIO

    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, max_edge / max(w, h))
        if scale < 1.0:
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))))
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ─────────────────────────────────────────────
# ComfyUI readiness — workers cold-start with ComfyUI booting in parallel
# (start.sh spawns it). Block on first invocation until it's serving.
# ─────────────────────────────────────────────

def wait_for_comfyui(timeout_s: int = 300) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(f"{COMFYUI_URL}/system_stats", timeout=2.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"ComfyUI not ready after {timeout_s}s at {COMFYUI_URL}")


# ─────────────────────────────────────────────
# Submit a workflow to ComfyUI and block until /history reports completion.
#
# We poll /history instead of using the /ws stream because the WS pattern has
# a race: the prompt may finish executing before we manage to connect, and
# then `ws.recv()` hangs forever waiting for a message that already fired.
# Polling adds ~0.5–1s overhead, which is invisible next to model load +
# inference time.
# ─────────────────────────────────────────────

async def submit_and_wait(workflow: dict, max_wait_s: float = 600.0) -> tuple[str, Path]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": workflow, "client_id": str(uuid.uuid4())},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"ComfyUI rejected workflow: {resp.text}")
        prompt_id = resp.json()["prompt_id"]

    deadline = time.time() + max_wait_s
    job_data: dict = {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            if time.time() > deadline:
                raise RuntimeError(
                    f"ComfyUI did not finish prompt {prompt_id} within {max_wait_s:.0f}s"
                )
            history = (await client.get(f"{COMFYUI_URL}/history/{prompt_id}")).json()
            job_data = history.get(prompt_id, {})
            if job_data.get("status", {}).get("completed"):
                break
            await asyncio.sleep(1.0)

    status_str = job_data.get("status", {}).get("status_str", "")
    if status_str == "error":
        for m in job_data.get("status", {}).get("messages", []):
            if m[0] == "execution_error":
                raise RuntimeError(m[1].get("exception_message", "ComfyUI execution error"))
        raise RuntimeError("ComfyUI execution error (no detail in history)")

    for node_output in job_data.get("outputs", {}).values():
        for key in ("images", "videos", "gifs"):
            if key in node_output:
                item = node_output[key][0]
                filename = item["filename"]
                subfolder = item.get("subfolder", "")
                path = OUTPUT_DIR / subfolder / filename if subfolder else OUTPUT_DIR / filename
                if path.exists():
                    return filename, path

    raise RuntimeError("ComfyUI completed but no output file was found in history")


# ─────────────────────────────────────────────
# Endpoint dispatchers
# ─────────────────────────────────────────────

async def run_t2i(inp: dict, job_id: str) -> dict:
    prompt = inp.get("prompt")
    if not prompt:
        raise ValueError("'prompt' is required for t2i")
    seed = inp.get("seed", -1)
    seed = seed if seed != -1 else uuid.uuid4().int % 2**32
    workflow = build_t2i_workflow(
        prompt=prompt,
        width=int(inp.get("width", 1024)),
        height=int(inp.get("height", 1024)),
        seed=seed,
        steps=int(inp.get("steps", 4)),
        cfg=float(inp.get("cfg", 1.0)),
        guidance=float(inp.get("guidance", 4.0)),
    )
    started = time.time()
    filename, src = await submit_and_wait(workflow)
    dest = _stage_output_to_volume(filename, src, job_id)
    wm_err = _apply_watermark(dest, inp.get("watermark"), inp.get("watermark_image", False))
    dest, delivery = _prepare_delivery_image(dest)
    result = {
        "image_path": str(dest),
        "filename": dest.name,
        "size_bytes": dest.stat().st_size,
        "image_delivery": delivery,
        "seed": seed,
        "duration_seconds": round(time.time() - started, 2),
    }
    if wm_err:
        result["watermark_warning"] = wm_err
    return result


async def run_flux_face_swap(inp: dict, job_id: str) -> dict:
    if not inp.get("target_image_b64") or not inp.get("face_image_b64"):
        raise ValueError("'target_image_b64' and 'face_image_b64' are required for flux/face-swap")

    seed = inp.get("seed", -1)
    seed = seed if seed != -1 else uuid.uuid4().int % 2**32
    aspect_ratio = inp.get("aspect_ratio", "original")
    megapixels = float(inp.get("megapixels", 2.0))

    if aspect_ratio != "original" and aspect_ratio not in ASPECT_RATIOS:
        raise ValueError(f"invalid aspect_ratio '{aspect_ratio}'; valid: original, {', '.join(ASPECT_RATIOS)}")

    target_bytes = _decode_image_b64(inp["target_image_b64"], "target_image_b64")
    face_bytes   = _decode_image_b64(inp["face_image_b64"],   "face_image_b64")

    inputs_for_filter = [
        (target_bytes, "target_image_b64"),
        (face_bytes,   "face_image_b64"),
    ]
    _require_detectable_face(
        _as_bool(inp.get("require_detectable_face", False)),
        [(face_bytes, "face_image_b64")],
    )
    _apply_face_filter("flux/face-swap", job_id, bool(inp.get("face_filter", False)), inputs_for_filter)
    _apply_logo_filter("flux/face-swap", job_id, bool(inp.get("logo_filter", False)), inputs_for_filter)

    if aspect_ratio != "original":
        w_r, h_r = ASPECT_RATIOS[aspect_ratio]
        target_w, target_h = compute_dimensions(w_r, h_r, megapixels)
        target_bytes = crop_to_aspect(target_bytes, target_w, target_h)

    target_filename = f"flux_target_{uuid.uuid4().hex}.png"
    face_filename = f"flux_face_{uuid.uuid4().hex}.png"
    (INPUT_DIR / target_filename).write_bytes(target_bytes)
    (INPUT_DIR / face_filename).write_bytes(face_bytes)

    workflow = get_flux_face_swap_workflow(
        target_filename, face_filename, seed,
        prompt=inp.get("prompt") or None,
        megapixels=megapixels,
        steps=int(inp.get("steps", 4)),
        cfg=float(inp.get("cfg", 1.0)),
        guidance=float(inp.get("guidance", 4.0)),
        lora_strength=float(inp.get("lora_strength", 1.0)),
    )

    started = time.time()
    try:
        filename, src = await submit_and_wait(workflow)
        dest = _stage_output_to_volume(filename, src, job_id)
        wm_err = _apply_watermark(dest, inp.get("watermark"), inp.get("watermark_image", False))
        dest, delivery = _prepare_delivery_image(dest)
        result = {
            "image_path": str(dest),
            "filename": dest.name,
            "size_bytes": dest.stat().st_size,
            "image_delivery": delivery,
            "seed": seed,
            "duration_seconds": round(time.time() - started, 2),
        }
        if wm_err:
            result["watermark_warning"] = wm_err
        return result
    finally:
        (INPUT_DIR / target_filename).unlink(missing_ok=True)
        (INPUT_DIR / face_filename).unlink(missing_ok=True)


async def run_flux_multi_face_swap(inp: dict, job_id: str) -> dict:
    if not inp.get("target_image_b64"):
        raise ValueError("'target_image_b64' is required for flux/multi-face-swap")
    faces_b64 = inp.get("face_images_b64")
    if not isinstance(faces_b64, list) or len(faces_b64) not in (1, 2):
        count = len(faces_b64) if isinstance(faces_b64, list) else 0
        raise ValueError(
            f"'face_images_b64' must contain 1 or 2 images for "
            f"flux/multi-face-swap; received {count}"
        )

    face_order = str(inp.get("face_order", "left-to-right"))
    if face_order not in MULTI_FACE_SWAP_ORDERS:
        raise ValueError(
            f"invalid face_order '{face_order}'; valid: "
            f"{', '.join(MULTI_FACE_SWAP_ORDERS)}"
        )
    target_face_indices = normalize_target_face_indices(
        inp.get("target_face_indices"),
        len(faces_b64),
    )
    aspect_ratio = str(inp.get("aspect_ratio", "original"))
    if aspect_ratio != "original" and aspect_ratio not in ASPECT_RATIOS:
        raise ValueError(
            f"invalid aspect_ratio '{aspect_ratio}'; valid: original, "
            f"{', '.join(ASPECT_RATIOS)}"
        )
    megapixels = float(inp.get("megapixels", 2.0))
    if not 0.5 <= megapixels <= 4.0:
        raise ValueError("megapixels must be between 0.5 and 4.0")
    prompt = inp.get("prompt")
    if prompt is not None and len(str(prompt)) > 2000:
        raise ValueError("prompt must be 2000 characters or fewer")

    seed = inp.get("seed", -1)
    seed = seed if seed != -1 else uuid.uuid4().int % 2**32
    target_bytes = _decode_image_b64(inp["target_image_b64"], "target_image_b64")
    face_bytes_list = [
        _decode_image_b64(value, f"face_images_b64[{index}]")
        for index, value in enumerate(faces_b64)
    ]

    face_inputs = [
        (face_bytes, f"face_images_b64[{index}]")
        for index, face_bytes in enumerate(face_bytes_list)
    ]
    inputs_for_filter = [(target_bytes, "target_image_b64"), *face_inputs]
    _require_detectable_face(
        _as_bool(inp.get("require_detectable_face", True)),
        face_inputs,
    )
    _apply_face_filter(
        "flux/multi-face-swap",
        job_id,
        _as_bool(inp.get("face_filter", True)),
        inputs_for_filter,
    )
    _apply_logo_filter(
        "flux/multi-face-swap",
        job_id,
        _as_bool(inp.get("logo_filter", True)),
        inputs_for_filter,
    )

    if aspect_ratio != "original":
        w_ratio, h_ratio = ASPECT_RATIOS[aspect_ratio]
        target_w, target_h = compute_dimensions(w_ratio, h_ratio, megapixels)
        target_bytes = crop_to_aspect(target_bytes, target_w, target_h)

    target_filename = f"flux_multi_target_{uuid.uuid4().hex}.png"
    face_filenames = [
        f"flux_multi_face_{uuid.uuid4().hex}_{index}.png"
        for index in range(len(face_bytes_list))
    ]
    target_path = INPUT_DIR / target_filename
    face_paths = [INPUT_DIR / filename for filename in face_filenames]
    staged_paths = [target_path, *face_paths]
    try:
        target_path.write_bytes(target_bytes)
        for path, face_bytes in zip(face_paths, face_bytes_list):
            path.write_bytes(face_bytes)

        workflow = build_flux_multi_face_swap_workflow(
            target_filename,
            face_filenames,
            seed,
            face_order=face_order,
            prompt=str(prompt) if prompt else None,
            megapixels=megapixels,
            steps=int(inp.get("steps", 4)),
            cfg=float(inp.get("cfg", 1.0)),
            guidance=float(inp.get("guidance", 4.0)),
            lora_strength=float(inp.get("lora_strength", 1.0)),
            target_face_indices=target_face_indices,
        )

        started = time.time()
        filename, src = await submit_and_wait(workflow)
        dest = _stage_output_to_volume(filename, src, job_id)
        if face_safety is None or not hasattr(face_safety, "get_face_bboxes"):
            raise RuntimeError("face detector unavailable for selected-person preservation")
        preserved, preserve_message = preserve_selected_faces(
            dest,
            target_path,
            face_order=face_order,
            target_face_indices=target_face_indices,
            detect_face_bboxes=face_safety.get_face_bboxes,
        )
        if not preserved:
            dest.unlink(missing_ok=True)
            raise RuntimeError(preserve_message)
        wm_err = _apply_watermark(
            dest,
            inp.get("watermark"),
            _as_bool(inp.get("watermark_image", False)),
        )
        dest, delivery = _prepare_delivery_image(dest)
        result = {
            "image_path": str(dest),
            "filename": dest.name,
            "size_bytes": dest.stat().st_size,
            "image_delivery": delivery,
            "seed": seed,
            "face_count": len(face_bytes_list),
            "face_order": face_order,
            "target_face_indices": target_face_indices,
            "duration_seconds": round(time.time() - started, 2),
        }
        if wm_err:
            result["watermark_warning"] = wm_err
        return result
    finally:
        for path in staged_paths:
            path.unlink(missing_ok=True)


async def run_flux_i2i(inp: dict, job_id: str) -> dict:
    images_b64 = inp.get("images_b64")
    if not isinstance(images_b64, list) or not (1 <= len(images_b64) <= 5):
        raise ValueError("'images_b64' must be a list of 1 to 5 base64-encoded images")

    prompt = inp.get("prompt") or ""
    seed = inp.get("seed", -1)
    seed = seed if seed != -1 else uuid.uuid4().int % 2**32

    # Decode all images first so we can run the face filter BEFORE writing
    # anything to disk (cheaper to reject early).
    decoded: list[tuple[bytes, str]] = []
    for idx, b64 in enumerate(images_b64):
        decoded.append((_decode_image_b64(b64, f"images_b64[{idx}]"), f"images_b64[{idx}]"))

    if inp.get("composition_mode") == "scene_blend" and len(decoded) > 1:
        raw_scene_idx = int(inp.get("scene_image_index", -1))
        scene_idx = len(decoded) - 1 if raw_scene_idx == -1 else max(0, min(raw_scene_idx, len(decoded) - 1))
        face_inputs = [item for idx, item in enumerate(decoded) if idx != scene_idx]
    else:
        face_inputs = decoded[:1]
    _require_detectable_face(
        _as_bool(inp.get("require_detectable_face", False)), face_inputs,
    )

    _apply_face_filter("flux/i2i", job_id, bool(inp.get("face_filter", False)), decoded)
    _apply_logo_filter("flux/i2i", job_id, bool(inp.get("logo_filter", False)), decoded)

    # Stage each input image to ComfyUI's input dir
    input_filenames: list[str] = []
    staged_paths: list[Path] = []
    try:
        for idx, (img_bytes, _) in enumerate(decoded):
            fn = f"flux_i2i_{uuid.uuid4().hex}_{idx}.png"
            p = INPUT_DIR / fn
            p.write_bytes(img_bytes)
            input_filenames.append(fn)
            staged_paths.append(p)

        workflow = build_flux_i2i_workflow(
            input_filenames, prompt, seed,
            megapixels=float(inp.get("megapixels", 2.0)),
            output_width=int(inp.get("width", 0)),
            output_height=int(inp.get("height", 0)),
            steps=int(inp.get("steps", 4)),
            cfg=float(inp.get("cfg", 1.0)),
            guidance=float(inp.get("guidance", 4.0)),
            lora_strength=float(inp.get("lora_strength", 0.0)),
        )

        started = time.time()
        filename, src = await submit_and_wait(workflow)
        dest = _stage_output_to_volume(filename, src, job_id)
        wm_err = _apply_watermark(dest, inp.get("watermark"), inp.get("watermark_image", False))
        dest, delivery = _prepare_delivery_image(dest)
        result = {
            "image_path": str(dest),
            "filename": dest.name,
            "size_bytes": dest.stat().st_size,
            "image_delivery": delivery,
            "seed": seed,
            "ref_count": len(images_b64),
            "duration_seconds": round(time.time() - started, 2),
        }
        if wm_err:
            result["watermark_warning"] = wm_err
        return result
    finally:
        for p in staged_paths:
            p.unlink(missing_ok=True)


ENDPOINTS = {
    "t2i": run_t2i,
    "flux/face-swap": run_flux_face_swap,
    "flux/multi-face-swap": run_flux_multi_face_swap,
    "flux/i2i": run_flux_i2i,
}


# ─────────────────────────────────────────────
# RunPod entrypoint
# ─────────────────────────────────────────────

_comfyui_ready = False


async def handler(event):
    global _comfyui_ready
    if not _comfyui_ready:
        wait_for_comfyui()
        _comfyui_ready = True

    inp = event.get("input") or {}
    endpoint = inp.get("endpoint")
    if endpoint not in ENDPOINTS:
        return {"error": f"unknown endpoint '{endpoint}'; valid: {list(ENDPOINTS)}"}

    job_id = event.get("id") or str(uuid.uuid4())
    try:
        result = await ENDPOINTS[endpoint](inp, job_id)
        # Inline delivery: when the caller sets return_b64=true (the load
        # balancer does — serverless has no persistent file server to fetch
        # /image/<file> from), attach the rendered image as base64 so the proxy
        # can hand the browser a data: URL directly. The image_path on the
        # network volume is still returned for the reaper / S3 fetchers.
        if isinstance(result, dict) and inp.get("return_b64") and result.get("image_path"):
            try:
                result["image_b64"] = _encode_result_b64(Path(result["image_path"]))
                result["mime"] = "image/jpeg"
            except Exception as e:
                result["b64_error"] = f"{type(e).__name__}: {e}"
        return result
    except FilterBlocked as e:
        resp = {
            "error": "blocked",
            "filter": e.filter_name,
            "reason": f"{e.label} matches blocked {e.filter_name}",
            "score": round(e.score, 4),
            "image_index": e.image_index,
        }
        if e.filter_name == "face":
            resp["matched_identity"] = e.matched
        else:
            resp["matched_logo"] = e.matched
        return resp
    except DetectableFaceRequired as e:
        return {
            "error": "image_quality",
            "error_code": "image_quality_insufficient",
            "reason": f"{e.label} does not contain a clearly detectable face.",
            "image_index": e.image_index,
        }
    except FaceValidationUnavailable:
        return {
            "error": "server_busy",
            "error_code": "server_overload",
            "reason": "Face validation is temporarily unavailable.",
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
