"""
RunPod serverless handler — Video worker (LTX-2.3 22B).

Endpoints supported via `event["input"]["endpoint"]`:
  - "ltx/i2v"  image-to-video
  - "ltx/t2v"  text-to-video

Models read from the network volume at /runpod-volume/runpod-slim/ComfyUI/models
(linked into ComfyUI's models dir by start.sh before the handler boots).

Output files are written to VOLUME_OUTPUTS. By default the worker keeps the
original /runpod-volume/outputs/<job-id>/<filename> layout. Set
VOLUME_OUTPUTS_INCLUDE_JOB_ID=false when VOLUME_OUTPUTS points at the persistent
pod's flat video folder (/runpod-volume/runpod-slim/ComfyUI/output/video), so
the CMS load balancer can serve the result through /video/<filename>.

Input (i2v):
  {
    "input": {
      "endpoint": "ltx/i2v",
      "image_b64": "iVBOR...",
      "prompt": "...",
      "negative_prompt": "...",       (optional)
      "preset": "fast",               (fast | quality)
      "aspect_ratio": "9:16",
      "width": 544, "height": 960,
      "length": 121, "fps": 24,
      "seed": -1,
      "audio": false,
      "enhance_prompt": true
    }
  }

Input (t2v):
  {
    "input": {
      "endpoint": "ltx/t2v",
      "prompt": "...",
      "negative_prompt": "...",       (optional)
      "preset": "fast",
      "aspect_ratio": "16:9",
      "width": 1280, "height": 720,
      "length": 121, "fps": 24,
      "seed": -1,
      "audio": false
    }
  }

Output (success):
  {
    "video_path": "/runpod-volume/outputs/<job-id>/ltx_i2v_42_00001_.mp4",
    "filename":   "ltx_i2v_42_00001_.mp4",
    "size_bytes": 18472104,
    "seed": 42,
    "duration_seconds": 87.4
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

sys.path.insert(0, "/app")
from workflows import (
    LTX_ASPECT_RATIOS,
    LTX_DEFAULT_NEGATIVE,
    LTX_PRESETS,
    build_ltx_i2v_workflow,
    build_ltx_t2v_workflow,
    compute_ltx_dimensions,
)
try:
    import watermark
except ImportError:
    watermark = None
try:
    import safety as face_safety
except ImportError:
    face_safety = None


COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", "/comfyui"))
OUTPUT_DIR = COMFY_ROOT / "output"
INPUT_DIR = COMFY_ROOT / "input"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VOLUME_OUTPUTS = Path(os.environ.get("VOLUME_OUTPUTS", "/runpod-volume/outputs"))
VOLUME_OUTPUTS.mkdir(parents=True, exist_ok=True)
VOLUME_OUTPUTS_INCLUDE_JOB_ID = (
    os.environ.get("VOLUME_OUTPUTS_INCLUDE_JOB_ID", "true").strip().lower()
    not in {"0", "false", "no", "off"}
)


def _decode_image_b64(b64: str, field_name: str) -> bytes:
    """Decode a base64 image, accepting both raw and data-URI (`data:image/png;base64,…`) forms.
    Raises ValueError with the field name if the input is malformed."""
    if not isinstance(b64, str):
        raise ValueError(f"'{field_name}' must be a base64 string, got {type(b64).__name__}")
    if b64.startswith("data:") and "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        return base64.b64decode(b64, validate=False)
    except Exception as e:
        raise ValueError(f"'{field_name}' is not valid base64: {e}") from e


class DetectableFaceRequired(Exception):
    pass


class FaceValidationUnavailable(Exception):
    pass


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _require_detectable_face(enabled: bool, image_bytes: bytes) -> None:
    if not enabled:
        return
    if face_safety is None:
        raise FaceValidationUnavailable("face validation module is unavailable")
    try:
        face_count = face_safety.detect_face_count(image_bytes)
    except RuntimeError as exc:
        raise FaceValidationUnavailable(str(exc)) from exc
    if face_count < 1:
        raise DetectableFaceRequired("image does not contain a clearly detectable face")


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


async def submit_and_wait(workflow: dict, max_wait_s: float = 600.0) -> tuple[str, Path]:
    """Submit a workflow to ComfyUI, poll /history until done. Race-free vs WS stream."""
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
        for key in ("videos", "gifs", "images"):
            if key in node_output:
                item = node_output[key][0]
                filename = item["filename"]
                subfolder = item.get("subfolder", "")
                path = OUTPUT_DIR / subfolder / filename if subfolder else OUTPUT_DIR / filename
                if path.exists():
                    return filename, path

    raise RuntimeError("ComfyUI completed but no output file was found in history")


def _validate_preset_and_aspect(preset: str, aspect_ratio: str) -> None:
    if preset not in LTX_PRESETS:
        raise ValueError(f"invalid preset '{preset}'; valid: {', '.join(LTX_PRESETS)}")
    if aspect_ratio != "original" and aspect_ratio not in LTX_ASPECT_RATIOS:
        raise ValueError(f"invalid aspect_ratio '{aspect_ratio}'; valid: original, {', '.join(LTX_ASPECT_RATIOS)}")


def _stage_output_to_volume(filename: str, src: Path, job_id: str) -> tuple[Path, str]:
    dest_dir = VOLUME_OUTPUTS / job_id if VOLUME_OUTPUTS_INCLUDE_JOB_ID else VOLUME_OUTPUTS
    dest_dir.mkdir(parents=True, exist_ok=True)
    staged_filename = filename
    dest = dest_dir / staged_filename
    if not VOLUME_OUTPUTS_INCLUDE_JOB_ID and dest.exists():
        staged_filename = f"{job_id}__{filename}"
        dest = dest_dir / staged_filename
    shutil.copy2(src, dest)
    return dest, staged_filename


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


async def run_ltx_i2v(inp: dict, job_id: str) -> dict:
    if not inp.get("image_b64"):
        raise ValueError("'image_b64' is required for ltx/i2v")

    preset = inp.get("preset", "fast")
    aspect_ratio = inp.get("aspect_ratio", "9:16")
    _validate_preset_and_aspect(preset, aspect_ratio)

    seed = inp.get("seed", -1)
    seed = seed if seed != -1 else uuid.uuid4().int % 2**32
    width, height = compute_ltx_dimensions(
        int(inp.get("width", 544)), int(inp.get("height", 960)), aspect_ratio
    )

    img_bytes = _decode_image_b64(inp["image_b64"], "image_b64")
    _require_detectable_face(
        _as_bool(inp.get("require_detectable_face", False)), img_bytes,
    )
    img_filename = f"ltx_i2v_{uuid.uuid4().hex}.png"
    img_path = INPUT_DIR / img_filename
    img_path.write_bytes(img_bytes)

    workflow = build_ltx_i2v_workflow(
        image_filename=img_filename,
        prompt=inp.get("prompt", ""),
        negative_prompt=inp.get("negative_prompt", LTX_DEFAULT_NEGATIVE),
        width=width, height=height,
        length=int(inp.get("length", 121)),
        fps=int(inp.get("fps", 24)),
        seed=seed,
        preset=preset,
        audio=bool(inp.get("audio", False)),
        enhance_prompt=bool(inp.get("enhance_prompt", True)),
        inplace_strength=float(inp.get("inplace_strength", 0.7)),
    )

    started = time.time()
    try:
        filename, src = await submit_and_wait(workflow)
        dest, staged_filename = _stage_output_to_volume(filename, src, job_id)
        wm_err = _apply_watermark(dest, inp.get("watermark"), inp.get("watermark_image", False))
        result = {
            "video_path": str(dest),
            "filename": staged_filename,
            "source_filename": filename,
            "size_bytes": dest.stat().st_size,
            "seed": seed,
            "duration_seconds": round(time.time() - started, 2),
        }
        if wm_err:
            result["watermark_warning"] = wm_err
        return result
    finally:
        img_path.unlink(missing_ok=True)


async def run_ltx_t2v(inp: dict, job_id: str) -> dict:
    prompt = inp.get("prompt")
    if not prompt:
        raise ValueError("'prompt' is required for ltx/t2v")

    preset = inp.get("preset", "fast")
    aspect_ratio = inp.get("aspect_ratio", "16:9")
    _validate_preset_and_aspect(preset, aspect_ratio)

    seed = inp.get("seed", -1)
    seed = seed if seed != -1 else uuid.uuid4().int % 2**32
    width, height = compute_ltx_dimensions(
        int(inp.get("width", 1280)), int(inp.get("height", 720)), aspect_ratio
    )

    workflow = build_ltx_t2v_workflow(
        prompt=prompt,
        negative_prompt=inp.get("negative_prompt", LTX_DEFAULT_NEGATIVE),
        width=width, height=height,
        length=int(inp.get("length", 121)),
        fps=int(inp.get("fps", 24)),
        seed=seed,
        preset=preset,
        audio=bool(inp.get("audio", False)),
    )

    started = time.time()
    filename, src = await submit_and_wait(workflow)
    dest, staged_filename = _stage_output_to_volume(filename, src, job_id)
    wm_err = _apply_watermark(dest, inp.get("watermark"), inp.get("watermark_image", False))
    result = {
        "video_path": str(dest),
        "filename": staged_filename,
        "source_filename": filename,
        "size_bytes": dest.stat().st_size,
        "seed": seed,
        "duration_seconds": round(time.time() - started, 2),
    }
    if wm_err:
        result["watermark_warning"] = wm_err
    return result


ENDPOINTS = {
    "ltx/i2v": run_ltx_i2v,
    "ltx/t2v": run_ltx_t2v,
}


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
        return await ENDPOINTS[endpoint](inp, job_id)
    except DetectableFaceRequired:
        return {
            "error": "image_quality",
            "error_code": "image_quality_insufficient",
            "reason": "image does not contain a clearly detectable face.",
            "image_index": 0,
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
