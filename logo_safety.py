"""
Logo / flag / symbol blocklist filter — parallel to safety.py but using
CLIP image embeddings instead of face-specific embeddings.

Why CLIP? Logos/flags don't have a stable "identity" model like faces do.
CLIP gives a semantic image embedding good for "this image is dominated
by the Apple logo" or "this image is the Korean flag". CLIP is a
whole-image embedding, so it may MISS small logos in a corner of a
larger photo. That's a known limit — accepting it as v1.

Layout on the network volume:
  /workspace/blocklist_logos/<logo_name>.png   (one image per blocked logo)
  /workspace/clip_models/...                   (model cache, downloaded once)

Hot-reload: re-scans the blocklist dir on every check, picks up admin
changes without restarting uvicorn or the serverless worker.

Threshold (default 0.85): CLIP cosine similarity is in [-1, 1]. For real
logos with rich detail (Nike swoosh, national flag, etc.):
  Same image, mild variation:  ~0.92-1.00
  Same logo, different photo:  ~0.85-0.92
  Visually similar but different: ~0.65-0.80
  Unrelated content:           ~0.10-0.50
0.85 errs toward fewer false positives. For SIMPLE geometric/abstract
logos (e.g. circle/triangle shapes), CLIP tends to over-cluster similar
silhouettes — admins should test with their actual blocklist and tune
LOGO_FILTER_THRESHOLD if they see false positives.
"""

from __future__ import annotations

import io
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, Optional


class LogoFilterResult(NamedTuple):
    blocked: bool
    matched_logo: Optional[str]
    score: float


# ─────────────────────────────────────────────
# Module-level singleton, lazy load, hot-reload via blocklist mtime
# ─────────────────────────────────────────────

_FILTER = None
_FILTER_BLOCKLIST_MTIME: float = 0.0
_FILTER_INIT_ERROR: Optional[str] = None
_CACHED_MODEL = None       # open_clip model + preprocess + device — survives reloads
_CACHED_THRESHOLD = 0.0


def _blocklist_dir() -> Path:
    return Path(os.environ.get("BLOCKLIST_LOGOS_DIR", "/workspace/blocklist_logos"))


def _blocklist_mtime() -> float:
    d = _blocklist_dir()
    if not d.is_dir():
        return 0.0
    latest = d.stat().st_mtime
    for f in d.iterdir():
        if f.is_file():
            latest = max(latest, f.stat().st_mtime)
    return latest


def _maybe_reload():
    global _FILTER, _FILTER_BLOCKLIST_MTIME
    current = _blocklist_mtime()
    if _FILTER is not None and current != _FILTER_BLOCKLIST_MTIME:
        _FILTER = None
    _build_filter()
    _FILTER_BLOCKLIST_MTIME = current


def _build_filter():
    global _FILTER, _FILTER_INIT_ERROR, _CACHED_MODEL, _CACHED_THRESHOLD
    try:
        import torch
        import open_clip
        from PIL import Image
    except ImportError as e:
        _FILTER_INIT_ERROR = f"logo filter requires open_clip_torch + torch: {e}"
        return

    blocklist_dir = _blocklist_dir()
    model_root    = os.environ.get("CLIP_MODEL_ROOT", "/workspace/clip_models")
    threshold     = float(os.environ.get("LOGO_FILTER_THRESHOLD", "0.85"))
    _CACHED_THRESHOLD = threshold

    # Build the CLIP pipeline once and reuse across blocklist reloads.
    if _CACHED_MODEL is None:
        try:
            # LOGO_FILTER_DEVICE=cpu forces CPU — on serverless the GPU is busy
            # with a resident FLUX model, so keep CLIP off it (avoids the same
            # VRAM contention that breaks the InsightFace filter). Default auto.
            device = "cpu" if os.environ.get("LOGO_FILTER_DEVICE", "").lower() == "cpu" \
                else ("cuda" if torch.cuda.is_available() else "cpu")
            # Use the QuickGELU variant — open_clip's openai/ViT-B-32 weights
            # were trained with QuickGELU; the default config uses standard
            # GELU and emits a warning + small accuracy degradation.
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32-quickgelu", pretrained="openai", cache_dir=model_root,
            )
            model = model.eval().to(device)
            _CACHED_MODEL = (model, preprocess, device, torch)
        except Exception as e:
            _FILTER_INIT_ERROR = f"failed to load open_clip ViT-B/32: {e}"
            return

    model, preprocess, device, torch = _CACHED_MODEL

    # Build embeddings for each blocklist entry
    blocklist: dict[str, "torch.Tensor"] = {}
    if blocklist_dir.is_dir():
        from PIL import Image
        for img_path in sorted(blocklist_dir.iterdir()):
            if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            try:
                im = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[logo-filter] WARN: cannot open {img_path.name}: {e}")
                continue
            try:
                t = preprocess(im).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model.encode_image(t)
                    emb = emb / emb.norm(dim=-1, keepdim=True)  # L2 normalize
                blocklist[img_path.stem] = emb[0].cpu()
            except Exception as e:
                print(f"[logo-filter] WARN: failed to embed {img_path.name}: {e}")
                continue

    _FILTER = {
        "model": model,
        "preprocess": preprocess,
        "device": device,
        "torch": torch,
        "blocklist": blocklist,
        "threshold": threshold,
    }
    print(f"[logo-filter] ready: {len(blocklist)} logos loaded from {blocklist_dir}, threshold={threshold}")


def check_image(image_bytes: bytes) -> LogoFilterResult:
    """Compute the input's CLIP embedding and compare against the logo blocklist."""
    _maybe_reload()
    if _FILTER is None:
        raise RuntimeError(_FILTER_INIT_ERROR or "logo filter unavailable")

    blocklist = _FILTER["blocklist"]
    if not blocklist:
        return LogoFilterResult(False, None, 0.0)

    model       = _FILTER["model"]
    preprocess  = _FILTER["preprocess"]
    device      = _FILTER["device"]
    torch       = _FILTER["torch"]
    threshold   = _FILTER["threshold"]

    try:
        from PIL import Image
        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return LogoFilterResult(False, None, 0.0)

    t = preprocess(im).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(t)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    query = emb[0].cpu()

    best_score = -1.0
    best_id: Optional[str] = None
    for name, ref in blocklist.items():
        score = float(torch.dot(query, ref).item())
        if score > best_score:
            best_score = score
            best_id = name

    blocked = best_score > threshold
    return LogoFilterResult(blocked, best_id, best_score)


# ─────────────────────────────────────────────
# Admin helper — validates an image is loadable (no face-detection
# precondition; logos may not contain any face).
# ─────────────────────────────────────────────

def validate_uploadable(image_bytes: bytes) -> Optional[str]:
    """Return None if the bytes are a loadable image, else an error message."""
    try:
        from PIL import Image
        Image.open(io.BytesIO(image_bytes)).verify()
        return None
    except Exception as e:
        return f"image is not a valid PNG/JPEG/WebP: {e}"
