"""
Face blocklist filter — compliance/safety check for image-generation endpoints.

Loads a directory of face images (one per blocked identity), computes
InsightFace embeddings, and provides a single check_image(bytes) method
that returns whether any face in an input image matches the blocklist.

Layout on the network volume:
  /workspace/blocklist/<identity_name>.png       (one face image per identity)
  /workspace/insightface_models/buffalo_l/...    (model cache, downloaded once)

Pod sees these at /workspace/; serverless workers see the same at
/runpod-volume/. The BLOCKLIST_DIR / MODEL_ROOT env vars override defaults.

Threshold (default 0.68): InsightFace embeddings are L2-normalized, so the
similarity score is in [-1, 1]. Same-identity scores are typically > 0.5;
different identities usually score lower, but demographic lookalikes can land
in the high 0.5s or low 0.6s. 0.68 is intentionally precision-first: only a
high-confidence match to an identity explicitly present in the Blocked Faces
list is rejected. Adjust via FACE_FILTER_THRESHOLD if later calibration has a
larger labelled test set.

Imported lazily by the handlers so a worker that never runs face-filtered
requests doesn't pay the import + model-load cost.
"""

from __future__ import annotations

import io
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, Optional


class FilterResult(NamedTuple):
    blocked: bool
    matched_identity: Optional[str]
    score: float
    face_count: int


# ─────────────────────────────────────────────
# Lazy module-level singleton — built on first check_image() call.
# Hot-reloads when the blocklist directory's mtime changes so admin
# uploads/deletes take effect on the next request without restarting
# uvicorn or the serverless worker.
# ─────────────────────────────────────────────

_FILTER = None
_FILTER_BLOCKLIST_MTIME: float = 0.0
_FILTER_INIT_ERROR: Optional[str] = None


def _blocklist_dir() -> Path:
    return Path(os.environ.get("BLOCKLIST_DIR", "/workspace/blocklist"))


def _blocklist_mtime() -> float:
    """Return the most recent mtime across the blocklist dir and its
    contents. We need both because adding/removing files updates the
    directory's mtime; replacing an existing file with new content does
    not (the dir mtime stays the same), but the file's own mtime updates."""
    d = _blocklist_dir()
    if not d.is_dir():
        return 0.0
    latest = d.stat().st_mtime
    for f in d.iterdir():
        if f.is_file():
            latest = max(latest, f.stat().st_mtime)
    return latest


def _maybe_reload():
    """If the blocklist on disk has changed since we built _FILTER, rebuild it.
    Cheap: one stat() per directory entry — negligible vs face-detection cost."""
    global _FILTER, _FILTER_BLOCKLIST_MTIME
    current_mtime = _blocklist_mtime()
    if _FILTER is not None and current_mtime != _FILTER_BLOCKLIST_MTIME:
        # Drop the cached filter so the next check_image rebuilds it. We
        # keep the InsightFace model loaded — it's expensive to re-init.
        _FILTER = None
    _build_filter()
    _FILTER_BLOCKLIST_MTIME = current_mtime


_CACHED_APP = None  # InsightFace FaceAnalysis instance — survives blocklist reloads

# Production is precision-first: a merely similar-looking person must not be
# rejected. This default is deliberately above the observed 0.55-0.60
# lookalike band while remaining below the usual score for a clear reference
# photo of the same identity.
DEFAULT_FACE_FILTER_THRESHOLD = 0.68


def is_confident_face_match(score: float, threshold: float) -> bool:
    """Return True only for a high-confidence Blocked Faces identity match."""
    return score > threshold


# ─────────────────────────────────────────────
# Shared detection fallback chain
#
# Why this exists: SCRFD (the detector behind buffalo_l) has known blind
# spots — old archival B&W portraits, washed-out faces, grainy/soft-focus
# photos, faces that occupy a small fraction of the canvas, oddly-cropped
# faces with heavy borders. The first detection pass often returns zero
# faces on these inputs even when a human can clearly see the face.
#
# Until 2026-05-25 the fallback chain only ran during blocklist BUILD —
# `check_image` used a naive single-pass detector. Result: a generation
# whose input image had any of the above traits silently bypassed the
# face filter (face_count=0 → return blocked=False → no match attempted).
# Admins saw the blocklist load 53/86 photos but had no visibility that
# generation-time checks were ALSO missing on many inputs.
#
# This helper unifies the chain so build AND check use the SAME
# preprocessing variants. The variants below are ordered by speed-to-
# effectiveness ratio so the common case (face detected on first pass)
# stays fast (~50ms) and only stubborn images pay the full price (~400ms).
# ─────────────────────────────────────────────

def _detect_with_fallbacks(app, pil_image, *, np, label_for_log: str = ""):
    """Run face detection with progressive preprocessing fallbacks.

    Returns (faces, fallback_used, detect_img_shape):
      faces           — InsightFace face objects from the variant that worked
      fallback_used   — None if original passed; else the name of the
                        preprocessing step that recovered (for telemetry)
      detect_img_shape — (height, width) of the image the detection ran on.
                        IMPORTANT: bbox coordinates are in this coordinate
                        system, so callers computing area ratios must use
                        these dims, not the original image's.

    Idempotent — never modifies the input PIL image.

    ── CRITICAL: channel-order convention ──
    InsightFace's SCRFD detector + ArcFace embedder both expect **BGR**
    input (OpenCV convention). They were trained on BGR via cv2.imread.
    PIL's Image.open().convert("RGB") + np.array() gives **RGB**, which
    feeds the model swapped R↔B channels. Effect:
       • detection RECALL drops sharply on borderline cases (archival
         B&W, faded photos, certain lighting) — the model can't find a
         face it would otherwise easily detect
       • match SCORES on detected faces shift unpredictably — could
         still match same-person but cross identity boundaries
    Symptom: crystal-clear front-facing portraits returning 0 detections
    while heavily-degraded photos detect fine. THE bug that bypassed
    To Lam / archival portraits in /Users/cyrus/Desktop/testblockedface.
    Fix: explicit RGB→BGR conversion before every app.get() call.
    """
    from PIL import Image as _Image, ImageOps as _ImageOps, ImageFilter as _IF

    def _to_bgr(pil_im):
        """PIL RGB → BGR numpy array. The slice [..., ::-1] reverses the
        last axis (the channel axis). .copy() ensures a contiguous array
        because InsightFace's cv2-backed kernels reject non-contiguous
        memory layouts."""
        return np.array(pil_im)[:, :, ::-1].copy()

    # Helper: is at least one face in the result "significant" (≥0.5%
    # of image area)? Below that, the detection is likely a low-confidence
    # noise blob from the permissive det_thresh=0.1 — accepting it as the
    # final answer causes downstream area filtering to drop us into the
    # "no significant face" bypass branch. Keep trying variants if so.
    #
    # ⚠️ 0.005 (0.5%) here is INTENTIONALLY 10× the MIN_FACE_AREA_RATIO_QUERY
    # (0.05%). The query-time area filter is super-permissive — it accepts
    # any face the detector emits — but THIS check is about "is this variant
    # giving us a real face or just noise". We want to keep trying for a
    # real face before accepting noise.
    SIGNIFICANT_AREA_RATIO = 0.005

    def _is_significant(face_list, h: int, w: int) -> bool:
        img_area = max(1, h * w)
        for f in face_list:
            x1, y1, x2, y2 = f.bbox
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if (area / img_area) >= SIGNIFICANT_AREA_RATIO:
                return True
        return False

    # Track the best (largest-face) fallback we've seen so far, so if no
    # variant produces a significant face we still return the biggest noise
    # blob as a last resort. None until we get our first detection.
    best_faces: list = []
    best_fallback: Optional[str] = None
    best_shape: tuple = (0, 0)
    best_area: float = -1.0

    def _maybe_remember(face_list, name, shape):
        """If `face_list` has a face larger than what we've seen, remember it."""
        nonlocal best_faces, best_fallback, best_shape, best_area
        h, w = shape
        img_area = max(1, h * w)
        for f in face_list:
            x1, y1, x2, y2 = f.bbox
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            ratio = area / img_area
            if ratio > best_area:
                best_area = ratio
                best_faces = face_list
                best_fallback = name
                best_shape = shape

    # Pass 1: original image, no preprocessing. Most images pass here
    # (now that channels are correct).
    arr = _to_bgr(pil_image)
    faces = app.get(arr)
    if faces:
        _maybe_remember(faces, None, arr.shape[:2])
        if _is_significant(faces, *arr.shape[:2]):
            return faces, None, arr.shape[:2]

    # Build the variant pipeline. Each tuple is (name, transform_fn). Order
    # matters — cheap + most-likely-to-work first. We DON'T try all of them
    # exhaustively; we return as soon as one succeeds. The naming should be
    # stable so the audit log lets admins spot trends ("everything is
    # recovering via clahe — start uploading clearer photos").
    def _gamma(im, value: float):
        """Apply gamma correction. <1.0 darkens midtones (helps washed-out
        photos), >1.0 brightens (helps backlit / silhouetted faces)."""
        arr_g = np.array(im).astype(np.float32) / 255.0
        arr_g = np.power(arr_g, value)
        return _Image.fromarray((arr_g * 255.0).clip(0, 255).astype(np.uint8))

    def _clahe(im):
        """Contrast-Limited Adaptive Histogram Equalization — the gold
        standard for resurrecting faces in faded archival photos. Operates
        on the L channel of LAB so colors stay sane."""
        try:
            import cv2  # opencv-python is already a dependency via insightface
            arr_c = np.array(im)
            lab = cv2.cvtColor(arr_c, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l2 = clahe.apply(l)
            merged = cv2.merge((l2, a, b))
            rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
            return _Image.fromarray(rgb)
        except Exception:
            # Fall back to global autocontrast if cv2 is unavailable.
            return _ImageOps.autocontrast(im, cutoff=2)

    def _center_crop_upscale(im, margin_frac: float, scale: int):
        w, h = im.size
        mx, my = int(w * margin_frac), int(h * margin_frac)
        cropped = im.crop((mx, my, w - mx, h - my))
        cw, ch = cropped.size
        return cropped.resize((cw * scale, ch * scale), _Image.LANCZOS)

    def _pad_out(im, factor: float, fill=(255, 255, 255)):
        """Surround the image with a solid border so the face occupies a
        smaller fraction of the canvas. Critical for the under-recognized
        SCRFD failure mode: face TOO BIG (filling >60% of frame). SCRFD's
        anchor boxes are tuned for faces at 10-30% of frame — when a face
        is the entire image (clear portrait crops, profile pics) the
        detector can fail. Padding (zoom out) brings the face into the
        anchor sweet spot.
          factor=1.5 → final canvas 1.5× the original (face goes from
                       100% → ~44% of canvas)
          factor=2.0 → final canvas 2× the original (face goes to 25%)
        White fill is chosen because most face-swap inputs have light
        backgrounds; for dark-background inputs the border could create
        an edge that SCRFD interprets as a structural feature, but in
        practice padding is harmless even when imperfect.
        """
        w, h = im.size
        nw, nh = int(w * factor), int(h * factor)
        canvas = _Image.new("RGB", (nw, nh), fill)
        # Paste centered.
        ox, oy = (nw - w) // 2, (nh - h) // 2
        canvas.paste(im, (ox, oy))
        return canvas

    variants = [
        # ── PAD-OUT VARIANTS FIRST (face too big) ──
        # Front-loaded because face-TOO-BIG is the #1 confirmed bypass
        # mode for our user base (tight portrait crops of public
        # figures). SCRFD's anchor boxes are tuned for faces at 10-30%
        # of frame — when a face fills >60%, the original-pass detector
        # returns 0 faces. Padding (zooming out) brings the face into
        # the anchor sweet spot. These are also CHEAP (just paste onto
        # a bigger canvas) so trying them first costs little even when
        # they're not needed. White first (common for headshot backdrops),
        # 2x for severe cases, black for dark-background photos.
        ("pad_white_1.5x",     lambda im: _pad_out(im, 1.5, fill=(255, 255, 255))),
        ("pad_white_2x",       lambda im: _pad_out(im, 2.0, fill=(255, 255, 255))),
        ("pad_black_2x",       lambda im: _pad_out(im, 2.0, fill=(0, 0, 0))),
        ("autocontrast",       lambda im: _ImageOps.autocontrast(im, cutoff=2)),
        ("autocontrast+sharpen", lambda im: _ImageOps.autocontrast(im, cutoff=2).filter(
            _IF.UnsharpMask(radius=1.5, percent=130, threshold=2))),
        # CLAHE is the biggest single recovery vector for archival photos.
        # At det_thresh=0.1 it can produce TINY noise detections — we now
        # require a "significant" face (≥0.5% of image area) before
        # returning, so CLAHE no longer hijacks the chain with noise blobs.
        ("clahe",              _clahe),
        # Gamma variants help when the face is silhouetted (gamma>1) or
        # the face is very bright on a dark background (gamma<1).
        ("gamma_0.7_dark",     lambda im: _gamma(im, 0.7)),
        ("gamma_1.4_bright",   lambda im: _gamma(im, 1.4)),
        # Upscales help with small faces — try clahe-then-upscale so we get
        # both benefits without a sixth pass.
        ("2x_upscale",         lambda im: im.resize((im.size[0] * 2, im.size[1] * 2), _Image.LANCZOS)),
        ("clahe+2x_upscale",   lambda im: _clahe(im).resize((im.size[0] * 2, im.size[1] * 2), _Image.LANCZOS)),
        ("4x_autocontrast",    lambda im: _ImageOps.autocontrast(im, cutoff=2).resize(
            (im.size[0] * 4, im.size[1] * 4), _Image.LANCZOS)),
        # CLAHE + pad-out — for archival portraits that are ALSO tight crops.
        ("clahe+pad_white_2x", lambda im: _pad_out(_clahe(im), 2.0, fill=(255, 255, 255))),
        # Crop fallbacks for photos with heavy frames / borders / busy
        # backgrounds that confuse the detector. Tries inner 80% then 60%.
        ("center_crop_80+2x",  lambda im: _center_crop_upscale(im, 0.10, 2)),
        ("center_crop_60+2x",  lambda im: _center_crop_upscale(im, 0.20, 2)),
    ]

    for name, transform in variants:
        try:
            transformed = transform(pil_image)
            # Same RGB→BGR conversion as pass 1 — InsightFace expects BGR.
            arr2 = _to_bgr(transformed)
            faces = app.get(arr2)
            if faces:
                _maybe_remember(faces, name, arr2.shape[:2])
                if _is_significant(faces, *arr2.shape[:2]):
                    # Log at INFO level so admins reading the pod log can
                    # see which variant recovered.
                    if label_for_log:
                        print(f"[face-filter] {label_for_log}: recovered via {name} (significant)")
                    return faces, name, arr2.shape[:2]
        except Exception as e:
            print(f"[face-filter] fallback {name} errored: {e}")

    # No variant produced a significant face. If we collected ANY detection
    # along the way (even a tiny one), return that — better to attempt
    # matching against a noise blob than return zero faces (zero-face means
    # the safety filter passes the image through). The matching threshold
    # (cosine ≥ 0.55 against a real identity) protects us from random
    # noise scoring as a real person.
    if best_faces:
        if label_for_log:
            print(f"[face-filter] {label_for_log}: only tiny detections found, "
                  f"best via {best_fallback} (area_ratio={best_area:.5f})")
        return best_faces, best_fallback, best_shape

    return [], None, arr.shape[:2]


def _build_filter():
    global _FILTER, _FILTER_INIT_ERROR, _CACHED_APP
    try:
        import numpy as np
        from PIL import Image
        from insightface.app import FaceAnalysis
    except ImportError as e:
        _FILTER_INIT_ERROR = f"face filter requested but dependencies missing: {e}. Install insightface + onnxruntime-gpu."
        return

    blocklist_dir = _blocklist_dir()
    model_root    = os.environ.get("INSIGHTFACE_MODEL_ROOT", "/workspace/insightface_models")
    # Cosine-similarity cutoff for "this detected face matches a blocked
    # identity". The older 0.55 default rejected unrelated demographic
    # lookalikes (a confirmed production false positive scored 0.5834 against
    # one Le Duan reference). Blocking is now precision-first: only strong
    # matches to the explicit Blocked Faces references are rejected. Multiple
    # reference photos per person improve recall without lowering this global
    # threshold. Override via FACE_FILTER_THRESHOLD only after calibration.
    threshold     = float(os.environ.get(
        "FACE_FILTER_THRESHOLD", str(DEFAULT_FACE_FILTER_THRESHOLD)
    ))
    # InsightFace's default detection threshold (0.5) — and even our earlier
    # bump to 0.3 — kept rejecting clearly-visible elderly faces. SCRFD-10G
    # under-trains on older subjects and on photos with washed-out colour,
    # so we set the floor much lower (0.1) and use a larger detection canvas
    # (1024² instead of 640²) which dramatically improves recall on small or
    # low-contrast faces. The MATCHING threshold (FACE_FILTER_THRESHOLD)
    # above is independent — it's a cosine-similarity cutoff on the embedding,
    # not on detection, so a more permissive detector doesn't loosen blocking.
    det_thresh    = float(os.environ.get("FACE_DETECTOR_THRESHOLD", "0.1"))
    det_size_edge = int(os.environ.get("FACE_DETECTOR_SIZE", "1024"))

    # Reuse the FaceAnalysis instance across reloads — the model load is
    # ~5s on first call. The blocklist itself is cheap to rescan.
    if _CACHED_APP is None:
        try:
            # Device: default GPU (CUDA→CPU fallback). On serverless the GPU is
            # shared with a resident FLUX model, so InsightFace's cuBLAS calls
            # fail mid-inference (cublasStatus_t / ONNXRuntimeError). Setting
            # FACE_FILTER_DEVICE=cpu forces CPU detection there (~1s/image, no
            # VRAM contention) — same models + blocklist from the volume.
            _cpu = os.environ.get("FACE_FILTER_DEVICE", "").lower() == "cpu"
            _providers = ["CPUExecutionProvider"] if _cpu else ["CUDAExecutionProvider", "CPUExecutionProvider"]
            _CACHED_APP = FaceAnalysis(name="buffalo_l", root=model_root, providers=_providers)
            _CACHED_APP.prepare(ctx_id=-1 if _cpu else 0, det_size=(det_size_edge, det_size_edge), det_thresh=det_thresh)
        except Exception as e:
            _FILTER_INIT_ERROR = f"failed to initialize InsightFace: {e}"
            return
    app = _CACHED_APP

    # Load blocklist. Each successful embedding is one identity the
    # filter can actually block at query time. If detection FAILS on a
    # blocklist image, that identity is unblockable until the file is
    # replaced — so we run the full `_detect_with_fallbacks` chain
    # (10 preprocessing variants including CLAHE for archival photos,
    # gamma for backlit shots, upscale for small faces, and crops for
    # bordered scans). Anything we still can't extract is recorded in
    # `skipped` so admins can see the gap via /admin/blocklist's
    # per-entry `loaded` flag.
    #
    # The SAME helper is used by `check_image` — so an admin uploading a
    # tricky reference photo can be confident that if it loaded here, the
    # detector will also catch the same identity in a generated image
    # (provided the generation isn't even MORE degraded than the
    # reference, which is rare for FLUX outputs).
    blocklist: dict[str, "np.ndarray"] = {}
    skipped: list[str] = []
    for img_path in sorted(blocklist_dir.iterdir()) if blocklist_dir.is_dir() else []:
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[face-filter] WARN: cannot open {img_path.name}: {e}")
            continue

        faces, fallback_used, _ = _detect_with_fallbacks(
            app, pil, np=np, label_for_log=img_path.name,
        )

        if not faces:
            skipped.append(img_path.name)
            print(f"[face-filter] WARN: no face detected in blocklist image {img_path.name} — skipping (will not be blockable!)")
            continue

        # Use the largest face if multiple are detected (assume the
        # blocklist image is well-cropped around one identity).
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        blocklist[img_path.stem] = face.normed_embedding
    if skipped:
        print(f"[face-filter] WARNING: {len(skipped)} blocklist images had no detectable face: {skipped[:10]}{'...' if len(skipped) > 10 else ''}")

    _FILTER = {
        "app": app,
        "blocklist": blocklist,
        "threshold": threshold,
        "np": np,
        "Image": Image,
        "skipped_files": skipped,
    }
    print(f"[face-filter] ready: {len(blocklist)} identities loaded from {blocklist_dir}, threshold={threshold}, skipped={len(skipped)}")


# Smallest fraction of the image area a detected bbox must cover to count
# as a "real" face during upload/face-count checks. Below this, we treat
# the detection as noise — at the permissive detector settings we run
# (det_thresh=0.1), SCRFD sometimes reports a tiny background artifact
# as a low-confidence face. 3% of the image area corresponds to a
# ~177×177 face inside a 1024×1024 frame.
MIN_FACE_AREA_RATIO = float(os.environ.get("FACE_MIN_AREA_RATIO", "0.03"))

# Separate (smaller) threshold for the BLOCKLIST QUERY path. We want
# to catch blocked identities even when they appear small in the
# input — a distant face in a group shot, or a target image where the
# blocked person is just one element. The upload check stays strict
# (so admins don't accidentally add bad photos as identity references),
# but matching is permissive so real blocks don't slip through just
# because the target face happens to be small.
#
# 2026-05-26 LOWERED from 0.005 → 0.0005 (50× more permissive). Real
# bypass case observed: user uploaded 542×542 thumbnails where the
# detected face bbox was ~30×30 (0.3% of image area). At the old 0.5%
# floor those went unmatched even though they cleanly resolved to
# Hun Sen / To Lam under closer inspection. 0.05% = ~18×18 in a 1024²
# frame — close to "anything the detector emits with det_thresh=0.1".
# False-positive risk from comparing tiny detections against the
# blocklist is bounded by the COSINE-SIMILARITY threshold (0.55), which
# is independent of size — random noise blobs won't accidentally score
# >0.55 against a real identity embedding even if they pass the area
# filter, so loosening here is safe.
MIN_FACE_AREA_RATIO_QUERY = float(os.environ.get("FACE_MIN_AREA_RATIO_QUERY", "0.0005"))


def _count_significant_faces(faces, img_area: int) -> int:
    """Count faces whose bbox covers at least MIN_FACE_AREA_RATIO of the
    image. Filters out background noise that the permissive detector
    sometimes catches. Returns 0 if `faces` is empty.
    """
    n = 0
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if img_area > 0 and (area / img_area) >= MIN_FACE_AREA_RATIO:
            n += 1
    return n


def _ensure_app():
    """Build (or reuse) the InsightFace detector WITHOUT loading the blocklist.
    Safe to call when face_filter is off — used by the optional face-refine pass.
    Returns the FaceAnalysis app, or None if init fails."""
    global _CACHED_APP
    if _CACHED_APP is not None:
        return _CACHED_APP
    try:
        from insightface.app import FaceAnalysis
        model_root    = os.environ.get("INSIGHTFACE_MODEL_ROOT", "/workspace/insightface_models")
        det_thresh    = float(os.environ.get("FACE_DETECTOR_THRESHOLD", "0.1"))
        det_size_edge = int(os.environ.get("FACE_DETECTOR_SIZE", "1024"))
        _cpu = os.environ.get("FACE_FILTER_DEVICE", "").lower() == "cpu"
        _providers = ["CPUExecutionProvider"] if _cpu else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        app = FaceAnalysis(name="buffalo_l", root=model_root, providers=_providers)
        app.prepare(ctx_id=-1 if _cpu else 0, det_size=(det_size_edge, det_size_edge), det_thresh=det_thresh)
        _CACHED_APP = app
    except Exception as e:
        print(f"[face-refine] InsightFace init failed: {e}")
        return None
    return _CACHED_APP


def get_largest_face_bbox(image_bytes: bytes):
    """Largest detected face as (x1, y1, x2, y2) integer pixel coords, or None.

    Reuses the same detector + preprocessing-fallback chain as the blocklist
    filter, but only returns geometry (no identity matching). Used by the
    optional 2nd-pass face refiner. Never raises."""
    try:
        import io
        import numpy as np
        from PIL import Image
        app = _ensure_app()
        if app is None:
            return None
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        faces, _, _ = _detect_with_fallbacks(app, pil, np=np, label_for_log="refine")
        if not faces:
            return None
        f = max(faces, key=lambda fc: (fc.bbox[2] - fc.bbox[0]) * (fc.bbox[3] - fc.bbox[1]))
        x1, y1, x2, y2 = (int(round(float(v))) for v in f.bbox)
        return (x1, y1, x2, y2)
    except Exception as e:
        print(f"[face-refine] bbox detection failed: {e}")
        return None


def get_status() -> dict:
    """Return the current filter state WITHOUT rebuilding.

    Used by /admin/blocklist GET to annotate each entry with whether
    it actually got loaded (vs being silently skipped). Lazily triggers
    a build on first call if the filter hasn't been initialized yet,
    so a fresh pod that's never run a generation can still answer.
    """
    _maybe_reload()
    if _FILTER is None:
        return {"initialized": False, "error": _FILTER_INIT_ERROR}
    skipped = _FILTER.get("skipped_files", [])
    return {
        "initialized": True,
        "threshold": _FILTER["threshold"],
        "blocklist_count": len(_FILTER["blocklist"]),
        "skipped_count": len(skipped),
        "skipped_files": list(skipped),  # full list, not truncated
    }


def force_reload_filter() -> dict:
    """Wipe the cached `_FILTER` and re-run `_build_filter()` immediately.

    Lets `/admin/reload-filter` apply config changes (FACE_FILTER_THRESHOLD,
    FACE_DETECTOR_THRESHOLD, FACE_MIN_AREA_RATIO env vars, or new blocklist
    files dropped on disk out-of-band) without restarting uvicorn.

    The underlying FaceAnalysis model instance (`_CACHED_APP`) is preserved
    so we don't pay the ~5s model-load tax on every reload; we only
    re-read the env-tunable knobs and re-scan the blocklist directory for
    embedding extraction.

    Returns a dict the admin endpoint can echo back so the caller sees the
    new threshold + identity count and knows the reload actually took.
    """
    global _FILTER, _FILTER_BLOCKLIST_MTIME
    # Force the next _maybe_reload to fall through to _build_filter even if
    # the blocklist dir mtime hasn't moved.
    _FILTER = None
    _FILTER_BLOCKLIST_MTIME = 0.0
    _build_filter()
    if _FILTER is None:
        return {
            "ok": False,
            "error": _FILTER_INIT_ERROR or "filter rebuild produced no state",
        }
    skipped = _FILTER.get("skipped_files", [])
    return {
        "ok": True,
        "threshold": _FILTER["threshold"],
        "blocklist_count": len(_FILTER["blocklist"]),
        # Surface a small sample so admins eyeballing the response can
        # confirm the entries they expect are loaded.
        "sample_identities": sorted(_FILTER["blocklist"].keys())[:10],
        # Hard misses: images on disk that produced no embedding, even
        # after autocontrast + 2× upscale fallbacks. These identities
        # are UNBLOCKABLE — the admin should replace them with a clearer
        # photo.
        "skipped_count": len(skipped),
        "skipped_files": skipped[:30],
    }


def detect_face_count(image_bytes: bytes) -> int:
    """Count significant faces in the image.

    Used by the admin upload endpoint to validate a blocklist entry.
    Uses the SAME `_detect_with_fallbacks` chain as `_build_filter` and
    `check_image` — critical invariant: if this returns N>=1, then
    `_build_filter` will also load this photo (same detector, same
    preprocessing chain, same input image). The reverse is also true:
    if a photo gets skipped at build time, it would have returned 0
    here too. No more "upload accepted but silently skipped at load".

    Filters detections by bbox area so background noise (low-confidence
    pseudo-faces from the permissive det_thresh=0.1) doesn't count as a
    second face and trigger a false "detected 2 faces" rejection.

    The area filter uses the dimensions of the IMAGE VARIANT that
    succeeded — `_detect_with_fallbacks` returns those — so a successful
    recovery via 2x upscale doesn't inflate area ratios against the
    smaller original.
    """
    _maybe_reload()
    if _FILTER is None:
        raise RuntimeError(_FILTER_INIT_ERROR or "face filter unavailable")
    app   = _FILTER["app"]
    np    = _FILTER["np"]
    Image = _FILTER["Image"]
    try:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return 0

    faces, fallback_used, (detect_h, detect_w) = _detect_with_fallbacks(
        app, pil, np=np, label_for_log="upload",
    )
    if not faces:
        return 0
    return _count_significant_faces(faces, max(1, detect_h * detect_w))


# Max edge length we store for a blocklist entry. Embeddings are computed by
# InsightFace at 112x112 internally, so downscaling above ~1024 has no effect
# on matching accuracy — it only saves disk + reload time.
BLOCKLIST_MAX_EDGE = int(os.environ.get("BLOCKLIST_MAX_EDGE", "1024"))


def normalize_blocklist_image(image_bytes: bytes) -> tuple[bytes, str]:
    """Decode, EXIF-rotate, downscale (if needed) and re-encode as PNG.

    Returns (png_bytes, ".png"). Raises ValueError on undecodable input so
    the caller can return a 400 instead of a 500. Lets the admin upload
    accept phone photos / 4K crops / odd formats without the caller worrying
    about request-body limits or storing 20 MB blobs on the network volume.
    """
    try:
        from PIL import Image, ImageOps
    except ImportError as e:
        raise RuntimeError(f"Pillow missing: {e}")
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)  # honor camera rotation
        img = img.convert("RGB")
    except Exception as e:
        raise ValueError(f"could not decode image: {e}")
    w, h = img.size
    longer = max(w, h)
    if longer > BLOCKLIST_MAX_EDGE:
        scale = BLOCKLIST_MAX_EDGE / longer
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), ".png"


def check_image(image_bytes: bytes) -> FilterResult:
    """Detect faces in an image and compare against the blocklist.

    Returns FilterResult(blocked, matched_identity, score, face_count).
    If the filter can't initialize (missing deps, etc.) this raises
    RuntimeError — the caller should decide whether to fail closed or open.

    ── 2026-05-25 — bypass fix ──
    Previously this function ran SCRFD once on the original image and
    returned (False, None, 0.0, 0) if detection failed. That silently
    bypassed the filter whenever the input image happened to confuse the
    detector — common on FLUX-generated inputs with unusual contrast,
    archival reference photos used as face-swap targets, or any image
    where the face is small / faded / oddly cropped.
    Now it runs the SAME fallback chain as `_build_filter` (autocontrast,
    sharpen, CLAHE, gamma, 2x/4x upscale, center-crops). If any variant
    recovers a detection, we proceed to embedding + matching. Plus every
    call writes one line to /workspace/face_filter_check.log with the
    outcome so a future bypass is auditable instead of invisible.
    """
    # Self-heal: pick up any blocklist additions/removals since last call.
    _maybe_reload()
    if _FILTER is None:
        raise RuntimeError(_FILTER_INIT_ERROR or "face filter unavailable")

    app       = _FILTER["app"]
    blocklist = _FILTER["blocklist"]
    threshold = _FILTER["threshold"]
    np        = _FILTER["np"]
    Image     = _FILTER["Image"]

    # An empty blocklist means nothing to compare against → never blocks.
    # Log it so an admin reviewing bypass.log can distinguish "filter ran
    # but blocklist empty" from "filter never ran".
    if not blocklist:
        _log_check(outcome="no_blocklist", score=0.0, identity=None, faces=0, fallback=None)
        return FilterResult(False, None, 0.0, 0)

    try:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        _log_check(outcome="decode_error", score=0.0, identity=None, faces=0, fallback=None)
        return FilterResult(False, None, 0.0, 0)

    # Run detection with the FULL fallback chain (same as _build_filter).
    # This is the actual bypass fix — without the fallbacks, any input
    # image SCRFD missed on first pass slipped through.
    raw_faces, fallback_used, (detect_h, detect_w) = _detect_with_fallbacks(
        app, pil, np=np, label_for_log="query",
    )
    if not raw_faces:
        # Truly no face anywhere — log so admins can audit. We can't match
        # against the blocklist if we can't extract an embedding; this is
        # the one bypass mode we CAN'T close without a second detector.
        _log_check(outcome="no_face_detected_even_with_fallbacks",
                   score=0.0, identity=None, faces=0, fallback=None)
        return FilterResult(False, None, 0.0, 0)

    # Filter detections by area before matching. CRITICAL: use the area
    # of the IMAGE WE ACTUALLY DETECTED ON (which may be a transformed
    # variant of the original — bboxes are in that coordinate system).
    # Using the original img_area would give wrong ratios when a fallback
    # variant succeeded after upscaling.
    img_area = max(1, detect_h * detect_w)

    # MIN_FACE_AREA_RATIO_QUERY (0.5%) is permissive enough to catch
    # distant subjects in group shots while still filtering true noise
    # from the permissive det_thresh=0.1.
    faces = [
        f for f in raw_faces
        if (max(0.0, f.bbox[2] - f.bbox[0]) * max(0.0, f.bbox[3] - f.bbox[1]) / img_area) >= MIN_FACE_AREA_RATIO_QUERY
    ]
    if not faces:
        # Detector saw only noise-tier blobs. Log so admins notice if
        # this happens often — it means MIN_FACE_AREA_RATIO_QUERY may
        # need tuning, or the input genuinely has no significant face.
        _log_check(outcome="only_noise_detections",
                   score=0.0, identity=None, faces=len(raw_faces), fallback=fallback_used)
        return FilterResult(False, None, 0.0, 0)

    best_score = -1.0
    best_id: Optional[str] = None
    for face in faces:
        emb = face.normed_embedding
        for identity, ref_emb in blocklist.items():
            # Both embeddings are L2-normalized → dot product = cosine similarity
            score = float(np.dot(emb, ref_emb))
            if score > best_score:
                best_score = score
                best_id = identity

    blocked = is_confident_face_match(best_score, threshold)
    _log_check(
        outcome="blocked" if blocked else "no_match",
        score=best_score, identity=best_id, faces=len(faces), fallback=fallback_used,
    )
    return FilterResult(blocked, best_id, best_score, len(faces))


# ─────────────────────────────────────────────
# check_image audit log — one line per call. Distinguishes the three
# bypass scenarios so admins can see WHY an image wasn't blocked:
#   • outcome=blocked              — match found, generation rejected
#   • outcome=no_match             — face detected, no blocklist match
#   • outcome=only_noise           — detector found blobs below area floor
#   • outcome=no_face_detected_*   — detector + 10 fallbacks all failed
#   • outcome=no_blocklist         — blocklist is empty (admin error)
#   • outcome=decode_error         — input bytes not a valid image
# Use `tail -f /workspace/face_filter_check.log | grep no_face_detected`
# to surface bypass-risk inputs in real time.
# ─────────────────────────────────────────────

FACE_FILTER_CHECK_LOG = Path(os.environ.get("FACE_FILTER_CHECK_LOG", "/workspace/face_filter_check.log"))


def _log_check(*, outcome: str, score: float, identity: Optional[str],
               faces: int, fallback: Optional[str]) -> None:
    """Append one line to the check log. Best-effort — never raises."""
    try:
        FACE_FILTER_CHECK_LOG.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        line = (
            f"{ts}\toutcome={outcome}"
            f"\tscore={score:.4f}"
            f"\tidentity={identity or '-'}"
            f"\tfaces={faces}"
            f"\tfallback={fallback or 'none'}\n"
        )
        with open(FACE_FILTER_CHECK_LOG, "a") as f:
            f.write(line)
    except Exception:
        pass  # never let logging break the filter


# ─────────────────────────────────────────────
# Bypass audit log — every face_filter=false call logs here for compliance
# ─────────────────────────────────────────────

BYPASS_LOG = Path(os.environ.get("BYPASS_LOG", "/workspace/face_filter_bypass.log"))


def log_bypass(job_id: str, endpoint: str, note: str = "") -> None:
    """Append a timestamped entry recording a face_filter=false bypass."""
    try:
        BYPASS_LOG.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        line = f"{ts}\t{endpoint}\t{job_id}\t{note}\n"
        with open(BYPASS_LOG, "a") as f:
            f.write(line)
    except Exception:
        # Audit log is best-effort — don't fail the request if disk is full.
        pass
