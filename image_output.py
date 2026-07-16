"""Small, high-quality image outputs for browser delivery.

ComfyUI saves FLUX results as large PNG files. That is useful as a local
master, but it is unnecessarily expensive to send to a phone and was adding
seconds to Pandie's result reveal. This module creates a visually faithful
JPEG derivative that normally lands between 25 KB and 150 KB.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps


DEFAULT_TARGET_MIN_BYTES = 25 * 1024
DEFAULT_TARGET_MAX_BYTES = 150 * 1024
DEFAULT_MAX_EDGES = (1600, 1440, 1280, 1120, 960, 840, 720, 640, 560, 480)
DEFAULT_MIN_QUALITY = 58
DEFAULT_MAX_QUALITY = 92


@dataclass(frozen=True)
class OptimizedImage:
    path: Path
    original_bytes: int
    output_bytes: int
    width: int
    height: int
    quality: int | None


def _jpeg_bytes(image: Image.Image, quality: int) -> bytes:
    buffer = BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=True,
        subsampling="4:2:0",
    )
    return buffer.getvalue()


def optimize_image_file(
    source: str | Path,
    *,
    target_min_bytes: int = DEFAULT_TARGET_MIN_BYTES,
    target_max_bytes: int = DEFAULT_TARGET_MAX_BYTES,
) -> OptimizedImage:
    """Create a compact JPEG beside ``source`` and return its metadata.

    The first (largest) canvas that can fit under ``target_max_bytes`` wins,
    with a binary search for the highest viable JPEG quality. The original is
    removed only after the derivative has been written successfully.
    """

    source_path = Path(source)
    original_bytes = source_path.stat().st_size

    with Image.open(source_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        source_width, source_height = image.size

        best_payload: bytes | None = None
        best_size = image.size
        best_quality: int | None = None

        for max_edge in DEFAULT_MAX_EDGES:
            scale = min(1.0, max_edge / max(source_width, source_height))
            width = max(1, round(source_width * scale))
            height = max(1, round(source_height * scale))
            candidate_image = image
            if (width, height) != image.size:
                candidate_image = image.resize((width, height), Image.Resampling.LANCZOS)

            low = DEFAULT_MIN_QUALITY
            high = DEFAULT_MAX_QUALITY
            under_target: tuple[bytes, int] | None = None

            while low <= high:
                quality = (low + high) // 2
                payload = _jpeg_bytes(candidate_image, quality)
                if best_payload is None or len(payload) < len(best_payload):
                    best_payload = payload
                    best_size = (width, height)
                    best_quality = quality

                if len(payload) <= target_max_bytes:
                    under_target = (payload, quality)
                    low = quality + 1
                else:
                    high = quality - 1

            if under_target is not None:
                payload, quality = under_target
                best_payload = payload
                best_size = (width, height)
                best_quality = quality
                break

    if best_payload is None:
        return OptimizedImage(
            path=source_path,
            original_bytes=original_bytes,
            output_bytes=original_bytes,
            width=source_width,
            height=source_height,
            quality=None,
        )

    # For an already tiny image, keep the source when converting would make it
    # larger. Otherwise the compact JPEG is the browser-facing artifact.
    if original_bytes <= target_max_bytes and len(best_payload) >= original_bytes:
        return OptimizedImage(
            path=source_path,
            original_bytes=original_bytes,
            output_bytes=original_bytes,
            width=source_width,
            height=source_height,
            quality=None,
        )

    output_path = source_path.with_suffix(".jpg")
    temp_path = output_path.with_suffix(".jpg.tmp")
    temp_path.write_bytes(best_payload)
    temp_path.replace(output_path)
    if output_path != source_path:
        source_path.unlink(missing_ok=True)

    # Images with large flat areas can legitimately be below the preferred
    # floor even at maximum quality. The upper bound matters for latency; the
    # lower bound is a quality target, not a reason to add meaningless bytes.
    _ = target_min_bytes
    return OptimizedImage(
        path=output_path,
        original_bytes=original_bytes,
        output_bytes=len(best_payload),
        width=best_size[0],
        height=best_size[1],
        quality=best_quality,
    )
