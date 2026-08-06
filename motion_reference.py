"""Safe loading and lightweight quality checks for motion-template videos."""

from __future__ import annotations

import asyncio
import subprocess
import uuid
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit

MOTION_REFERENCE_MAX_BYTES = 100 * 1024 * 1024
MOTION_IDENTITY_SAMPLE_FPS = 4
MOTION_IDENTITY_MAX_SAMPLED_FRAMES = 60
MOTION_IDENTITY_MEDIAN_THRESHOLD = 0.30
MOTION_IDENTITY_STABLE_FRAME_THRESHOLD = 0.22
MOTION_IDENTITY_STABLE_FRACTION_THRESHOLD = 0.70
MOTION_QUALITY_SAMPLE_FPS = 5
MOTION_QUALITY_WIDTH = 144
MOTION_QUALITY_HEIGHT = 240
MOTION_QUALITY_MEAN_DELTA_THRESHOLD = 0.75
MOTION_QUALITY_MOVING_FRACTION_THRESHOLD = 0.70
_PUBLIC_TEMPLATE_PATH = "/storage/v1/object/public/template-assets/"
_VIDEO_CONTENT_TYPES = {
    "application/octet-stream",
    "video/mp4",
    "video/quicktime",
    "video/webm",
    "video/x-matroska",
}
_VIDEO_EXTENSIONS = {".gif", ".mkv", ".mov", ".mp4", ".webm"}


class MotionReferenceValidationError(ValueError):
    """The caller supplied a URL outside the public template-assets bucket."""


class MotionReferenceTooLargeError(ValueError):
    """The reference exceeds the pod's bounded download size."""


class MotionReferenceDownloadError(RuntimeError):
    """The trusted reference could not be downloaded from storage."""


def summarize_motion_identity(
    scores: list[float],
    *,
    sampled_frames: int,
) -> dict[str, Any]:
    """Summarize per-frame identity scores with the production thresholds."""

    metrics: dict[str, Any] = {
        "available": True,
        "sample_rate_fps": MOTION_IDENTITY_SAMPLE_FPS,
        "max_sampled_frames": MOTION_IDENTITY_MAX_SAMPLED_FRAMES,
        "detected_frames": len(scores),
        "sampled_frames": sampled_frames,
    }
    required_detections = max(3, int(sampled_frames * 0.7 + 0.999))
    if len(scores) < required_detections:
        return {
            **metrics,
            "passed": False,
            "reason": "face was not detectable throughout the clip",
        }

    ordered = sorted(scores)
    middle = len(ordered) // 2
    median_score = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    stable_fraction = sum(
        score >= MOTION_IDENTITY_STABLE_FRAME_THRESHOLD for score in scores
    ) / len(scores)
    passed = (
        median_score >= MOTION_IDENTITY_MEDIAN_THRESHOLD
        and stable_fraction >= MOTION_IDENTITY_STABLE_FRACTION_THRESHOLD
    )
    return {
        **metrics,
        "passed": passed,
        "median_similarity": round(median_score, 4),
        "minimum_similarity": round(min(scores), 4),
        "stable_frame_fraction": round(stable_fraction, 3),
        "reason": None if passed else "the generated face changed during the motion",
    }


def assess_motion_identity(
    source_image_path: Path,
    video_path: Path,
    *,
    face_detector: Any,
    output_dir: Path,
) -> dict[str, Any]:
    """Measure identity at 4 fps across the complete result, up to 60 frames."""

    base_metrics = {
        "sample_rate_fps": MOTION_IDENTITY_SAMPLE_FPS,
        "max_sampled_frames": MOTION_IDENTITY_MAX_SAMPLED_FRAMES,
    }
    if face_detector is None or not hasattr(face_detector, "get_largest_face_embedding"):
        return {
            **base_metrics,
            "available": False,
            "passed": False,
            "reason": "identity detector unavailable",
        }

    try:
        reference = face_detector.get_largest_face_embedding(source_image_path.read_bytes())
        if reference is None:
            return {
                **base_metrics,
                "available": True,
                "passed": False,
                "reason": "no source face detected",
            }

        prefix = f"motion_identity_{uuid.uuid4().hex}"
        pattern = output_dir / f"{prefix}_%03d.jpg"
        extracted = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(video_path),
                "-vf", f"fps={MOTION_IDENTITY_SAMPLE_FPS}",
                "-frames:v", str(MOTION_IDENTITY_MAX_SAMPLED_FRAMES),
                str(pattern),
            ],
            capture_output=True,
            timeout=120,
        )
        frame_paths = sorted(output_dir.glob(f"{prefix}_*.jpg"))
        if extracted.returncode != 0 or not frame_paths:
            for frame_path in frame_paths:
                frame_path.unlink(missing_ok=True)
            return {
                **base_metrics,
                "available": True,
                "passed": False,
                "reason": "could not sample output faces",
            }

        scores: list[float] = []
        try:
            for frame_path in frame_paths:
                embedding = face_detector.get_largest_face_embedding(frame_path.read_bytes())
                if embedding is not None:
                    scores.append(float(sum(
                        float(left) * float(right)
                        for left, right in zip(reference, embedding)
                    )))
        finally:
            for frame_path in frame_paths:
                frame_path.unlink(missing_ok=True)

        return summarize_motion_identity(scores, sampled_frames=len(frame_paths))
    except Exception as exc:
        return {
            **base_metrics,
            "available": False,
            "passed": False,
            "reason": str(exc),
        }


def assess_generated_motion(video_path: Path) -> dict[str, Any]:
    """Measure low-resolution frame deltas across the complete output video."""

    frame_size = MOTION_QUALITY_WIDTH * MOTION_QUALITY_HEIGHT
    base_metrics: dict[str, Any] = {
        "sample_rate_fps": MOTION_QUALITY_SAMPLE_FPS,
        "mean_frame_delta_threshold": MOTION_QUALITY_MEAN_DELTA_THRESHOLD,
        "moving_frame_fraction_threshold": MOTION_QUALITY_MOVING_FRACTION_THRESHOLD,
    }
    try:
        analysis = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", str(video_path),
                "-vf", (
                    f"fps={MOTION_QUALITY_SAMPLE_FPS},"
                    f"scale={MOTION_QUALITY_WIDTH}:{MOTION_QUALITY_HEIGHT},format=gray"
                ),
                "-f", "rawvideo", "-pix_fmt", "gray", "-",
            ],
            capture_output=True,
            timeout=120,
        )
        raw = analysis.stdout
        frame_count = len(raw) // frame_size
        if analysis.returncode != 0 or len(raw) % frame_size or frame_count < 2:
            return {
                **base_metrics,
                "available": analysis.returncode == 0,
                "passed": False,
                "sampled_frames": frame_count,
                "mean_frame_delta": 0.0,
                "moving_frame_fraction": 0.0,
                "reason": "could not assess motion across the output clip",
            }

        frames = [
            memoryview(raw)[index * frame_size:(index + 1) * frame_size]
            for index in range(frame_count)
        ]
        deltas = [
            sum(
                abs(current - previous)
                for current, previous in zip(frames[index], frames[index - 1])
            ) / frame_size
            for index in range(1, frame_count)
        ]
        mean_delta = sum(deltas) / len(deltas)
        moving_fraction = sum(
            delta >= MOTION_QUALITY_MEAN_DELTA_THRESHOLD for delta in deltas
        ) / len(deltas)
        reasons: list[str] = []
        if mean_delta < MOTION_QUALITY_MEAN_DELTA_THRESHOLD:
            reasons.append("output is nearly static")
        if moving_fraction < MOTION_QUALITY_MOVING_FRACTION_THRESHOLD:
            reasons.append("motion is not sustained across the clip")
        return {
            **base_metrics,
            "available": True,
            "passed": not reasons,
            "sampled_frames": frame_count,
            "mean_frame_delta": round(mean_delta, 4),
            "moving_frame_fraction": round(moving_fraction, 4),
            "reason": "; ".join(reasons) if reasons else None,
        }
    except Exception as exc:
        return {
            **base_metrics,
            "available": False,
            "passed": False,
            "sampled_frames": 0,
            "mean_frame_delta": 0.0,
            "moving_frame_fraction": 0.0,
            "reason": str(exc),
        }


def validate_motion_reference_url(value: str) -> str:
    """Allow only HTTPS objects in a Supabase public template-assets bucket."""

    url = value.strip()
    try:
        parsed = urlsplit(url)
        hostname = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError as exc:
        raise MotionReferenceValidationError("Invalid motion reference URL.") from exc

    decoded_path = unquote(parsed.path)
    if (
        parsed.scheme.lower() != "https"
        or not hostname.endswith(".supabase.co")
        or port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not decoded_path.startswith(_PUBLIC_TEMPLATE_PATH)
        or decoded_path == _PUBLIC_TEMPLATE_PATH
        or ".." in PurePosixPath(decoded_path).parts
    ):
        raise MotionReferenceValidationError(
            "Motion references must be public Supabase template-assets videos."
        )
    return url


def motion_reference_extension(url: str) -> str:
    """Return a safe video extension for a validated storage URL."""

    suffix = PurePosixPath(unquote(urlsplit(url).path)).suffix.lower()
    return suffix if suffix in _VIDEO_EXTENSIONS else ".mp4"


async def download_motion_reference(
    value: str,
    *,
    max_bytes: int = MOTION_REFERENCE_MAX_BYTES,
) -> bytes:
    """Download a trusted template reference over HTTP/1.1 with bounded retries.

    Browsers may negotiate HTTP/3 with Supabase/Cloudflare and fail a ranged
    MP4 transfer with ERR_QUIC_PROTOCOL_ERROR. httpx intentionally uses its
    default HTTP/1.1 transport here, keeping that large transfer off the client.
    """

    # Imported lazily so URL validation remains usable in lightweight tooling
    # that does not install the API server's HTTP dependencies.
    import httpx

    url = validate_motion_reference_url(value)
    timeout = httpx.Timeout(90.0, connect=15.0, read=60.0, write=30.0, pool=15.0)
    last_error: Exception | None = None

    for attempt in range(2):
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=False,
                trust_env=False,
            ) as client:
                async with client.stream(
                    "GET",
                    url,
                    headers={
                        "Accept": "video/*,application/octet-stream;q=0.8",
                        "User-Agent": "Pandie-Motion-Pod/1.0",
                    },
                ) as response:
                    if response.status_code != 200:
                        raise MotionReferenceDownloadError(
                            f"Template storage returned HTTP {response.status_code}."
                        )

                    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                    if content_type and content_type not in _VIDEO_CONTENT_TYPES:
                        raise MotionReferenceDownloadError(
                            "Template storage returned a non-video response."
                        )

                    content_length = response.headers.get("content-length")
                    if content_length:
                        try:
                            declared_bytes = int(content_length)
                        except ValueError as exc:
                            raise MotionReferenceDownloadError(
                                "Template storage returned an invalid content length."
                            ) from exc
                        if declared_bytes > max_bytes:
                            raise MotionReferenceTooLargeError(
                                "Motion reference exceeds the 100 MB limit."
                            )

                    chunks: list[bytes] = []
                    downloaded = 0
                    async for chunk in response.aiter_bytes():
                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            raise MotionReferenceTooLargeError(
                                "Motion reference exceeds the 100 MB limit."
                            )
                        chunks.append(chunk)

                    payload = b"".join(chunks)
                    if not payload:
                        raise MotionReferenceDownloadError(
                            "Template storage returned an empty video."
                        )
                    return payload
        except MotionReferenceTooLargeError:
            raise
        except MotionReferenceDownloadError as exc:
            last_error = exc
        except httpx.HTTPError as exc:
            last_error = exc

        if attempt == 0:
            await asyncio.sleep(0.5)

    raise MotionReferenceDownloadError(
        "Could not download the motion template video. Please retry."
    ) from last_error
