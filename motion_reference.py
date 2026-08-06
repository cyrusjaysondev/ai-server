"""Safe server-side loading for motion-template reference videos."""

from __future__ import annotations

import asyncio
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit

MOTION_REFERENCE_MAX_BYTES = 100 * 1024 * 1024
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
