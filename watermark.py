"""
watermark.py — overlay a short text label or the Metfone GenAI logo onto generated
images and videos.

Public API:
    apply(path, text="AI") -> None
        Text overlay. No-op when `text` is None or empty after stripping.

    apply_logo(path) -> None
        Composite the Metfone GenAI logo onto the file. No-op when the logo asset
        is missing (setup.sh fetches it on every pod boot — see LOGO_PATH).

Both modify the file at `path` in place. Callers can chain them to stack
text over the logo. Failures are kept tolerant — a missing logo file or
broken ffmpeg leaves the unwatermarked file intact so the API can still
return a usable result.

Style:
  * Text: bold white with a black outline at the bottom-right. Size scales
    to ~4% of image height (min 24 px), padded ~3% from the edges.
  * Logo: PNG with transparency, placed at the bottom-right. Width scales
    to ~12% of the image's shorter side (min 40 px) so the detailed wordmark is
    readable without dominating the frame.

Video paths prefer NVENC and retry through libx264 when CUDA is unavailable.
Audio, if present, is stream-copied so we don't degrade it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import unicodedata
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv"}

_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
# Elegant serif for the horoscope caption (vs the sans logo/watermark text) — an
# upmarket "card" look instead of a heavy subtitle. EB Garamond (OFL, free for
# commercial use) is fetched to the volume by setup.sh; if it's missing on a
# fresh pod we fall back to the bundled DejaVu Serif, then the sans font.
_CAPTION_FONT_PATH = "/workspace/assets/fonts/EBGaramond.ttf"
# EB Garamond covers Latin + Vietnamese but has NO Khmer glyphs (they'd render as
# tofu boxes). Khmer captions use Noto Serif Khmer instead — same elegant serif
# look, OFL-licensed, fetched to the volume by setup.sh. Its first variable axis
# is also Weight, so the same SemiBold pin below applies.
_CAPTION_FONT_KHMER = "/workspace/assets/fonts/NotoSerifKhmer.ttf"
_CAPTION_FONT_FALLBACKS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    _FONT_PATH,
)
_CAPTION_FONT_WEIGHT = 600  # EB Garamond is a variable font — pin SemiBold for readability

# The single fixed-asset brand mark. setup.sh fetches it to the network
# volume so every pod / serverless worker sees it; the URL is duplicated
# here so apply_logo() can self-heal (lazy-download) when called on a pod
# where setup.sh hasn't run the asset step yet.
LOGO_PATH = Path("/workspace/assets/metfone_genai_watermark.png")
LOGO_URL = (
    "https://raw.githubusercontent.com/cyrus688/ai-server/"
    "main/assets/metfone_genai_watermark.png"
)

# Fraction of the image's shorter side used as the visible logo width. This
# mark contains a small wordmark, so 7% was unreadable on 480p/544px outputs.
_LOGO_SCALE = 0.12
# Padding from the side edges, as a fraction of the shorter side.
_EDGE_PAD = 0.03
# Extra padding from the bottom edge — larger so the logo stays above the
# app's download bar / player controls on mobile (roughly 8 % of shorter side).
_BOTTOM_PAD = 0.08


def apply(path, text: str = "AI") -> None:
    """Overlay `text` on the file at `path` in place. No-op on empty text."""
    if text is None:
        return
    text = text.strip()
    if not text:
        return

    p = Path(path)
    ext = p.suffix.lower()
    if ext in IMAGE_EXTS:
        _apply_image(p, text)
    elif ext in VIDEO_EXTS:
        _apply_video(p, text)
    # Unknown extensions (e.g. .gif from a future workflow): leave alone
    # rather than ship a broken file.


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(_FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def _has_khmer(text: str) -> bool:
    """True if any char is in the Khmer Unicode block (U+1780–U+17FF)."""
    return any(0x1780 <= ord(c) <= 0x17FF for c in text)


def _load_caption_font(size: int, text: str = "") -> ImageFont.ImageFont:
    # Pick a font that actually has glyphs for the caption's script: Khmer text
    # needs Noto Serif Khmer (EB Garamond / DejaVu render Khmer as tofu); Latin +
    # Vietnamese stay on EB Garamond (Noto Serif Khmer lacks some Vietnamese
    # diacritics, so it's NOT a universal substitute).
    primary = (_CAPTION_FONT_KHMER,) if _has_khmer(text) else ()
    for path in (*primary, _CAPTION_FONT_PATH, *_CAPTION_FONT_FALLBACKS):
        try:
            f = ImageFont.truetype(path, size)
        except OSError:
            continue
        # Variable fonts (EB Garamond, Noto Serif Khmer) ship multiple weights and
        # have Weight as their first axis — pin a heavier, readable one. Harmless
        # no-op (caught) for static fonts like DejaVu.
        try:
            f.set_variation_by_axes([_CAPTION_FONT_WEIGHT])
        except Exception:
            pass
        return f
    return ImageFont.load_default()


def _apply_image(p: Path, text: str) -> None:
    img = Image.open(p)
    original_mode = img.mode
    img = img.convert("RGBA")
    draw = ImageDraw.Draw(img)

    font_size = max(24, int(img.height * 0.04))
    stroke = max(2, font_size // 12)
    font = _load_font(font_size)

    # textbbox returns ink bounds; we want to right-align by the ink box width.
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad = max(8, int(min(img.width, img.height) * 0.03))
    # bbox[0]/bbox[1] are the bbox origin offset relative to draw origin —
    # subtract so the visual edge lands at (x, y) rather than the origin.
    x = img.width - text_w - pad - bbox[0]
    y = img.height - text_h - pad - bbox[1]

    draw.text(
        (x, y), text,
        font=font,
        fill=(255, 255, 255, 255),
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, 255),
    )

    suffix = p.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        img.convert("RGB").save(p, quality=95)
    elif suffix == ".webp":
        img.save(p, quality=95)
    else:
        # Preserve original mode where possible (avoid bloating greyscale PNGs to RGBA)
        if original_mode != "RGBA":
            img.convert(original_mode if original_mode in ("RGB", "L", "P") else "RGB").save(p)
        else:
            img.save(p)


def _probe_height(p: Path) -> int:
    """Return the video's frame height, or 720 as a safe fallback."""
    return _probe_dimensions(p)[1]


# ─── Encoder selection ─────────────────────────────────────────────────────


def _detect_nvenc() -> bool:
    """Probe once for `h264_nvenc` support. RunPod's CUDA ffmpeg builds
    almost always have it; the legacy 5090 driver definitely does. Cached
    per process so we don't re-shell on every encode.
    """
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-encoders"],
            capture_output=True, text=True, timeout=5,
        )
        return "h264_nvenc" in out.stdout
    except Exception:
        return False


_HAS_NVENC = _detect_nvenc()


def _video_encode_args(*, prefer_gpu: bool = True) -> list[str]:
    """Return ffmpeg `-c:v ...` + preset args.

    GPU re-encode of a 2-5 s clip lands in well under a second on a 5090,
    vs. 30-80 s with software libx264. Fall back to the latter only if
    NVENC isn't available (CPU-only pods, or stripped ffmpeg builds).
    """
    if prefer_gpu and _HAS_NVENC:
        # `-cq` controls quality (lower = better). 22 is visually
        # indistinguishable from the source for a watermark pass.
        # `p4` is the balanced NVENC preset; p1 is fastest, p7 is best.
        return [
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-cq", "22",
            "-tune", "hq",
        ]
    return [
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
    ]


def _run_video_ffmpeg(prefix: list[str], suffix: list[str]) -> subprocess.CompletedProcess[str]:
    """Run an encode, retrying with libx264 when NVENC cannot initialize."""
    proc = subprocess.run(
        [*prefix, *_video_encode_args(), *suffix],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0 or not _HAS_NVENC:
        return proc

    # Some ffmpeg builds advertise NVENC even when the CUDA driver is not
    # currently reachable. A CPU retry keeps post-processing from failing.
    Path(suffix[-1]).unlink(missing_ok=True)
    print("[watermark] NVENC encode failed; retrying with libx264")
    return subprocess.run(
        [*prefix, *_video_encode_args(prefer_gpu=False), *suffix],
        capture_output=True,
        text=True,
    )


def _probe_dimensions(p: Path) -> tuple[int, int]:
    """Return (width, height), or (1280, 720) as a safe fallback.

    We probe once before building each watermark filter so the filter
    expression contains only concrete integers — ffmpeg's `-filter_complex`
    parser treats commas as filter separators, so `max(8, min(w,h)*0.03)`
    style expressions get mis-tokenised. Computing in Python sidesteps the
    whole class of bugs.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-loglevel", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0:s=x", str(p)],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            txt = out.stdout.strip()
            if "x" in txt:
                w, h = txt.split("x", 1)
                if w.isdigit() and h.isdigit():
                    return int(w), int(h)
    except Exception:
        pass
    return 1280, 720


def _apply_video(p: Path, text: str) -> None:
    # Escape ffmpeg filtergraph metacharacters in `text`. Order matters:
    # backslash first, then the others.
    safe = (
        text.replace("\\", r"\\\\")
            .replace(":", r"\:")
            .replace("'", r"\'")
            .replace(",", r"\,")
    )
    # Keep the original extension at the end — ffmpeg picks its muxer from
    # the filename suffix (foo.mp4 → mp4 muxer). foo.mp4.wm.tmp would fail.
    tmp = p.with_name(f"{p.stem}.wm{p.suffix}")
    # ffmpeg drawtext's borderw / fontsize need integers, not expressions.
    # Probe once and compute proportionate values (~4.5% font height, ≥2px border).
    height = _probe_height(p)
    fontsize = max(18, height // 22)
    borderw = max(2, height // 300)
    drawtext = (
        f"drawtext=fontfile={_FONT_PATH}:"
        f"text='{safe}':"
        f"fontcolor=white:fontsize={fontsize}:"
        f"borderw={borderw}:bordercolor=black:"
        f"x=w-tw-h/30:y=h-th-h/30"
    )
    prefix = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(p),
        "-vf", drawtext,
    ]
    suffix = [
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(tmp),
    ]
    proc = _run_video_ffmpeg(prefix, suffix)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg drawtext failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()[-500:] or '(no stderr)'}"
        )
    shutil.move(str(tmp), str(p))


# ─── Logo overlay (image asset, not text) ─────────────────────────────────


def apply_logo(path) -> None:
    """Composite the Metfone GenAI logo onto the file at `path` in place.

    Self-healing: if LOGO_PATH doesn't exist (e.g. setup.sh's asset step
    didn't run on this pod), we lazy-download it once from LOGO_URL and
    cache it on the volume so every subsequent request — and every other
    worker sharing the volume — finds it instantly.

    If the download itself fails we log and skip the overlay; the
    unwatermarked file is still a valid result.
    """
    if not LOGO_PATH.exists():
        if not _try_download_logo():
            return  # download already logged the reason

    p = Path(path)
    ext = p.suffix.lower()
    if ext in IMAGE_EXTS:
        _apply_image_logo(p)
    elif ext in VIDEO_EXTS:
        _apply_video_logo(p)


def _try_download_logo() -> bool:
    """Fetch LOGO_URL → LOGO_PATH. Returns True on success.

    Uses a .tmp file + atomic rename so two concurrent jobs can't see a
    half-written PNG. If the temp already exists (another worker mid-fetch)
    we just bail and let the next request retry — the alternative is
    serialising every first-image-watermark request behind a file lock.
    """
    try:
        LOGO_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = LOGO_PATH.with_suffix(LOGO_PATH.suffix + ".tmp")
        if tmp.exists():
            print(f"[watermark] {tmp} already exists — another worker downloading; skipping this round")
            return False
        print(f"[watermark] logo missing at {LOGO_PATH}; fetching from {LOGO_URL}")
        with urllib.request.urlopen(LOGO_URL, timeout=15) as resp:
            data = resp.read()
        if not data:
            print("[watermark] download returned empty body")
            return False
        tmp.write_bytes(data)
        os.replace(tmp, LOGO_PATH)
        print(f"[watermark] logo cached at {LOGO_PATH} ({len(data)} bytes)")
        return True
    except Exception as exc:
        print(f"[watermark] logo download failed: {exc}")
        return False


def _logo_content_bbox() -> tuple[int, int, int, int] | None:
    """Return the non-transparent logo bounds, excluding canvas padding."""
    try:
        with Image.open(LOGO_PATH) as logo:
            rgba = logo.convert("RGBA")
            return rgba.getchannel("A").getbbox()
    except Exception:
        return None


def _apply_image_logo(p: Path) -> None:
    base = Image.open(p)
    original_mode = base.mode
    base = base.convert("RGBA")

    logo = Image.open(LOGO_PATH).convert("RGBA")
    content_bbox = logo.getchannel("A").getbbox()
    if content_bbox:
        logo = logo.crop(content_bbox)
    shorter = min(base.width, base.height)
    target_w = max(40, int(shorter * _LOGO_SCALE))
    ratio = target_w / logo.width
    target_h = max(8, int(logo.height * ratio))
    logo = logo.resize((target_w, target_h), Image.LANCZOS)

    pad = max(8, int(shorter * _EDGE_PAD))
    bottom_pad = max(8, int(shorter * _BOTTOM_PAD))
    x = base.width - target_w - pad
    y = base.height - target_h - bottom_pad
    # Use the logo's own alpha as the paste mask so PNG transparency is honoured.
    base.alpha_composite(logo, dest=(x, y))

    ext = p.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        base.convert("RGB").save(p, quality=95)
    elif ext == ".webp":
        base.save(p, quality=95)
    elif original_mode != "RGBA":
        target_mode = original_mode if original_mode in ("RGB", "L", "P") else "RGB"
        base.convert(target_mode).save(p)
    else:
        base.save(p)


def _apply_video_logo(p: Path) -> None:
    """Overlay LOGO_PATH onto the video at `p` via ffmpeg `overlay`.

    The visible logo is sized to ~12 % of the video's shorter side. Side padding is
    ~3 % of the shorter side; bottom padding is ~8 % so the mark clears the
    app's download bar / player controls on mobile. We probe the input first
    and bake concrete integers into the filter string so ffmpeg's filtergraph
    parser (which treats commas as filter separators) doesn't mistokenise
    `max(..., ...)` expressions.
    """
    w, h = _probe_dimensions(p)
    shorter = min(w, h)
    pad = max(8, int(shorter * _EDGE_PAD))
    bottom_pad = max(8, int(shorter * _BOTTOM_PAD))
    target_w = max(40, int(shorter * _LOGO_SCALE))

    tmp = p.with_name(f"{p.stem}.wm{p.suffix}")
    content_bbox = _logo_content_bbox()
    crop_filter = ""
    if content_bbox:
        left, top, right, bottom = content_bbox
        crop_filter = f"crop={right - left}:{bottom - top}:{left}:{top},"
    # main_w / main_h / overlay_w / overlay_h are valid overlay-filter
    # variables and contain no commas; safe to leave as expressions.
    filter_complex = (
        f"[1:v]{crop_filter}scale={target_w}:-1[wm];"
        f"[0:v][wm]overlay=x=main_w-overlay_w-{pad}:"
        f"y=main_h-overlay_h-{bottom_pad}:format=auto"
    )
    prefix = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(p),
        "-i", str(LOGO_PATH),
        "-filter_complex", filter_complex,
    ]
    suffix = [
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(tmp),
    ]
    proc = _run_video_ffmpeg(prefix, suffix)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg overlay failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()[-500:] or '(no stderr)'}"
        )
    shutil.move(str(tmp), str(p))


# ─── Caption overlay (styled lower-third body text, e.g. horoscope copy) ────
#
# Design (fixed server-side so every caption looks identical regardless of
# caller): centered, word-wrapped white text with a heavy black stroke in the
# lower third. No background panel ("body only") — the stroke keeps it legible
# over busy scenes. The SAME renderer is used for stills and video so an image
# and its animated version share pixel-identical styling. Videos fade the
# caption in ~1s after the start; stills are static.

# Caption sizing as fractions of the frame's shorter side.
_CAPTION_FONT_SCALE = 0.052     # ~5.2% of the shorter side
_CAPTION_BOTTOM_PAD = 0.14      # text block sits ~14% above the bottom edge (clears the video player controls)
_CAPTION_SIDE_PAD = 0.07        # wrap within (1 - 2*side_pad) of the width
_CAPTION_LINE_SPACING = 1.25    # multiple of the font's line height
# Fade-in timing for video captions (seconds).
_CAPTION_FADE_START = 1.0
_CAPTION_FADE_DUR = 0.6

# Optional zodiac icon + gold divider stacked above the caption text. Assets
# live on the shared network volume (uploaded once; visible to every pod).
ZODIAC_DIR = Path("/workspace/assets/zodiac-overlays")
_VALID_SIGNS = {
    "aries", "taurus", "gemini", "cancer", "leo", "virgo",
    "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces",
}
_CAPTION_ICON_SCALE = 0.16      # zodiac glyph width as fraction of shorter side
_CAPTION_DIVIDER_SCALE = 0.62   # gold divider width as fraction of frame width
_CAPTION_ELEM_GAP = 0.012       # vertical gap between icon/divider/text


def _resolve_zodiac_icon(icon_sign: str | None) -> Path | None:
    """Map a sign name (e.g. 'taurus', 'Taurus ♉') to its icon path, or None.
    Sanitised to the 12 known signs so the caller can't path-traverse."""
    if not icon_sign:
        return None
    sign = "".join(c for c in icon_sign.lower() if c.isalpha())
    if sign not in _VALID_SIGNS:
        return None
    p = ZODIAC_DIR / "icons" / f"{sign}.png"
    return p if p.exists() else None


def _grapheme_clusters(s: str) -> list[str]:
    """Split a string into rendering clusters so we never break between a base
    char and its combining marks. Handles Khmer COENG (U+17D2), which pulls the
    following consonant into a subscript cluster — important because Khmer has no
    spaces between words, so long captions must break mid-token at safe points."""
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        cl = s[i]
        i += 1
        while i < n:
            c = s[i]
            if c == "្":  # Khmer coeng: attach it AND the consonant it subscripts
                cl += c
                i += 1
                if i < n:
                    cl += s[i]
                    i += 1
            elif unicodedata.category(c) in ("Mn", "Mc", "Me"):  # combining marks
                cl += c
                i += 1
            else:
                break
        out.append(cl)
    return out


def _break_long_token(token: str, fits, max_width: int) -> list[str]:
    """Break one over-wide token into cluster-aligned chunks that each fit
    `max_width`. Used for scripts without spaces (Khmer, CJK, Thai…)."""
    chunks: list[str] = []
    cur = ""
    for cl in _grapheme_clusters(token):
        trial = cur + cl
        if not cur or fits(trial, max_width):
            cur = trial
        else:
            chunks.append(cur)
            cur = cl
    if cur:
        chunks.append(cur)
    return chunks or [token]


def _wrap_text_to_width(text: str, font, draw, stroke: int, max_width: int) -> list[str]:
    """Greedy word-wrap so each rendered line fits within `max_width` px. Words
    that are themselves wider than `max_width` (e.g. an un-spaced Khmer clause)
    are broken at safe cluster boundaries instead of overflowing the frame."""
    def fits(s: str, w: int) -> bool:
        bbox = draw.textbbox((0, 0), s, font=font, stroke_width=stroke)
        return (bbox[2] - bbox[0]) <= w

    lines: list[str] = []
    for paragraph in text.splitlines() or [text]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        cur = ""
        for w in words:
            trial = f"{cur} {w}".strip()
            if fits(trial, max_width) or not cur:
                # The word may fit appended; but if it stands alone and is still
                # too wide, break it at cluster boundaries.
                if not cur and not fits(w, max_width):
                    pieces = _break_long_token(w, fits, max_width)
                    lines.extend(pieces[:-1])
                    cur = pieces[-1]
                else:
                    cur = trial
            else:
                lines.append(cur)
                if not fits(w, max_width):
                    pieces = _break_long_token(w, fits, max_width)
                    lines.extend(pieces[:-1])
                    cur = pieces[-1]
                else:
                    cur = w
        if cur:
            lines.append(cur)
    return lines


def _scaled_asset(path: Path, target_w: int) -> Image.Image | None:
    """Load an RGBA asset and scale it to target_w preserving aspect ratio."""
    try:
        img = Image.open(path).convert("RGBA")
        h = max(1, int(img.height * (target_w / img.width)))
        return img.resize((max(1, target_w), h), Image.LANCZOS)
    except Exception:
        return None


def _render_caption_overlay(text: str, width: int, height: int,
                            icon_sign: str | None = None) -> Image.Image:
    """Return a transparent RGBA image (width x height) with the caption stacked
    in the lower third: optional zodiac glyph, optional gold divider, then the
    centered, stroked white text. The icon/divider only appear when a valid
    zodiac sign is given (horoscope use); otherwise it's plain body text."""
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    shorter = min(width, height)
    font_size = max(26, int(shorter * _CAPTION_FONT_SCALE))
    stroke = max(1, font_size // 22)   # thin edge — readability without the heavy "subtitle" border
    font = _load_caption_font(font_size, text)

    max_width = max(1, int(width * (1 - 2 * _CAPTION_SIDE_PAD)))
    lines = _wrap_text_to_width(text, font, draw, stroke, max_width)

    ascent, descent = font.getmetrics()
    line_h = int((ascent + descent) * _CAPTION_LINE_SPACING)
    text_block_h = line_h * len(lines) if lines else 0
    gap = max(6, int(shorter * _CAPTION_ELEM_GAP))

    # Optional zodiac glyph + gold divider (only when a valid sign is given,
    # and only the divider when the icon resolves — they're a horoscope pair).
    icon_img = div_img = None
    icon_path = _resolve_zodiac_icon(icon_sign)
    if icon_path is not None:
        icon_img = _scaled_asset(icon_path, max(32, int(shorter * _CAPTION_ICON_SCALE)))
        div_path = ZODIAC_DIR / "divider-gold.png"
        if icon_img is not None and div_path.exists():
            div_img = _scaled_asset(div_path, max(32, int(width * _CAPTION_DIVIDER_SCALE)))

    total_h = text_block_h
    if icon_img is not None:
        total_h += icon_img.height + gap
    if div_img is not None:
        total_h += div_img.height + gap
    if total_h == 0:
        return overlay

    bottom_pad = max(12, int(shorter * _CAPTION_BOTTOM_PAD))
    y = height - bottom_pad - total_h

    if icon_img is not None:
        overlay.alpha_composite(icon_img, dest=((width - icon_img.width) // 2, y))
        y += icon_img.height + gap
    if div_img is not None:
        overlay.alpha_composite(div_img, dest=((width - div_img.width) // 2, y))
        y += div_img.height + gap

    # Elegant "card" caption (not a heavy subtitle): a soft, blurred drop shadow
    # for depth + readability on any backdrop, then warm cream-gold serif text
    # with a thin dark edge — matching the gold glyph + divider.
    CREAM = (245, 232, 198, 255)
    STROKE_C = (38, 26, 10, 210)
    SHADOW_C = (0, 0, 0, 165)
    sh_off = max(2, int(font_size * 0.05))

    placed = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke)
        x = (width - (bbox[2] - bbox[0])) // 2 - bbox[0]
        placed.append((x, y - bbox[1], line))
        y += line_h

    shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    for x, ty, line in placed:
        sdraw.text((x + sh_off, ty + sh_off), line, font=font,
                   fill=SHADOW_C, stroke_width=stroke, stroke_fill=SHADOW_C)
    shadow = shadow.filter(ImageFilter.GaussianBlur(max(2, font_size // 10)))
    overlay.alpha_composite(shadow)

    for x, ty, line in placed:
        draw.text((x, ty), line, font=font, fill=CREAM,
                  stroke_width=stroke, stroke_fill=STROKE_C)
    return overlay


def apply_caption(path, text: str, fade_in: bool = True,
                  icon_sign: str | None = None) -> None:
    """Overlay `text` as a styled lower-third caption on the file at `path`,
    in place. When `icon_sign` is a valid zodiac sign, a gold glyph + divider
    are stacked above the text. Images: static. Videos: fades in ~1s after
    start. No-op on empty text/unknown extension; failures leave the original
    intact. NOTE: an empty/None `text` is still a no-op even if icon_sign is
    set — pass at least a space if you want icon-only."""
    if text is None:
        return
    text = text.strip()
    if not text:
        return
    p = Path(path)
    ext = p.suffix.lower()
    if ext in IMAGE_EXTS:
        _apply_image_caption(p, text, icon_sign)
    elif ext in VIDEO_EXTS:
        _apply_video_caption(p, text, fade_in, icon_sign)


def _apply_image_caption(p: Path, text: str, icon_sign: str | None = None) -> None:
    base = Image.open(p)
    original_mode = base.mode
    base = base.convert("RGBA")
    overlay = _render_caption_overlay(text, base.width, base.height, icon_sign)
    base.alpha_composite(overlay)

    ext = p.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        base.convert("RGB").save(p, quality=95)
    elif ext == ".webp":
        base.save(p, quality=95)
    elif original_mode != "RGBA":
        base.convert(original_mode if original_mode in ("RGB", "L", "P") else "RGB").save(p)
    else:
        base.save(p)


def _apply_video_caption(p: Path, text: str, fade_in: bool,
                         icon_sign: str | None = None) -> None:
    w, h = _probe_dimensions(p)
    overlay = _render_caption_overlay(text, w, h, icon_sign)
    cap_png = p.with_name(f"{p.stem}.caption.png")
    overlay.save(cap_png)

    tmp = p.with_name(f"{p.stem}.cap{p.suffix}")
    # The overlay PNG is full-frame (text in the lower third, transparent
    # elsewhere). It's a single still, so `-loop 1` turns it into a continuous
    # stream we can fade over time and overlay across every frame; `shortest=1`
    # ends the output with the video (not the looping image).
    if fade_in:
        filter_complex = (
            f"[1:v]fade=in:st={_CAPTION_FADE_START}:d={_CAPTION_FADE_DUR}:alpha=1[cap];"
            f"[0:v][cap]overlay=0:0:format=auto:shortest=1"
        )
    else:
        filter_complex = "[0:v][1:v]overlay=0:0:format=auto:shortest=1"
    prefix = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(p),
        "-loop", "1", "-i", str(cap_png),
        "-filter_complex", filter_complex,
    ]
    suffix = [
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(tmp),
    ]
    proc = _run_video_ffmpeg(prefix, suffix)
    cap_png.unlink(missing_ok=True)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg caption overlay failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()[-500:] or '(no stderr)'}"
        )
    shutil.move(str(tmp), str(p))
