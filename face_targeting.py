"""Target-position helpers for one- or two-person face replacement."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Callable, Iterable, Sequence


VALID_TARGET_FACE_INDICES = (0, 1)


def normalize_target_face_indices(value, face_count: int) -> list[int]:
    """Return one target index per uploaded face.

    Multipart callers send a comma-separated string (``"0,1"``), while the
    serverless handler receives a JSON list. An omitted value preserves the
    original behavior: uploads map to target slots 0, then 1.
    """
    if face_count not in (1, 2):
        raise ValueError(f"face_count must be 1 or 2, got {face_count}")

    if value is None or value == "":
        indices = list(range(face_count))
    elif isinstance(value, str):
        normalized = value.strip()
        if normalized.startswith("[") and normalized.endswith("]"):
            normalized = normalized[1:-1]
        if not normalized:
            indices = list(range(face_count))
        else:
            try:
                indices = [int(part.strip()) for part in normalized.split(",")]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "target_face_indices must be a comma-separated list containing 0 and/or 1"
                ) from exc
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        try:
            indices = [int(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "target_face_indices must contain integer slot indices"
            ) from exc
    else:
        raise ValueError(
            "target_face_indices must be a comma-separated string or a JSON list"
        )

    if len(indices) != face_count:
        raise ValueError(
            "target_face_indices must contain exactly one index for each face image"
        )
    if len(set(indices)) != len(indices):
        raise ValueError("target_face_indices cannot contain duplicate slots")
    if any(index not in VALID_TARGET_FACE_INDICES for index in indices):
        raise ValueError("target_face_indices values must be 0 or 1")
    return indices


def order_face_bboxes(
    bboxes: Iterable[Sequence[int | float]],
    face_order: str,
) -> list[tuple[int, int, int, int]]:
    """Sort detected faces using the same positional convention as the prompt."""
    normalized = [
        tuple(int(round(float(value))) for value in bbox)
        for bbox in bboxes
    ]

    if face_order == "left-to-right":
        return sorted(normalized, key=lambda box: (box[0] + box[2]) / 2)
    if face_order == "right-to-left":
        return sorted(normalized, key=lambda box: (box[0] + box[2]) / 2, reverse=True)
    if face_order == "top-to-bottom":
        return sorted(normalized, key=lambda box: (box[1] + box[3]) / 2)
    if face_order == "bottom-to-top":
        return sorted(normalized, key=lambda box: (box[1] + box[3]) / 2, reverse=True)
    if face_order == "largest-first":
        return sorted(
            normalized,
            key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
            reverse=True,
        )
    raise ValueError(f"unsupported face_order '{face_order}'")


def preserve_selected_faces(
    image_path: str | Path,
    template_path: str | Path,
    *,
    face_order: str,
    target_face_indices: Sequence[int],
    detect_face_bboxes: Callable[[bytes], Sequence[Sequence[int | float]] | None],
) -> tuple[bool, str]:
    """Composite only selected generated heads onto the original template.

    The generative model may slightly redraw every face in a group photo. This
    function makes that impossible in the delivered result: the original
    template supplies every pixel except feathered head masks for explicitly
    selected target slots. Unselected head regions are additionally protected
    from overlap when two people are close together.
    """
    from PIL import Image, ImageChops, ImageDraw, ImageFilter

    output_path = Path(image_path)
    base_path = Path(template_path)
    generated = Image.open(output_path).convert("RGB")
    width, height = generated.size
    template = Image.open(base_path).convert("RGB")
    if template.size != (width, height):
        template = template.resize((width, height), Image.Resampling.LANCZOS)

    encoded_template = io.BytesIO()
    template.save(encoded_template, format="PNG")
    detected = detect_face_bboxes(encoded_template.getvalue()) or []
    ordered = order_face_bboxes(detected, face_order)
    highest_index = max(target_face_indices)
    if len(ordered) <= highest_index:
        return (
            False,
            f"template has {len(ordered)} detectable faces; target slot {highest_index} is unavailable",
        )

    selected = set(target_face_indices)
    replacement_mask = Image.new("L", (width, height), 0)
    replacement_draw = ImageDraw.Draw(replacement_mask)
    protection_mask = Image.new("L", (width, height), 0)
    protection_draw = ImageDraw.Draw(protection_mask)
    selected_face_size = 1

    for index, (x1, y1, x2, y2) in enumerate(ordered[:2]):
        face_width = max(1, x2 - x1)
        face_height = max(1, y2 - y1)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        if index in selected:
            selected_face_size = max(selected_face_size, face_width, face_height)
            replacement_draw.ellipse(
                [
                    center_x - face_width * 1.15,
                    center_y - face_height * 1.55,
                    center_x + face_width * 1.15,
                    center_y + face_height * 1.35,
                ],
                fill=255,
            )
        else:
            # Protect a slightly larger area around every untouched head so a
            # nearby selected mask can never alter its face or hair.
            protection_draw.ellipse(
                [
                    center_x - face_width * 1.35,
                    center_y - face_height * 1.75,
                    center_x + face_width * 1.35,
                    center_y + face_height * 1.50,
                ],
                fill=255,
            )

    feather = max(5, int(selected_face_size * 0.14))
    replacement_mask = replacement_mask.filter(ImageFilter.GaussianBlur(feather))
    if selected != {0, 1}:
        protection_mask = protection_mask.filter(
            ImageFilter.GaussianBlur(max(3, feather // 2))
        )
        replacement_mask = ImageChops.subtract(replacement_mask, protection_mask)

    composited = template.copy()
    composited.paste(generated, (0, 0), replacement_mask)
    composited.save(output_path)
    return True, f"preserved template pixels outside target slots {sorted(selected)}"
