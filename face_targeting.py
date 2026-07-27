"""Backward-compatible exports for multi-person face targeting helpers.

The canonical implementations live in ``workflows.py`` because every existing
pod hot-refresh downloads that file. Keeping this wrapper lets older imports
and tests continue to work without making a first-time live deployment depend
on downloading an additional module.
"""

from workflows import (
    VALID_TARGET_FACE_INDICES,
    normalize_target_face_indices,
    order_face_bboxes,
    preserve_selected_faces,
)

__all__ = [
    "VALID_TARGET_FACE_INDICES",
    "normalize_target_face_indices",
    "order_face_bboxes",
    "preserve_selected_faces",
]
