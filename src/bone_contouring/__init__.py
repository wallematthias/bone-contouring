"""SimpleITK-first contour and bone-mask generation."""

from .parameters import (
    ContourParameters,
    InnerContourParameters,
    OuterContourParameters,
    SegmentationParameters,
)
from .presets import load_preset, resolve_preset
from .api import GeneratedMasks, generate_bone_segmentation, generate_masks_from_image

__all__ = [
    "ContourParameters",
    "GeneratedMasks",
    "InnerContourParameters",
    "OuterContourParameters",
    "SegmentationParameters",
    "generate_bone_segmentation",
    "generate_masks_from_image",
    "load_preset",
    "resolve_preset",
]
