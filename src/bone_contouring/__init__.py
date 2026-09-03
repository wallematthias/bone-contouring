"""SimpleITK-first contour and bone-mask generation."""

from importlib.metadata import PackageNotFoundError, version

from .parameters import (
    ContourParameters,
    InnerContourParameters,
    OuterContourParameters,
    SegmentationParameters,
)
from .presets import load_preset, resolve_preset
from .api import GeneratedMasks, generate_bone_segmentation, generate_masks_from_image
from .batch import BoneContouringBatchRow, discover_bone_contouring_batch, run_bone_contouring_batch

try:
    __version__ = version("bone-contouring")
except PackageNotFoundError:  # pragma: no cover - only used from an uninstalled source tree
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "ContourParameters",
    "BoneContouringBatchRow",
    "GeneratedMasks",
    "InnerContourParameters",
    "OuterContourParameters",
    "SegmentationParameters",
    "discover_bone_contouring_batch",
    "generate_bone_segmentation",
    "generate_masks_from_image",
    "load_preset",
    "resolve_preset",
    "run_bone_contouring_batch",
]
