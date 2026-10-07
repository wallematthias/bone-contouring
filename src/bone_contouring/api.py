"""SimpleITK-first public contour and mask-generation API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import SimpleITK as sitk

from ._arrays import (
    numpy_xyz_to_sitk_binary,
    segment_bone_xyz,
    sitk_to_numpy_xyz,
)
from .parameters import ContourParameters
from .presets import resolve_preset


@dataclass(slots=True)
class GeneratedMasks:
    """Geometry-preserving masks generated from one input image."""

    seg: sitk.Image
    full: sitk.Image
    trab: sitk.Image
    cort: sitk.Image
    material: sitk.Image
    mask_provenance: dict[str, str]
    metadata: dict[str, Any]


def _validate_image(image: sitk.Image, label: str) -> None:
    if not isinstance(image, sitk.Image):
        raise TypeError(f"{label} must be a SimpleITK.Image.")
    if image.GetDimension() != 3:
        raise ValueError(f"{label} must be three-dimensional, got {image.GetDimension()}D.")


def _geodesic_outer_contour(
    density_xyz: np.ndarray,
    parameters: ContourParameters,
    spacing_xyz: tuple[float, float, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        from hrpqct_geodesic_contour import contour
    except ImportError as exc:
        raise RuntimeError(
            "Geodesic outer contouring requires the optional 'hrpqct-geodesic-contour' dependency."
        ) from exc
    full_mask, support_masks = contour(
        density_xyz,
        voxel_size_mm=spacing_xyz,
        bone_threshold=parameters.outer.geodesic_bone_threshold,
        fill_holes=parameters.outer.geodesic_fill_holes,
    )
    full = np.asarray(full_mask, dtype=bool)
    if full.shape != density_xyz.shape:
        raise ValueError("Geodesic contour output shape must match the input image shape.")
    return full, {"support_mask_count": len(support_masks)}


def _numpy_xyz_to_sitk_label(label_xyz: np.ndarray, reference: sitk.Image) -> sitk.Image:
    """Create a uint8 label image with geometry copied from ``reference``."""
    image = sitk.GetImageFromArray(np.transpose(np.asarray(label_xyz, dtype=np.uint8), (2, 1, 0)))
    image.CopyInformation(reference)
    return sitk.Cast(image, sitk.sitkUInt8)


def generate_masks_from_image(
    image: sitk.Image,
    parameters: ContourParameters | None = None,
    *,
    segmentation_image: sitk.Image | None = None,
) -> GeneratedMasks:
    """Generate `seg`, `full`, `trab`, and `cort` masks from a 3D SimpleITK image."""
    _validate_image(image, "image")
    params = parameters or resolve_preset()
    density_xyz = sitk_to_numpy_xyz(image)
    spacing_xyz = tuple(float(value) for value in image.GetSpacing())
    segmentation_source = image if segmentation_image is None else segmentation_image
    _validate_image(segmentation_source, "segmentation_image")
    segmentation_xyz = sitk_to_numpy_xyz(segmentation_source)
    if segmentation_xyz.shape != density_xyz.shape:
        raise ValueError("segmentation_image size must match image size.")
    segmentation_method = params.segmentation.method.strip().lower()
    if segmentation_method in {"global", "seg_gauss"}:
        segmentation_method = "gauss"
    # Retain requested support provenance, although it no longer drives standard envelopes.
    support_method = (params.segmentation.contour_support_method or segmentation_method).strip().lower()
    if support_method in {"global", "seg_gauss"}:
        support_method = "gauss"

    outer_method = params.outer.contour_method.strip().lower()
    outer_metadata: dict[str, Any] = {}
    from .parameters import STANDARD_ALGORITHM_REVISION
    if outer_method == "standard":
        from . import _stable_3d
        full_xyz, outer_metadata = _stable_3d.outer_contour_xyz(
            density_xyz, params.outer, params.stable_3d, spacing_xyz=spacing_xyz)
        outer_metadata["algorithm_revision"] = STANDARD_ALGORITHM_REVISION
    elif outer_method == "geodesic":
        full_xyz, outer_metadata = _geodesic_outer_contour(density_xyz, params, spacing_xyz)
    elif outer_method == "buie":
        from . import _buie
        full_xyz, outer_metadata = _buie.outer_contour_xyz(
            density_xyz, params.outer.periosteal_threshold, params.buie)
    elif outer_method == "ipl_match":
        from . import _ipl_match
        full_xyz, outer_metadata = _ipl_match.outer_contour_xyz(
            density_xyz, params.outer, spacing_xyz=spacing_xyz)
    elif outer_method == "stable_3d":
        from . import _stable_3d
        full_xyz, outer_metadata = _stable_3d.outer_contour_xyz(
            density_xyz, params.outer, params.stable_3d, spacing_xyz=spacing_xyz)
    else:
        raise ValueError(f"Unsupported outer contour method: {params.outer.contour_method!r}.")

    inner_method = params.inner.contour_method.strip().lower()
    fallback = {"applied": False, "reason": None}
    inner_metadata: dict[str, Any] = {}
    if inner_method == "none":
        trab_xyz = full_xyz.copy()
        cort_xyz = np.zeros_like(full_xyz)
    elif inner_method == "standard":
        from . import _ipl
        trab_xyz, cort_xyz, inner_metadata = _ipl.inner_contour_xyz(
            density_xyz, full_xyz, site=params.site, parameters=params.inner)
        inner_metadata["standard_algorithm_revision"] = STANDARD_ALGORITHM_REVISION
        inner_metadata["scanner_transfer_unvalidated"] = params.modality != "xct2" or params.site not in {"radius", "tibia"}
        from ._stable_3d import mask_quality_xyz
        inner_metadata["quality"] = mask_quality_xyz(trab_xyz, spacing_xyz, params.stable_3d)
    elif inner_method == "buie":
        from . import _buie
        trab_xyz, cort_xyz, inner_metadata = _buie.inner_contour_xyz(
            density_xyz, full_xyz, params.inner.endosteal_threshold, params.buie)
    elif inner_method == "ipl_match":
        from . import _ipl_match
        trab_xyz, cort_xyz, inner_metadata = _ipl_match.inner_contour_xyz(
            density_xyz, full_xyz, params.inner, params.buie)
    elif inner_method == "ipl":
        if params.modality != "xct2" or params.site not in {"radius", "tibia"}:
            raise ValueError("IPL candidate requires XCTII radius or tibia.")
        from . import _ipl
        trab_xyz, cort_xyz, inner_metadata = _ipl.inner_contour_xyz(
            density_xyz, full_xyz, site=params.site)
    elif inner_method == "stable_3d":
        from . import _stable_3d
        trab_xyz, cort_xyz, inner_metadata = _stable_3d.inner_contour_xyz(
            density_xyz, full_xyz, params.inner, params.buie, params.stable_3d,
            spacing_xyz=spacing_xyz)
    else:
        raise ValueError(f"Unsupported inner contour method: {params.inner.contour_method!r}.")

    seg_xyz = segment_bone_xyz(
        segmentation_xyz,
        full_xyz,
        trab_xyz,
        cort_xyz,
        params.segmentation,
        spacing_xyz=spacing_xyz,
    )
    material_xyz = np.zeros(seg_xyz.shape, dtype=np.uint8)
    seg_bool = np.asarray(seg_xyz, dtype=bool)
    material_xyz[seg_bool & np.asarray(trab_xyz, dtype=bool)] = 100
    material_xyz[seg_bool & np.asarray(cort_xyz, dtype=bool)] = 127
    metadata = {
        "modality": params.modality,
        "site": params.site,
        "segmentation_method": params.segmentation.method,
        "contour_support_method": support_method,
        "periosteal_contour_method": outer_method,
        "endosteal_contour_method": inner_method,
        "endosteal_fallback": fallback,
        "outer_contour": outer_metadata,
        "inner_contour": inner_metadata,
        "contour_support": outer_metadata,
        "material_labels": {
            "100": "segmentation_intersect_trabecular_mask",
            "127": "segmentation_intersect_cortical_mask",
        },
    }
    return GeneratedMasks(
        seg=numpy_xyz_to_sitk_binary(seg_xyz, image),
        full=numpy_xyz_to_sitk_binary(full_xyz, image),
        trab=numpy_xyz_to_sitk_binary(trab_xyz, image),
        cort=numpy_xyz_to_sitk_binary(cort_xyz, image),
        material=_numpy_xyz_to_sitk_label(material_xyz, image),
        mask_provenance={
            "seg": "generated",
            "full": "generated",
            "trab": "generated",
            "cort": "generated",
            "fea-input": "generated_from_seg_trab_cort",
        },
        metadata=metadata,
    )


def generate_bone_segmentation(
    image: sitk.Image,
    parameters: ContourParameters | None = None,
    *,
    full_mask: sitk.Image | None = None,
    trab_mask: sitk.Image | None = None,
    cort_mask: sitk.Image | None = None,
) -> sitk.Image:
    """Generate a binary bone segmentation, optionally within supplied masks."""
    _validate_image(image, "image")
    params = parameters or resolve_preset()
    if full_mask is None and trab_mask is None and cort_mask is None:
        return generate_masks_from_image(image, params).seg
    if full_mask is None or trab_mask is None or cort_mask is None:
        raise ValueError("full_mask, trab_mask, and cort_mask must be supplied together.")
    for label, mask in (("full_mask", full_mask), ("trab_mask", trab_mask), ("cort_mask", cort_mask)):
        _validate_image(mask, label)
        if mask.GetSize() != image.GetSize():
            raise ValueError(f"{label} size must match image size.")
    segmentation_xyz = segment_bone_xyz(
        sitk_to_numpy_xyz(image),
        sitk_to_numpy_xyz(full_mask) > 0,
        sitk_to_numpy_xyz(trab_mask) > 0,
        sitk_to_numpy_xyz(cort_mask) > 0,
        params.segmentation,
        spacing_xyz=tuple(float(value) for value in image.GetSpacing()),
    )
    return numpy_xyz_to_sitk_binary(segmentation_xyz, image)
