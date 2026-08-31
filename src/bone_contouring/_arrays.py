"""Internal x/y/z array algorithms and SimpleITK conversion helpers."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from .laplace_hamming import LaplaceHammingParameters, laplace_hamming_binarize_xyz
from .parameters import InnerContourParameters, OuterContourParameters, SegmentationParameters


def sitk_to_numpy_xyz(image: sitk.Image) -> np.ndarray:
    """Convert a SimpleITK image from z/y/x storage to contiguous x/y/z data."""
    return np.ascontiguousarray(np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0)))


def numpy_xyz_to_sitk_binary(mask_xyz: np.ndarray, reference: sitk.Image) -> sitk.Image:
    """Create a uint8 binary image with geometry copied from ``reference``."""
    image = sitk.GetImageFromArray(np.transpose(np.asarray(mask_xyz, dtype=np.uint8), (2, 1, 0)))
    image.CopyInformation(reference)
    return sitk.Cast(image > 0, sitk.sitkUInt8)


def numpy_xyz_to_sitk_scalar(
    image_xyz: np.ndarray, spacing_xyz: tuple[float, float, float] | None = None
) -> sitk.Image:
    """Create a float32 image from x/y/z scalar data."""
    image = sitk.GetImageFromArray(np.transpose(np.asarray(image_xyz, dtype=np.float32), (2, 1, 0)))
    if spacing_xyz is not None:
        image.SetSpacing(tuple(float(value) for value in spacing_xyz))
    return image


def remove_small_components_xyz(mask_xyz: np.ndarray, min_size_voxels: int) -> np.ndarray:
    """Remove 6-connected components smaller than ``min_size_voxels``."""
    mask = np.asarray(mask_xyz, dtype=bool)
    if min_size_voxels <= 0 or not np.any(mask):
        return np.ascontiguousarray(mask)
    image = sitk.GetImageFromArray(np.transpose(mask.astype(np.uint8), (2, 1, 0)))
    labels = sitk.ConnectedComponent(image, False)
    retained = sitk.RelabelComponent(labels, minimumObjectSize=int(min_size_voxels), sortByObjectSize=False)
    return np.ascontiguousarray(np.transpose(sitk.GetArrayFromImage(retained > 0), (2, 1, 0)).astype(bool))


def largest_component_xyz(mask_xyz: np.ndarray) -> np.ndarray:
    """Return the largest 6-connected component, preserving an empty mask."""
    mask = np.asarray(mask_xyz, dtype=bool)
    if not np.any(mask):
        return np.ascontiguousarray(mask)
    image = sitk.GetImageFromArray(np.transpose(mask.astype(np.uint8), (2, 1, 0)))
    labels = sitk.RelabelComponent(sitk.ConnectedComponent(image, False), sortByObjectSize=True)
    return np.ascontiguousarray(np.transpose(sitk.GetArrayFromImage(labels == 1), (2, 1, 0)).astype(bool))


def smooth_xyz(
    image_xyz: np.ndarray,
    *,
    sigma: float,
    spacing_xyz: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Smooth an image with the same voxel-relative sigma convention as Timelapsed."""
    if sigma <= 0:
        return np.asarray(image_xyz, dtype=np.float32).copy()
    image = numpy_xyz_to_sitk_scalar(image_xyz, spacing_xyz)
    spacing = image.GetSpacing()
    smoothed = sitk.SmoothingRecursiveGaussian(image, float(sigma) * min(spacing))
    return sitk_to_numpy_xyz(smoothed)


def adaptive_threshold_xyz(
    density_xyz: np.ndarray,
    *,
    spacing_xyz: tuple[float, float, float] | None = None,
    low_threshold: float = 190.0,
    high_threshold: float = 450.0,
    block_size: int = 13,
    min_size_voxels: int = 64,
) -> np.ndarray:
    """Apply Schulte-style combined adaptive thresholding in x/y/z order."""
    density = np.asarray(density_xyz, dtype=np.float32)
    if density.ndim != 3:
        raise ValueError(f"adaptive_threshold_xyz expects a 3D array, got ndim={density.ndim}.")
    if block_size % 2 == 0:
        raise ValueError(f"block_size must be odd, got {block_size}.")
    image = numpy_xyz_to_sitk_scalar(density, spacing_xyz)
    local_mean = sitk_to_numpy_xyz(sitk.BoxMean(image, [block_size // 2] * 3))
    filtered = smooth_xyz(density, sigma=1.0, spacing_xyz=spacing_xyz)
    low_mask = filtered > float(low_threshold)
    result = (filtered * low_mask > local_mean * low_mask) | (filtered > float(high_threshold))
    return remove_small_components_xyz(result, min_size_voxels)


def _laplace_hamming_parameters(params: SegmentationParameters) -> LaplaceHammingParameters:
    return LaplaceHammingParameters(
        low_pass_cutoff=params.laplace_hamming_low_pass_cutoff,
        high_pass_cutoff=params.laplace_hamming_high_pass_cutoff,
        laplace_epsilon=params.laplace_hamming_epsilon,
        hamming_amplitude=params.laplace_hamming_amplitude,
        amplification=params.laplace_hamming_amplification,
        input_offset=params.laplace_hamming_input_offset,
        ipl_float_max=params.laplace_hamming_ipl_float_max,
        int16_max=params.laplace_hamming_int16_max,
        threshold=params.laplace_hamming_threshold,
        min_size_voxels=params.laplace_hamming_min_size_voxels,
        backend=params.laplace_hamming_backend,
    )


def segment_bone_xyz(
    image_xyz: np.ndarray,
    full_mask_xyz: np.ndarray,
    trab_mask_xyz: np.ndarray,
    cort_mask_xyz: np.ndarray,
    parameters: SegmentationParameters,
    *,
    spacing_xyz: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Generate a cleaned bone segmentation constrained to the full mask."""
    full = np.asarray(full_mask_xyz, dtype=bool)
    if not parameters.enabled:
        return np.ascontiguousarray(full)
    method = parameters.method.strip().lower()
    if method in {"global", "seg_gauss"}:
        method = "gauss"
    if method == "gauss":
        filtered = smooth_xyz(image_xyz, sigma=parameters.gaussian_sigma, spacing_xyz=spacing_xyz)
        segmentation = ((filtered >= parameters.trab_threshold) & np.asarray(trab_mask_xyz, dtype=bool)) | (
            (filtered >= parameters.cort_threshold) & np.asarray(cort_mask_xyz, dtype=bool)
        )
    elif method == "adaptive":
        segmentation = adaptive_threshold_xyz(
            image_xyz,
            spacing_xyz=spacing_xyz,
            low_threshold=parameters.adaptive_low_threshold,
            high_threshold=parameters.adaptive_high_threshold,
            block_size=parameters.adaptive_block_size,
            min_size_voxels=parameters.min_size_voxels,
        )
    elif method == "laplace_hamming":
        segmentation = laplace_hamming_binarize_xyz(
            image_xyz,
            full_mask_xyz=full,
            spacing_xyz=spacing_xyz,
            parameters=_laplace_hamming_parameters(parameters),
        )
    else:
        raise ValueError(f"Unsupported segmentation method: {parameters.method!r}.")
    segmentation = remove_small_components_xyz(segmentation & full, parameters.min_size_voxels)
    if method != "laplace_hamming" and parameters.keep_largest_component:
        segmentation = largest_component_xyz(segmentation)
    return np.ascontiguousarray(segmentation, dtype=bool)


def contour_support_xyz(
    image_xyz: np.ndarray,
    parameters: SegmentationParameters,
    *,
    spacing_xyz: tuple[float, float, float] | None = None,
    full_mask_xyz: np.ndarray | None = None,
    role: str = "outer",
) -> np.ndarray | None:
    """Create a temporary contour-support mask from the selected segmentation method."""
    if not parameters.enabled:
        return None
    method = parameters.method.strip().lower()
    if method in {"global", "seg_gauss"}:
        method = "gauss"
    if full_mask_xyz is None:
        full_mask = np.ones(np.asarray(image_xyz).shape, dtype=bool)
    else:
        full_mask = np.asarray(full_mask_xyz, dtype=bool)
    if method == "gauss":
        filtered = smooth_xyz(image_xyz, sigma=parameters.gaussian_sigma, spacing_xyz=spacing_xyz)
        threshold = parameters.trab_threshold if role == "outer" else parameters.cort_threshold
        support = filtered >= float(threshold)
    elif method == "adaptive":
        support = adaptive_threshold_xyz(
            image_xyz,
            spacing_xyz=spacing_xyz,
            low_threshold=parameters.adaptive_low_threshold,
            high_threshold=parameters.adaptive_high_threshold,
            block_size=parameters.adaptive_block_size,
            min_size_voxels=parameters.min_size_voxels,
        )
    elif method == "laplace_hamming":
        support = laplace_hamming_binarize_xyz(
            image_xyz,
            full_mask_xyz=full_mask,
            spacing_xyz=spacing_xyz,
            parameters=_laplace_hamming_parameters(parameters),
        )
    else:
        return None
    return np.ascontiguousarray(np.asarray(support, dtype=bool) & full_mask)


def _apply_xy_morphology(mask_xyz: np.ndarray, radius: int, operation: str) -> np.ndarray:
    """Apply a 2D binary morphology operation independently to every stack slice."""
    mask = np.asarray(mask_xyz, dtype=bool)
    if radius <= 0:
        return np.ascontiguousarray(mask)
    output = np.zeros_like(mask)
    for z_index in range(mask.shape[2]):
        slice_image = sitk.GetImageFromArray(mask[:, :, z_index].T.astype(np.uint8))
        if operation == "close":
            processed = sitk.BinaryMorphologicalClosing(slice_image, [radius, radius])
        elif operation == "open":
            processed = sitk.BinaryMorphologicalOpening(slice_image, [radius, radius])
        elif operation == "erode":
            processed = sitk.BinaryErode(slice_image, [radius, radius])
        else:  # pragma: no cover - private caller supplies fixed operations
            raise ValueError(f"Unsupported morphology operation: {operation}.")
        output[:, :, z_index] = sitk.GetArrayFromImage(processed).T > 0
    return output


def fill_holes_xy(mask_xyz: np.ndarray) -> np.ndarray:
    """Fill holes in each axial slice, including holes at terminal slices."""
    mask = np.asarray(mask_xyz, dtype=bool)
    output = np.zeros_like(mask)
    for z_index in range(mask.shape[2]):
        slice_image = sitk.GetImageFromArray(mask[:, :, z_index].T.astype(np.uint8))
        output[:, :, z_index] = sitk.GetArrayFromImage(sitk.BinaryFillhole(slice_image)).T > 0
    return output


def outer_contour_xyz(
    density_xyz: np.ndarray,
    parameters: OuterContourParameters,
    *,
    spacing_xyz: tuple[float, float, float] | None = None,
    support_mask_xyz: np.ndarray | None = None,
) -> np.ndarray:
    """Create a component-cleaned, hole-filled periosteal mask in x/y/z order."""
    density = np.asarray(density_xyz, dtype=np.float32)
    if density.ndim != 3:
        raise ValueError(f"outer_contour_xyz expects a 3D array, got ndim={density.ndim}.")
    if support_mask_xyz is None and not np.any(density != 0):
        return np.zeros_like(density, dtype=bool)
    if support_mask_xyz is not None:
        thresholded = np.asarray(support_mask_xyz, dtype=bool)
        if thresholded.shape != density.shape:
            raise ValueError("support_mask_xyz shape must match density_xyz shape.")
    elif parameters.use_adaptive_threshold:
        thresholded = adaptive_threshold_xyz(density, spacing_xyz=spacing_xyz, min_size_voxels=0)
    else:
        thresholded = smooth_xyz(density, sigma=parameters.gaussian_sigma, spacing_xyz=spacing_xyz) >= float(
            parameters.periosteal_threshold
        )
    if support_mask_xyz is None:
        thresholded &= density != 0
    thresholded = largest_component_xyz(thresholded)
    thresholded = _apply_xy_morphology(thresholded, parameters.periosteal_kernel_size, "close")
    thresholded = _apply_xy_morphology(thresholded, parameters.periosteal_open_radius, "open")
    if parameters.fill_holes:
        thresholded = fill_holes_xy(thresholded)
    return np.ascontiguousarray(thresholded, dtype=bool)


def _site_trabecular_close_radius(site: str) -> int:
    return 15 if site.strip().lower() == "radius" else 25


def inner_contour_xyz(
    density_xyz: np.ndarray,
    full_mask_xyz: np.ndarray,
    parameters: InnerContourParameters,
    *,
    spacing_xyz: tuple[float, float, float] | None = None,
    support_mask_xyz: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive trabecular and cortical masks within a full periosteal mask."""
    density = np.asarray(density_xyz, dtype=np.float32)
    full = np.asarray(full_mask_xyz, dtype=bool)
    if density.shape != full.shape:
        raise ValueError("density_xyz and full_mask_xyz must have matching shapes.")
    if not np.any(full):
        empty = np.zeros_like(full)
        return empty, empty
    if support_mask_xyz is not None:
        cortical = np.asarray(support_mask_xyz, dtype=bool)
        if cortical.shape != full.shape:
            raise ValueError("support_mask_xyz shape must match density_xyz shape.")
    elif parameters.use_adaptive_threshold:
        cortical = adaptive_threshold_xyz(density, spacing_xyz=spacing_xyz, min_size_voxels=0)
    else:
        cortical = smooth_xyz(density, sigma=parameters.gaussian_sigma, spacing_xyz=spacing_xyz) >= float(
            parameters.endosteal_threshold
        )
    cortical &= full
    if parameters.peel >= min(full.shape[:2]):
        inner_support = np.zeros_like(full)
    else:
        inner_support = _apply_xy_morphology(full, parameters.peel, "erode")
    trabecular = largest_component_xyz(inner_support & ~cortical)
    close_radius = parameters.trabecular_close_radius
    if close_radius is None:
        close_radius = _site_trabecular_close_radius(parameters.site)
    if close_radius > 0 and np.any(trabecular):
        trabecular = _apply_xy_morphology(trabecular, close_radius, "close")
        trabecular &= inner_support
    cortical_mask = full & ~trabecular
    return np.ascontiguousarray(trabecular), np.ascontiguousarray(cortical_mask)
