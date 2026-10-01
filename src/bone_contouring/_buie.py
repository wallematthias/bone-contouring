"""Buie (2007) dual-threshold stages, without archive downsampling.

Arrays use x/y/z order. Compiled SciPy filters support the literal even-sized
VTK ellipsoid and anchor; no Python voxel loops or additional compiler needed.
Gaussian support is interpreted as VTK radius factors, not a (3,3,1) box.
"""
from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

import numpy as np
from scipy import ndimage as ndi

from .parameters import BuieParameters


def validate_parameters(parameters: BuieParameters) -> None:
    if not isinstance(parameters.fully_connected, (bool, np.bool_)):
        raise ValueError("buie.fully_connected must be boolean (False: 6-connectivity; True: 26).")
    for name in ("median_kernel_size", "periosteal_kernel_size", "endosteal_kernel_size"):
        size = getattr(parameters, name)
        if len(size) != 3 or any(isinstance(n, bool) or not isinstance(n, (int, np.integer))
                                 or n < 1 for n in size):
            raise ValueError(f"buie.{name} must contain three positive integer dimensions.")
        if name == "median_kernel_size" and any(n % 2 == 0 for n in size):
            raise ValueError("buie.median_kernel_size dimensions must be odd.")
    if not np.isfinite(parameters.gaussian_sigma) or parameters.gaussian_sigma < 0:
        raise ValueError("buie.gaussian_sigma must be finite and nonnegative.")
    factors = parameters.gaussian_radius_factors
    if len(factors) != 3 or any(not np.isfinite(n) or n < 0 for n in factors):
        raise ValueError("buie.gaussian_radius_factors must contain three finite nonnegative values.")
    if not np.isfinite(parameters.final_threshold) or not 0 <= parameters.final_threshold <= 255:
        raise ValueError("buie.final_threshold must be within [0, 255].")


def _ellipsoid(size: tuple[int, int, int]) -> np.ndarray:
    axes = np.ogrid[tuple(slice(0, n) for n in size)]
    distance = sum(((axis - (n - 1) / 2) / (n / 2)) ** 2
                   for axis, n in zip(axes, size))
    return distance <= 1


def morphology_xyz(mask: np.ndarray, size: tuple[int, int, int], *, dilate: bool) -> np.ndarray:
    """VTK-style correlation footprint, even anchor floor(size/2), clipped edges.

    Maximum/minimum filters deliberately avoid binary_dilation's reflection of
    even footprints. Outside voxels are neutral (0 for dilation, 1 for erosion).
    """
    if size[2] == 1:
        # The ellipsoid is a union of contiguous X intervals, one per Y row.
        # Decomposition is exact, including even anchors and neutral edges.
        # Each interval uses a compiled O(N) running max/min, avoiding O(N*K)
        # scans of all footprint pixels. Never downsample or approximate a disk.
        source = np.ascontiguousarray(np.asarray(mask, dtype=bool).transpose(2, 1, 0))
        result = np.full_like(source, not dilate)
        footprint = _ellipsoid(size)[:, :, 0]
        function_1d = ndi.maximum_filter1d if dilate else ndi.minimum_filter1d
        combine = np.logical_or if dilate else np.logical_and
        for row in range(size[1]):
            indices = np.flatnonzero(footprint[:, row])
            if not indices.size:
                continue
            length = int(indices.size)
            origin = size[0] // 2 - int(indices[0]) - length // 2
            filtered = function_1d(source, length, axis=2, origin=origin,
                                   mode="constant", cval=0 if dilate else 1)
            offset = row - size[1] // 2
            start, stop = max(0, -offset), min(source.shape[1], source.shape[1] - offset)
            if stop > start:
                destination = result[:, start:stop, :]
                combine(destination, filtered[:, start + offset:stop + offset, :], out=destination)
        return result.transpose(2, 1, 0)
    function = ndi.maximum_filter if dilate else ndi.minimum_filter
    return function(np.asarray(mask, dtype=bool), footprint=_ellipsoid(size),
                    mode="constant", cval=0 if dilate else 1)


def fill_from_exterior_xyz(mask: np.ndarray, *, fully_connected: bool) -> np.ndarray:
    """Complement the largest background component touching the XY perimeter.

    Connectivity is volumetric (6 or 26). No exterior layer is padded across
    scan ends: that would connect the marrow to the exterior in an open stack.
    """
    labels, count = ndi.label(~np.asarray(mask, dtype=bool),
                             structure=ndi.generate_binary_structure(3, 3 if fully_connected else 1))
    if not count:
        return np.ones(mask.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    boundary = np.unique(np.concatenate((labels[0].ravel(), labels[-1].ravel(),
                                         labels[:, 0].ravel(), labels[:, -1].ravel())))
    boundary = boundary[boundary != 0]
    if not boundary.size:
        # No non-bone exterior at XY edges (e.g. the bone occupies the FOV).
        return np.ones(mask.shape, dtype=bool)
    exterior = boundary[np.argmax(sizes[boundary])]
    return labels != exterior


def smooth_exterior_xyz(marrow: np.ndarray, parameters: BuieParameters) -> np.ndarray:
    """Smooth white exterior (255), black marrow (0), with clipped normalization.

    VTK axis order z,y,x and uint8 intermediate quantization. Interpreting the
    published support as radius factors gives floor(sigma*factor) voxel radii.
    Values within 1e-10 below an integer are snapped before truncation to avoid
    eroding constant white regions through roundoff. This is deliberately not
    a promise of bitwise equivalence to a particular VTK/IPL build at ties.
    """
    result = (~marrow).astype(np.uint8) * 255
    sigma = parameters.gaussian_sigma
    if sigma == 0:
        return result
    for axis in (2, 1, 0):
        radius = int(sigma * parameters.gaussian_radius_factors[axis])
        if radius == 0:
            continue
        offsets = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-offsets ** 2 / (2 * sigma ** 2))
        kernel /= kernel.sum()
        weights = ndi.correlate1d(np.ones(result.shape[axis]), kernel, mode="constant", cval=0)
        shape = [1, 1, 1]
        shape[axis] = result.shape[axis]
        filtered = ndi.correlate1d(result, kernel, axis=axis, mode="constant", cval=0,
                                  output=np.float64)
        filtered /= weights.reshape(shape)
        np.clip(filtered, 0, 255, out=filtered)
        filtered += 1e-10
        result = filtered.astype(np.uint8)
    return result


def _timed(timings, name, function, *args, **kwargs):
    start = perf_counter()
    result = function(*args, **kwargs)
    timings[name] = perf_counter() - start
    return result


def outer_contour_xyz(density, threshold, parameters):
    validate_parameters(parameters)
    if not np.isfinite(threshold):
        raise ValueError("Buie periosteal threshold must be finite.")
    timings = {}
    bone = _timed(timings, "threshold_1", np.greater_equal, density, threshold)
    bone = _timed(timings, "median", ndi.median_filter, bone,
                  size=parameters.median_kernel_size, mode="nearest")
    bone = _timed(timings, "periosteal_dilate", morphology_xyz, bone,
                  parameters.periosteal_kernel_size, dilate=True)
    bone = _timed(timings, "periosteal_connectivity", fill_from_exterior_xyz, bone,
                  fully_connected=parameters.fully_connected)
    full = _timed(timings, "periosteal_erode", morphology_xyz, bone,
                  parameters.periosteal_kernel_size, dilate=False)
    return full, {"support": "raw_dual_threshold", "threshold": float(threshold),
                  "parameters": asdict(parameters), "stage_seconds": timings,
                  "median_boundary": "nearest", "morphology_boundary": "clipped_neutral",
                  "exterior_selection": "largest_background_touching_xy_perimeter"}


def inner_contour_xyz(density, full, threshold, parameters):
    validate_parameters(parameters)
    if not np.isfinite(threshold):
        raise ValueError("Buie endosteal threshold must be finite.")
    timings = {}
    marrow = _timed(timings, "threshold_2", np.less, density, threshold)
    marrow &= full
    marrow = _timed(timings, "endosteal_dilate", morphology_xyz, marrow,
                    parameters.endosteal_kernel_size, dilate=True)
    marrow = _timed(timings, "endosteal_connectivity", fill_from_exterior_xyz, marrow,
                    fully_connected=parameters.fully_connected)
    marrow = _timed(timings, "endosteal_erode", morphology_xyz, marrow,
                    parameters.endosteal_kernel_size, dilate=False)
    smoothed = _timed(timings, "endosteal_gaussian", smooth_exterior_xyz, marrow, parameters)
    trab = (smoothed < parameters.final_threshold) & full
    cort = full & ~trab
    return trab, cort, {"threshold": float(threshold), "stage_seconds": timings,
                        "gaussian_support_interpretation": "VTK_radius_factors",
                        "gaussian_polarity": "white_exterior_255_black_marrow_0",
                        "final_threshold_rule": "trab = smoothed_exterior < lower_threshold",
                        "empty_trabecular_mask": bool(np.any(full) and not np.any(trab))}
