"""Experimental topology-first contours with bounded 3D surface regularization.

The initial compartment envelopes are axial; final distance-field smoothing is
volumetric. Neither this method nor its advisory QA promises IPL equivalence.
"""
from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

import numpy as np
from scipy import ndimage as ndi
import SimpleITK as sitk

from . import _arrays, _buie
from .parameters import Stable3DParameters


def _triple(value, name, *, positive=False):
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must have three finite values.') from exc
    if a.shape != (3,) or not np.isfinite(a).all() or (a <= 0 if positive else a < 0).any():
        raise ValueError(f'{name} must have three finite {"positive" if positive else "nonnegative"} values.')
    return tuple(float(v) for v in a)


def validate_parameters(p):
    _triple(p.outer_sigma_mm, 'stable_3d.outer_sigma_mm')
    _triple(p.inner_sigma_mm, 'stable_3d.inner_sigma_mm')
    for name in ('max_boundary_shift_mm', 'area_jump_fraction', 'adjacent_boundary_limit_mm'):
        value = getattr(p, name)
        if not np.isfinite(value) or value < 0 or (name != 'max_boundary_shift_mm' and value == 0):
            raise ValueError(f'stable_3d.{name} must be finite and {"nonnegative" if name == "max_boundary_shift_mm" else "positive"}.')


def regularize_mask_xyz(mask, spacing_xyz, sigma_mm, max_shift_mm):
    """Smooth a symmetric, physical signed-distance field, positive inside.

    Averaging complementary Maurer fields avoids the one-sided zero-boundary
    convention's systematic half-voxel erosion. Only existing boundary-band
    voxels may change; this is a local change bound, not a global Hausdorff or
    topology guarantee. No artificial exterior caps are added at the scan ends.
    """
    spacing = _triple(spacing_xyz, 'spacing_xyz', positive=True)
    sigma = _triple(sigma_mm, 'sigma_mm')
    if not np.isfinite(max_shift_mm) or max_shift_mm < 0:
        raise ValueError('max_shift_mm must be finite and nonnegative.')
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3 or min(mask.shape) == 0:
        raise ValueError('mask must be a nonempty 3D array.')
    if not any(sigma) or max_shift_mm == 0 or not mask.any() or mask.all():
        return np.ascontiguousarray(mask.copy())
    # Lateral padding makes a crop-touching boundary finite; no z padding avoids
    # eroding a compartment merely because it reaches a terminal scan slice.
    padded = np.pad(mask, ((1, 1), (1, 1), (0, 0)))
    image = sitk.GetImageFromArray(padded.transpose(2, 1, 0).astype(np.uint8))
    image.SetSpacing(spacing)
    inside = sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(
        image, insideIsPositive=False, squaredDistance=False, useImageSpacing=True))
    outside = sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(
        sitk.Not(image), insideIsPositive=False, squaredDistance=False, useImageSpacing=True))
    outside -= inside
    outside *= 0.5
    del inside, image
    field = outside.transpose(2, 1, 0)
    filtered = ndi.gaussian_filter(field, sigma=tuple(s/h for s, h in zip(sigma, spacing)), mode='nearest')
    result = np.where(np.abs(field) <= max_shift_mm, filtered > 0, padded)
    return np.ascontiguousarray(result[1:-1, 1:-1, :])


def mask_quality_xyz(mask, spacing_xyz, p):
    """Advisory axial topology and neighboring-boundary checks, never a fallback.

    Real taper, motion, and crop truncation can trigger warnings. An absence of
    warnings is not proof of anatomical correctness.
    """
    validate_parameters(p)
    spacing = _triple(spacing_xyz, 'spacing_xyz', positive=True)
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3 or min(mask.shape) == 0:
        raise ValueError('mask must be a nonempty 3D array.')
    areas = mask.sum(axis=(0, 1)).astype(np.int64)
    relative = np.abs(np.diff(areas)) / np.maximum(np.minimum(areas[:-1], areas[1:]), 1)
    empty = np.flatnonzero(areas == 0).tolist()
    area_jumps = np.flatnonzero(relative > p.area_jump_fraction).tolist()
    fragments, holes, touches, distances = [], [], [], []
    previous_edge = previous_dt = None
    for z in range(mask.shape[2]):
        layer = mask[:, :, z]
        if ndi.label(layer)[1] > 1:
            fragments.append(z)
        count = int((ndi.binary_fill_holes(layer) & ~layer).sum())
        if count:
            holes.append({'z': z, 'voxels': count})
        if layer[0, :].any() or layer[-1, :].any() or layer[:, 0].any() or layer[:, -1].any():
            touches.append(z)
        edge = layer & ~ndi.binary_erosion(layer)
        dt = ndi.distance_transform_edt(~edge, sampling=spacing[:2]) if edge.any() else None
        if previous_edge is not None and previous_edge.any() and edge.any():
            total = float(previous_dt[edge].sum() + dt[previous_edge].sum())
            distances.append({'z0': z-1, 'z1': z, 'mean_mm': total / (int(edge.sum()) + int(previous_edge.sum()))})
        previous_edge, previous_dt = edge, dt
    boundary_jumps = [r for r in distances if r['mean_mm'] > p.adjacent_boundary_limit_mm]
    warnings = []
    for condition, name in ((not mask.any(), 'empty_compartment'), (bool(empty), 'empty_slices'),
                            (bool(area_jumps), 'area_jump'), (bool(boundary_jumps), 'boundary_jump'),
                            (bool(fragments), 'fragmented_slices'), (bool(holes), 'axial_holes'),
                            (bool(touches), 'xy_crop_boundary_contact')):
        if condition:
            warnings.append(name)
    return {'warnings': warnings, 'flags_are_advisory': True, 'area_voxels_by_slice': areas.tolist(),
            'empty_slices': empty, 'area_jump_pairs_z0': area_jumps,
            'fragmented_slices': fragments, 'axial_holes': holes, 'xy_crop_contact_slices': touches,
            'boundary_jump_pairs': boundary_jumps, 'scored_adjacent_pairs': len(distances),
            'worst_adjacent_pair': max(distances, key=lambda r:r['mean_mm']) if distances else None}


def _density_and_threshold(density, threshold, sigma):
    density = np.asarray(density, dtype=np.float32)
    if density.ndim != 3 or min(density.shape) == 0 or not np.isfinite(density).all():
        raise ValueError('density must be a finite nonempty 3D array.')
    if not np.isfinite(threshold) or not np.isfinite(sigma) or sigma < 0:
        raise ValueError('threshold and Gaussian sigma must be finite; sigma must be nonnegative.')
    return density


def outer_contour_xyz(density, parameters, stable, *, spacing_xyz):
    validate_parameters(stable)
    spacing = _triple(spacing_xyz, 'spacing_xyz', positive=True)
    density = _density_and_threshold(density, parameters.periosteal_threshold, parameters.gaussian_sigma)
    radius = parameters.periosteal_kernel_size
    if isinstance(radius, (bool, np.bool_)) or not isinstance(radius, (int, np.integer)) or radius < 0:
        raise ValueError('periosteal_kernel_size must be a nonnegative integer radius.')
    timings = {}
    support = _buie._timed(timings, 'density_gaussian', _arrays.smooth_xyz, density,
                           sigma=parameters.gaussian_sigma, spacing_xyz=spacing)
    support = support >= parameters.periosteal_threshold
    full = _buie._timed(timings, 'bone_connectivity', _arrays.largest_component_xyz, support)
    full = _buie._timed(timings, 'periosteal_close_xy', _arrays._apply_xy_morphology, full, int(radius), 'close')
    # Deliberately no pre-fill opening. A thin closed shell must remain closed.
    full = _buie._timed(timings, 'periosteal_fill_xy', _arrays.fill_holes_xy, full)
    full = _buie._timed(timings, 'periosteal_distance_regularization', regularize_mask_xyz,
                        full, spacing, stable.outer_sigma_mm, stable.max_boundary_shift_mm)
    # Smoothing must not reintroduce enclosed holes in a compartment envelope.
    full = _buie._timed(timings, 'periosteal_final_fill_xy', _arrays.fill_holes_xy, full)
    quality = _buie._timed(timings, 'quality', mask_quality_xyz, full, spacing, stable)
    return full, {'experimental': True, 'native_ipl_equivalence': False,
                  'support': 'independent_gaussian_density', 'parameters': asdict(stable),
                  'threshold': float(parameters.periosteal_threshold), 'density_gaussian_sigma_voxels': float(parameters.gaussian_sigma),
                  'close_xy_radius_voxels': int(radius), 'stage_seconds': timings, 'quality': quality,
                  'ignored_legacy_controls': ['periosteal_open_radius', 'fill_holes', 'use_adaptive_threshold']}


def inner_contour_xyz(density, full, parameters, buie, stable, *, spacing_xyz):
    validate_parameters(stable)
    _buie.validate_parameters(buie)
    spacing = _triple(spacing_xyz, 'spacing_xyz', positive=True)
    density = _density_and_threshold(density, parameters.endosteal_threshold, parameters.gaussian_sigma)
    full = np.asarray(full, dtype=bool)
    if full.shape != density.shape:
        raise ValueError('full and density must have matching 3D shapes.')
    timings = {}
    filtered = _buie._timed(timings, 'density_gaussian', _arrays.smooth_xyz, density,
                            sigma=parameters.gaussian_sigma, spacing_xyz=spacing)
    marrow = (filtered < parameters.endosteal_threshold) & full
    initial = int(marrow.sum())
    start = perf_counter()
    labels, count = ndi.label(marrow, ndi.generate_binary_structure(3, 3 if buie.fully_connected else 1))
    if count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        marrow = labels == sizes.argmax()
    del labels
    timings['primary_marrow_connectivity'] = perf_counter() - start
    selected = int(marrow.sum())
    marrow = _buie._timed(timings, 'endosteal_dilate', _buie.morphology_xyz,
                          marrow, buie.endosteal_kernel_size, dilate=True)
    marrow = _buie._timed(timings, 'endosteal_fill_xy', _arrays.fill_holes_xy, marrow)
    marrow = _buie._timed(timings, 'endosteal_erode', _buie.morphology_xyz,
                          marrow, buie.endosteal_kernel_size, dilate=False)
    marrow &= full
    trab = _buie._timed(timings, 'endosteal_distance_regularization', regularize_mask_xyz,
                        marrow, spacing, stable.inner_sigma_mm, stable.max_boundary_shift_mm)
    trab &= full
    trab = _buie._timed(timings, 'endosteal_final_fill_xy', _arrays.fill_holes_xy, trab)
    trab &= full
    quality = _buie._timed(timings, 'quality', mask_quality_xyz, trab, spacing, stable)
    return trab, full & ~trab, {'experimental': True, 'native_ipl_equivalence': False,
                              'threshold': float(parameters.endosteal_threshold),
                              'density_gaussian_sigma_voxels': float(parameters.gaussian_sigma),
                              'parameters': asdict(stable), 'closing_dimensions_xyz': buie.endosteal_kernel_size,
                              'initial_marrow_voxels': initial, 'selected_marrow_voxels': selected,
                              'initial_marrow_component_count': int(count), 'stage_seconds': timings, 'quality': quality,
                              'ignored_legacy_controls': ['peel', 'trabecular_close_radius', 'endosteal_kernel_size', 'use_adaptive_threshold'],
                              'deviations_from_buie': ['prefiltered density marrow seed', 'largest marrow component',
                                                       'axial fill before erosion', 'final axial envelope filling',
                                                       'symmetric signed-distance smoothing instead of uint8 100/255 smoothing']}
