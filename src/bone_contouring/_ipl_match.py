"""Experimental image-only IPL matching, explicitly deviating from Buie.

No reference mask enters this algorithm. Only XCT2 tibia agreement has been
investigated; the name denotes the objective, not native IPL equivalence.
"""
from dataclasses import asdict, replace
from time import perf_counter

import numpy as np
from scipy import ndimage as ndi

from . import _arrays, _buie


def outer_contour_xyz(density, parameters, *, spacing_xyz):
    """Standard Gaussian/primary-bone/axial cleanup, independent of tissue segmentation."""
    if not np.isfinite(parameters.periosteal_threshold):
        raise ValueError('IPL-match periosteal threshold must be finite.')
    if not np.isfinite(parameters.gaussian_sigma) or parameters.gaussian_sigma < 0:
        raise ValueError('IPL-match outer Gaussian sigma must be finite and nonnegative.')
    p = replace(parameters, use_adaptive_threshold=False)
    start = perf_counter()
    full = _arrays.outer_contour_xyz(density, p, spacing_xyz=spacing_xyz)
    return full, {'support': 'independent_gaussian_density', 'parameters': asdict(p),
                  'stage_seconds': {'outer_contour': perf_counter() - start},
                  'experimental': True, 'native_ipl_equivalence': False,
                  'deviations_from_buie': ['Gaussian prefilter before threshold 1',
                                          'largest bone component before axial closing/opening',
                                          'axial hole filling instead of exterior connectivity']}


def inner_contour_xyz(density, full, parameters, buie):
    """Peel ROI, select primary marrow, Buie closing/smoothing, then axial fill."""
    _buie.validate_parameters(buie)
    if (isinstance(parameters.peel, (bool, np.bool_))
            or not isinstance(parameters.peel, (int, np.integer)) or parameters.peel < 0):
        raise ValueError('IPL-match inner peel must be a nonnegative integer radius.')
    if not np.isfinite(parameters.endosteal_threshold):
        raise ValueError('IPL-match endosteal threshold must be finite.')
    density, full = np.asarray(density), np.asarray(full, dtype=bool)
    if density.ndim != 3 or density.shape != full.shape:
        raise ValueError('IPL-match density and full mask must have matching 3D shapes.')
    timings = {}
    roi = _buie._timed(timings, 'periosteal_peel_xy', _arrays._apply_xy_morphology,
                      full, int(parameters.peel), 'erode')
    marrow = (density < parameters.endosteal_threshold) & roi
    initial_voxels = int(marrow.sum())
    start = perf_counter()
    labels, count = ndi.label(marrow, structure=ndi.generate_binary_structure(
        3, 3 if buie.fully_connected else 1))
    if count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        marrow = labels == sizes.argmax()
    else:
        marrow = np.zeros_like(full)
    del labels
    timings['primary_marrow_connectivity'] = perf_counter() - start
    selected_voxels = int(marrow.sum())
    if selected_voxels:
        marrow = _buie._timed(timings, 'endosteal_dilate', _buie.morphology_xyz,
                             marrow, buie.endosteal_kernel_size, dilate=True)
        marrow = _buie._timed(timings, 'endosteal_connectivity', _buie.fill_from_exterior_xyz,
                             marrow, fully_connected=buie.fully_connected)
        marrow = _buie._timed(timings, 'endosteal_erode', _buie.morphology_xyz,
                             marrow, buie.endosteal_kernel_size, dilate=False)
        smoothed = _buie._timed(timings, 'endosteal_gaussian', _buie.smooth_exterior_xyz, marrow, buie)
        trab = (smoothed < buie.final_threshold) & roi
    else:
        trab = np.zeros_like(full)
    start = perf_counter()
    filled_voxels = 0
    for z in range(trab.shape[2]):
        filled = ndi.binary_fill_holes(trab[:, :, z]) & roi[:, :, z]
        filled_voxels += int((filled & ~trab[:, :, z]).sum())
        trab[:, :, z] = filled
    timings['trabecular_fill_xy'] = perf_counter() - start
    return trab, full & ~trab, {
        'experimental': True, 'native_ipl_equivalence': False,
        'threshold': float(parameters.endosteal_threshold), 'peel_xy_radius': int(parameters.peel),
        'parameters': asdict(buie), 'stage_seconds': timings,
        'initial_marrow_voxels': initial_voxels, 'initial_marrow_component_count': int(count),
        'selected_marrow_voxels': selected_voxels, 'filled_hole_voxels': filled_voxels,
        'empty_trabecular_mask': bool(np.any(full) and not np.any(trab)),
        'gaussian_support_interpretation': 'VTK_radius_factors',
        'gaussian_polarity': 'white_exterior_255_black_marrow_0',
        'final_threshold_rule': 'trab = smoothed_exterior < lower_threshold, then fill XY holes',
        'deviations_from_buie': ['axial periosteal peel before marrow selection',
                                'largest marrow component before closing',
                                'axial hole filling after Gaussian threshold']}
