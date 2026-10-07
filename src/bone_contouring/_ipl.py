"""IPL STEP_1 translation and its shared-standard compartment adaptation.

Recipe attribution: Steven K. Boyd (2017), Danielle E. Whittier (2018/2019).
The supplied script's STEP_1 is commented out and consumes an existing outer
GOBJ; it does not generate that contour. Here ``full`` supplies that same ROI.

The fixed candidate preserves the script's sequence and site constants; the
shared standard accepts preset/custom settings and enforces the final peel.
Native metric 11, GOBJ peel
rasterization, Gaussian/bounding-box margins and ties have not been validated
against IPL. Morphology uses voxel-centre Euclidean balls, background in XY,
and edge-replicated Z continuation; the finite Gaussian uses reflected borders.
Z continuation repairs artificial scan-end caps, not verified native behavior.
Do not claim scanner equivalence.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy import ndimage as ndi
import SimpleITK as sitk

from . import _arrays

ALGORITHM_REVISION = 'ipl_step1_euclidean_v2_z_continuation'


def largest_component(mask):
    """Rank-one face-connected component; never select background when empty."""
    labels, count = ndi.label(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def component_range(mask, minimum, maximum):
    """Face-connected voxel-count extraction, inclusive; zero maximum is unbounded."""
    labels, _ = ndi.label(mask)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum
    if maximum:
        keep &= sizes <= maximum
    keep[0] = False
    return keep[labels]


def slice_majority(mask):
    """Keep axial face-connected components occupying 50–100% of foreground."""
    out = np.zeros_like(mask, dtype=bool)
    for z in range(mask.shape[2]):
        layer = mask[:, :, z]
        total = int(layer.sum())
        if total:
            out[:, :, z] = component_range(layer, (total+1)//2, total)
    return out


def peel_xy(mask, distance):
    """Rasterized XY contour peel with an explicit exterior at cropped edges."""
    if isinstance(distance, (bool, np.bool_)) or not isinstance(distance, (int, np.integer)) or distance < 0:
        raise ValueError('Peel must be a nonnegative integer voxel radius.')
    if distance == 0:
        return np.asarray(mask, dtype=bool).copy()
    padded = np.pad(mask, ((distance, distance), (distance, distance), (0, 0)))
    peeled = _arrays._apply_xy_morphology(padded, distance, 'erode')
    return np.ascontiguousarray(peeled[distance:-distance, distance:-distance, :])


def _distance_outside(mask):
    # Exact float32 Maurer distances avoid a float64 SciPy EDT + index workspace
    # on large stacks. Array-axis order has no effect on voxel-unit distances.
    image = sitk.GetImageFromArray(mask.transpose(2, 1, 0).astype(np.uint8))
    distance = sitk.SignedMaurerDistanceMap(
        image, insideIsPositive=False, squaredDistance=False, useImageSpacing=False)
    return sitk.GetArrayFromImage(distance).transpose(2, 1, 0)


def morphology(mask, distance, operation):
    """Euclidean candidate operator with temporary margins, returned on input grid.

    XY exterior is background; Z replicates the terminal slices so scan ends
    are not treated as anatomical caps. The Z margin covers the entire operator
    (two radii for open/close), then is cropped back. Close computes dilation on
    the expanded grid before erosion without clipping temporary expansion.
    """
    if operation not in {'erode', 'dilate', 'open', 'close'}:
        raise ValueError(f'Unsupported morphology operation: {operation!r}.')
    if isinstance(distance, bool) or not isinstance(distance, (int, np.integer)) or distance < 0:
        raise ValueError('Morphology distance must be a nonnegative integer.')
    mask = np.asarray(mask, dtype=bool)
    if not mask.any() or distance == 0:
        return mask.copy()
    margin = int(distance)+1
    steps = {'erode': ('erode',), 'dilate': ('dilate',),
             'open': ('erode', 'dilate'), 'close': ('dilate', 'erode')}[operation]
    z_margin = len(steps)*int(distance)+1
    work = np.pad(mask, ((margin, margin), (margin, margin), (0, 0)))
    work = np.pad(work, ((0, 0), (0, 0), (z_margin, z_margin)), mode='edge')
    for step in steps:
        if not work.any():
            break
        if step == 'dilate':
            work = _distance_outside(work) <= distance
        else:
            work = _distance_outside(~work) > distance
    return np.ascontiguousarray(work[margin:-margin, margin:-margin, z_margin:-z_margin])


def inner_contour_xyz(density, full, *, site, progress=None, parameters=None):
    """Run STEP_1, or its tunable shared-standard adaptation on a full ROI.

    Without parameters this remains the fixed XCTII radius/tibia candidate.
    Standard knee uses the existing 36-voxel final close and tibia corner rule;
    neither that adaptation nor XCTI transfer is validated native IPL behavior.
    """
    if site not in ({'radius', 'tibia'} if parameters is None else {'radius', 'tibia', 'knee', 'none'}):
        raise ValueError('IPL candidate requires XCTII radius or tibia.')
    density = np.asarray(density, dtype=np.float32)
    full = np.asarray(full, dtype=bool)
    if density.ndim != 3 or min(density.shape) == 0 or density.shape != full.shape or not np.isfinite(density).all():
        raise ValueError('Density and full mask must be matching finite nonempty 3D arrays.')
    minimum, close_distance = (800, 30) if site in {'radius', 'none'} else (200000, 50)
    threshold, sigma, peel = 500., 2., 6
    if parameters is not None:
        threshold, sigma, peel = parameters.endosteal_threshold, parameters.gaussian_sigma, parameters.peel
        if site == 'knee':
            close_distance = 36
        if parameters.trabecular_close_radius is not None:
            close_distance = parameters.trabecular_close_radius
        if not np.isfinite(threshold) or not np.isfinite(sigma) or sigma < 0:
            raise ValueError('Endosteal threshold must be finite and sigma finite and nonnegative.')
        if isinstance(close_distance, (bool, np.bool_)) or not isinstance(close_distance, (int, np.integer)) or close_distance < 0:
            raise ValueError('Final close radius must be a nonnegative integer.')
    metadata = dict(candidate_name='IPL', algorithm_revision=ALGORITHM_REVISION,
                    experimental=parameters is None, native_ipl_equivalence=False,
                    source_recipe='IPL_UPAT_CALGARY_EVAL_XT2_NOREG.COM:STEP_1',
                    outer_roi='supplied_full_mask_not_generated_by_script',
                    cortical_seed_sigma_voxels=sigma, cortical_seed_support_voxels=3,
                    cortical_seed_thresholds_mgHA_cm3=[threshold, 3000], minimum_cortical_peel_xy_voxels=peel,
                    erosion_dilation_distance_voxels=3, first_close_open_distance_voxels=15,
                    corner_component_minimum_voxels=minimum, corner_final_range_voxels=[1, 800],
                    final_close_distance_voxels=close_distance,
                    morphology_metric='voxel_centre_euclidean_ball_unverified_against_IPL_metric_11',
                    boundary='zero_xy_exterior; edge_replicated_stack_ends; expanded_close_grid',
                    unverified_native_details=['metric_11', 'GOBJ_peel_rasterization',
                                               'Gaussian_support_and_margins', 'bounding_box_margins',
                                               'stack_end_boundary_handling', 'component_ties'],
                    trab_voxels_outside_peel_after_final_cortical_cleanup=0,
                    stage_seconds={}, stage_voxels={})

    def stage(name, fn, *args, **kwargs):
        if progress is not None:
            progress(name)
        start = perf_counter()
        out = fn(*args, **kwargs)
        metadata['stage_seconds'][name] = perf_counter()-start
        metadata['stage_voxels'][name] = int(np.count_nonzero(out))
        return out

    trab = np.zeros_like(full)
    if full.any():
        # Preserve the script's bounding-box domain for inversion and rank
        # extraction: exterior corners participate before masking back to all.
        bounds = ndi.find_objects(full.astype(np.uint8))[0]
        # Preserve known empty Z layers; continuation belongs at scan ends,
        # never at a newly cropped interior ROI boundary.
        bounds = (*bounds[:2], slice(0, full.shape[2]))
        all_mask = np.ascontiguousarray(full[bounds])
        masked_density = np.where(all_mask, density[bounds], 0.)
        filtered = stage('gross_cortical_seed_gaussian', _arrays.smooth_xyz,
                         masked_density, sigma=sigma, support=3)
        seed = (filtered >= threshold) & (filtered <= 3000)
        del filtered, masked_density
        allowed = stage('outer_peel_xy', peel_xy, all_mask, peel)
        marrow = (all_mask & ~seed) & allowed
        # /set trab 0 127 inverts both background and foreground, including
        # outside-all within the bounding box. Mask only AFTER cortical CL.
        cortex = stage('inverted_cortical_rank_one', largest_component, ~marrow) & all_mask
        marrow = stage('trabecular_rank_one', largest_component, all_mask & ~cortex)
        del seed, cortex
        marrow = stage('trab_erode_3', morphology, marrow, 3, 'erode')
        marrow = stage('eroded_trab_rank_one', largest_component, marrow)
        marrow = stage('trab_dilate_3', morphology, marrow, 3, 'dilate')
        marrow = stage('trab_close_15', morphology, marrow, 15, 'close') & allowed
        # Crop XY before opening; retain the true Z domain so an internal
        # foreground bound cannot become an artificial continuation boundary.
        opened = np.zeros_like(marrow)
        if marrow.any():
            inner_bounds = ndi.find_objects(marrow.astype(np.uint8))[0]
            inner_bounds = (*inner_bounds[:2], slice(0, marrow.shape[2]))
            opened[inner_bounds] = stage('trab_open_15', morphology,
                                         marrow[inner_bounds], 15, 'open')
        corners = stage('corners_erode_3', morphology, marrow & ~opened, 3, 'erode')
        del marrow
        corners = stage('eroded_corner_number_filter', component_range, corners, minimum, 0)
        corners = stage('corners_dilate_3', morphology, corners, 3, 'dilate')
        corners = stage('dilated_corner_number_filter', component_range, corners, minimum, 0)
        corners = stage('literal_final_corner_filter_1_to_800', component_range, corners, 1, 800)
        metadata['corner_voxels_after_final_range'] = int(corners.sum())
        marrow = stage('site_specific_final_close', morphology, opened | corners, close_distance, 'close') & allowed
        del opened, corners
        marrow = stage('trab_slice_50_to_100_percent', slice_majority, marrow)
        cortex = stage('cort_slice_50_to_100_percent', slice_majority, all_mask & ~marrow)
        trab[bounds] = all_mask & ~cortex
        # The literal final complement reassigns discarded cortical components
        # to TRAB, even outside the previous peel. Report, do not silently repair.
        metadata['trab_voxels_outside_peel_after_final_cortical_cleanup'] = int(
            (trab[bounds] & ~allowed).sum())
        if parameters is not None:
            # The shared standard promises a minimum rim, unlike the literal
            # candidate's final complement which can reassign small fragments.
            trab[bounds] &= allowed
    metadata['minimum_peel_enforced_after_cleanup'] = parameters is not None
    if parameters is not None:
        metadata['ignored_legacy_controls'] = ['endosteal_kernel_size', 'use_adaptive_threshold',
                                               'buie', 'stable_3d.inner_sigma_mm']
    metadata.setdefault('corner_voxels_after_final_range', 0)
    metadata['empty_trabecular_mask'] = not bool(trab.any())
    return trab, full & ~trab, metadata
