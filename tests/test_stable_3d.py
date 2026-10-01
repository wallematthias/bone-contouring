"""Behavioral regressions for the opt-in topology-first 3D contours."""
import numpy as np
import pytest
import SimpleITK as sitk

from bone_contouring import generate_masks_from_image, resolve_preset


def params():
    return resolve_preset(modality='xct2', site='radius', segmentation='gauss',
                          outer_contour='stable_3d', inner_contour='stable_3d')


def image(density):
    im = sitk.GetImageFromArray(density.transpose(2, 1, 0).astype(np.float32))
    im.SetSpacing((.06, .07, .08))
    im.SetOrigin((1., 2., 3.))
    im.SetDirection((0., -1., 0., 1., 0., 0., 0., 0., 1.))
    return im


def shell():
    d = np.zeros((41, 41, 7), np.float32)
    d[8:33, 8:33, :] = 900
    d[9:32, 9:32, :] = 0
    return d


def test_fixed_candidate_settings_and_legacy_methods_remain_separate():
    p = params()
    assert p.inner.endosteal_threshold == 380.
    assert p.buie.endosteal_kernel_size == (31, 31, 1)
    legacy = resolve_preset(outer_contour='buie', inner_contour='buie')
    assert legacy.buie.endosteal_kernel_size == (10, 10, 1)


def test_symmetric_distance_does_not_move_a_flat_axial_invariant_boundary():
    from bone_contouring import _stable_3d
    m = np.zeros((41, 45, 9), bool)
    m[8:33, 8:37, :] = True
    out = _stable_3d.regularize_mask_xyz(m, (.06, .07, .08), (.03, .03, .12), .12)
    assert np.array_equal(out[10:31, :, :], m[10:31, :, :])
    assert np.array_equal(out[:, 10:35, :], m[:, 10:35, :])


@pytest.mark.parametrize('spacing', [(0., .06, .06), (.06, float('nan'), .06), (.06, .06)])
def test_invalid_physical_spacing_is_rejected(spacing):
    from bone_contouring import _stable_3d
    with pytest.raises(ValueError):
        _stable_3d.regularize_mask_xyz(shell() > 0, spacing, (0., 0., .12), .12)


@pytest.mark.parametrize('stage', ['outer', 'inner'])
def test_invalid_density_threshold_is_rejected(stage):
    p = params()
    if stage == 'outer':
        p.outer.periosteal_threshold = float('nan')
    else:
        p.inner.endosteal_threshold = float('nan')
    with pytest.raises(ValueError):
        generate_masks_from_image(image(shell()), p)


def test_thin_shell_is_filled_before_any_cleanup_can_open_it():
    # Catches applying the legacy pre-fill opening, which removes this shell.
    p = params()
    p.outer.gaussian_sigma = 0
    p.outer.periosteal_kernel_size = 0
    p.outer.periosteal_open_radius = 2
    p.stable_3d.outer_sigma_mm = (0., 0., 0.)
    p.segmentation.enabled = False
    masks = generate_masks_from_image(image(shell()), p)
    want = np.zeros((7, 41, 41), bool)
    want[:, 8:33, 8:33] = True
    assert np.array_equal(sitk.GetArrayFromImage(masks.full).astype(bool), want)


def test_stable_outer_is_independent_of_tissue_segmentation_settings():
    p = params()
    first = generate_masks_from_image(image(shell()), p)
    p.segmentation.trab_threshold = p.segmentation.cort_threshold = 10000
    p.segmentation.gaussian_sigma = 5
    second = generate_masks_from_image(image(shell()), p)
    assert np.array_equal(sitk.GetArrayFromImage(first.full), sitk.GetArrayFromImage(second.full))
    assert resolve_preset().outer.contour_method == 'standard'


def test_signed_distance_smoothing_reduces_alternating_slice_jitter():
    from bone_contouring import _stable_3d
    x, y, z = np.indices((45, 45, 15))
    radius = np.where(z % 2, 13., 11.)
    mask = (x - 22)**2 + (y - 22)**2 < radius**2
    p = params().stable_3d
    smoothed = _stable_3d.regularize_mask_xyz(mask, (.06, .06, .06), (0., 0., .12), p.max_boundary_shift_mm)
    assert np.count_nonzero(np.diff(smoothed.astype(int), axis=2)) < np.count_nonzero(np.diff(mask.astype(int), axis=2))
    assert smoothed[22, 22, :].all() and not smoothed[0, 0, :].any()


@pytest.mark.parametrize('value', [False, True])
def test_uniform_masks_and_zero_sigma_do_not_create_phantom_boundaries(value):
    from bone_contouring import _stable_3d
    mask = np.full((9, 11, 3), value)
    out = _stable_3d.regularize_mask_xyz(mask, (.06, .07, .08), (.03, .03, .12), .12)
    assert np.array_equal(out, mask)
    mask[2:7, 2:9, :] = not value
    out = _stable_3d.regularize_mask_xyz(mask, (.06, .07, .08), (0., 0., 0.), .12)
    assert np.array_equal(out, mask)


def test_inner_does_not_impose_the_legacy_fixed_cortical_peel():
    # One-voxel cortex must not be replaced with a mandatory three-voxel rim.
    p = params()
    p.outer.gaussian_sigma = p.inner.gaussian_sigma = 0
    p.outer.periosteal_kernel_size = 0
    p.stable_3d.outer_sigma_mm = p.stable_3d.inner_sigma_mm = (0., 0., 0.)
    p.buie.endosteal_kernel_size = (1, 1, 1)
    p.segmentation.enabled = False
    masks = generate_masks_from_image(image(shell()), p)
    trab = sitk.GetArrayFromImage(masks.trab).astype(bool)
    assert trab[:, 10, 20].all()
    assert not trab[:, 8, 20].any()
    p.inner.peel = 10
    again = generate_masks_from_image(image(shell()), p)
    assert np.array_equal(trab, sitk.GetArrayFromImage(again.trab).astype(bool))


def test_empty_marrow_is_reported_without_synthetic_fallback():
    p = params()
    p.outer.gaussian_sigma = 0
    p.outer.periosteal_kernel_size = 0
    p.segmentation.enabled = False
    d = np.zeros((31, 31, 7), np.float32)
    d[5:26, 5:26, :] = 900
    masks = generate_masks_from_image(image(d), p)
    assert not sitk.GetArrayFromImage(masks.trab).any()
    assert np.array_equal(sitk.GetArrayFromImage(masks.cort), sitk.GetArrayFromImage(masks.full))
    assert masks.metadata['endosteal_fallback']['applied'] is False
    assert 'empty_compartment' in masks.metadata['inner_contour']['quality']['warnings']


def test_partition_and_geometry_are_preserved():
    im = image(shell())
    masks = generate_masks_from_image(im, params())
    f, t, c = [sitk.GetArrayFromImage(m).astype(bool) for m in (masks.full, masks.trab, masks.cort)]
    assert np.array_equal(t | c, f) and not (t & c).any()
    for m in (masks.full, masks.trab, masks.cort):
        assert m.GetSpacing() == im.GetSpacing()
        assert m.GetOrigin() == im.GetOrigin()
        assert m.GetDirection() == im.GetDirection()


def test_quality_flags_a_single_slice_compartment_collapse():
    from bone_contouring import _stable_3d
    m = np.zeros((31, 31, 7), bool)
    m[5:26, 5:26, :] = True
    m[:, :, 3] = False
    q = _stable_3d.mask_quality_xyz(m, (.06, .06, .06), params().stable_3d)
    assert 'empty_slices' in q['warnings'] and 'area_jump' in q['warnings']
    assert q['empty_slices'] == [3]


@pytest.mark.parametrize('field,value', [
    ('outer_sigma_mm', (-.1, 0., 0.)), ('inner_sigma_mm', (0., float('nan'), 0.)),
    ('outer_sigma_mm', (0., 0.)), ('max_boundary_shift_mm', float('inf')),
    ('area_jump_fraction', 0.), ('adjacent_boundary_limit_mm', -1.)])
def test_invalid_stable_parameters_are_rejected(field, value):
    p = params()
    setattr(p.stable_3d, field, value)
    with pytest.raises(ValueError):
        generate_masks_from_image(image(shell()), p)


def test_profile_roundtrip_and_settings_hash_include_stable_controls(tmp_path):
    from bone_imaging_derivatives import save_json_profile
    from bone_contouring import load_preset
    from bone_contouring.batch import _settings_hash
    save_json_profile('bone-contouring', 'Stable Test', {
        'schema': 'bone-contour-recipe-v1', 'modality': 'xct2', 'site': 'radius',
        'methods': {'periosteal_contour': 'stable_3d', 'endosteal_contour': 'stable_3d'},
        'parameters': {'stable_3d': {'inner_sigma_mm': [.03, .03, .18]}}}, root=tmp_path)
    p = load_preset('Stable Test', profile_root=tmp_path)
    assert tuple(p.stable_3d.inner_sigma_mm) == (.03, .03, .18)
    before = _settings_hash(p)
    p.stable_3d.inner_sigma_mm = (.03, .03, .12)
    assert _settings_hash(p) != before
    before = _settings_hash(p)
    p.buie.endosteal_kernel_size = (7, 7, 1)
    assert _settings_hash(p) != before


def test_benchmark_adjacent_distance_scores_a_known_shift_and_omits_empty_pairs():
    # Catches zero-scoring missing contours or losing physical pixel spacing.
    from benchmarks.benchmark_buie import adjacent_surface_distances
    a = np.zeros((3, 7, 7), bool)
    a[0, 3, 3] = True
    a[1, 3, 4] = True
    measured = adjacent_surface_distances(a, (.07, .06))
    assert measured['mean_mm'] == pytest.approx(.06)
    assert measured['max_pair_mean_mm'] == pytest.approx(.06)
    assert measured['omitted_pairs'] == 1
