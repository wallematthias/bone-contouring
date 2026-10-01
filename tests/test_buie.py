"""Buie stages: independent small-volume references, not saved-output snapshots."""
from dataclasses import asdict

import numpy as np
import pytest
import SimpleITK as sitk

from bone_contouring import generate_masks_from_image, resolve_preset


def test_buie_is_opt_in_and_preserves_default():
    assert resolve_preset().outer.contour_method == "standard"
    params = resolve_preset(modality="xct2", site="tibia", segmentation="gauss",
                            outer_contour="buie", inner_contour="buie")
    assert asdict(params)["buie"]["periosteal_kernel_size"] == (15, 15, 1)
    assert params.buie.endosteal_kernel_size == (10, 10, 1)


def _literal_morph(mask, size, dilate):
    """VTK neighborhood: ellipsoid centered (size-1)/2; anchor floor(size/2)."""
    offsets = []
    for index in np.ndindex(size):
        if sum(((i - (n - 1) / 2) / (n / 2)) ** 2
               for i, n in zip(index, size)) <= 1:
            offsets.append(tuple(i - n // 2 for i, n in zip(index, size)))
    result = np.empty_like(mask)
    for index in np.ndindex(mask.shape):
        values = []
        for offset in offsets:
            neighbor = tuple(i + d for i, d in zip(index, offset))
            if all(0 <= i < n for i, n in zip(neighbor, mask.shape)):
                values.append(mask[neighbor])
        result[index] = any(values) if dilate else all(values)
    return result


@pytest.mark.parametrize("size", [(3, 3, 1), (4, 4, 1), (10, 10, 1), (3, 3, 3)])
@pytest.mark.parametrize("dilate", [True, False])
def test_morphology_matches_literal_vtk_anchor_and_edges(size, dilate):
    from bone_contouring._buie import morphology_xyz
    mask = np.random.default_rng(42).random((11, 12, 3)) > .55
    assert np.array_equal(morphology_xyz(mask, size, dilate=dilate),
                          _literal_morph(mask, size, dilate))


def _ring():
    x, y, z = np.indices((65, 65, 9))
    r = np.hypot(x - 32, y - 32)
    density = np.where((r >= 18) & (r <= 25), 900., 0.).astype(np.float32)
    # A cortical canal narrower than the closing footprint, and thin trabeculae.
    density[(x == 32) & (y > 32)] = 0
    density[(r < 18) & (x % 8 == 0)] = 400
    image = sitk.GetImageFromArray(density.transpose(2, 1, 0))
    image.SetSpacing((.06, .07, .08))
    image.SetOrigin((1., 2., 3.))
    image.SetDirection((0., -1., 0., 1., 0., 0., 0., 0., 1.))
    return image


def test_buie_partition_geometry_and_threshold_support_independence():
    params = resolve_preset(segmentation="gauss", outer_contour="buie", inner_contour="buie")
    params.segmentation.enabled = False
    image = _ring()
    masks = generate_masks_from_image(image, params)
    full, trab, cort = [sitk.GetArrayFromImage(im).astype(bool)
                        for im in (masks.full, masks.trab, masks.cort)]
    assert full[:, 32, 32].all() and trab[:, 32, 32].all()
    assert cort[:, 32, 55].all()
    assert np.array_equal(trab | cort, full)
    assert not (trab & cort).any()
    for im in (masks.full, masks.trab, masks.cort):
        assert im.GetSpacing() == image.GetSpacing()
        assert im.GetOrigin() == image.GetOrigin()
        assert im.GetDirection() == image.GetDirection()
    params.segmentation.trab_threshold = 10000
    params.segmentation.cort_threshold = 10000
    params.outer.gaussian_sigma = 20
    params.inner.peel = 15
    again = generate_masks_from_image(image, params)
    assert np.array_equal(full, sitk.GetArrayFromImage(again.full).astype(bool))
    assert np.array_equal(trab, sitk.GetArrayFromImage(again.trab).astype(bool))
    assert masks.metadata["outer_contour"]["support"] == "raw_dual_threshold"
    assert masks.metadata["endosteal_fallback"]["applied"] is False


def test_connectivity_is_volumetric_without_virtual_scan_end_padding():
    from bone_contouring._buie import fill_from_exterior_xyz
    shell = np.zeros((13, 13, 3), dtype=bool)
    shell[3:10, 3:10, :] = True
    shell[4:9, 4:9, :] = False
    assert fill_from_exterior_xyz(shell, fully_connected=False)[6, 6, :].all()
    shell[3, 6, 1] = False  # One axial opening connects all marrow slices.
    assert not fill_from_exterior_xyz(shell, fully_connected=False)[6, 6, :].any()


def test_final_smoothing_uses_white_exterior_and_lower_100():
    from bone_contouring._buie import smooth_exterior_xyz
    from bone_contouring import BuieParameters
    marrow = np.zeros((21, 21, 7), dtype=bool)
    marrow[5:16, 5:16, :] = True
    p = BuieParameters(gaussian_sigma=0)
    assert np.array_equal(smooth_exterior_xyz(marrow, p) < p.final_threshold, marrow)
    p.gaussian_sigma = 3
    result = smooth_exterior_xyz(marrow, p)
    assert result[10, 10, 3] < 100
    assert result[5, 10, 3] >= 100  # The 100 threshold contracts, not expands, marrow.
    assert np.array_equal(result[:, :, 0], result[:, :, -1])


@pytest.mark.parametrize("field,value", [("median_kernel_size", (2, 3, 1)),
    ("endosteal_kernel_size", (0, 10, 1)), ("gaussian_sigma", -1),
    ("final_threshold", 256), ("gaussian_radius_factors", (3, float('nan'), 1)),
    ("fully_connected", "false"), ("fully_connected", 6)])
def test_invalid_buie_settings_are_rejected(field, value):
    params = resolve_preset(segmentation="gauss", outer_contour="buie", inner_contour="buie")
    setattr(params.buie, field, value)
    with pytest.raises(ValueError):
        generate_masks_from_image(_ring(), params)


def test_empty_image_stays_empty_without_synthetic_compartment_fallback():
    p = resolve_preset(segmentation="gauss", outer_contour="buie", inner_contour="buie")
    p.segmentation.enabled = False
    masks = generate_masks_from_image(sitk.Image([15, 15, 3], sitk.sitkFloat32), p)
    assert not sitk.GetArrayFromImage(masks.full).any()
    assert not sitk.GetArrayFromImage(masks.trab).any()
    assert not sitk.GetArrayFromImage(masks.cort).any()


def test_thresholds_act_on_raw_density_not_a_gaussian_image():
    from bone_contouring._buie import outer_contour_xyz
    from bone_contouring import BuieParameters
    density = np.zeros((9, 9, 2), dtype=np.float32)
    density[4, 4, :] = 301
    p = BuieParameters(median_kernel_size=(1, 1, 1), periosteal_kernel_size=(1, 1, 1))
    below, _ = outer_contour_xyz(density, 300, p)
    above, _ = outer_contour_xyz(density, 302, p)
    assert np.array_equal(below, density >= 300)
    assert not above.any()


def test_finite_support_gaussian_matches_independent_clipped_reference():
    from bone_contouring._buie import smooth_exterior_xyz
    from bone_contouring import BuieParameters
    marrow = np.random.default_rng(71).random((6, 7, 5)) > .4
    p = BuieParameters(gaussian_sigma=1, gaussian_radius_factors=(2, 2, 1))
    expected = (~marrow).astype(np.uint8) * 255
    for axis in (2, 1, 0):
        result = np.empty_like(expected)
        radius = int(p.gaussian_sigma * p.gaussian_radius_factors[axis])
        for index in np.ndindex(marrow.shape):
            value, weight = 0., 0.
            for offset in range(-radius, radius + 1):
                neighbor = list(index)
                neighbor[axis] += offset
                if 0 <= neighbor[axis] < marrow.shape[axis]:
                    w = np.exp(-offset ** 2 / (2 * p.gaussian_sigma ** 2))
                    value += w * expected[tuple(neighbor)]
                    weight += w
            result[index] = np.floor(np.clip(value / weight, 0, 255) + 1e-10)
        expected = result
    actual = smooth_exterior_xyz(marrow, p)
    assert np.array_equal(actual, expected)


def test_gaussian_preserves_constant_white_at_clipped_boundaries():
    from bone_contouring._buie import smooth_exterior_xyz
    from bone_contouring import BuieParameters
    result = smooth_exterior_xyz(np.zeros((5, 5, 3), dtype=bool),
                                 BuieParameters(gaussian_sigma=1))
    assert np.all(result == 255)


@pytest.mark.parametrize('schema', ['bone-contour-recipe-v1', 'bone-contouring-profile-v1'])
def test_saved_buie_profile_restores_kernels_and_thresholds(tmp_path, schema):
    from bone_imaging_derivatives import save_json_profile
    from bone_contouring import load_preset
    payload = {'modality': 'xct2', 'site': 'tibia',
               'methods': {'bone_segmentation': 'gauss', 'periosteal_contour': 'buie',
                           'endosteal_contour': 'buie'},
               'parameters': {'buie': {'endosteal_kernel_size': [8, 8, 1]},
                              'inner': {'endosteal_threshold': 475}}}
    document = dict(payload, schema=schema) if schema == 'bone-contour-recipe-v1' else {
        'schema': schema, 'contour_parameters': payload}
    save_json_profile('bone-contouring', 'Test Buie', document, root=tmp_path)
    restored = load_preset('Test Buie', profile_root=tmp_path)
    assert restored.outer.contour_method == 'buie'
    assert restored.inner.endosteal_threshold == 475
    assert tuple(restored.buie.endosteal_kernel_size) == (8, 8, 1)


@pytest.mark.parametrize('outer,inner', [('buie', 'none'), ('standard', 'buie')])
def test_buie_stages_can_be_selected_independently(outer, inner):
    p = resolve_preset(segmentation='gauss', outer_contour=outer, inner_contour=inner)
    p.segmentation.enabled = False
    m = generate_masks_from_image(_ring(), p)
    assert m.metadata['periosteal_contour_method'] == outer
    assert m.metadata['endosteal_contour_method'] == inner


def test_batch_settings_hash_tracks_effective_kernel_in_standard_and_buie():
    from bone_contouring.batch import _parameters_payload, _settings_hash
    p = resolve_preset()
    original = _settings_hash(p)
    p.buie.endosteal_kernel_size = (8, 8, 1)
    assert _settings_hash(p) != original
    assert _parameters_payload(p)['buie']['endosteal_kernel_size'] == (8, 8, 1)
    p.inner.contour_method = 'buie'
    before = _settings_hash(p)
    assert _parameters_payload(p)['buie']['endosteal_kernel_size'] == (8, 8, 1)
    p.buie.endosteal_kernel_size = (10, 10, 1)
    assert _settings_hash(p) != before


def test_profile_loaded_string_connectivity_is_not_silently_truthy(tmp_path):
    from bone_imaging_derivatives import save_json_profile
    from bone_contouring import load_preset
    save_json_profile('bone-contouring', 'Bad Connectivity', {
        'schema': 'bone-contour-recipe-v1', 'modality': 'xct2', 'site': 'tibia',
        'methods': {'bone_segmentation': 'gauss', 'periosteal_contour': 'buie',
                    'endosteal_contour': 'buie'},
        'parameters': {'buie': {'fully_connected': 'false'}}}, root=tmp_path)
    p = load_preset('Bad Connectivity', profile_root=tmp_path)
    with pytest.raises(ValueError, match='fully_connected'):
        generate_masks_from_image(_ring(), p)
