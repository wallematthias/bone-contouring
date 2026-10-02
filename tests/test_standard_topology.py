"""Every standard preset is topology-first, not an optional legacy switch."""
import numpy as np
import pytest
import SimpleITK as sitk

from bone_contouring import generate_masks_from_image, resolve_preset
from bone_contouring.batch import _parameters_payload, _settings_hash


@pytest.mark.parametrize('site', ['radius', 'tibia'])
def test_standard_defaults_and_versioned_hash(site):
    p = resolve_preset(modality='xct2', site=site, segmentation='gauss')
    assert p.outer.contour_method == p.inner.contour_method == 'standard'
    assert p.outer.periosteal_threshold == 320
    assert p.inner.endosteal_threshold == 380
    assert p.outer.gaussian_sigma == p.inner.gaussian_sigma == .8
    assert p.buie.endosteal_kernel_size == (31, 31, 1)
    assert p.stable_3d.inner_sigma_mm == (.03, .03, .06)
    payload = _parameters_payload(p)
    assert payload['standard_algorithm'] == 'topology_first_v4'
    first = _settings_hash(p)
    p.stable_3d.inner_sigma_mm = (.03, .03, .12)
    assert _settings_hash(p) != first
    first = _settings_hash(p)
    p.buie.endosteal_kernel_size = (29, 29, 1)
    assert _settings_hash(p) != first


@pytest.mark.parametrize('modality,site', [('xct1', 'radius'), ('xct1', 'tibia'),
                                         ('xct1', 'knee'), ('xct2', 'knee')])
def test_other_presets_keep_density_settings_but_version_the_repaired_algorithm(modality, site):
    p = resolve_preset(modality=modality, site=site)
    assert p.outer.periosteal_threshold == 300
    assert p.inner.endosteal_threshold == 500
    assert p.outer.gaussian_sigma == p.inner.gaussian_sigma == 1.5
    assert p.buie.endosteal_kernel_size == (10, 10, 1)
    if modality == 'xct1' and site != 'knee':
        assert p.segmentation.laplace_hamming_threshold == 15000
    payload = _parameters_payload(p)
    assert payload['standard_algorithm'] == 'topology_first_v4'
    first = _settings_hash(p)
    p.stable_3d.inner_sigma_mm = (.03, .03, .04)
    assert _settings_hash(p) != first


@pytest.mark.parametrize('modality', ['xct1', 'xct2'])
@pytest.mark.parametrize('site', ['radius', 'tibia', 'knee'])
def test_standard_preserves_a_thin_shell_and_is_independent_of_tissue_thresholds(modality, site):
    density = np.zeros((7, 41, 41), np.float32)
    density[:, 8:33, 8:33] = 900
    density[:, 9:32, 9:32] = 0
    image = sitk.GetImageFromArray(density)
    image.SetSpacing((.06, .06, .06))
    p = resolve_preset(modality=modality, site=site, segmentation='gauss')
    p.outer.gaussian_sigma = p.inner.gaussian_sigma = 0
    p.outer.periosteal_kernel_size = 0
    p.buie.endosteal_kernel_size = (1, 1, 1)
    p.inner.peel = 0  # Isolate density-derived contours without the optional minimum rim.
    p.stable_3d.outer_sigma_mm = p.stable_3d.inner_sigma_mm = (0, 0, 0)
    p.segmentation.trab_threshold = p.segmentation.cort_threshold = 10000
    out = generate_masks_from_image(image, p)
    expected = np.zeros_like(density, bool)
    expected[:, 8:33, 8:33] = True
    assert np.array_equal(sitk.GetArrayFromImage(out.full).astype(bool), expected)
    assert np.array_equal(sitk.GetArrayFromImage(out.cort).astype(bool), density > 0)
    assert not sitk.GetArrayFromImage(out.seg).any()
    assert out.metadata['outer_contour']['algorithm_revision'] == 'topology_first_v4'
    assert out.metadata['inner_contour']['algorithm_revision'] == 'topology_first_v4'
    p.outer.contour_method = p.inner.contour_method = 'stable_3d'
    explicit = generate_masks_from_image(image, p)
    for role in ('full', 'trab', 'cort'):
        assert np.array_equal(sitk.GetArrayFromImage(getattr(out, role)),
                              sitk.GetArrayFromImage(getattr(explicit, role)))


@pytest.mark.parametrize('modality', ['xct1', 'xct2'])
@pytest.mark.parametrize('site', ['radius', 'tibia', 'knee'])
def test_default_peel_prevents_full_trab_collapse_even_after_regularization(monkeypatch, modality, site):
    from bone_contouring import _arrays, _stable_3d
    # A shell between the two thresholds otherwise becomes marrow as well as full.
    density = np.zeros((7, 41, 41), np.float32)
    density[:, 8:33, 8:33] = 350
    density[:, 9:32, 9:32] = 0
    p = resolve_preset(modality=modality, site=site, segmentation='gauss')
    assert p.inner.peel == 3
    p.outer.gaussian_sigma = p.inner.gaussian_sigma = 0
    p.outer.periosteal_kernel_size = 0
    p.buie.endosteal_kernel_size = (1, 1, 1)
    p.segmentation.enabled = False
    # Simulate a smoothing pinhole in an ROI reaching full; cleanup must both
    # fill the hole and enforce the rim, including the first and last slices.
    def damaged_regularization(mask, *args):
        result = mask.copy()
        result[20, 20, 3] = False
        return result
    monkeypatch.setattr(_stable_3d, 'regularize_mask_xyz', damaged_regularization)
    out = generate_masks_from_image(sitk.GetImageFromArray(density), p)
    full, trab, cort = [_arrays.sitk_to_numpy_xyz(getattr(out, role)).astype(bool)
                        for role in ('full', 'trab', 'cort')]
    expected = _arrays._apply_xy_morphology(full, 3, 'erode')
    assert np.array_equal(trab, expected)
    assert trab[20, 20, :].all()  # No artificial Z-end caps.
    assert not trab[10, 20, :].any()
    assert trab[11, 20, :].all()
    assert np.array_equal(cort, full & ~trab) and cort.any()
    assert out.metadata['inner_contour']['quality']['axial_holes'] == []
    assert out.metadata['inner_contour']['peel_xy_radius_voxels'] == 3
    assert 'peel' not in out.metadata['inner_contour']['ignored_legacy_controls']
    first = _settings_hash(p)
    p.inner.peel = 0
    assert _settings_hash(p) != first


@pytest.mark.parametrize('modality', ['xct1', 'xct2'])
@pytest.mark.parametrize('site', ['radius', 'tibia', 'knee'])
def test_post_regularization_pinholes_are_filled_in_both_envelopes(monkeypatch, modality, site):
    # A one-voxel hole occurred in SAMPLE355 at reduced Z smoothing. Isolate
    # the smoothing/cleanup contract, without relying on a private patient image.
    from bone_contouring import _stable_3d
    def introduce_pinhole(mask, *args):
        damaged = mask.copy()
        damaged[20, 20, 3] = False
        return damaged
    monkeypatch.setattr(_stable_3d, 'regularize_mask_xyz', introduce_pinhole)
    density = np.zeros((7, 41, 41), np.float32)
    density[:, 8:33, 8:33] = 900
    density[:, 9:32, 9:32] = 0
    image = sitk.GetImageFromArray(density)
    p = resolve_preset(modality=modality, site=site, segmentation='gauss')
    p.outer.gaussian_sigma = p.inner.gaussian_sigma = 0
    p.outer.periosteal_kernel_size = 0
    p.buie.endosteal_kernel_size = (1, 1, 1)
    p.segmentation.enabled = False
    out = generate_masks_from_image(image, p)
    for role in ('full', 'trab'):
        assert sitk.GetArrayFromImage(getattr(out, role))[3, 20, 20]
        assert out.metadata['outer_contour' if role == 'full' else 'inner_contour']['quality']['axial_holes'] == []


@pytest.mark.parametrize('depth', [1, 2, 3])
def test_standard_smoothing_handles_short_stacks(depth):
    image = sitk.GetImageFromArray(np.full((depth, 9, 11), 900, np.float32))
    image.SetSpacing((.06, .07, .08))
    p = resolve_preset(modality='xct1', site='knee', segmentation='gauss')
    p.segmentation.enabled = False
    p.outer.periosteal_kernel_size = 0
    out = generate_masks_from_image(image, p)
    assert sitk.GetArrayFromImage(out.full).all()
    assert not sitk.GetArrayFromImage(out.trab).any()
    assert np.array_equal(sitk.GetArrayFromImage(out.cort), sitk.GetArrayFromImage(out.full))


@pytest.mark.parametrize('modality', ['xct1', 'xct2'])
@pytest.mark.parametrize('site', ['radius', 'tibia', 'knee'])
def test_peripheral_low_density_path_does_not_turn_cortex_into_only_the_peel(modality, site):
    # Without a seed peel, the peripheral low-density layer connects to marrow
    # through a pore. Dilating/filling that layer swallows the dense cortex.
    # Removing only the final peel leaves this regression undetected by QA.
    from bone_contouring import _stable_3d
    density = np.zeros((65, 65, 7), np.float32)
    full = np.zeros_like(density, bool)
    full[8:57, 8:57, :] = True
    density[10:55, 10:55, :] = 900
    density[20:45, 20:45, :] = 0
    density[43:57, 31:34, :] = 0  # One path from marrow to peripheral layer.
    p = resolve_preset(modality=modality, site=site)
    p.inner.gaussian_sigma = 0
    p.stable_3d.inner_sigma_mm = (0, 0, 0)
    trab, cort, _ = _stable_3d.inner_contour_xyz(
        density, full, p.inner, p.buie, p.stable_3d, spacing_xyz=(.082, .082, .082))
    # Hand-selected dense cortex well beyond the 3-voxel minimum rim, opposite
    # the pore, must survive on all slices, including both scan ends.
    assert cort[15, 32, :].all()
    assert not trab[15, 32, :].any()
    assert trab[32, 32, :].all()
    assert not trab[10, 32, :].any()
    assert np.array_equal(trab | cort, full) and not (trab & cort).any()
