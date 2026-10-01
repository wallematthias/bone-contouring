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
    assert payload['standard_algorithm'] == 'topology_first_v1'
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
    assert payload['standard_algorithm'] == 'topology_first_v1'
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
    p.stable_3d.outer_sigma_mm = p.stable_3d.inner_sigma_mm = (0, 0, 0)
    p.segmentation.trab_threshold = p.segmentation.cort_threshold = 10000
    out = generate_masks_from_image(image, p)
    expected = np.zeros_like(density, bool)
    expected[:, 8:33, 8:33] = True
    assert np.array_equal(sitk.GetArrayFromImage(out.full).astype(bool), expected)
    assert np.array_equal(sitk.GetArrayFromImage(out.cort).astype(bool), density > 0)
    assert not sitk.GetArrayFromImage(out.seg).any()
    assert out.metadata['outer_contour']['algorithm_revision'] == 'topology_first_v1'
    assert out.metadata['inner_contour']['algorithm_revision'] == 'topology_first_v1'
    p.outer.contour_method = p.inner.contour_method = 'stable_3d'
    explicit = generate_masks_from_image(image, p)
    for role in ('full', 'trab', 'cort'):
        assert np.array_equal(sitk.GetArrayFromImage(getattr(out, role)),
                              sitk.GetArrayFromImage(getattr(explicit, role)))


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
