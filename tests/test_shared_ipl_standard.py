"""The promoted standard must use one repaired compartment sequence."""
import numpy as np
import pytest
import SimpleITK as sitk

from bone_contouring import generate_masks_from_image, resolve_preset, load_preset
from bone_contouring import _ipl
from bone_contouring.batch import _parameters_payload, _settings_hash


@pytest.mark.parametrize('modality', ['xct1', 'xct2'])
@pytest.mark.parametrize('site', ['radius', 'tibia', 'knee'])
def test_all_standard_recipes_run_repaired_sequence_with_filled_partition(modality, site):
    x, y, z = np.indices((85, 85, 9))
    r = np.hypot(x-42, y-42)
    density = np.where((r >= 29) & (r <= 35), 900., 0.).astype(np.float32)
    image = sitk.GetImageFromArray(density.transpose(2, 1, 0))
    p = resolve_preset(modality=modality, site=site, segmentation='gauss')
    p.outer.gaussian_sigma = 0
    p.outer.periosteal_kernel_size = 0
    p.stable_3d.outer_sigma_mm = (0, 0, 0)
    p.segmentation.enabled = False
    out = generate_masks_from_image(image, p)
    f, t, c = [sitk.GetArrayFromImage(getattr(out, role)) > 0 for role in ('full', 'trab', 'cort')]
    assert out.metadata['inner_contour']['source_recipe'].endswith(':STEP_1')
    assert t[:, 42, 42].all() and c[:, 42, 75].all()
    assert np.array_equal(t | c, f) and not (t & c).any()
    assert np.array_equal(t[0], t[-1])
    assert out.metadata['inner_contour']['minimum_cortical_peel_xy_voxels'] == 6
    assert out.metadata['inner_contour']['quality']['axial_holes'] == []


@pytest.mark.parametrize('profile', ['XtremeCTI', 'XtremeCTII', 'XtremeCTII-LH', 'XtremeCTII-Geodesic'])
@pytest.mark.parametrize('site', ['radius', 'tibia', 'knee'])
def test_named_and_resolved_recipes_have_same_effective_contour_settings(profile, site):
    p = load_preset(profile, site=site)
    expected = resolve_preset(modality=p.modality, site=site, segmentation=p.segmentation.method,
                              outer_contour=p.outer.contour_method)
    assert p.inner == expected.inner
    assert p.inner.gaussian_sigma == 2
    assert p.inner.endosteal_threshold == (150 if site == 'knee' else 500)
    assert p.inner.trabecular_close_radius == {'radius': 30, 'tibia': 50, 'knee': 36}[site]
    assert p.inner.peel == 6


def test_custom_threshold_sigma_peel_and_close_are_effective_and_recorded():
    from bone_contouring.parameters import InnerContourParameters
    density = np.zeros((85, 85, 3), np.float32)
    full = np.ones_like(density, bool)
    p = InnerContourParameters(endosteal_threshold=123, gaussian_sigma=0, peel=0,
                                trabecular_close_radius=0)
    trab, cort, meta = _ipl.inner_contour_xyz(density, full, site='radius', parameters=p)
    assert trab[:, :, 0].any()
    assert np.array_equal(trab | cort, full)
    assert meta['cortical_seed_thresholds_mgHA_cm3'] == [123, 3000]
    assert meta['cortical_seed_sigma_voxels'] == 0
    assert meta['minimum_cortical_peel_xy_voxels'] == 0
    assert meta['final_close_distance_voxels'] == 0
    density.fill(200)
    low = _ipl.inner_contour_xyz(density, full, site='radius', parameters=p)[0]
    p.endosteal_threshold = 300
    high = _ipl.inner_contour_xyz(density, full, site='radius', parameters=p)[0]
    assert not low.any() and high.any()


def test_zero_peel_returns_independent_unchanged_mask():
    full = np.ones((9, 11, 2), bool)
    out = _ipl.peel_xy(full, 0)
    np.testing.assert_array_equal(out, full)
    assert not np.shares_memory(out, full)


def test_standard_hash_includes_effective_compartment_algorithm():
    p = resolve_preset(modality='xct2', site='radius')
    payload = _parameters_payload(p)
    assert payload['standard_inner_algorithm'] == _ipl.ALGORITHM_REVISION
    old = _settings_hash(p)
    p.inner.peel = 5
    assert _settings_hash(p) != old
