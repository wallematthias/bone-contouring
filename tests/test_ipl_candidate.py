"""Literal IPL STEP_1 candidate, independently of the stable default."""
import numpy as np
import pytest
from scipy import ndimage as ndi

from bone_contouring import resolve_preset


def test_candidate_is_opt_in_and_only_supports_script_sites():
    p = resolve_preset(modality='xct2', site='tibia', inner_contour='IPL')
    assert p.inner.contour_method == 'ipl'
    assert resolve_preset(modality='xct2', site='tibia').inner.contour_method == 'standard'
    for modality, site in [('xct1', 'radius'), ('xct2', 'knee')]:
        with pytest.raises(ValueError, match='XCTII radius or tibia'):
            resolve_preset(modality=modality, site=site, inner_contour='ipl')


def test_batch_settings_hash_records_fixed_candidate_algorithm():
    from bone_contouring.batch import _parameters_payload
    p = resolve_preset(modality='xct2', site='tibia', inner_contour='ipl')
    payload = _parameters_payload(p)
    assert payload['ipl_candidate_algorithm'] == 'ipl_step1_euclidean_v2_z_continuation'


def test_rank_and_number_filters_use_face_connectivity_and_inclusive_limits():
    from bone_contouring import _ipl
    a = np.zeros((12, 12, 3), bool)
    a[1:4, 1:4, 1] = True  # Nine voxels.
    a[4:6, 4:6, 1] = True  # Four voxels; diagonal contact is not connectivity.
    assert _ipl.largest_component(a).sum() == 9
    assert _ipl.component_range(a, 4, 4).sum() == 4
    assert _ipl.component_range(a, 9, 0).sum() == 9
    assert not _ipl.largest_component(np.zeros_like(a)).any()
    assert not _ipl.component_range(np.zeros_like(a), 1, 800).any()


def test_slice_cleanup_is_relative_to_each_slices_total_foreground():
    from bone_contouring import _ipl
    a = np.zeros((15, 15, 2), bool)
    a[1:4, 1:4, 0] = True
    a[10:12, 10:12, 0] = True
    a[1:3, 1:3, 1] = True
    a[10:12, 10:12, 1] = True
    out = _ipl.slice_majority(a)
    assert out[:, :, 0].sum() == 9
    assert out[:, :, 1].sum() == 8  # Both exactly 50% are retained.


def test_peel_treats_crop_exterior_as_outside_the_contour_not_foreground():
    from bone_contouring import _ipl
    full = np.ones((31, 33, 3), bool)
    peeled = _ipl.peel_xy(full, 6)
    assert not peeled[:6].any()
    assert not peeled[-6:].any()
    assert not peeled[:, :6].any()
    assert not peeled[:, -6:].any()
    assert peeled[15, 16, :].all()  # XY peel does not cap the stack ends.


@pytest.mark.parametrize('operation', ['erode', 'dilate', 'open', 'close'])
def test_euclidean_candidate_operator_matches_ball_with_xy_background_and_z_continuation(operation):
    from bone_contouring import _ipl
    a = np.zeros((19, 21, 17), bool)
    a[0:15, 4:17, 0:14] = True
    a[7:10, 8:11, 5:8] = False
    r = 3
    grid = np.indices((2*r+1,)*3) - r
    ball = (grid*grid).sum(axis=0) <= r*r
    # Two sequential ball operations reach up to 2*r along Z. Extend only
    # terminal slices; XY remains true background, including cropped edges.
    pad = np.pad(a, ((r+1, r+1), (r+1, r+1), (0, 0)))
    pad = np.pad(pad, ((0, 0), (0, 0), (2*r+1, 2*r+1)), mode='edge')
    fn = {'erode': ndi.binary_erosion, 'dilate': ndi.binary_dilation,
          'open': ndi.binary_opening, 'close': ndi.binary_closing}[operation]
    expected = fn(pad, structure=ball)[r+1:-r-1, r+1:-r-1, 2*r+1:-2*r-1]
    np.testing.assert_array_equal(_ipl.morphology(a, r, operation), expected)
    assert not _ipl.morphology(np.zeros_like(a), r, operation).any()


@pytest.mark.parametrize('site', ['radius', 'tibia'])
def test_constant_cylinder_has_no_artificial_endosteal_scan_end_caps(site):
    from bone_contouring import _ipl
    x, y, _ = np.indices((85, 85, 49))
    radial = np.hypot(x-42, y-42)
    full = radial <= 37
    density = np.where(full & (radial >= 28), 1000., 0.).astype(np.float32)
    trab, cort, _ = _ipl.inner_contour_xyz(density, full, site=site)
    assert trab[:, :, 24].any()
    for z in range(full.shape[2]):
        np.testing.assert_array_equal(trab[:, :, z], trab[:, :, 24])
        np.testing.assert_array_equal(cort[:, :, z], cort[:, :, 24])


def test_z_continuation_does_not_erase_real_empty_internal_slices():
    from bone_contouring import _ipl
    mask = np.zeros((19, 19, 21), bool)
    mask[4:15, 4:15, :8] = True
    mask[4:15, 4:15, 13:] = True
    eroded = _ipl.morphology(mask, 2, 'erode')
    assert eroded[9, 9, 0] and eroded[9, 9, -1]
    assert not eroded[:, :, 8:13].any()


def test_script_pipeline_partitions_full_without_changing_outer_roi():
    from bone_contouring import _ipl, _arrays
    x, y, z = np.indices((85, 85, 49))
    radial = np.hypot(x-42, y-42)
    full = radial <= 37
    density = np.where(full & (radial >= 28), 1000., 0.).astype(np.float32)
    before = full.copy()
    trab, cort, metadata = _ipl.inner_contour_xyz(density, full, site='tibia')
    np.testing.assert_array_equal(full, before)
    np.testing.assert_array_equal(trab | cort, full)
    assert not (trab & cort).any()
    assert trab[42, 42, 24]
    assert cort[76, 42, 24]
    assert not (trab & ~_arrays._apply_xy_morphology(full, 6, 'erode')).any()
    assert metadata['corner_voxels_after_final_range'] == 0
    assert metadata['native_ipl_equivalence'] is False
    assert metadata['final_close_distance_voxels'] == 50


def test_empty_marrow_does_not_become_full_after_dilation():
    from bone_contouring import _ipl
    full = np.ones((31, 31, 31), bool)
    trab, cort, metadata = _ipl.inner_contour_xyz(np.full(full.shape, 2000.), full, site='radius')
    assert not trab.any()
    np.testing.assert_array_equal(cort, full)
    assert metadata['empty_trabecular_mask']


def test_literal_final_cortical_cleanup_can_reassign_small_rim_fragments_to_trab():
    from bone_contouring import _ipl
    full = np.zeros((33, 33, 3), bool)
    full[:20, :20, :] = True
    full[26:29, 26:29, :] = True
    trab, cort, metadata = _ipl.inner_contour_xyz(np.full(full.shape, 2000.), full, site='tibia')
    assert trab[26:29, 26:29, :].all()
    assert not trab[:20, :20, :].any()
    np.testing.assert_array_equal(trab | cort, full)
    assert metadata['trab_voxels_outside_peel_after_final_cortical_cleanup'] == 27


def test_public_api_dispatches_candidate_and_preserves_geometry():
    import SimpleITK as sitk
    from bone_contouring import generate_masks_from_image
    x, y, _ = np.indices((85, 85, 49))
    radius = np.hypot(x-42, y-42)
    image = sitk.GetImageFromArray(np.where((radius >= 28) & (radius <= 37), 1000., 0.).transpose(2, 1, 0))
    image.SetSpacing((.06, .06, .06))
    image.SetOrigin((1., 2., 3.))
    result = generate_masks_from_image(image, resolve_preset(modality='xct2', site='tibia',
                                                            segmentation='gauss', inner_contour='ipl'))
    assert result.metadata['endosteal_contour_method'] == 'ipl'
    assert result.metadata['inner_contour']['candidate_name'] == 'IPL'
    for role in ('full', 'trab', 'cort', 'seg'):
        mask = getattr(result, role)
        assert mask.GetSpacing() == image.GetSpacing()
        assert mask.GetOrigin() == image.GetOrigin()
