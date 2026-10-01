"""IPL-matching deviations tested on hand-built compartment fixtures."""
import numpy as np
import pytest
import SimpleITK as sitk

from bone_contouring import BuieParameters, InnerContourParameters, generate_masks_from_image, resolve_preset


def _image():
    x, y, _ = np.indices((65, 65, 7))
    radius = np.hypot(x - 32, y - 32)
    density = np.where((radius >= 18) & (radius <= 25), 900., 0.).astype(np.float32)
    im = sitk.GetImageFromArray(density.transpose(2, 1, 0))
    im.SetSpacing((.06, .07, .08))
    im.SetOrigin((1., 2., 3.))
    im.SetDirection((0., -1., 0., 1., 0., 0., 0., 0., 1.))
    return im


def test_opt_in_ipl_outer_does_not_depend_on_segmentation_support():
    # Catches accidentally routing this stage through standard's aligned support.
    p = resolve_preset(modality='xct2', site='tibia', segmentation='gauss',
                       outer_contour='ipl_match', inner_contour='none')
    p.segmentation.enabled = False
    first = generate_masks_from_image(_image(), p)
    p.segmentation.enabled = True
    p.segmentation.trab_threshold = 10000
    p.segmentation.cort_threshold = 10000
    p.segmentation.gaussian_sigma = 5
    second = generate_masks_from_image(_image(), p)
    assert sitk.GetArrayFromImage(first.full)[:, 32, 32].all()
    assert np.array_equal(sitk.GetArrayFromImage(first.full), sitk.GetArrayFromImage(second.full))
    assert resolve_preset().outer.contour_method == 'standard'


def test_inner_excludes_disconnected_peripheral_marrow_and_fills_dense_island():
    # Catches missing pre-closing component selection or missing axial filling.
    from bone_contouring import _ipl_match
    full = np.zeros((61, 61, 5), dtype=bool)
    full[3:58, 3:58, :] = True
    density = np.full(full.shape, 900., dtype=np.float32)
    density[3:5, 3:58, :] = 0  # Disconnected low-density periosteal band.
    density[10:51, 10:51, :] = 0
    density[29:32, 29:32, :] = 900  # Tissue island, not a compartment hole.
    # A one-slice corridor connects the island to exterior in 3D. Buie's
    # exterior connectivity therefore leaves axial holes on the other slices.
    density[:32, 29:32, 2] = 900
    p = InnerContourParameters(peel=0)
    b = BuieParameters(endosteal_kernel_size=(1, 1, 1), gaussian_sigma=0)
    trab, cort, meta = _ipl_match.inner_contour_xyz(density, full, p, b)
    expected = np.zeros_like(full)
    expected[10:51, 10:51, :] = True
    expected[10:32, 29:32, 2] = False
    assert np.array_equal(trab, expected)
    assert np.array_equal(cort, full & ~expected)
    assert meta['filled_hole_voxels'] == 36


def test_empty_marrow_does_not_turn_background_label_into_foreground():
    # Catches argmax of all-zero component counts selecting label zero.
    from bone_contouring import _ipl_match
    full = np.ones((9, 9, 3), dtype=bool)
    trab, cort, meta = _ipl_match.inner_contour_xyz(
        np.full(full.shape, 900.), full, InnerContourParameters(peel=0), BuieParameters())
    assert not trab.any()
    assert np.array_equal(cort, full)
    assert meta['empty_trabecular_mask'] is True


def test_peel_prevents_thin_cortex_from_merging_marrows_inside_and_outside():
    # Catches a missing peel or unbounded filling that reaches the periosteum.
    from bone_contouring import _ipl_match
    full = np.zeros((21, 21, 3), bool)
    full[2:19, 2:19, :] = True
    density = np.zeros(full.shape)
    density[5:16, 5:16, :] = 900
    density[6:15, 6:15, :] = 0
    b = BuieParameters(endosteal_kernel_size=(1, 1, 1), gaussian_sigma=0)
    trab, cort, _ = _ipl_match.inner_contour_xyz(density, full, InnerContourParameters(peel=3), b)
    assert trab[10, 10, :].all()
    assert not trab[3, 10, :].any()
    assert cort[3, 10, :].all()


@pytest.mark.parametrize('connected,second_selected', [(False, False), (True, True)])
def test_marrow_connectivity_honors_6_vs_26_choice(connected, second_selected):
    from bone_contouring import _ipl_match
    full = np.ones((11, 11, 3), bool)
    density = np.full(full.shape, 900.)
    density[2:5, 2:5, :] = 0
    density[5:8, 5:8, :] = 0
    b = BuieParameters(endosteal_kernel_size=(1, 1, 1), gaussian_sigma=0,
                       fully_connected=connected)
    trab, _, _ = _ipl_match.inner_contour_xyz(density, full, InnerContourParameters(peel=0), b)
    assert bool(trab[6, 6, 1]) is second_selected


def test_empty_full_and_shape_mismatch_are_handled():
    from bone_contouring import _ipl_match
    full = np.zeros((9, 9, 3), bool)
    t, c, _ = _ipl_match.inner_contour_xyz(np.zeros(full.shape), full,
                                         InnerContourParameters(), BuieParameters())
    assert not (t | c).any()
    with pytest.raises(ValueError, match='matching 3D'):
        _ipl_match.inner_contour_xyz(np.zeros((9, 9)), full, InnerContourParameters(), BuieParameters())


@pytest.mark.parametrize('peel', [-1, 1.5, True])
def test_invalid_ipl_peel_is_rejected(peel):
    from bone_contouring import _ipl_match
    with pytest.raises(ValueError, match='peel'):
        _ipl_match.inner_contour_xyz(np.zeros((9, 9, 3)), np.ones((9, 9, 3), bool),
                                    InnerContourParameters(peel=peel), BuieParameters())


def test_ipl_partition_preserves_geometry_and_does_not_apply_standard_fallback():
    p = resolve_preset(modality='xct2', site='tibia', segmentation='gauss',
                       outer_contour='ipl_match', inner_contour='ipl_match')
    p.segmentation.enabled = False
    image = _image()
    masks = generate_masks_from_image(image, p)
    f, t, c = [sitk.GetArrayFromImage(im).astype(bool) for im in (masks.full, masks.trab, masks.cort)]
    assert t[:, 32, 32].all() and c[:, 32, 55].all()
    assert np.array_equal(t | c, f) and not (t & c).any()
    for im in (masks.full, masks.trab, masks.cort):
        assert im.GetSpacing() == image.GetSpacing()
        assert im.GetOrigin() == image.GetOrigin()
        assert im.GetDirection() == image.GetDirection()
    assert masks.metadata['endosteal_fallback']['applied'] is False


def test_ipl_profile_restores_shared_morphology_and_updates_settings_hash(tmp_path):
    from bone_imaging_derivatives import save_json_profile
    from bone_contouring import load_preset
    from bone_contouring.batch import _settings_hash
    save_json_profile('bone-contouring', 'IPL Test', {
        'schema': 'bone-contour-recipe-v1', 'modality': 'xct2', 'site': 'tibia',
        'methods': {'periosteal_contour': 'ipl_match', 'endosteal_contour': 'ipl_match'},
        'parameters': {'buie': {'endosteal_kernel_size': [7, 7, 1]},
                       'inner': {'endosteal_threshold': 490, 'peel': 2}}}, root=tmp_path)
    p = load_preset('IPL Test', profile_root=tmp_path)
    before = _settings_hash(p)
    assert p.inner.peel == 2 and p.inner.endosteal_threshold == 490
    assert tuple(p.buie.endosteal_kernel_size) == (7, 7, 1)
    p.buie.endosteal_kernel_size = (9, 9, 1)
    assert _settings_hash(p) != before


def test_empty_marrow_image_api_reports_all_cortex_without_fallback():
    p = resolve_preset(segmentation='gauss', outer_contour='ipl_match', inner_contour='ipl_match')
    p.outer.gaussian_sigma = 0
    p.outer.periosteal_kernel_size = 0
    p.outer.periosteal_open_radius = 0
    p.segmentation.enabled = False
    density = np.zeros((21, 21, 5), np.float32)
    density[3:18, 3:18, :] = 900
    masks = generate_masks_from_image(sitk.GetImageFromArray(density.transpose(2, 1, 0)), p)
    assert not sitk.GetArrayFromImage(masks.trab).any()
    assert np.array_equal(sitk.GetArrayFromImage(masks.full), sitk.GetArrayFromImage(masks.cort))
    assert masks.metadata['inner_contour']['empty_trabecular_mask'] is True
    assert masks.metadata['endosteal_fallback']['applied'] is False


def test_inner_only_selection_ignores_standard_inner_gaussian_fields():
    p = resolve_preset(segmentation='gauss', outer_contour='standard', inner_contour='ipl_match')
    p.segmentation.enabled = False
    first = generate_masks_from_image(_image(), p)
    p.inner.gaussian_sigma = 30
    p.inner.trabecular_close_radius = 30
    second = generate_masks_from_image(_image(), p)
    assert np.array_equal(sitk.GetArrayFromImage(first.trab), sitk.GetArrayFromImage(second.trab))
    assert first.metadata['periosteal_contour_method'] == 'standard'


@pytest.mark.parametrize('section,field,value', [
    ('outer', 'periosteal_threshold', float('nan')),
    ('outer', 'gaussian_sigma', float('inf')),
    ('outer', 'gaussian_sigma', -1),
    ('inner', 'endosteal_threshold', float('inf'))])
def test_invalid_ipl_threshold_and_sigma_are_rejected(section, field, value):
    p = resolve_preset(segmentation='gauss', outer_contour='ipl_match', inner_contour='ipl_match')
    setattr(getattr(p, section), field, value)
    with pytest.raises(ValueError, match='finite'):
        generate_masks_from_image(_image(), p)
