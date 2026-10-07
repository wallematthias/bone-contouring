from __future__ import annotations

import sys
import types

import numpy as np
import SimpleITK as sitk

from bone_contouring import ContourParameters, generate_bone_segmentation, generate_masks_from_image, resolve_preset
from bone_contouring._arrays import sitk_to_numpy_xyz


def _image_from_xyz(values: np.ndarray) -> sitk.Image:
    image = sitk.GetImageFromArray(np.transpose(values.astype(np.float32), (2, 1, 0)))
    image.SetSpacing((0.061, 0.062, 0.063))
    image.SetOrigin((1.0, 2.0, 3.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0))
    return image


def _ring_image() -> sitk.Image:
    shape = (33, 33, 7)
    x, y, _z = np.indices(shape)
    radius = np.sqrt((x - 16) ** 2 + (y - 16) ** 2)
    values = np.zeros(shape, dtype=np.float32)
    values[(radius >= 8) & (radius <= 11)] = 900.0
    return _image_from_xyz(values)


def _large_ring_image() -> sitk.Image:
    # The documented 15-voxel opening needs a marrow diameter >30 voxels.
    x, y, _ = np.indices((85, 85, 7))
    r = np.hypot(x-42, y-42)
    values = np.where((r >= 29) & (r <= 35), 900., 0.).astype(np.float32)
    values[r <= 5] = 600  # Interior tissue for the material label test.
    return _image_from_xyz(values)


def _standard_outer_parameters() -> ContourParameters:
    params = ContourParameters()
    params.outer.use_adaptive_threshold = False
    params.outer.periosteal_threshold = 300.0
    params.outer.periosteal_kernel_size = 1
    params.outer.periosteal_open_radius = 0
    params.segmentation.method = "gauss"
    params.segmentation.gaussian_sigma = 0.0
    params.segmentation.trab_threshold = 500.0
    params.segmentation.cort_threshold = 500.0
    params.segmentation.min_size_voxels = 0
    return params


def test_standard_contours_and_gaussian_tissue_each_filter_raw_density(monkeypatch):
    """Contour prefilters must never become the source of the tissue prefilter."""
    from bone_contouring import _arrays

    image = _ring_image()
    raw = sitk_to_numpy_xyz(image).copy()
    calls = []
    smooth = _arrays.smooth_xyz

    def record(source, *, sigma, spacing_xyz=None, support=None):
        calls.append((np.array(source, copy=True), sigma, support))
        if support is None:
            return smooth(source, sigma=sigma, spacing_xyz=spacing_xyz)
        return smooth(source, sigma=sigma, spacing_xyz=spacing_xyz, support=support)

    monkeypatch.setattr(_arrays, "smooth_xyz", record)
    params = resolve_preset(modality="xct2", site="radius", segmentation="gauss")
    params.segmentation.min_size_voxels = 0
    params.segmentation.keep_largest_component = False
    masks = generate_masks_from_image(image, params)
    assert [(sigma, support) for _, sigma, support in calls] == [(.8, None), (2., 3), (.8, 1)]
    np.testing.assert_array_equal(calls[0][0], raw)
    np.testing.assert_array_equal(calls[-1][0], raw)
    np.testing.assert_array_equal(sitk_to_numpy_xyz(image), raw)
    from scipy.ndimage import gaussian_filter
    once = gaussian_filter(raw, .8, radius=1, mode='reflect')
    trab, cort, full = (sitk_to_numpy_xyz(mask) > 0 for mask in (masks.trab, masks.cort, masks.full))
    expected = (((once >= 320) & trab) | ((once >= 450) & cort)) & full
    assert expected.any()
    np.testing.assert_array_equal(sitk_to_numpy_xyz(masks.seg) > 0, expected)


def test_generate_masks_fills_full_mask_holes_and_preserves_geometry() -> None:
    """A periosteal ring must become a solid geometry-preserving full mask."""
    image = _ring_image()
    params = _standard_outer_parameters()
    params.inner.contour_method = "none"

    masks = generate_masks_from_image(image, params)

    full = sitk_to_numpy_xyz(masks.full) > 0
    assert full[16, 16, 3]
    for mask in (masks.seg, masks.full, masks.trab, masks.cort):
        assert mask.GetSpacing() == image.GetSpacing()
        assert mask.GetOrigin() == image.GetOrigin()
        assert mask.GetDirection() == image.GetDirection()
    material = sitk.GetArrayFromImage(masks.material)
    assert set(np.unique(material)).issubset({0, 100, 127})
    assert np.count_nonzero(material == 100) > 0


def test_material_labelmap_encodes_trabecular_and_cortical_bone() -> None:
    """The FEA material labelmap should split segmented bone by trab/cort masks."""
    image = _large_ring_image()
    params = _standard_outer_parameters()
    params.inner.contour_method = "standard"
    params.inner.endosteal_threshold = 500.0
    params.inner.endosteal_kernel_size = 1
    params.inner.peel = 2

    masks = generate_masks_from_image(image, params)

    material = sitk.GetArrayFromImage(masks.material)
    assert np.count_nonzero(material == 127) > 0
    assert np.count_nonzero(material == 100) > 0


def test_standard_outer_contour_ignores_gaussian_tissue_support() -> None:
    """A tissue support threshold must not override the density envelope threshold."""
    values = np.zeros((21, 21, 3), dtype=np.float32)
    values[5:16, 5:16, :] = 310.0
    image = _image_from_xyz(values)
    params = ContourParameters()
    params.outer.contour_method = "standard"
    params.outer.use_adaptive_threshold = True
    params.outer.periosteal_threshold = 900.0
    params.outer.periosteal_kernel_size = 0
    params.outer.periosteal_open_radius = 0
    params.inner.contour_method = "none"
    params.segmentation.method = "gauss"
    params.segmentation.gaussian_sigma = 0.0
    params.segmentation.trab_threshold = 300.0
    params.segmentation.use_segmentation_aligned_contour_support = True

    masks = generate_masks_from_image(image, params)

    full = sitk_to_numpy_xyz(masks.full) > 0
    assert not full.any()
    assert masks.metadata["contour_support"]["support"] == "independent_gaussian_density"


def test_laplace_hamming_uses_native_tissue_image_not_for_contour_support(monkeypatch) -> None:
    """The density contour and native-unit tissue source remain independent."""
    from bone_contouring import _arrays

    density = np.zeros((11, 11, 3), dtype=np.float32)
    density[2:9, 2:9, :] = 900.0
    native = np.zeros_like(density)
    native[3:8, 3:8, :] = 20000.0
    image = _image_from_xyz(density)
    segmentation_image = _image_from_xyz(native)
    seen = {}

    def fake_lh(image_xyz, *, full_mask_xyz, spacing_xyz, parameters):
        seen["max"] = float(np.max(image_xyz))
        return image_xyz > 10000

    monkeypatch.setattr(_arrays, "laplace_hamming_binarize_xyz", fake_lh)
    params = ContourParameters()
    params.outer.contour_method = "standard"
    params.outer.periosteal_kernel_size = 0
    params.outer.periosteal_open_radius = 0
    params.inner.contour_method = "none"
    params.segmentation.method = "laplace_hamming"
    params.segmentation.use_segmentation_aligned_contour_support = True

    masks = generate_masks_from_image(image, params, segmentation_image=segmentation_image)

    full = sitk_to_numpy_xyz(masks.full) > 0
    assert seen["max"] == 20000.0
    assert full[5, 5, 1]
    assert sitk_to_numpy_xyz(masks.seg)[5, 5, 1]
    assert masks.metadata["contour_support"]["support"] == "independent_gaussian_density"


def test_laplace_hamming_tissue_filter_runs_once_after_contouring(monkeypatch) -> None:
    """Independent envelopes must not duplicate the native-unit tissue filter."""
    from bone_contouring import _arrays

    density = np.zeros((11, 11, 5), dtype=np.float32)
    density[2:9, 2:9, :] = 900.0
    native = np.zeros_like(density)
    native[3:8, 3:8, :] = 20000.0
    image = _image_from_xyz(density)
    segmentation_image = _image_from_xyz(native)
    calls = {"count": 0}

    def fake_lh(image_xyz, *, full_mask_xyz, spacing_xyz, parameters):
        calls["count"] += 1
        return image_xyz > 10000

    monkeypatch.setattr(_arrays, "laplace_hamming_binarize_xyz", fake_lh)
    params = ContourParameters()
    params.outer.contour_method = "standard"
    params.outer.periosteal_kernel_size = 0
    params.outer.periosteal_open_radius = 0
    params.outer.gaussian_sigma = 0
    params.stable_3d.outer_sigma_mm = (0, 0, 0)
    params.inner.contour_method = "none"
    params.segmentation.method = "laplace_hamming"
    params.segmentation.use_segmentation_aligned_contour_support = True

    masks = generate_masks_from_image(image, params, segmentation_image=segmentation_image)

    assert calls["count"] == 1
    assert np.array_equal(sitk_to_numpy_xyz(masks.seg) > 0, native > 10000)
    assert sitk_to_numpy_xyz(masks.full)[2, 2, 2]
    assert not sitk_to_numpy_xyz(masks.seg)[2, 2, 2]


def test_none_inner_contour_assigns_full_mask_to_trabecular_compartment() -> None:
    """The explicit `none` choice must not synthesize a cortical compartment."""
    params = _standard_outer_parameters()
    params.inner.contour_method = "none"

    masks = generate_masks_from_image(_large_ring_image(), params)

    assert np.array_equal(sitk_to_numpy_xyz(masks.trab), sitk_to_numpy_xyz(masks.full))
    assert not np.any(sitk_to_numpy_xyz(masks.cort))
    assert masks.metadata["endosteal_contour_method"] == "none"


def test_oversized_standard_peel_does_not_fall_back_to_full_trab() -> None:
    """An empty peeled ROI must retain cortex, not silently undo the peel."""
    params = _standard_outer_parameters()
    params.inner.contour_method = "standard"
    params.inner.peel = 100

    masks = generate_masks_from_image(_ring_image(), params)

    assert not np.any(sitk_to_numpy_xyz(masks.trab))
    assert np.array_equal(sitk_to_numpy_xyz(masks.cort), sitk_to_numpy_xyz(masks.full))
    assert masks.metadata["endosteal_fallback"]["applied"] is False
    params.inner.peel = 0
    without_peel = generate_masks_from_image(_large_ring_image(), params)
    assert np.any(sitk_to_numpy_xyz(without_peel.trab))


def test_generate_bone_segmentation_returns_a_geometry_preserving_mask() -> None:
    """The standalone segmentation entry point must retain the input image geometry."""
    values = np.zeros((9, 9, 5), dtype=np.float32)
    values[2:6, 2:6, 1:4] = 800.0
    image = _image_from_xyz(values)
    params = _standard_outer_parameters()

    segmentation = generate_bone_segmentation(image, params)

    assert sitk_to_numpy_xyz(segmentation)[3, 3, 2] == 1
    assert segmentation.GetSpacing() == image.GetSpacing()
    assert segmentation.GetOrigin() == image.GetOrigin()


def test_geodesic_outer_contour_uses_optional_adapter_output(monkeypatch) -> None:
    """The geodesic choice must use the optional package result as the full mask."""
    image = _ring_image()
    expected = np.zeros((33, 33, 7), dtype=bool)
    expected[8:25, 8:25, :] = True

    def contour(_density: np.ndarray, **_kwargs: object) -> tuple[np.ndarray, list[np.ndarray]]:
        return expected, [expected]

    monkeypatch.setitem(sys.modules, "hrpqct_geodesic_contour", types.SimpleNamespace(contour=contour))
    params = _standard_outer_parameters()
    params.outer.contour_method = "geodesic"
    params.inner.contour_method = "none"

    masks = generate_masks_from_image(image, params)

    assert np.array_equal(sitk_to_numpy_xyz(masks.full) > 0, expected)
    assert masks.metadata["periosteal_contour_method"] == "geodesic"
