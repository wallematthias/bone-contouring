from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from bone_contouring import ContourParameters, SegmentationParameters, generate_bone_segmentation, resolve_preset
from bone_contouring._arrays import adaptive_threshold_xyz, segment_bone_xyz
from bone_contouring.laplace_hamming import LaplaceHammingParameters, laplace_hamming_binarize_xyz


@pytest.mark.parametrize("preset", [None, "xct1", "xct2"])
def test_default_gaussian_tissue_segmentation_matches_one_1_2_voxel_filter(preset):
    """Default tissue thresholds operate on raw density smoothed exactly once."""
    density = np.random.default_rng(42).uniform(0, 700, (9, 25, 25)).astype(np.float32)
    density[:, 5:10, 4:21] += 500
    image = sitk.GetImageFromArray(density)
    image.SetSpacing((.0607,) * 3)
    trab = np.zeros_like(density, dtype=np.uint8)
    trab[:, :, :13] = 1
    cort = 1 - trab
    masks = {}
    for role, array in (("full", trab | cort), ("trab", trab), ("cort", cort)):
        masks[role + "_mask"] = sitk.GetImageFromArray(array)
        masks[role + "_mask"].CopyInformation(image)
    parameters = ContourParameters() if preset is None else resolve_preset(modality=preset, segmentation="gauss")
    # Isolate smoothing/thresholds from the separately tested component cleanup.
    parameters.segmentation.min_size_voxels = 0
    parameters.segmentation.keep_largest_component = False
    smoothed = sitk.SmoothingRecursiveGaussian(image, 1.2 * .0607)
    once = sitk.GetArrayFromImage(smoothed)
    twice = sitk.GetArrayFromImage(sitk.SmoothingRecursiveGaussian(smoothed, 1.2 * .0607))
    def threshold(values):
        return ((values >= 320) & (trab > 0)) | ((values >= 450) & (cort > 0))

    expected = threshold(once)
    assert np.any(expected & (trab > 0)) and np.any(expected & (cort > 0))
    assert not np.array_equal(expected, threshold(twice)), "Fixture must detect double filtering"
    result = generate_bone_segmentation(image, parameters, **masks)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(result) > 0, expected)


def test_gaussian_segmentation_cleans_small_components_and_stays_in_full_mask() -> None:
    """A segmentation regression must not keep isolated noise or escape `full`."""
    image = np.zeros((9, 9, 9), dtype=np.float32)
    image[2:5, 2:5, 2:5] = 800.0
    image[7, 7, 7] = 800.0
    full = np.zeros_like(image, dtype=bool)
    full[1:6, 1:6, 1:6] = True
    params = SegmentationParameters(
        method="gauss",
        gaussian_sigma=0.0,
        trab_threshold=500.0,
        cort_threshold=500.0,
        min_size_voxels=4,
    )

    result = segment_bone_xyz(image, full, full, full, params, spacing_xyz=(1.0, 1.0, 1.0))

    assert result[2:5, 2:5, 2:5].all()
    assert not result[7, 7, 7]
    assert not np.any(result & ~full)


@pytest.mark.parametrize("method", ["gauss", "adaptive"])
@pytest.mark.parametrize("legacy_keep_largest", [False, True])
def test_tissue_segmentation_keeps_disconnected_bone_even_with_legacy_largest_flag(method, legacy_keep_largest):
    """SEG is not an FEA connectivity-filtered mask; retain both valid islands."""
    image = np.zeros((25, 25, 9), dtype=np.float32)
    image[2:8, 2:8, 2:7] = 900
    image[16:21, 16:21, 2:7] = 900
    image[12, 12, 4] = 900
    full = np.ones_like(image, dtype=bool)
    params = SegmentationParameters(method=method, gaussian_sigma=0, min_size_voxels=64,
                                    keep_largest_component=legacy_keep_largest)
    result = segment_bone_xyz(image, full, full, np.zeros_like(full), params)
    assert result[4, 4, 4] and result[18, 18, 4]
    assert not result[12, 12, 4]  # Small-noise removal is a separate, retained policy.
    labels = sitk.RelabelComponent(sitk.ConnectedComponent(sitk.GetImageFromArray(result.astype(np.uint8))))
    assert int(sitk.GetArrayViewFromImage(labels).max()) == 2


def test_laplace_hamming_binarization_respects_full_mask_and_component_limit() -> None:
    """Laplace-Hamming output must be constrained even when bright voxels exist outside full."""
    image = np.zeros((8, 8, 8), dtype=np.float32)
    image[2:4, 2:4, 2:4] = 900.0
    image[6, 6, 6] = 900.0
    full = np.zeros_like(image, dtype=bool)
    full[1:5, 1:5, 1:5] = True
    params = LaplaceHammingParameters(
        low_pass_cutoff=1.0,
        laplace_epsilon=0.0,
        hamming_amplitude=0.0,
        ipl_float_max=10000.0,
        int16_max=10000.0,
        threshold=500.0,
        min_size_voxels=2,
    )

    result = laplace_hamming_binarize_xyz(
        image,
        full_mask_xyz=full,
        spacing_xyz=(1.0, 1.0, 1.0),
        parameters=params,
    )

    assert result[2:4, 2:4, 2:4].all()
    assert not result[6, 6, 6]


def test_adaptive_threshold_rejects_even_window_sizes() -> None:
    """An even local window has no center voxel and must fail explicitly."""
    with pytest.raises(ValueError, match="odd"):
        adaptive_threshold_xyz(
            np.zeros((5, 5, 5), dtype=np.float32),
            block_size=4,
        )
