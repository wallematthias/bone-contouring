from __future__ import annotations

import numpy as np
import pytest

from bone_contouring import SegmentationParameters
from bone_contouring._arrays import adaptive_threshold_xyz, segment_bone_xyz
from bone_contouring.laplace_hamming import LaplaceHammingParameters, laplace_hamming_binarize_xyz


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
