"""Exact morphology and dispatch-budget regressions for published defaults."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("skimage")
from scipy import ndimage as ndi
from skimage.measure import label
from skimage.morphology import binary_dilation, binary_erosion


SOURCE = Path(__file__).resolve().parents[1] / "src/bone_contouring/unet/_postprocessing.py"
spec = importlib.util.spec_from_file_location("unet_postprocessing", SOURCE)
postprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(postprocessing)


def _reference_largest(mask, background=False):
    labels = label(~mask if background else mask, background=0)
    counts = np.bincount(labels.ravel())[1:]
    kept = labels == (counts.argmax() + 1) if counts.size else np.zeros_like(mask)
    return ~kept if background else kept


def _reference_islands(mask, n):
    result = np.pad(mask, ((1, 1), (1, 1), (0, 0)))
    for _ in range(n):
        result = binary_erosion(result)
    result = _reference_largest(result)
    for _ in range(n):
        result = binary_dilation(result)
    return result[1:-1, 1:-1, :]


def _reference_gaps(mask, n):
    pad = 2 * n if n else 1
    result = np.pad(mask, ((pad, pad), (pad, pad), (0, 0)))
    for _ in range(n):
        result = binary_dilation(result)
    result = _reference_largest(result, background=True)
    for _ in range(n):
        result = binary_erosion(result)
    return _reference_largest(result[pad:-pad, pad:-pad, :], background=True)


@pytest.mark.parametrize("n", [0, 1, 3, 8, 15])
@pytest.mark.parametrize("kind", ["random", "solid", "empty", "boundary", "strided"])
def test_batched_morphology_matches_repeated_steps_at_boundaries(n, kind):
    # Wrong connectivity, border values, or iterations=0-as-convergence fail here.
    rng = np.random.default_rng(710)
    mask = rng.random((37, 41, 7)) > .3
    if kind == "solid":
        mask[:] = True
    elif kind == "empty":
        mask[:] = False
    elif kind == "boundary":
        mask[:] = False
        mask[:29, :33, :] = True
        mask[10:15, 10:15, 2:5] = False
    elif kind == "strided":
        mask = mask.transpose(1, 0, 2)[:, ::-1, :]
    original = mask.copy()
    np.testing.assert_array_equal(postprocessing._islands(mask, n), _reference_islands(mask, n))
    np.testing.assert_array_equal(postprocessing._gaps(mask, n), _reference_gaps(mask, n))
    np.testing.assert_array_equal(mask, original)


def test_postprocessing_batches_repeated_scipy_morphology_dispatches(monkeypatch):
    # Reintroducing Python loops over full-volume single-step calls violates this
    # deterministic performance budget. Spies still execute the real morphology.
    calls = []
    for name in ("binary_dilation", "binary_erosion"):
        original = getattr(ndi, name)

        def observe(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(ndi, name, observe)
    x, y, z = np.ogrid[:64, :72, :9]
    trab = np.broadcast_to((x-31)**2 + (y-35)**2 < 17**2, (64, 72, 9)).copy()
    full = np.broadcast_to((x-31)**2 + (y-35)**2 < 24**2, trab.shape).copy()
    cort = full & ~trab
    result_cort, result_trab = postprocessing.postprocess(np.where(cort, .8, -.6), cort, trab)
    assert result_cort.any() and result_trab.any()
    assert not (result_cort & result_trab).any()
    assert len(calls) <= 49


def test_postprocessing_reports_stages_without_changing_masks():
    # Dropping progress forwarding or altering the masks when reporting fails.
    x, y, z = np.ogrid[:64, :72, :9]
    trab = np.broadcast_to((x-31)**2 + (y-35)**2 < 17**2, (64, 72, 9)).copy()
    full = np.broadcast_to((x-31)**2 + (y-35)**2 < 24**2, trab.shape).copy()
    cort = full & ~trab
    image = np.where(cort, .8, -.6)
    stages = []
    actual = postprocessing.postprocess(image, cort, trab, progress=stages.append)
    expected = postprocessing.postprocess(image, cort, trab)
    assert len(stages) == 4
    assert all(isinstance(stage, str) and stage for stage in stages)
    for result, want in zip(actual, expected):
        np.testing.assert_array_equal(result, want)


@pytest.mark.parametrize("slice_count", [1, 2, 3, 9, 31])
def test_open_scan_ends_preserve_compartments_in_uniform_phantom(slice_count):
    """Changing the Z border to exterior background must not strip scan ends."""
    x, y = np.ogrid[:72, :72]
    radius2 = (x - 35)**2 + (y - 35)**2
    trab = np.repeat((radius2 < 17**2)[:, :, None], slice_count, axis=2)
    full = np.repeat((radius2 < 25**2)[:, :, None], slice_count, axis=2)
    cort = full & ~trab
    result_cort, result_trab = postprocessing.postprocess(np.where(cort, .8, -.6), cort, trab)
    assert result_trab[35, 35, :].all()
    assert result_cort[35, 58, :].all()
    assert not (result_cort & result_trab).any()
    for mask in (result_cort, result_trab):
        for z in range(slice_count):
            np.testing.assert_array_equal(mask[:, :, z], mask[:, :, slice_count // 2])
