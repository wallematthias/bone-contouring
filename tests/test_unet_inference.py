"""Regression checks for the published runtime, independent of legacy VTK I/O."""
import ast
import sys
from pathlib import Path

import numpy as np
import pytest
torch = pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests/fixtures"))


def test_preprocessing_matches_original_and_preserves_orientation():
    from bone_contouring.unet.inference import prepare_volume
    from unet_legacy.SamplePadder import SamplePadder
    from unet_legacy.SampleStandardizer import SampleStandardizer
    density = np.arange(3 * 7 * 11, dtype=float).reshape(3, 7, 11) * 10 - 600
    original = SampleStandardizer(-400, 1400)(SamplePadder(8)({
        "image": density.transpose(2, 1, 0), "image_position": [4, 8, 12],
    }))["image"]
    prepared, crop = prepare_volume(density)
    np.testing.assert_array_equal(prepared, original)
    np.testing.assert_array_equal(prepared[crop].transpose(2, 1, 0),
                                  np.clip((2 * density - 1000) / 1800, -1, 1))


@pytest.mark.parametrize("z,want", [(0, [0,0,0,2,2]), (2, [0,1,2,4,4]), (4, [2,3,4,4,4])])
def test_slice_stack_matches_published_legacy_channels(z, want):
    from bone_contouring.unet.inference import slice_indices
    assert slice_indices(z, 5) == want


def test_device_explicit_cuda_never_silently_falls_back(monkeypatch):
    from bone_contouring.unet.inference import resolve_device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert str(resolve_device("auto")) == "cpu"
    assert str(resolve_device("cpu")) == "cpu"
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")
    with pytest.raises(ValueError, match="device"):
        resolve_device("mystery")


def test_auto_prefers_cuda_then_mps_then_cpu(monkeypatch):
    from bone_contouring.unet.inference import resolve_device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert str(resolve_device("auto")) == "mps"
    assert str(resolve_device("mps")) == "mps"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert str(resolve_device("auto")) == "cuda"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS"):
        resolve_device("mps")


def test_morphology_matches_original_default_pipeline():
    from bone_contouring.unet._postprocessing import postprocess
    from skimage.morphology import binary_dilation, binary_erosion
    from skimage.measure import label
    from skimage.filters import median
    # Execute the actual legacy scientific functions, without its unused VTK
    # and plotting imports. This oracle does not call our implementation.
    tree = ast.parse((ROOT / "tests/fixtures/unet_legacy/postprocessing.py").read_text())
    names = {"keep_largest_connected_component_skimage", "remove_islands_from_mask",
             "fill_in_gaps_in_mask", "dilate_and_subtract", "extract_bone",
             "iterative_filter", "postprocess_masks_iterative"}
    namespace = dict(np=np, binary_dilation=binary_dilation,
                     binary_erosion=binary_erosion, sklabel=label, median=median)
    exec(compile(ast.Module([n for n in tree.body if isinstance(n, ast.FunctionDef)
                            and n.name in names], type_ignores=[]), "legacy", "exec"), namespace)
    x, y, z = np.indices((64, 72, 9))
    trab = (x-31)**2 + (y-35)**2 < 17**2
    full = (x-31)**2 + (y-35)**2 < 24**2
    trab[31,35,4] = False
    cort = full & ~trab
    density = np.where(cort, 0.8, -0.6)
    expected = namespace["postprocess_masks_iterative"](density, cort, trab)
    actual = postprocess(density, cort, trab)
    for result, want in zip(actual, expected):
        np.testing.assert_array_equal(result, want)


def test_corrupt_weight_override_is_rejected(tmp_path, monkeypatch):
    from bone_contouring.unet.weights import weights_path
    weights = tmp_path / "bad.pth"
    weights.write_bytes(b"not published weights")
    monkeypatch.setenv("HRPQCT_SEGMENTATION_WEIGHTS", str(weights))
    with pytest.raises(ValueError, match="checksum"):
        weights_path()


def test_shared_model_directory_reuses_verified_weights(tmp_path, monkeypatch):
    from bone_contouring.unet import weights
    import hashlib
    model = tmp_path / "models" / "radius_tibia_final.pth"
    model.parent.mkdir()
    model.write_bytes(b"small checksum fixture")
    monkeypatch.delenv("HRPQCT_SEGMENTATION_WEIGHTS", raising=False)
    monkeypatch.setenv("HRPQCT_SEGMENTATION_MODEL_DIR", str(model.parent))
    monkeypatch.setattr(weights, "WEIGHTS_SHA256", hashlib.sha256(model.read_bytes()).hexdigest())
    def no_network(*args, **kwargs):
        raise AssertionError("Verified cached weights must not access network")
    monkeypatch.setattr(weights, "urlopen", no_network)
    assert weights.weights_path() == model


def test_bad_density_is_rejected_before_model_loading():
    from bone_contouring.unet.inference import prepare_volume
    for invalid in (np.zeros((3, 4)), np.zeros((0,4,5)), np.full((2,3,4), np.nan)):
        with pytest.raises(ValueError):
            prepare_volume(invalid)


def test_model_architecture_accepts_original_state_dict():
    from bone_contouring.unet._model import UNet
    from unet_legacy.UNet import UNet as LegacyUNet
    legacy = LegacyUNet(5, 2, [32,64,128,256], 16, .1).eval()
    model = UNet().eval()
    model.load_state_dict(legacy.state_dict(), strict=True)
    data = torch.arange(5*16*24, dtype=torch.float32).reshape(1,5,16,24)/1000
    with torch.inference_mode():
        torch.testing.assert_close(model(data), legacy(data), rtol=0, atol=0)


def test_numpy_abi_fallback_preserves_tensor_values(monkeypatch):
    from bone_contouring.unet.inference import _TensorBridge
    def incompatible(*args):
        raise RuntimeError("Numpy is not available")
    monkeypatch.setattr(torch, "from_numpy", incompatible)
    bridge = _TensorBridge()
    values = np.linspace(-1, 1, 35, dtype=np.float32).reshape(5, 7)
    tensor = bridge.input(values)
    np.testing.assert_array_equal(tensor.tolist(), values)
    np.testing.assert_array_equal(bridge.mask(tensor < 0), values < 0)


def test_segmenter_forwards_postprocessing_stages_to_scene_and_batch():
    from bone_contouring.unet.inference import Segmenter, _TensorBridge

    class AnalyticCompartments(torch.nn.Module):
        # Replace only the expensive learned predictor; execute tensor transfer,
        # masking, morphology, cropping, and progress forwarding for real.
        def forward(self, channels):
            x = torch.arange(channels.shape[-2], device=channels.device)[:, None]
            y = torch.arange(channels.shape[-1], device=channels.device)[None, :]
            radius2 = (x-31.5)**2 + (y-31.5)**2
            return torch.stack((radius2-26**2, radius2-18**2))[None]

    segmenter = Segmenter.__new__(Segmenter)
    segmenter.device = torch.device("cpu")
    segmenter._bridge = _TensorBridge()
    segmenter.model = AnalyticCompartments()
    messages = []
    masks = segmenter.segment(np.full((3, 64, 64), 300.), progress=messages.append)
    assert masks["full"].shape == (3, 64, 64)
    assert masks["trab"].any() and masks["cort"].any()
    assert not (masks["trab"] & masks["cort"]).any()
    for stage in ("1/4", "2/4", "3/4", "4/4"):
        assert any(stage in message for message in messages)
    assert any("Post-processing completed" in message for message in messages)
