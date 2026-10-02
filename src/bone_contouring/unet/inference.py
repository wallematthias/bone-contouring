"""Fixed-default BMD arrays -> compartment masks using the published model."""
import os

import numpy as np
import torch

from ._model import UNet
from ._postprocessing import postprocess
from .weights import weights_path


def resolve_device(device="auto"):
    if device not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("device must be auto, cpu, cuda, or mps")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no compatible CUDA device is available.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but this Python/PyTorch runtime has no available MPS device.")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(device)


def prepare_volume(density_zyx):
    density = np.asarray(density_zyx)
    if density.ndim != 3 or not all(density.shape) or not np.issubdtype(density.dtype, np.number):
        raise ValueError("Expected a nonempty 3D BMD array in ZYX order.")
    if not np.isfinite(density).all():
        raise ValueError("BMD voxels must be finite.")
    xyz = density.transpose(2,1,0).astype(np.float64, copy=False)
    side = ((max(xyz.shape[:2]) + 7) // 8) * 8
    xpad, ypad = side - xyz.shape[0], side - xyz.shape[1]
    padded = np.pad(xyz, ((xpad//2, xpad-xpad//2),(ypad//2, ypad-ypad//2),(0,0)), mode="edge")
    padded = (2 * np.clip(padded, -400, 1400) - 1000) / 1800
    crop = (slice(xpad//2, xpad//2+xyz.shape[0]),
            slice(ypad//2, ypad//2+xyz.shape[1]), slice(None))
    return padded, crop


def slice_indices(z, count):
    # Deliberately preserve SingleImageDataset's duplicated upper slice. Changing
    # to a symmetric stack is a scientific/model change, not dependency removal.
    return [max(z-2,0), max(z-1,0), z, min(z+2,count-1), min(z+2,count-1)]


class _TensorBridge:
    """Support Intel macOS PyTorch 2.2 with NumPy 2 without a global downgrade.

    Input is a writable contiguous float32 buffer; output is already boolean.
    The fallback changes only array transport, not inference or thresholds.
    """
    def __init__(self):
        try:
            torch.from_numpy(np.zeros(1, dtype=np.float32)).numpy()
            self.numpy_available = True
        except RuntimeError:
            self.numpy_available = False

    def input(self, array):
        if self.numpy_available:
            return torch.from_numpy(array)
        return torch.frombuffer(array, dtype=torch.float32).reshape(array.shape)

    def mask(self, tensor):
        return tensor.numpy() if self.numpy_available else np.asarray(tensor.tolist(), dtype=bool)


class Segmenter:
    """Load the published model once, reuse it for one or many BMD volumes."""
    def __init__(self, device="auto"):
        self.device = resolve_device(device)
        self._bridge = _TensorBridge()
        if self.device.type == "cpu":
            torch.set_num_threads(min(8, os.cpu_count() or 1))
        self.model = UNet()
        self.model.load_state_dict(torch.load(weights_path(), map_location="cpu", weights_only=True), strict=True)
        self.model.to(self.device).eval()

    def segment(self, density_zyx, progress=None):
        image, crop = prepare_volume(density_zyx)
        cort = np.zeros(image.shape, dtype=bool)
        trab = np.zeros(image.shape, dtype=bool)
        report = progress or (lambda message: None)
        report(f"Device: {self.device}; predicting {image.shape[2]} slices")
        with torch.inference_mode():
            for z in range(image.shape[2]):
                channels = np.ascontiguousarray(image[:,:,slice_indices(z,image.shape[2])].transpose(2,0,1), dtype=np.float32)
                phi = self.model(self._bridge.input(channels).unsqueeze(0).to(self.device))[0].cpu()
                cort[:,:,z] = self._bridge.mask((phi[0] < 0) & (phi[1] > 0))
                trab[:,:,z] = self._bridge.mask(phi[1] < 0)
                report(f"Slices {z+1}/{image.shape[2]}")
        report("Post-processing with published defaults...")
        cort, trab = postprocess(image, cort, trab)
        masks = {"cort": cort[crop], "trab": trab[crop]}
        for role, mask in masks.items():
            if not np.any(mask):
                raise ValueError(f"Model produced an empty {role} compartment. Check scan calibration and model applicability.")
        masks["full"] = masks["cort"] | masks["trab"]
        return {role: np.ascontiguousarray(mask.transpose(2,1,0), dtype=np.uint8)
                for role, mask in masks.items()}
