"""Laplace-Hamming HR-pQCT segmentation on x/y/z NumPy arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk


@dataclass(slots=True)
class LaplaceHammingParameters:
    """Parameters for the IPL-style Laplace-Hamming binarization."""

    low_pass_cutoff: float = 0.3
    high_pass_cutoff: float = 0.0
    laplace_epsilon: float = 0.45
    hamming_amplitude: float = 1.0
    amplification: float = 1.0
    input_offset: float = 0.0
    ipl_float_max: float = 200000.0
    int16_max: float = 32767.0
    threshold: float = 15564.0
    min_size_voxels: int = 70
    backend: str = "cpu"


def _remove_small_components_6(binary: np.ndarray, min_size_voxels: int) -> np.ndarray:
    if min_size_voxels <= 0 or not np.any(binary):
        return np.ascontiguousarray(binary, dtype=bool)
    image = sitk.GetImageFromArray(np.transpose(np.asarray(binary, dtype=np.uint8), (2, 1, 0)))
    labels = sitk.ConnectedComponent(image, False)
    retained = sitk.RelabelComponent(labels, minimumObjectSize=int(min_size_voxels), sortByObjectSize=False)
    array = sitk.GetArrayFromImage(retained > 0)
    return np.ascontiguousarray(np.transpose(array.astype(bool), (2, 1, 0)))


def _mirror_pad_to_power_of_two(array: np.ndarray) -> tuple[np.ndarray, tuple[slice, slice, slice]]:
    pad_widths: list[tuple[int, int]] = []
    slices: list[slice] = []
    for size in array.shape:
        target = 1 if size <= 1 else 2 ** int(np.ceil(np.log2(size)))
        total = target - size
        lower = total // 2
        pad_widths.append((lower, total - lower))
        slices.append(slice(lower, lower + size))
    return np.pad(array, pad_widths, mode="reflect"), tuple(slices)  # type: ignore[return-value]


def laplace_hamming_filter_xyz(
    image_xyz: np.ndarray,
    *,
    spacing_xyz: tuple[float, float, float] | None = None,
    parameters: LaplaceHammingParameters | None = None,
) -> np.ndarray:
    """Apply the Laplace-Hamming frequency-domain filter to an x/y/z image."""
    p = parameters or LaplaceHammingParameters()
    backend = p.backend.strip().lower()
    if backend not in {"cpu", "auto"}:
        raise RuntimeError("Only the CPU Laplace-Hamming backend is available in bone-contouring.")

    pixels = np.asarray(image_xyz, dtype=np.float64) + float(p.input_offset)
    if pixels.ndim != 3:
        raise ValueError(f"Laplace-Hamming expects a 3D array, got ndim={pixels.ndim}.")
    spacing = np.asarray(spacing_xyz or (0.0607, 0.0607, 0.0607), dtype=np.float64)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("spacing_xyz must contain three positive values.")
    nyquist_min = 1.0 / (2.0 * float(np.min(spacing)))
    low_pass_frequency = float(p.low_pass_cutoff) * 2.0 * nyquist_min
    high_pass_frequency = float(p.high_pass_cutoff) * 2.0 * nyquist_min
    if low_pass_frequency <= 0:
        raise ValueError("Laplace-Hamming low_pass_cutoff must be positive.")

    axes = [np.fft.fftfreq(size, d=float(spacing[index])) for index, size in enumerate(pixels.shape)]
    kx, ky, kz = np.meshgrid(*axes, indexing="ij")
    frequency_squared = kx * kx + ky * ky + kz * kz
    frequency = np.sqrt(frequency_squared)
    in_band = (frequency < low_pass_frequency) & (frequency >= high_pass_frequency)
    half_amplitude = float(p.hamming_amplitude) * 0.5
    window = np.where(
        in_band,
        (1.0 - half_amplitude) + half_amplitude * np.cos(np.pi * frequency / low_pass_frequency),
        0.0,
    )
    kernel = (
        float(p.amplification)
        * ((2.0 * np.pi) ** 2)
        * ((1.0 - float(p.laplace_epsilon)) + float(p.laplace_epsilon) * frequency_squared)
        * window
    )
    return np.real(np.fft.ifftn(np.fft.fftn(pixels) * kernel))


def laplace_hamming_binarize_xyz(
    image_xyz: np.ndarray,
    *,
    full_mask_xyz: np.ndarray | None = None,
    spacing_xyz: tuple[float, float, float] | None = None,
    parameters: LaplaceHammingParameters | None = None,
) -> np.ndarray:
    """Return a component-cleaned Laplace-Hamming bone mask in x/y/z order."""
    p = parameters or LaplaceHammingParameters()
    original = np.asarray(image_xyz)
    if original.ndim != 3:
        raise ValueError(f"Laplace-Hamming expects a 3D array, got ndim={original.ndim}.")
    extended = np.pad(original, ((1, 1), (1, 1), (1, 1)), mode="edge")
    padded, original_slices = _mirror_pad_to_power_of_two(extended)
    filtered = laplace_hamming_filter_xyz(padded, spacing_xyz=spacing_xyz, parameters=p)
    scaled = np.rint(
        np.clip(
            filtered * (float(p.int16_max) / float(p.ipl_float_max)),
            -float(p.int16_max),
            float(p.int16_max),
        )
    ).astype(np.int16)
    binary = (scaled >= float(p.threshold)) & (scaled <= float(p.int16_max))
    binary = binary[original_slices][1:-1, 1:-1, 1:-1]
    if full_mask_xyz is not None:
        full_mask = np.asarray(full_mask_xyz, dtype=bool)
        if full_mask.shape != binary.shape:
            raise ValueError("full_mask_xyz shape must match image_xyz shape.")
        binary &= full_mask
    return _remove_small_components_6(binary, int(p.min_size_voxels))
