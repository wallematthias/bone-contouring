"""Stable configuration types for contour and mask generation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class OuterContourParameters:
    """Controls periosteal (full-mask) contour generation."""

    contour_method: str = "standard"
    periosteal_threshold: float = 300.0
    periosteal_kernel_size: int = 5
    periosteal_open_radius: int = 2
    gaussian_sigma: float = 1.5
    use_adaptive_threshold: bool = True
    fill_holes: bool = True
    geodesic_bone_threshold: float = 250.0
    geodesic_fill_holes: bool = True


@dataclass(slots=True)
class InnerContourParameters:
    """Controls endosteal contour generation and compartment partitioning."""

    contour_method: str = "standard"
    site: str = "radius"
    endosteal_threshold: float = 500.0
    endosteal_kernel_size: int = 3
    gaussian_sigma: float = 1.5
    use_adaptive_threshold: bool = False
    peel: int = 3
    trabecular_close_radius: int | None = None


@dataclass(slots=True)
class SegmentationParameters:
    """Controls final bone segmentation within the full mask."""

    enabled: bool = True
    method: str = "gauss"
    contour_support_method: str = ""
    gaussian_sigma: float = 0.8
    trab_threshold: float = 320.0
    cort_threshold: float = 450.0
    adaptive_low_threshold: float = 190.0
    adaptive_high_threshold: float = 450.0
    adaptive_block_size: int = 13
    min_size_voxels: int = 64
    keep_largest_component: bool = True
    laplace_hamming_low_pass_cutoff: float = 0.3
    laplace_hamming_high_pass_cutoff: float = 0.0
    laplace_hamming_threshold: float = 15564.0
    laplace_hamming_epsilon: float = 0.45
    laplace_hamming_amplitude: float = 1.0
    laplace_hamming_amplification: float = 1.0
    laplace_hamming_input_offset: float = 0.0
    laplace_hamming_ipl_float_max: float = 200000.0
    laplace_hamming_int16_max: float = 32767.0
    laplace_hamming_min_size_voxels: int = 70
    laplace_hamming_backend: str = "cpu"
    use_segmentation_aligned_contour_support: bool = False


@dataclass(slots=True)
class ContourParameters:
    """Complete configuration for full, compartment, and bone masks."""

    modality: str = "xct1"
    site: str = "radius"
    outer: OuterContourParameters = field(default_factory=OuterContourParameters)
    inner: InnerContourParameters = field(default_factory=InnerContourParameters)
    segmentation: SegmentationParameters = field(default_factory=SegmentationParameters)
