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
    gaussian_sigma: float = 2.0
    use_adaptive_threshold: bool = False
    peel: int = 6  # Minimum cortical compartment rim: axial XY erosion radius.
    trabecular_close_radius: int | None = None


@dataclass(slots=True)
class SegmentationParameters:
    """Controls final bone segmentation within the full mask."""

    enabled: bool = True
    method: str = "gauss"
    contour_support_method: str = ""
    gaussian_sigma: float = 0.8
    gaussian_support: int = 1  # Finite kernel radius in voxels, not a contour prefilter.
    trab_threshold: float = 320.0
    cort_threshold: float = 450.0
    adaptive_low_threshold: float = 190.0
    adaptive_high_threshold: float = 450.0
    adaptive_block_size: int = 13
    min_size_voxels: int = 64
    # Legacy profile/API compatibility only; tissue SEG ignores this field.
    # Largest-component selection belongs to downstream FEA preparation.
    keep_largest_component: bool = False
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
class BuieParameters:
    """Buie Fig. 1 kernels in voxels (dimensions, NOT radii).

    Density thresholds remain in ``outer`` and ``inner`` and must use the
    input image's calibrated units. Gaussian support is interpreted as VTK
    radius factors, giving radii (9, 9, 3) at sigma 3. No IPL equivalence is
    implied by these defaults.
    """

    median_kernel_size: tuple[int, int, int] = (3, 3, 1)
    periosteal_kernel_size: tuple[int, int, int] = (15, 15, 1)
    endosteal_kernel_size: tuple[int, int, int] = (10, 10, 1)
    gaussian_sigma: float = 3.0
    gaussian_radius_factors: tuple[float, float, float] = (3.0, 3.0, 1.0)
    final_threshold: float = 100.0
    fully_connected: bool = False


@dataclass(slots=True)
class Stable3DParameters:
    """Physical-unit boundary regularization and advisory quality controls.

    The standard outer stage uses the outer settings and advisory controls.
    Inner regularization belongs only to the explicit ``stable_3d`` method,
    not the shared IPL-style standard. The boundary-change limit restricts
    smoothing, not the preceding contour repair.
    """

    outer_sigma_mm: tuple[float, float, float] = (0.03, 0.03, 0.06)
    inner_sigma_mm: tuple[float, float, float] = (0.03, 0.03, 0.06)
    max_boundary_shift_mm: float = 0.12
    area_jump_fraction: float = 0.25
    adjacent_boundary_limit_mm: float = 0.3


@dataclass(slots=True)
class ContourParameters:
    """Complete configuration for full, compartment, and bone masks."""

    modality: str = "xct1"
    site: str = "radius"
    outer: OuterContourParameters = field(default_factory=OuterContourParameters)
    inner: InnerContourParameters = field(default_factory=InnerContourParameters)
    segmentation: SegmentationParameters = field(default_factory=SegmentationParameters)
    buie: BuieParameters = field(default_factory=BuieParameters)
    stable_3d: Stable3DParameters = field(default_factory=Stable3DParameters)


STANDARD_ALGORITHM_REVISION = "shared_ipl_standard_v1"
