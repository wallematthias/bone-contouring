from __future__ import annotations

from bone_contouring import (
    ContourParameters,
    InnerContourParameters,
    OuterContourParameters,
    SegmentationParameters,
    load_preset,
    resolve_preset,
)


def test_root_api_exports_parameter_types_and_preset_helpers() -> None:
    """A missing root export would make the advertised stable API unusable."""
    assert ContourParameters is not None
    assert SegmentationParameters is not None
    assert OuterContourParameters is not None
    assert InnerContourParameters is not None
    assert callable(load_preset)
    assert callable(resolve_preset)


def test_resolve_preset_composes_each_requested_dimension() -> None:
    """Independent preset dimensions must not silently override one another."""
    params = resolve_preset(
        modality="xct1",
        site="radius",
        segmentation="laplace_hamming",
        outer_contour="geodesic",
        inner_contour="none",
    )

    assert params.modality == "xct1"
    assert params.site == "radius"
    assert params.segmentation.method == "laplace_hamming"
    assert params.outer.contour_method == "geodesic"
    assert params.inner.contour_method == "none"
    assert params.inner.site == "radius"
    assert params.segmentation.use_segmentation_aligned_contour_support is True


def test_xct1_preset_uses_standard_periosteal_contour_defaults() -> None:
    """XtremeCT I full-mask contouring uses the scanner-specific kernel/open defaults."""
    params = resolve_preset(
        modality="xct1",
        site="radius",
        segmentation="laplace_hamming",
        outer_contour="standard",
        inner_contour="standard",
    )

    assert params.outer.periosteal_threshold == 300.0
    assert params.outer.periosteal_kernel_size == 12
    assert params.outer.periosteal_open_radius == 1
    assert params.outer.use_adaptive_threshold is False
    assert params.segmentation.laplace_hamming_threshold == 15000.0


def test_resolved_presets_are_independent_instances() -> None:
    """A caller's parameter edit must not contaminate later preset resolution."""
    first = load_preset("xct2-tibia-gauss-standard-standard")
    first.outer.periosteal_threshold = 999.0
    second = load_preset("xct2-tibia-gauss-standard-standard")

    assert first.segmentation.method == "gauss"
    assert second.outer.periosteal_threshold != 999.0


def test_xct1_knee_keeps_default_laplace_hamming_threshold() -> None:
    """The more inclusive XCT1 LH threshold is scoped to radius/tibia."""
    params = resolve_preset(
        modality="xct1",
        site="knee",
        segmentation="laplace_hamming",
        outer_contour="standard",
        inner_contour="standard",
    )

    assert params.segmentation.laplace_hamming_threshold == 15564.0
