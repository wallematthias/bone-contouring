from __future__ import annotations

from importlib.metadata import version

import bone_contouring
import numpy as np
import pytest
import SimpleITK as sitk
from bone_contouring import (
    ContourParameters,
    InnerContourParameters,
    OuterContourParameters,
    SegmentationParameters,
    load_preset,
    resolve_preset,
)


def test_scene_unet_recipe_cannot_silently_run_as_standard_contouring():
    from bone_contouring.presets import _params_from_user_profile
    with pytest.raises(ValueError, match="U-Net"):
        _params_from_user_profile({"schema": "bone-contour-recipe-v1", "contouring_method": "unet",
                                  "modality": "xct2", "site": "radius"})


def test_custom_manual_recipe_preserves_units_and_no_site():
    from bone_contouring.presets import _params_from_user_profile

    params = _params_from_user_profile({
        "schema": "bone-contour-recipe-v1", "modality": "custom", "site": "none",
        "methods": {"bone_segmentation": "seg_gauss", "periosteal_contour": "standard",
                    "endosteal_contour": "standard"},
        "parameters": {"outer": {"periosteal_threshold": 1200},
                       "inner": {"endosteal_threshold": 1800},
                       "segmentation": {"trab_threshold": 1500, "cort_threshold": 2000}},
    })
    assert params.modality == "custom" and params.site == "none"
    assert params.outer.periosteal_threshold == 1200
    assert params.inner.endosteal_threshold == 1800
    assert params.segmentation.trab_threshold == 1500
    assert params.segmentation.cort_threshold == 2000


def test_custom_does_not_apply_xct2_radius_calibration():
    params = resolve_preset(modality="custom", site="radius", segmentation="gauss")
    assert params.outer.periosteal_threshold == OuterContourParameters().periosteal_threshold
    assert params.inner.endosteal_threshold == InnerContourParameters().endosteal_threshold


def test_named_user_microct_profile_keeps_custom_settings_in_batch(tmp_path):
    from bone_imaging_derivatives import save_json_profile

    save_json_profile("bone-contouring", "My Micro-CT", {
        "schema": "bone-contour-recipe-v1", "modality": "custom", "site": "none",
        "methods": {"bone_segmentation": "seg_gauss", "periosteal_contour": "standard",
                    "endosteal_contour": "standard"},
        "parameters": {"inner": {"endosteal_threshold": 1800},
                       "segmentation": {"trab_threshold": 1500}},
    }, root=tmp_path)
    params = load_preset("My Micro-CT", site="tibia", profile_root=tmp_path)
    assert params.modality == "custom"
    assert params.inner.endosteal_threshold == 1800
    assert params.segmentation.trab_threshold == 1500


def test_root_api_exports_parameter_types_and_preset_helpers() -> None:
    """A missing root export would make the advertised stable API unusable."""
    assert ContourParameters is not None
    assert SegmentationParameters is not None
    assert OuterContourParameters is not None
    assert InnerContourParameters is not None
    assert callable(load_preset)
    assert callable(resolve_preset)


def test_root_api_exports_installed_package_version() -> None:
    """The import-level version should match the installed package metadata."""
    assert bone_contouring.__version__ == version("bone-contouring")


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

    assert params.outer.periosteal_threshold == 250.0
    assert params.outer.periosteal_kernel_size == 12
    assert params.outer.periosteal_open_radius == 1
    assert params.outer.use_adaptive_threshold is False
    assert params.segmentation.laplace_hamming_threshold == 15000.0


@pytest.mark.parametrize("site,density", [("radius", 275.0), ("tibia", 275.0), ("knee", 175.0)])
@pytest.mark.parametrize("profile", ["resolved", "XtremeCTI", "encoded"])
def test_xct1_standard_presets_retain_low_density_outer_shell(site, density, profile):
    """Scene/API and both batch preset routes must not discard lower-density cortex."""
    if profile == "resolved":
        params = resolve_preset(modality="xct1", site=site)
    elif profile == "encoded":
        params = load_preset(f"xct1-{site}-laplace_hamming-standard-standard")
    else:
        params = load_preset(profile, site=site)
    image_array = np.zeros((7, 41, 41), dtype=np.float32)
    image_array[:, 8:33, 8:33] = density
    image_array[:, 9:32, 9:32] = 0
    params.outer.gaussian_sigma = 0
    params.outer.periosteal_kernel_size = 0
    params.stable_3d.outer_sigma_mm = (0, 0, 0)
    params.inner.contour_method = "none"
    params.segmentation.enabled = False
    masks = bone_contouring.generate_masks_from_image(sitk.GetImageFromArray(image_array), params)
    expected = np.zeros_like(image_array, dtype=bool)
    expected[:, 8:33, 8:33] = True
    assert np.array_equal(sitk.GetArrayFromImage(masks.full).astype(bool), expected)


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


def test_named_xtremect_profiles_encode_scanner_defaults() -> None:
    """Named profiles should expose the stable scanner recipes used by Slicer batch."""
    xct1 = load_preset("XtremeCTI", site="radius")
    xct2 = load_preset("XtremeCTII", site="tibia")
    xct2_geodesic = load_preset("XtremeCTII-Geodesic", site="radius")
    xct2_lh = load_preset("XtremeCTII-LH", site="radius")

    assert xct1.modality == "xct1"
    assert xct1.segmentation.method == "laplace_hamming"
    assert xct1.segmentation.contour_support_method == ""
    assert xct1.inner.peel == 6

    assert xct2.modality == "xct2"
    assert xct2.segmentation.method == "gauss"
    assert xct2.segmentation.contour_support_method == ""
    assert xct2.outer.contour_method == "standard"
    assert xct2.inner.peel == 6
    assert xct2.segmentation.gaussian_sigma == .8
    assert xct2.segmentation.gaussian_support == 1

    assert xct2_geodesic.modality == "xct2"
    assert xct2_geodesic.segmentation.method == "gauss"
    assert xct2_geodesic.outer.contour_method == "geodesic"
    assert xct2_geodesic.inner.contour_method == "standard"

    assert xct2_lh.modality == "xct2"
    assert xct2_lh.segmentation.method == "laplace_hamming"
    assert xct2_lh.segmentation.contour_support_method == "gauss"
    assert xct2_lh.inner.peel == 6


def test_finite_filter_support_is_part_of_batch_provenance():
    from bone_contouring.batch import _parameters_payload, _settings_hash
    params = load_preset('XtremeCTII')
    original_hash = _settings_hash(params)
    assert _parameters_payload(params)['segmentation']['gaussian_support'] == 1
    params.segmentation.gaussian_support = 2
    assert _settings_hash(params) != original_hash


def test_load_preset_can_resolve_user_saved_profile_from_shared_registry(tmp_path) -> None:
    """Custom Slicer contour profiles should be loadable by name through the shared registry."""
    from bone_imaging_derivatives import save_json_profile

    save_json_profile(
        "bone-contouring",
        "Lab XCT2 LH",
        {
            "schema": "bone-contour-recipe-v1",
            "modality": "xct2",
            "site": "radius",
            "methods": {
                "bone_segmentation": "laplace_hamming",
                "periosteal_contour": "standard",
                "endosteal_contour": "standard",
            },
            "parameters": {
                "segmentation": {
                    "contour_support_method": "gauss",
                    "laplace_hamming_threshold": 14900.0,
                },
                "outer": {
                    "periosteal_threshold": 275.0,
                },
            },
        },
        root=tmp_path,
    )

    params = load_preset("Lab XCT2 LH", profile_root=tmp_path)

    assert params.modality == "xct2"
    assert params.site == "radius"
    assert params.segmentation.method == "laplace_hamming"
    assert params.segmentation.contour_support_method == "gauss"
    assert params.segmentation.laplace_hamming_threshold == 14900.0
    assert params.outer.periosteal_threshold == 275.0
