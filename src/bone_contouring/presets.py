"""Composable presets for supported bone contouring choices."""

from __future__ import annotations

from .parameters import ContourParameters

_MODALITIES = {"xct1", "xct2"}
_SITES = {"radius", "tibia", "knee"}
_SEGMENTATION_METHODS = {"laplace_hamming", "gauss", "adaptive"}
_OUTER_CONTOUR_METHODS = {"standard", "geodesic"}
_INNER_CONTOUR_METHODS = {"standard", "none"}


def _choice(value: str, allowed: set[str], label: str) -> str:
    normalized = value.strip().lower()
    if normalized not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"Unsupported {label} {value!r}; choose one of: {choices}.")
    return normalized


def resolve_preset(
    *,
    modality: str = "xct1",
    site: str = "radius",
    segmentation: str = "laplace_hamming",
    outer_contour: str = "standard",
    inner_contour: str = "standard",
) -> ContourParameters:
    """Compose a fresh parameter object from supported method dimensions."""
    modality = _choice(modality, _MODALITIES, "modality")
    site = _choice(site, _SITES, "site")
    segmentation = _choice(segmentation, _SEGMENTATION_METHODS, "segmentation")
    outer_contour = _choice(outer_contour, _OUTER_CONTOUR_METHODS, "outer contour")
    inner_contour = _choice(inner_contour, _INNER_CONTOUR_METHODS, "inner contour")

    params = ContourParameters(modality=modality, site=site)
    params.inner.site = site
    params.segmentation.method = segmentation
    params.segmentation.use_segmentation_aligned_contour_support = True
    params.outer.contour_method = outer_contour
    params.outer.use_adaptive_threshold = False
    params.inner.contour_method = inner_contour
    params.inner.use_adaptive_threshold = False

    if modality == "xct1":
        params.outer.periosteal_kernel_size = 12
        params.outer.periosteal_open_radius = 1
        if site in {"radius", "tibia"} and segmentation == "laplace_hamming":
            params.segmentation.laplace_hamming_threshold = 15000.0
    else:
        params.outer.periosteal_kernel_size = 5
        params.outer.periosteal_open_radius = 2
    return params


def load_preset(name: str) -> ContourParameters:
    """Load a preset encoded as ``modality-site-segmentation-outer-inner``."""
    parts = name.strip().lower().split("-")
    if len(parts) != 5:
        raise ValueError(
            "Preset names use 'modality-site-segmentation-outer-inner', for example "
            "'xct1-radius-laplace_hamming-standard-standard'."
        )
    return resolve_preset(
        modality=parts[0],
        site=parts[1],
        segmentation=parts[2],
        outer_contour=parts[3],
        inner_contour=parts[4],
    )
