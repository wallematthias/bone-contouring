"""Composable presets and named profile files for bone contouring."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import json
from importlib import resources
from pathlib import Path
from typing import Any

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
    contour_support_method: str = "",
) -> ContourParameters:
    """Compose a fresh parameter object from supported method dimensions."""
    modality = _choice(modality, _MODALITIES, "modality")
    site = _choice(site, _SITES, "site")
    segmentation = _choice(segmentation, _SEGMENTATION_METHODS, "segmentation")
    if contour_support_method:
        contour_support_method = _choice(contour_support_method, _SEGMENTATION_METHODS, "contour support")
    outer_contour = _choice(outer_contour, _OUTER_CONTOUR_METHODS, "outer contour")
    inner_contour = _choice(inner_contour, _INNER_CONTOUR_METHODS, "inner contour")

    params = ContourParameters(modality=modality, site=site)
    params.inner.site = site
    params.segmentation.method = segmentation
    params.segmentation.contour_support_method = contour_support_method
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


def load_preset(name: str, *, site: str = "", profile_root: Path | None = None) -> ContourParameters:
    """Load a named profile or an encoded ``modality-site-segmentation-outer-inner`` preset."""
    profile = _load_named_profile(name)
    if profile is not None:
        requested_site = _canonical_site(site or str(profile.get("default_site", "")))
        return _params_from_shipped_profile(name, profile, requested_site)

    user_profile = _load_user_profile(name, root=profile_root)
    if user_profile is not None:
        return _params_from_user_profile(user_profile, override_site=site)

    parts = name.strip().lower().split("-")
    if len(parts) != 5:
        raise ValueError(
            "Preset names use a shipped profile such as 'XtremeCTI', 'XtremeCTII', or 'XtremeCTII-LH', "
            "or 'modality-site-segmentation-outer-inner', for example "
            "'xct1-radius-laplace_hamming-standard-standard'."
        )
    return resolve_preset(
        modality=parts[0],
        site=parts[1],
        segmentation=parts[2],
        outer_contour=parts[3],
        inner_contour=parts[4],
    )


def _params_from_shipped_profile(name: str, profile: dict[str, Any], requested_site: str) -> ContourParameters:
    site_profiles = profile.get("sites") or {}
    if requested_site not in site_profiles:
        choices = ", ".join(sorted(site_profiles))
        raise ValueError(f"Profile {name!r} has no site {requested_site!r}; choose one of: {choices}.")
    params = resolve_preset(
        modality=str(profile["modality"]),
        site=requested_site,
        segmentation=str(profile["segmentation"]),
        contour_support_method=str(profile.get("contour_support_method") or ""),
        outer_contour=str(profile.get("outer_contour", "standard")),
        inner_contour=str(profile.get("inner_contour", "standard")),
    )
    _update_dataclass(params.outer, site_profiles[requested_site].get("outer", {}))
    _update_dataclass(params.inner, site_profiles[requested_site].get("inner", {}))
    _update_dataclass(params.segmentation, profile.get("segmentation_parameters", {}))
    _update_dataclass(params.segmentation, site_profiles[requested_site].get("segmentation", {}))
    params.modality = str(profile["modality"])
    params.site = requested_site
    params.inner.site = requested_site
    params.segmentation.method = str(profile["segmentation"])
    params.segmentation.contour_support_method = str(profile.get("contour_support_method") or "")
    return params


def _params_from_user_profile(profile: dict[str, Any], *, override_site: str = "") -> ContourParameters:
    schema = str(profile.get("schema") or "")
    if schema == "bone-contour-recipe-v1":
        methods = dict(profile.get("methods") or {})
        site = _canonical_site(override_site or str(profile.get("site") or "radius"))
        params = resolve_preset(
            modality=str(profile.get("modality") or "xct2"),
            site=site,
            segmentation=_recipe_segmentation_method(methods.get("bone_segmentation", "gauss")),
            outer_contour=str(methods.get("periosteal_contour") or "standard"),
            inner_contour=str(methods.get("endosteal_contour") or "standard"),
        )
        payload = dict(profile.get("parameters") or {})
        _update_dataclass(params.outer, payload.get("outer", {}))
        _update_dataclass(params.inner, payload.get("inner", {}))
        _update_dataclass(params.segmentation, payload.get("segmentation", {}))
        return params
    if schema == "bone-contouring-profile-v1":
        payload = dict(profile.get("contour_parameters") or {})
        methods = dict(payload.get("methods") or {})
        site = _canonical_site(override_site or str(payload.get("site") or profile.get("site") or "radius"))
        params = resolve_preset(
            modality=str(payload.get("modality") or profile.get("modality") or "xct2"),
            site=site,
            segmentation=_recipe_segmentation_method(methods.get("bone_segmentation", "gauss")),
            outer_contour=str(methods.get("periosteal_contour") or "standard"),
            inner_contour=str(methods.get("endosteal_contour") or "standard"),
        )
        settings = dict(payload.get("parameters") or {})
        _update_dataclass(params.outer, settings.get("outer", {}))
        _update_dataclass(params.inner, settings.get("inner", {}))
        _update_dataclass(params.segmentation, settings.get("segmentation", {}))
        return params
    raise ValueError(f"Unsupported bone contouring profile schema: {schema!r}.")


def _recipe_segmentation_method(value: str) -> str:
    return {"seg_gauss": "gauss", "none": "gauss"}.get(str(value or "").strip().lower(), str(value or "gauss"))


def _canonical_site(site: str) -> str:
    normalized = str(site or "").strip().lower().replace("_", "").replace("-", "")
    if normalized.startswith("radius"):
        return "radius"
    if normalized.startswith("tibia"):
        return "tibia"
    if normalized.startswith("knee"):
        return "knee"
    return _choice(str(site or "radius"), _SITES, "site")


def _load_named_profile(name: str) -> dict[str, Any] | None:
    normalized = str(name or "").strip().lower().replace("_", "").replace("-", "")
    with resources.files("bone_contouring").joinpath("profiles/shipped_profiles.json").open(
        "r",
        encoding="utf-8",
    ) as stream:
        profiles = json.load(stream)
    for profile_name, payload in profiles.items():
        if profile_name.lower().replace("_", "").replace("-", "") == normalized:
            return dict(payload)
    return None


def _load_user_profile(name: str, *, root: Path | None) -> dict[str, Any] | None:
    try:
        from bone_imaging_derivatives import list_profiles, load_profile_payload
    except Exception:
        return None
    requested = str(name or "").strip().lower()
    requested_token = requested.replace("_", "-").replace(" ", "-")
    for record in list_profiles("bone-contouring", root=root):
        if record.kind != "json":
            continue
        names = {record.name.lower(), record.path.stem.lower()}
        names.add(record.path.stem.lower().replace("-", " "))
        names.add(record.path.stem.lower().replace("_", "-"))
        if requested in names or requested_token in names:
            return load_profile_payload(record)
    return None


def _update_dataclass(instance, values: dict[str, Any]) -> None:
    if not values:
        return
    if not is_dataclass(instance):
        raise TypeError("instance must be a dataclass instance")
    valid_names = {field.name for field in fields(instance)}
    for key, value in values.items():
        if key not in valid_names:
            raise ValueError(f"Unknown profile parameter {key!r} for {type(instance).__name__}.")
        setattr(instance, key, value)
