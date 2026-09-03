"""Batch workflow for normalized XCT bone-contouring datasets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re
from typing import Any

import SimpleITK as sitk
import numpy as np
from bone_imaging_derivatives import (
    BatchArtifact,
    DerivativeManifest,
    DerivativeRecord,
    discover_derivative_artifacts,
    discover_raw_xct_images,
    read_manifest,
    write_manifest,
)
from bone_imaging_derivatives.layout import manifest_path, record_output_path, voi_token

from .api import GeneratedMasks, generate_masks_from_image
from .parameters import ContourParameters
from .presets import load_preset, resolve_preset

_FAMILY = "BoneContours"
_SHORT_TO_RECORD_ROLE = {
    "seg": "bone_segmentation",
    "full": "periosteal_mask",
    "trab": "trabecular_mask",
    "cort": "cortical_mask",
    "fea-materials": "material_labelmap",
}
_RECORD_TO_SHORT_ROLE = {value: key for key, value in _SHORT_TO_RECORD_ROLE.items()}


@dataclass(frozen=True)
class BoneContouringBatchRow:
    """One normalized source image and its current contouring status."""

    image: BatchArtifact
    status: str
    outputs: tuple[BatchArtifact, ...] = ()


def discover_bone_contouring_batch(dataset_root) -> tuple[BoneContouringBatchRow, ...]:
    """Return contouring rows as ``ready`` or ``loadable`` for a normalized dataset."""
    root = _dataset_root(Path(dataset_root).expanduser().resolve())
    images = discover_raw_xct_images(root)
    outputs = discover_derivative_artifacts(root, _FAMILY)
    rows: list[BoneContouringBatchRow] = []
    for image in images:
        matching = tuple(output for output in outputs if output.key == image.key)
        status = "loadable" if matching else "ready"
        rows.append(BoneContouringBatchRow(image=image, status=status, outputs=matching))
    return tuple(rows)


def run_bone_contouring_batch(
    dataset_root,
    *,
    modality: str = "xct1",
    site: str = "radius",
    segmentation: str = "laplace_hamming",
    outer_contour: str = "standard",
    inner_contour: str = "standard",
    profile: str = "",
    parameters: ContourParameters | None = None,
    subject_id: str = "",
    session_id: str = "",
    voi: str = "",
    output_root=None,
    force: bool = False,
    dry_run: bool = False,
    progress: Callable[[str], None] | None = None,
) -> list[DerivativeRecord]:
    """Generate ``BoneContours`` derivatives for every selected normalized XCT image."""
    root = _dataset_root(Path(dataset_root).expanduser().resolve())
    output_dataset_root = _dataset_root(Path(output_root).expanduser().resolve()) if output_root else root
    cases = _filter_rows(
        discover_bone_contouring_batch(root),
        subject_id=subject_id,
        session_id=session_id,
        voi=voi,
    )
    if not cases:
        raise ValueError("No normalized XCT image row was found for bone contouring")
    if dry_run:
        return []

    existing_records = _read_existing_records(output_dataset_root)
    output_records: list[DerivativeRecord] = []
    for row in cases:
        image_record = row.image
        if row.status == "loadable" and not force:
            output_records.extend(_records_for_existing_outputs(existing_records, image_record))
            _emit(progress, f"reused {image_record.path.name}")
            continue
        _emit(progress, f"generating {image_record.path.name}")
        row_site = _site_from_voi(site, image_record.key.voi)
        params = parameters or _resolve_batch_parameters(
            profile=profile,
            modality=modality,
            site=row_site,
            segmentation=segmentation,
            outer_contour=outer_contour,
            inner_contour=inner_contour,
        )
        image, segmentation_image, source_metadata = _read_image_inputs(image_record.path, params)
        generated = generate_masks_from_image(image, params, segmentation_image=segmentation_image)
        input_id = image_record.path.name
        settings_hash = _settings_hash(params)
        for short_role, mask, content_type in _generated_outputs(generated):
            record_role = _SHORT_TO_RECORD_ROLE[short_role]
            output_path = _output_path(output_dataset_root, image_record, short_role, content_type)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_format = "aim" if _is_aim_path(image_record.path) else "nifti"
            if output_format == "aim":
                _write_aim_output(mask, image_record.path, output_path, source_metadata, short_role, content_type)
            else:
                if content_type == "mask":
                    sitk.WriteImage(sitk.Cast(mask > 0, sitk.sitkUInt8), str(output_path))
                else:
                    sitk.WriteImage(sitk.Cast(mask, sitk.sitkUInt8), str(output_path))
            sidecar_path = _sidecar_path(output_path)
            sidecar_path.write_text(
                json.dumps(
                    _sidecar_payload(
                        generated,
                        params,
                        short_role,
                        settings_hash,
                        source_metadata=source_metadata,
                        output_format=output_format,
                    ),
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )
            output_records.append(
                DerivativeRecord(
                    derivative=_FAMILY,
                    role=record_role,
                    subject_id=image_record.key.subject_id,
                    site=image_record.key.voi,
                    session_id=image_record.key.session_id,
                    stack_index=image_record.key.stack_index,
                    space="native",
                    path=output_path,
                    source="generated",
                    inputs=(input_id,),
                    metadata={"short_role": short_role},
                    content_type=content_type,
                    settings_hash=settings_hash,
                    software={"name": "bone-contouring", "version": _package_version()},
                )
            )

    records = _merge_records(existing_records, output_records)
    write_manifest(
        DerivativeManifest.create(
            _FAMILY,
            output_dataset_root,
            {"name": "bone-contouring", "version": _package_version()},
            tuple(records),
        ),
        manifest_path(output_dataset_root, _FAMILY),
    )
    return output_records


def _dataset_root(root: Path) -> Path:
    return root.parent if root.name == "derivatives" else root


def _is_aim_path(path: Path) -> bool:
    name = Path(path).name.lower()
    return name.endswith(".aim") or re.search(r"\.aim;\d+$", name, re.IGNORECASE) is not None


def _site_from_voi(site: str, voi: str) -> str:
    if str(site or "").strip().lower() not in {"", "auto"}:
        return str(site).strip().lower()
    normalized = voi_token(voi)
    if normalized.startswith("radius"):
        return "radius"
    if normalized.startswith("tibia"):
        return "tibia"
    if normalized.startswith("knee"):
        return "knee"
    return "radius"


def _resolve_batch_parameters(
    *,
    profile: str,
    modality: str,
    site: str,
    segmentation: str,
    outer_contour: str,
    inner_contour: str,
) -> ContourParameters:
    normalized_profile = str(profile or "").strip()
    if normalized_profile and normalized_profile.lower() not in {"standard", "custom"}:
        return load_preset(normalized_profile, site=site)
    return resolve_preset(
        modality=modality,
        site=site,
        segmentation=segmentation,
        outer_contour=outer_contour,
        inner_contour=inner_contour,
    )


def _read_source_image(path: Path) -> sitk.Image:
    """Read a normalized source image, using py_aimio for Scanco AIM inputs."""
    image, _segmentation_image, _metadata = _read_image_inputs(path, None)
    return image


def _read_image_inputs(path: Path, parameters: ContourParameters | None) -> tuple[sitk.Image, sitk.Image | None, dict[str, Any]]:
    """Read density image plus optional native AIM image needed by Laplace-Hamming."""
    path = Path(path)
    if not _is_aim_path(path):
        return sitk.ReadImage(str(path)), None, {}
    density_image, density_metadata = _read_aim_image(path, density=True)
    method = (parameters.segmentation.method if parameters is not None else "").strip().lower()
    if method != "laplace_hamming":
        return density_image, None, density_metadata
    native_image, native_metadata = _read_aim_image(path, density=False)
    native_image = sitk.Cast(native_image, sitk.sitkInt16)
    if native_image.GetSize() != density_image.GetSize():
        raise ValueError(
            "AIM native and density reads produced different image sizes: "
            f"native={native_image.GetSize()}, density={density_image.GetSize()}."
        )
    native_image.CopyInformation(density_image)
    return density_image, native_image, native_metadata or density_metadata


def _read_aim_image(path: Path, *, density: bool) -> tuple[sitk.Image, dict[str, Any]]:
    try:
        import py_aimio
    except Exception as exc:
        raise RuntimeError(
            "AIM input requires aimio-py/py_aimio. Install aimio-py in the active Python environment "
            "or import the scan through the Scanco I/O setup first."
        ) from exc
    array_zyx, metadata = py_aimio.read_aim(str(path), density=bool(density))
    metadata = dict(metadata or {})
    image = sitk.GetImageFromArray(np.asarray(array_zyx))
    spacing = metadata.get("spacing") or metadata.get("element_size")
    if spacing is not None and len(spacing) == 3:
        image.SetSpacing(tuple(float(value) for value in spacing))
    origin = metadata.get("origin")
    if origin is not None and len(origin) == 3:
        image.SetOrigin(tuple(float(value) for value in origin))
    direction = metadata.get("direction")
    if direction is not None and len(direction) == 9:
        image.SetDirection(tuple(float(value) for value in direction))
    return image, metadata


def _write_aim_output(
    output_image: sitk.Image,
    source_path: Path,
    output_path: Path,
    source_metadata: dict[str, Any],
    role: str,
    content_type: str,
) -> None:
    try:
        import py_aimio
    except Exception as exc:
        raise RuntimeError("AIM contour output requires aimio-py/py_aimio in the active Python environment.") from exc
    metadata = dict(source_metadata or {})
    metadata["source_file"] = str(source_path)
    metadata["contour_role"] = str(role)
    metadata["content_type"] = str(content_type)
    metadata["unit"] = "native"
    metadata["dimensions"] = tuple(int(value) for value in output_image.GetSize())
    metadata["spacing"] = tuple(float(value) for value in output_image.GetSpacing())
    metadata["element_size"] = tuple(float(value) for value in output_image.GetSpacing())
    metadata["origin"] = tuple(float(value) for value in output_image.GetOrigin())
    metadata["direction"] = tuple(float(value) for value in output_image.GetDirection())
    if content_type == "mask":
        array_zyx = (127 * (sitk.GetArrayFromImage(output_image) > 0)).astype(np.int8)
    else:
        array_zyx = sitk.GetArrayFromImage(output_image).astype(np.int8, copy=False)
    py_aimio.write_aim(str(output_path), array_zyx, metadata, unit="native")


def _sidecar_path(path: Path) -> Path:
    if path.name.lower().endswith(".nii.gz"):
        return path.with_name(path.name[:-7] + ".json")
    return path.with_suffix(path.suffix + ".json")


def _filter_rows(
    rows: tuple[BoneContouringBatchRow, ...],
    *,
    subject_id: str,
    session_id: str,
    voi: str,
) -> tuple[BoneContouringBatchRow, ...]:
    subject_filter = _clean_filter(subject_id)
    session_filter = _clean_filter(session_id)
    voi_filter = voi_token(voi) if voi else ""
    selected = []
    for row in rows:
        key = row.image.key
        if subject_filter and _clean_filter(key.subject_id) != subject_filter:
            continue
        if session_filter and _clean_filter(key.session_id) != session_filter:
            continue
        if voi_filter and key.voi != voi_filter:
            continue
        selected.append(row)
    return tuple(selected)


def _clean_filter(value: str) -> str:
    return str(value or "").strip().removeprefix("sub-").removeprefix("ses-")


def _read_existing_records(root: Path) -> list[DerivativeRecord]:
    path = manifest_path(root, _FAMILY)
    if not path.exists():
        return []
    return list(read_manifest(path).records)


def _records_for_existing_outputs(records: list[DerivativeRecord], image: BatchArtifact) -> list[DerivativeRecord]:
    return [
        record
        for record in records
        if (
            record.subject_id == image.key.subject_id
            and record.session_id == image.key.session_id
            and voi_token(record.site) == image.key.voi
            and record.stack_index == image.key.stack_index
            and record.role in set(_SHORT_TO_RECORD_ROLE.values())
        )
    ]


def _generated_outputs(generated: GeneratedMasks):
    yield "seg", generated.seg, "mask"
    yield "full", generated.full, "mask"
    yield "trab", generated.trab, "mask"
    yield "cort", generated.cort, "mask"
    yield "fea-materials", generated.material, "label"


def _output_path(root: Path, image: BatchArtifact, short_role: str, content_type: str) -> Path:
    session_part = f"ses-{image.key.session_id}"
    stack_part = f"_stack-{image.key.stack_index:02d}" if image.key.stack_index is not None else ""
    extension = ".AIM" if _is_aim_path(image.path) else ".nii.gz"
    suffix = "label" if content_type == "label" else "mask"
    filename = (
        f"sub-{image.key.subject_id}_{session_part}_voi-{voi_token(image.key.voi)}"
        f"{stack_part}_desc-{short_role}_{suffix}{extension}"
    )
    return record_output_path(root, _FAMILY, image.key.subject_id, image.key.voi, session_part, filename)


def _sidecar_payload(
    generated: GeneratedMasks,
    parameters: ContourParameters,
    short_role: str,
    settings_hash: str,
    *,
    source_metadata: dict[str, Any] | None = None,
    output_format: str = "nifti",
) -> dict[str, Any]:
    return {
        "schema": "bone-contour-mask-provenance-v1",
        "role": _SHORT_TO_RECORD_ROLE[short_role],
        "short_role": short_role,
        "settings_hash": settings_hash,
        "output_format": output_format,
        "algorithm_metadata": generated.metadata,
        "mask_provenance": generated.mask_provenance,
        "parameters": _parameters_payload(parameters),
        "source_metadata": dict(source_metadata or {}),
        "software": {"name": "bone-contouring", "version": _package_version()},
    }


def _parameters_payload(parameters: ContourParameters) -> dict[str, Any]:
    return {
        "modality": parameters.modality,
        "site": parameters.site,
        "segmentation": asdict(parameters.segmentation),
        "outer": asdict(parameters.outer),
        "inner": asdict(parameters.inner),
    }


def _settings_hash(parameters: ContourParameters) -> str:
    payload = json.dumps(_parameters_payload(parameters), sort_keys=True)
    return sha256(payload.encode("utf-8")).hexdigest()[:20]


def _merge_records(existing: list[DerivativeRecord], generated: list[DerivativeRecord]) -> list[DerivativeRecord]:
    generated_ids = {
        (
            record.role,
            record.subject_id,
            record.session_id,
            record.site,
            record.stack_index,
        )
        for record in generated
    }
    kept = [
        record
        for record in existing
        if (
            record.role,
            record.subject_id,
            record.session_id,
            record.site,
            record.stack_index,
        )
        not in generated_ids
    ]
    return kept + generated


def _package_version() -> str:
    try:
        return version("bone-contouring")
    except PackageNotFoundError:
        return "0+unknown"


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
