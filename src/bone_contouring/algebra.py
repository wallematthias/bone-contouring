"""Batch mask and label algebra for existing contour artifacts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from bone_imaging_derivatives import (
    BatchArtifact,
    DerivativeManifest,
    DerivativeRecord,
    discover_derivative_artifacts,
    discover_raw_xct_images,
    manifest_path,
    preferred_contours,
    read_manifest,
    write_manifest,
)

from .batch import (
    _FAMILY,
    _SHORT_TO_RECORD_ROLE,
    _dataset_root,
    _is_aim_path,
    _output_path,
    _package_version,
    _read_aim_image,
    _sidecar_path,
    _write_aim_output,
)


@dataclass(frozen=True)
class MaskLabelAlgebraRow:
    """One source image plus contour inputs available for derivation."""

    image: BatchArtifact
    contours: dict[str, BatchArtifact]
    status: str
    derivable_roles: tuple[str, ...] = ()
    outputs: tuple[BatchArtifact, ...] = ()


def discover_mask_label_algebra_batch(dataset_root) -> tuple[MaskLabelAlgebraRow, ...]:
    """Discover normalized rows where existing masks can derive more artifacts."""
    root = _dataset_root(Path(dataset_root).expanduser().resolve())
    images = discover_raw_xct_images(root)
    contours = _discover_contour_inputs(root)
    outputs = discover_derivative_artifacts(root, _FAMILY)
    rows: list[MaskLabelAlgebraRow] = []
    for image in images:
        selected = dict(preferred_contours(contours, image.key).selected)
        matching_outputs = tuple(
            output
            for output in outputs
            if output.key == image.key and _is_mask_label_algebra_output(output)
        )
        derivable = _derivable_roles(selected, matching_outputs)
        if derivable:
            status = "ready"
        elif matching_outputs:
            status = "loadable"
        else:
            status = "missing"
        rows.append(
            MaskLabelAlgebraRow(
                image=image,
                contours=selected,
                status=status,
                derivable_roles=tuple(derivable),
                outputs=matching_outputs,
            )
        )
    return tuple(rows)


def run_mask_label_algebra_batch(
    dataset_root,
    *,
    subject_id: str = "",
    session_id: str = "",
    voi: str = "",
    output_root=None,
    force: bool = False,
    dry_run: bool = False,
    progress: Callable[[str], None] | None = None,
) -> list[DerivativeRecord]:
    """Derive missing masks and FEA material labels from existing contours.

    This workflow performs only algebra on already available masks: it can
    derive `full`, `trab`, or `cort` from any consistent pair and can create
    the FEA material labelmap from a bone segmentation intersected with
    trabecular and cortical ROIs.
    """
    root = _dataset_root(Path(dataset_root).expanduser().resolve())
    output_dataset_root = _dataset_root(Path(output_root).expanduser().resolve()) if output_root else root
    rows = _filter_rows(
        discover_mask_label_algebra_batch(root),
        subject_id=subject_id,
        session_id=session_id,
        voi=voi,
    )
    if not rows:
        raise ValueError("No normalized row with mask/label algebra inputs was found")
    if dry_run:
        return []

    existing_records = _read_existing_records(output_dataset_root)
    output_records: list[DerivativeRecord] = []
    for row in rows:
        if row.status == "loadable" and not force:
            output_records.extend(_records_for_existing_outputs(existing_records, row.image))
            _emit(progress, f"reused mask algebra outputs for {row.image.path.name}")
            continue
        generated = _derive_outputs(row, force=force)
        if not generated:
            _emit(progress, f"nothing derivable for {row.image.path.name}")
            continue
        _emit(progress, f"deriving {', '.join(generated)} for {row.image.path.name}")
        source_metadata = _source_metadata(row.image.path)
        for short_role, image in generated.items():
            content_type = "label" if short_role == "fea-materials" else "mask"
            record_role = _SHORT_TO_RECORD_ROLE[short_role]
            output_path = _output_path(output_dataset_root, row.image, short_role, content_type)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_format = "aim" if _is_aim_path(row.image.path) else "nifti"
            if output_format == "aim":
                _write_aim_output(image, row.image.path, output_path, source_metadata, short_role, content_type)
            elif content_type == "mask":
                sitk.WriteImage(sitk.Cast(image > 0, sitk.sitkUInt8), str(output_path))
            else:
                sitk.WriteImage(sitk.Cast(image, sitk.sitkUInt8), str(output_path))
            _sidecar_path(output_path).write_text(
                json.dumps(
                    _sidecar_payload(short_role, row, output_format),
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
                    subject_id=row.image.key.subject_id,
                    site=row.image.key.voi,
                    session_id=row.image.key.session_id,
                    stack_index=row.image.key.stack_index,
                    space="native",
                    path=output_path,
                    source="derived",
                    inputs=tuple(str(artifact.path) for artifact in row.contours.values()),
                    metadata={"short_role": short_role, "workflow": "mask_label_algebra"},
                    content_type=content_type,
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


def _discover_contour_inputs(root: Path) -> tuple[BatchArtifact, ...]:
    return (
        *discover_derivative_artifacts(root, "IPLContours"),
        *discover_derivative_artifacts(root, "ImportedContours"),
    )


def _is_mask_label_algebra_output(artifact: BatchArtifact) -> bool:
    if str(artifact.metadata.get("workflow") or "") == "mask_label_algebra":
        return True
    if str(artifact.metadata.get("short_role") or "") == "fea-materials":
        return True
    return False


def _derivable_roles(contours: dict[str, BatchArtifact], existing_outputs: tuple[BatchArtifact, ...]) -> list[str]:
    existing_short = {
        str(output.metadata.get("short_role") or "").strip()
        for output in existing_outputs
    }
    existing_short.update(_short_role_from_record_role(output.role) for output in existing_outputs)
    roles = set(contours)
    derivable: list[str] = []
    if "full" not in roles and {"trab", "cort"} <= roles and "full" not in existing_short:
        derivable.append("full")
    if "trab" not in roles and {"full", "cort"} <= roles and "trab" not in existing_short:
        derivable.append("trab")
    if "cort" not in roles and {"full", "trab"} <= roles and "cort" not in existing_short:
        derivable.append("cort")
    if "segmentation" in roles and _can_resolve_trab_cort(roles | set(derivable)) and "fea-materials" not in existing_short:
        derivable.append("fea-materials")
    return derivable


def _can_resolve_trab_cort(roles: set[str]) -> bool:
    return {"trab", "cort"} <= roles or {"full", "cort"} <= roles or {"full", "trab"} <= roles


def _derive_outputs(row: MaskLabelAlgebraRow, *, force: bool) -> dict[str, sitk.Image]:
    contours = dict(row.contours)
    existing_short = {
        str(output.metadata.get("short_role") or "").strip()
        for output in row.outputs
    }
    existing_short.update(_short_role_from_record_role(output.role) for output in row.outputs)
    images: dict[str, sitk.Image] = {
        role: _read_label_image(artifact.path, row.image.path)
        for role, artifact in contours.items()
        if role in {"segmentation", "full", "trab", "cort"}
    }
    derived: dict[str, sitk.Image] = {}
    if ("full" not in images or force) and {"trab", "cort"} <= set(images) and (force or "full" not in existing_short):
        derived["full"] = _binary_or(images["trab"], images["cort"])
        images["full"] = derived["full"]
    if ("trab" not in images or force) and {"full", "cort"} <= set(images) and (force or "trab" not in existing_short):
        derived["trab"] = _binary_and_not(images["full"], images["cort"])
        images["trab"] = derived["trab"]
    if ("cort" not in images or force) and {"full", "trab"} <= set(images) and (force or "cort" not in existing_short):
        derived["cort"] = _binary_and_not(images["full"], images["trab"])
        images["cort"] = derived["cort"]
    if (
        "segmentation" in images
        and {"trab", "cort"} <= set(images)
        and (force or "fea-materials" not in existing_short)
    ):
        derived["fea-materials"] = _material_labelmap(images["segmentation"], images["trab"], images["cort"])
    return derived


def _read_label_image(path: Path, reference_path: Path) -> sitk.Image:
    if _is_aim_path(path):
        image, _metadata = _read_aim_image(path, density=False)
    else:
        image = sitk.ReadImage(str(path))
    image = sitk.Cast(image > 0, sitk.sitkUInt8)
    reference = _read_reference_image(reference_path)
    if _same_geometry(image, reference):
        return image
    return sitk.Resample(
        image,
        reference,
        sitk.Transform(reference.GetDimension(), sitk.sitkIdentity),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


def _read_reference_image(path: Path) -> sitk.Image:
    if _is_aim_path(path):
        image, _metadata = _read_aim_image(path, density=True)
        return image
    return sitk.ReadImage(str(path))


def _source_metadata(path: Path) -> dict[str, Any]:
    if not _is_aim_path(path):
        return {}
    _image, metadata = _read_aim_image(path, density=False)
    return metadata


def _same_geometry(left: sitk.Image, right: sitk.Image) -> bool:
    return (
        left.GetSize() == right.GetSize()
        and left.GetSpacing() == right.GetSpacing()
        and left.GetOrigin() == right.GetOrigin()
        and left.GetDirection() == right.GetDirection()
    )


def _binary_or(left: sitk.Image, right: sitk.Image) -> sitk.Image:
    return sitk.Cast(sitk.Or(left > 0, right > 0), sitk.sitkUInt8)


def _binary_and_not(left: sitk.Image, right: sitk.Image) -> sitk.Image:
    return sitk.Cast(sitk.And(left > 0, sitk.Not(right > 0)), sitk.sitkUInt8)


def _material_labelmap(segmentation: sitk.Image, trab: sitk.Image, cort: sitk.Image) -> sitk.Image:
    seg_arr = sitk.GetArrayFromImage(segmentation) > 0
    trab_arr = sitk.GetArrayFromImage(trab) > 0
    cort_arr = sitk.GetArrayFromImage(cort) > 0
    material = np.zeros(seg_arr.shape, dtype=np.uint8)
    material[seg_arr & trab_arr] = 100
    material[seg_arr & cort_arr] = 127
    out = sitk.GetImageFromArray(material)
    out.CopyInformation(segmentation)
    return sitk.Cast(out, sitk.sitkUInt8)


def _sidecar_payload(short_role: str, row: MaskLabelAlgebraRow, output_format: str) -> dict[str, Any]:
    return {
        "schema": "bone-contour-mask-label-algebra-v1",
        "role": _SHORT_TO_RECORD_ROLE[short_role],
        "short_role": short_role,
        "output_format": output_format,
        "algorithm_metadata": {
            "workflow": "mask_label_algebra",
            "operation": _operation_for_role(short_role),
        },
        "inputs": {
            role: str(artifact.path)
            for role, artifact in sorted(row.contours.items())
        },
        "software": {"name": "bone-contouring", "version": _package_version()},
    }


def _operation_for_role(short_role: str) -> str:
    return {
        "full": "trab_or_cort",
        "trab": "full_and_not_cort",
        "cort": "full_and_not_trab",
        "fea-materials": "segmentation_intersect_trab_cort",
    }.get(short_role, "unknown")


def _filter_rows(
    rows: tuple[MaskLabelAlgebraRow, ...],
    *,
    subject_id: str,
    session_id: str,
    voi: str,
) -> tuple[MaskLabelAlgebraRow, ...]:
    subject_filter = _clean_filter(subject_id)
    session_filter = _clean_filter(session_id)
    voi_filter = _clean_voi(voi)
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


def _clean_voi(value: str) -> str:
    return "".join(ch for ch in str(value or "").strip().lower().removeprefix("voi-") if ch.isalnum())


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
            and _clean_voi(record.site) == image.key.voi
            and record.stack_index == image.key.stack_index
            and record.metadata.get("workflow") == "mask_label_algebra"
        )
    ]


def _short_role_from_record_role(role: str) -> str:
    return {
        "bone_segmentation": "seg",
        "periosteal_mask": "full",
        "trabecular_mask": "trab",
        "cortical_mask": "cort",
        "material_labelmap": "fea-materials",
    }.get(str(role or ""), "")


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


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
