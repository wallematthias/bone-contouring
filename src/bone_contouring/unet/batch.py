"""One normalized U-Net contour case plus fixed XCTII LH tissue segmentation."""
import json
from dataclasses import asdict
from pathlib import Path

from bone_imaging_derivatives import (
    DerivativeManifest, DerivativeRecord, completed_unet_masks,
    discover_derivative_artifacts, discover_raw_xct_images, preferred_contours,
    read_manifest, write_manifest,
)
from bone_imaging_derivatives.layout import manifest_path

from .aim import image_stem, output_paths, run_batch


def lh_segmentation_matches(path, voi):
    """Require provenance for the fixed XCTII LH recipe, not just a SEG role."""
    from ..presets import load_preset
    site = next((site for site in ("radius", "tibia") if str(voi).lower().startswith(site)), None)
    if site is None:
        return False
    try:
        payload = json.loads(Path(path).with_suffix(".AIM.json").read_text())
        parameters = payload["parameters"]
        expected = load_preset("XtremeCTII-LH", site=site)
        return (parameters["modality"] == "xct2" and parameters["site"] == site
                and parameters["segmentation"] == asdict(expected.segmentation))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_normalized_case(source, output, dataset_root, device="auto"):
    """Reuse complete U-Net compartments, preserve other masks, and fill missing SEG.

    The standalone raw AIM command remains compartment-only. This normalized
    dataset recipe uses native AIM units and the shipped XCTII LH defaults.
    """
    root = Path(dataset_root).expanduser().resolve()
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    matches = [item for item in discover_raw_xct_images(root) if item.path.resolve() == source]
    if len(matches) != 1 or matches[0].source == "virtual":
        raise ValueError("U-Net requires one normalized physical AIM image.")
    image = matches[0]
    site = next((site for site in ("radius", "tibia") if image.key.voi.lower().startswith(site)), None)
    if site is None:
        raise ValueError("Published U-Net batch defaults support radius and tibia only.")
    expected = root / "derivatives/BoneContours" / f"sub-{image.key.subject_id}" / f"ses-{image.key.session_id}" / "xct"
    if output != expected:
        raise ValueError("Normalized U-Net output must use the case's BoneContours directory.")
    targets = output_paths(output, image_stem(source))
    marker = targets["provenance"]
    paths = completed_unet_masks(marker)
    available = [artifact for family in ("ImportedContours", "IPLContours", "BoneContours")
                 for artifact in discover_derivative_artifacts(root, family)]
    selection = preferred_contours(available, image.key)
    if selection.review_roles:
        raise ValueError("Conflicting contour roles need review before U-Net processing.")
    existing_seg = selection.selected.get("segmentation")
    if existing_seg is not None and existing_seg.path.is_file() and not lh_segmentation_matches(existing_seg.path, image.key.voi):
        raise FileExistsError("Existing SEG is preserved but does not have matching XCTII LH provenance/defaults.")
    if paths:
        provenance = json.loads(marker.read_text())
        if Path(provenance.get("source", "")).resolve() != source:
            raise ValueError("U-Net provenance does not match the selected source.")
        for role, path in zip(("full", "trab", "cort"), paths):
            selected = selection.selected.get(role)
            if selected is not None and selected.path.resolve() != Path(path).resolve():
                raise FileExistsError("Existing imported/conflicting contours are preserved; refusing to mix masks.")
    else:
        if any(path.exists() for path in targets.values()) or any(role in selection.selected for role in ("full", "trab", "cort")):
            raise FileExistsError("Existing/incomplete contours are preserved; refusing to overwrite masks.")
        run_batch([source], output, device)
        paths = completed_unet_masks(marker)
        if not paths:
            raise RuntimeError("U-Net outputs are incomplete; no combined results were published.")
        provenance = json.loads(marker.read_text())

    software = {"name": "bone-contouring", "version": str(provenance["version"])}
    records = [DerivativeRecord(
        "BoneContours", role, image.key.subject_id, image.key.voi,
        image.key.session_id, image.key.stack_index, "native", Path(path), "generated",
        inputs=(source.relative_to(root).as_posix(),), content_type="mask", software=software,
        metadata={key: provenance.get(key) for key in ("method", "model", "device", "weights_sha256", "source_geometry")},
    ) for role, path in zip(("periosteal_mask", "trabecular_mask", "cortical_mask"), paths)]
    manifest = manifest_path(root, "BoneContours")
    existing = list(read_manifest(manifest).records) if manifest.exists() else []
    replacements = {record.path.resolve() for record in records}
    merged = [record for record in existing if record.path.resolve() not in replacements and record.path.is_file()] + records
    write_manifest(DerivativeManifest.create("BoneContours", root, software, records=tuple(merged)), manifest)

    from ..batch import run_bone_contouring_batch
    return run_bone_contouring_batch(root, profile="XtremeCTII-LH", site=site, image_path=source)
