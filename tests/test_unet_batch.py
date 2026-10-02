import json
import sys
import types

import numpy as np
import py_aimio
import pytest


def _case(root, voi="radiusleft", stack=1):
    source = root / "sub-001/ses-1/xct" / f"sub-001_ses-1_voi-{voi}_stack-{stack:02d}_xct.AIM"
    source.parent.mkdir(parents=True, exist_ok=True)
    full = np.zeros((5, 30, 32), dtype=np.uint8)
    full[:, 4:26, 4:28] = 1
    trab = np.zeros_like(full)
    trab[:, 7:23, 7:25] = 1
    metadata = dict(position=(4, 7, 11), offset=(0, 0, 0), element_size=(.061,) * 3,
                    processing_log="Mu_Scaling 8192\nHU: mu water 0.24\nDensity: slope 1500\nDensity: intercept -100\n")
    py_aimio.write_aim(str(source), full.astype(np.int16) * 8000, metadata, unit="native")
    output = root / "derivatives/BoneContours/sub-001/ses-1/xct"
    return source, output, dict(full=full, trab=trab, cort=full-trab), metadata


@pytest.mark.parametrize("voi", ["radiusleft", "tibiaright"])
def test_normalized_unet_adds_lh_and_resumes_without_inference(tmp_path, monkeypatch, voi):
    from bone_contouring.unet.batch import run_normalized_case
    from bone_contouring import batch
    from bone_imaging_derivatives import read_manifest
    source, output, masks, metadata = _case(tmp_path, voi)
    other, _, _, _ = _case(tmp_path, voi, stack=2)
    calls = []
    class Segmenter:
        def __init__(self, device):
            self.device = device
            calls.append("inference")
        def segment(self, image, progress):
            return masks
    monkeypatch.setitem(sys.modules, "bone_contouring.unet.inference", types.SimpleNamespace(Segmenter=Segmenter))
    original_lh = batch.generate_bone_segmentation
    def check_lh(image, parameters, **kwargs):
        assert parameters.segmentation.method == "laplace_hamming"
        assert np.max(py_aimio.read_aim(str(source), density=False)[0]) == 8000
        assert np.max(__import__("SimpleITK").GetArrayFromImage(image)) == 8000
        calls.append("lh")
        return original_lh(image, parameters, **kwargs)
    monkeypatch.setattr(batch, "generate_bone_segmentation", check_lh)
    run_normalized_case(source, output, tmp_path, "cpu")
    manifest_path = tmp_path / "derivatives/BoneContours/manifest.json"
    records = read_manifest(manifest_path).records
    assert len(records) == 5
    assert {record.stack_index for record in records} == {1}
    assert sum(record.metadata.get("method") == "unet" for record in records) == 3
    originals = {record.path: record.path.read_bytes() for record in records if record.content_type == "mask" and record.role != "bone_segmentation"}
    seg = next(record for record in records if record.role == "bone_segmentation")
    assert json.loads(seg.path.with_suffix(".AIM.json").read_text())["parameters"]["segmentation"]["method"] == "laplace_hamming"
    seg.path.unlink()
    seg.path.with_suffix(".AIM.json").unlink()
    run_normalized_case(source, output, tmp_path, "cpu")
    assert calls == ["inference", "lh", "lh"]
    assert all(path.read_bytes() == before for path, before in originals.items())
    assert len(read_manifest(manifest_path).records) == 5
    run_normalized_case(source, output, tmp_path, "cpu")
    assert calls == ["inference", "lh", "lh"]


def test_normalized_unet_rejects_partial_outputs_before_loading_model(tmp_path):
    from bone_contouring.unet.batch import run_normalized_case
    source, output, _, _ = _case(tmp_path)
    output.mkdir(parents=True)
    (output / source.stem.replace("_xct", "_desc-full_mask.AIM")).touch()
    with pytest.raises(FileExistsError):
        run_normalized_case(source, output, tmp_path)


def test_normalized_unet_rejects_existing_gaussian_seg_without_overwriting(tmp_path):
    from bone_contouring.unet.batch import run_normalized_case
    from bone_contouring.unet.aim import write_masks
    from bone_contouring.batch import run_bone_contouring_batch
    from bone_imaging_derivatives import DerivativeManifest, DerivativeRecord, write_manifest
    source, output, masks, metadata = _case(tmp_path)
    paths = write_masks(output, source.stem, masks, metadata, source=source, device="cpu")
    records = [DerivativeRecord("BoneContours", role, "001", "radiusleft", "1", 1, "native", paths[short], "generated", content_type="mask")
               for short, role in (("full", "periosteal_mask"), ("trab", "trabecular_mask"), ("cort", "cortical_mask"))]
    write_manifest(DerivativeManifest.create("BoneContours", tmp_path, {"name":"test", "version":"1"}, tuple(records)),
                   tmp_path / "derivatives/BoneContours/manifest.json")
    run_bone_contouring_batch(tmp_path, modality="xct2", segmentation="gauss", image_path=source)
    before = {path: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError, match="LH"):
        run_normalized_case(source, output, tmp_path)
    assert all(path.read_bytes() == contents for path, contents in before.items())
