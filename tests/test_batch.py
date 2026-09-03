from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from bone_imaging_derivatives import read_manifest


def _write_image(path: Path) -> None:
    values = np.zeros((9, 9, 5), dtype=np.float32)
    values[2:7, 2:7, 1:4] = 900.0
    image = sitk.GetImageFromArray(np.transpose(values, (2, 1, 0)))
    image.SetSpacing((0.061, 0.062, 0.063))
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))


def test_run_bone_contouring_batch_writes_masks_sidecars_and_manifest(tmp_path: Path) -> None:
    """Normalized image rows should become manifest-backed BoneContours outputs."""
    from bone_contouring.batch import run_bone_contouring_batch

    image_path = tmp_path / "sub-001" / "ses-001" / "xct" / "sub-001_ses-001_voi-radiusleft_xct.nii.gz"
    _write_image(image_path)

    records = run_bone_contouring_batch(
        tmp_path,
        modality="xct2",
        site="radius",
        segmentation="gauss",
        inner_contour="none",
    )

    manifest = read_manifest(tmp_path / "derivatives" / "BoneContours" / "manifest.json")
    roles = {record.role for record in records}

    assert roles == {
        "bone_segmentation",
        "periosteal_mask",
        "trabecular_mask",
        "cortical_mask",
        "material_labelmap",
    }
    assert len(manifest.records) == 5
    assert all(record.derivative == "BoneContours" for record in manifest.records)
    assert all(record.site == "radiusleft" for record in manifest.records)
    assert all(record.path.exists() for record in manifest.records)
    assert all(record.path.parent.name == "xct" for record in manifest.records)
    sidecars = [record.path.with_name(record.path.name[:-7] + ".json") for record in manifest.records]
    assert all(path.exists() for path in sidecars)
    assert any(record.path.name.endswith("_desc-fea-materials_label.nii.gz") for record in manifest.records)
    assert json.loads(sidecars[0].read_text(encoding="utf-8"))["algorithm_metadata"]["modality"] == "xct2"


def test_discover_bone_contouring_batch_marks_existing_outputs_loadable(tmp_path: Path) -> None:
    """A completed contour row should report loadable instead of runnable."""
    from bone_contouring.batch import discover_bone_contouring_batch, run_bone_contouring_batch

    image_path = tmp_path / "sub-001" / "ses-001" / "xct" / "sub-001_ses-001_voi-radiusleft_xct.nii.gz"
    _write_image(image_path)

    before = discover_bone_contouring_batch(tmp_path)
    run_bone_contouring_batch(
        tmp_path,
        modality="xct2",
        site="radius",
        segmentation="gauss",
        inner_contour="none",
    )
    after = discover_bone_contouring_batch(tmp_path)

    assert before[0].status == "ready"
    assert after[0].status == "loadable"


def test_aim_batch_input_uses_py_aimio_density_reader(monkeypatch) -> None:
    """AIM files should be read through py_aimio, not SimpleITK ImageFileReader."""
    import sys
    import types

    from bone_contouring import batch

    calls = []

    class DummyAimio(types.SimpleNamespace):
        def read_aim(self, path, density=False, hu=False):
            calls.append((path, density, hu))
            values = np.zeros((3, 4, 5), dtype=np.float32)
            return values, {
                "spacing": (0.061, 0.062, 0.063),
                "origin": (1.0, 2.0, 3.0),
                "direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            }

    def fail_sitk_read(_path):
        raise AssertionError("AIM input should not be read with SimpleITK")

    monkeypatch.setitem(sys.modules, "py_aimio", DummyAimio())
    monkeypatch.setattr(batch.sitk, "ReadImage", fail_sitk_read)

    image = batch._read_source_image(Path("scan.AIM"))

    assert calls == [("scan.AIM", True, False)]
    assert image.GetSize() == (5, 4, 3)
    assert image.GetSpacing() == (0.061, 0.062, 0.063)
    assert image.GetOrigin() == (1.0, 2.0, 3.0)


def test_aim_batch_input_writes_aim_masks(monkeypatch, tmp_path: Path) -> None:
    """AIM sources should keep AIM mask outputs and use native values for Laplace-Hamming."""
    import sys
    import types

    from bone_contouring.batch import run_bone_contouring_batch

    image_path = tmp_path / "sub-001" / "ses-001" / "xct" / "sub-001_ses-001_voi-radiusleft_xct.AIM"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"AIM")
    writes = []

    class DummyAimio(types.SimpleNamespace):
        def read_aim(self, path, density=False, hu=False):
            values = np.zeros((5, 9, 9), dtype=np.float32 if density else np.int16)
            values[:, 2:7, 2:7] = 900.0 if density else 16000
            return values, {
                "spacing": (0.061, 0.062, 0.063),
                "origin": (1.0, 2.0, 3.0),
                "direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            }

        def write_aim(self, path, array, metadata, unit=None):
            writes.append((Path(path), np.asarray(array).dtype, int(np.count_nonzero(array)), dict(metadata), unit))
            Path(path).write_bytes(b"MASK")

    monkeypatch.setitem(sys.modules, "py_aimio", DummyAimio())

    records = run_bone_contouring_batch(
        tmp_path,
        modality="xct1",
        site="radius",
        segmentation="laplace_hamming",
        inner_contour="none",
    )

    assert len(records) == 5
    assert all(record.path.suffix == ".AIM" for record in records)
    assert all(record.path.parent.name == "xct" for record in records)
    assert len(writes) == 5
    assert all(dtype == np.dtype("int8") for _path, dtype, _count, _metadata, _unit in writes)
    nonzero_by_role = {
        path.name.split("_desc-", 1)[1].split("_mask", 1)[0]: count
        for path, _dtype, count, _metadata, _unit in writes
    }
    assert nonzero_by_role["seg"] > 0
    assert nonzero_by_role["full"] > 0
    assert nonzero_by_role["trab"] > 0
    assert any(path.name.endswith("_desc-fea-materials_label.AIM") for path, _dtype, _count, _metadata, _unit in writes)
    assert all(metadata["unit"] == "native" for _path, _dtype, _count, metadata, _unit in writes)
    assert all(unit == "native" for _path, _dtype, _count, _metadata, unit in writes)


def test_cli_run_batch_writes_bone_contours_manifest(tmp_path: Path) -> None:
    """The PyPI package should expose a stable Slicer-callable batch command."""
    image_path = tmp_path / "sub-001" / "ses-001" / "xct" / "sub-001_ses-001_voi-radiusleft_xct.nii.gz"
    _write_image(image_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "bone_contouring.cli",
            "run-batch",
            str(tmp_path),
            "--modality",
            "xct2",
            "--site",
            "radius",
            "--segmentation",
            "gauss",
            "--inner-contour",
            "none",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "wrote 5 BoneContours artifact(s)" in completed.stdout
    assert (tmp_path / "derivatives" / "BoneContours" / "manifest.json").exists()
