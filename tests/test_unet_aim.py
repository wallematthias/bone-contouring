from pathlib import Path
import json

import numpy as np
import pytest
py_aimio = pytest.importorskip("py_aimio")


def test_publication_flushes_writable_handles_for_windows(tmp_path, monkeypatch):
    """Windows rejects flushing read-only handles after successful inference."""
    import errno
    from bone_contouring.unet import aim

    real_open = Path.open
    real_fsync = aim.os.fsync
    writable_handles = {}

    def tracked_open(path, *args, **kwargs):
        stream = real_open(path, *args, **kwargs)
        writable_handles[stream.fileno()] = stream.writable()
        return stream

    def windows_fsync(fd):
        if not writable_handles.get(fd, False):
            raise OSError(errno.EBADF, "Bad file descriptor")
        real_fsync(fd)

    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(aim.os, "fsync", windows_fsync)
    trab = np.zeros((3, 7, 9), dtype=np.uint8)
    trab[:, 2:5, 3:6] = 1
    cort = np.zeros_like(trab)
    cort[:, 1, 1:7] = 1
    masks = dict(full=trab | cort, trab=trab, cort=cort)
    meta = dict(position=[0]*3, offset=[0]*3, element_size=[.061]*3, unit="native")

    paths = aim.write_masks(tmp_path, "scan", masks, meta,
                            source=tmp_path / "raw.AIM", device="cpu")

    assert all(path.is_file() for path in paths.values())
    for role, expected in masks.items():
        actual, _ = py_aimio.read_aim(str(paths[role]))
        np.testing.assert_array_equal(actual, expected * 127)
    assert json.loads(paths["provenance"].read_text())["method"] == "unet"
    assert not list(tmp_path.glob(".unet-*"))


def test_interrupted_publication_never_exposes_truncated_masks(tmp_path, monkeypatch):
    from bone_contouring.unet import aim
    import py_aimio
    import numpy as np
    meta = dict(position=[0, 0, 0], offset=[0, 0, 0], element_size=[.061]*3, unit='native')
    trab = np.zeros((3, 7, 9), dtype=np.uint8)
    trab[:, 2:5, 3:6] = 1
    cort = np.zeros_like(trab)
    cort[:, 1, 1:7] = 1
    masks = dict(full=trab|cort, trab=trab, cort=cort)
    real_link = aim.os.link
    def interrupted(source, destination):
        if destination.name.endswith('_desc-trab_mask.AIM'):
            raise OSError('interrupted')
        real_link(source, destination)
    monkeypatch.setattr(aim.os, 'link', interrupted)
    with pytest.raises(OSError, match='interrupted'):
        aim.write_masks(tmp_path, 'scan', masks, meta, source='/raw/scan.AIM', device='cpu')
    paths = aim.output_paths(tmp_path, 'scan')
    assert not paths['provenance'].exists()
    assert paths['full_sidecar'].exists()
    actual, _ = py_aimio.read_aim(str(paths['full']))
    np.testing.assert_array_equal(actual, masks['full']*127)
    assert not paths['trab'].exists()


def test_publication_race_does_not_overwrite_another_writer(tmp_path, monkeypatch):
    from bone_contouring.unet import aim
    import numpy as np
    trab = np.zeros((3, 7, 9), dtype=np.uint8)
    trab[:, 2:5, 3:6] = 1
    cort = np.zeros_like(trab)
    cort[:, 1, 1:7] = 1
    real_link = aim.os.link
    def racing(source, destination):
        if destination.name.endswith('_desc-full_mask.AIM'):
            destination.write_bytes(b'other writer')
        real_link(source, destination)
    monkeypatch.setattr(aim.os, 'link', racing)
    with pytest.raises(FileExistsError):
        aim.write_masks(tmp_path, 'scan', dict(full=trab|cort, trab=trab, cort=cort),
                        dict(position=[0]*3, offset=[0]*3, element_size=[.061]*3, unit='native'),
                        source='/raw/scan.AIM', device='cpu')
    assert (tmp_path/'scan_desc-full_mask.AIM').read_bytes() == b'other writer'
    assert not (tmp_path/'scan_UNET.json').exists()


def test_directory_discovery_ignores_masks_and_preserves_relative_paths(tmp_path):
    from bone_contouring.unet.aim import plan_batch
    raw = tmp_path / "raw"
    for name in ("one/scan.AIM", "two/scan.AIM", "one/scan_SEG.AIM", "one/scan_CORT_MASK.AIM", "ignore.txt"):
        file = raw / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    jobs = plan_batch([raw], tmp_path / "results")
    assert [(j.source.relative_to(raw).as_posix(), j.output.relative_to(tmp_path/"results").as_posix())
            for j in jobs] == [("one/scan.AIM", "one"), ("two/scan.AIM", "two")]


def test_raw_and_existing_outputs_are_rejected_before_inference(tmp_path):
    from bone_contouring.unet.aim import plan_batch
    source = tmp_path / "raw" / "scan.AIM"
    source.parent.mkdir()
    source.touch()
    for out in (source.parent, source.parent / "results"):
        with pytest.raises(ValueError, match="input"):
            plan_batch([source], out)
    results = tmp_path / "results"
    results.mkdir()
    (results/"scan_desc-cort_mask.AIM").touch()
    with pytest.raises(FileExistsError):
        plan_batch([source], results)


def test_colliding_explicit_files_and_mask_inputs_are_rejected(tmp_path):
    from bone_contouring.unet.aim import plan_batch
    files = []
    for directory in ("a","b"):
        path = tmp_path / directory / "same.AIM"
        path.parent.mkdir()
        path.touch()
        files.append(path)
    with pytest.raises(ValueError, match="collid"):
        plan_batch(files, tmp_path/"results")
    mask = tmp_path / "scan_TRAB_MASK.AIM"
    mask.touch()
    with pytest.raises(ValueError, match="image"):
        plan_batch([mask], tmp_path/"results")


def test_native_aim_outputs_preserve_geometry_not_density_scaled(tmp_path):
    from bone_contouring.unet.aim import write_masks
    meta = {"position": (4,7,11), "offset": (1,2,3), "element_size": (.061,.062,.063),
            "processing_log": "Mu_Scaling 8192\nHU: mu water 0.24\nDensity: slope 1500\nDensity: intercept -100\n",
            "unit": "BMD"}
    trab = np.zeros((3,7,9), dtype=np.uint8)
    trab[:,2:5,3:6] = 1
    cort = np.zeros_like(trab)
    cort[:,1,1:7] = 1
    masks = dict(trab=trab, cort=cort, full=trab|cort)
    paths = write_masks(tmp_path, "scan", masks, meta, source=Path("/original/scan.AIM"), device="cpu")
    for role in masks:
        array, actual = py_aimio.read_aim(str(paths[role]))
        np.testing.assert_array_equal(array, masks[role]*127)
        for key in ("position", "offset", "element_size"):
            if key == "element_size":
                np.testing.assert_allclose(actual[key], meta[key], rtol=1e-6)
            else:
                assert actual[key] == meta[key]
    assert json.loads((tmp_path/"scan_UNET.json").read_text())["model"] == "radius_tibia_final"
    assert paths["cort"].name == "scan_desc-cort_mask.AIM"
    for role in masks:
        sidecar = json.loads(paths[role].with_suffix(".AIM.json").read_text())
        assert sidecar["short_role"] == role
        assert sidecar["software"]["name"] == "bone-contouring"
        assert sidecar["source_geometry"]["position"] == [4, 7, 11]


def test_normalized_scan_uses_bone_contours_names(tmp_path):
    from bone_contouring.unet.aim import output_paths
    paths = output_paths(tmp_path, "sub-001_ses-1_voi-radiusleft_stack-02_xct")
    assert paths["full"].name == "sub-001_ses-1_voi-radiusleft_stack-02_desc-full_mask.AIM"
    assert paths["trab"].name == "sub-001_ses-1_voi-radiusleft_stack-02_desc-trab_mask.AIM"
    assert paths["cort"].name == "sub-001_ses-1_voi-radiusleft_stack-02_desc-cort_mask.AIM"


def test_existing_sidecar_alone_prevents_inference(tmp_path):
    from bone_contouring.unet.aim import plan_batch
    source = tmp_path / "raw" / "scan.AIM"
    source.parent.mkdir()
    source.touch()
    results = tmp_path / "results"
    results.mkdir()
    (results/"scan_desc-cort_mask.AIM.json").write_text("{}")
    with pytest.raises(FileExistsError):
        plan_batch([source], results)


def test_cli_exposes_only_io_and_device(capsys):
    from bone_contouring.unet.cli import main
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--device" in help_text and "--output" in help_text
    assert "--threshold" not in help_text and "--model" not in help_text


def test_directory_symlink_cannot_write_beside_actual_raw_scan(tmp_path):
    from bone_contouring.unet.aim import plan_batch
    raw,study=tmp_path/"raw",tmp_path/"study"
    raw.mkdir()
    study.mkdir()
    scan=raw/"scan.AIM"
    scan.touch()
    (study/"scan.AIM").symlink_to(scan)
    with pytest.raises(ValueError,match="input"):
        plan_batch([study],raw)


def test_output_child_symlink_cannot_redirect_publication_into_raw(tmp_path):
    from bone_contouring.unet.aim import plan_batch
    raw = tmp_path / "raw"
    scan = raw / "one/scan.AIM"
    scan.parent.mkdir(parents=True)
    scan.touch()
    output = tmp_path / "results"
    output.mkdir()
    (output / "one").symlink_to(scan.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="input"):
        plan_batch([raw], output)
