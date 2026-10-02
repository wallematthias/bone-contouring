import subprocess
import sys

import pytest


def test_unet_namespace_is_available_without_loading_optional_runtime():
    result = subprocess.run([sys.executable, '-c',
        'import sys; import bone_contouring.unet; '
        'assert not any(m in sys.modules for m in ("torch", "skimage", "py_aimio"))'],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_unet_command_help_has_only_io_and_device(capsys):
    from bone_contouring.cli import main
    with pytest.raises(SystemExit) as exc:
        main(['unet', '--help'])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert '--device' in help_text and '--output' in help_text
    assert '--threshold' not in help_text and '--model' not in help_text


def test_unet_command_dispatches_to_optional_cli(monkeypatch):
    from bone_contouring.cli import main
    from bone_contouring.unet import cli
    seen = []
    monkeypatch.setattr(cli, "main", lambda args: seen.append(args) or 0)
    assert main(["unet", "scan.AIM", "--output", "results", "--device", "mps"]) == 0
    assert seen == [["scan.AIM", "--output", "results", "--device", "mps"]]
