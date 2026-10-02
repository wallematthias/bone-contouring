"""One published model, verified cache, and offline weight override."""
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

MODEL_LABEL = "radius_tibia_final"
WEIGHTS_URL = "https://zenodo.org/api/records/14755838/files/radius_tibia_final.pth/content"
WEIGHTS_SHA256 = "62765161705a6b366c994ce07c5419531d6ca6cee8c09437a550279f774440a2"


def _verify(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()
    if digest != WEIGHTS_SHA256:
        raise ValueError(f"Model checksum does not match the published radius/tibia weights: {path}")
    return path


def weights_path():
    """Download once or reuse verified weights; env override supports offline SSH."""
    override = os.environ.get("HRPQCT_SEGMENTATION_WEIGHTS")
    if override:
        return _verify(Path(override).expanduser().resolve())
    model_dir = os.environ.get("HRPQCT_SEGMENTATION_MODEL_DIR")
    cache = (Path(model_dir).expanduser().resolve() if model_dir else
             Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "hrpqct-segmentation")
    destination = cache / f"{MODEL_LABEL}.pth"
    if destination.exists():
        return _verify(destination)
    cache.mkdir(parents=True, exist_ok=True)
    print("Downloading published radius/tibia weights (cached for offline reuse)...", flush=True)
    with tempfile.NamedTemporaryFile(dir=cache, suffix=".pth", delete=False) as temp:
        temporary = Path(temp.name)
        try:
            with urlopen(WEIGHTS_URL, timeout=60) as source:
                while chunk := source.read(1024 * 1024):
                    temp.write(chunk)
            temp.close()
            _verify(temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination
