"""Published radius/tibia U-Net inference (GPL-3.0-only)."""
from .. import __version__


def __getattr__(name):
    if name == "Segmenter":
        from .inference import Segmenter
        return Segmenter
    raise AttributeError(name)
