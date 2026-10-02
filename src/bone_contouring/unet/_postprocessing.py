"""Default morphology from Nathan Neeteson's utils/postprocessing.py (GPL).

All arrays here are XYZ, as in the original implementation. Scientific
defaults are internal, not user-tunable. VTK/plotting alternatives are omitted.
"""
import numpy as np
from skimage.filters import median
from skimage.measure import label
from skimage.morphology import binary_dilation, binary_erosion


def _largest(mask, background=False):
    labels = label(~mask if background else mask, background=0)
    counts = np.bincount(labels.ravel())[1:]
    # Degenerate intermediate erosion can remove everything; do not interpret
    # a zero label as foreground or silently produce a full-volume mask.
    kept = labels == (int(np.argmax(counts)) + 1) if counts.size else np.zeros_like(mask)
    return ~kept if background else kept


def _islands(mask, n):
    result = np.pad(mask, ((1,1),(1,1),(0,0)))
    for _ in range(n):
        result = binary_erosion(result)
    result = _largest(result)
    for _ in range(n):
        result = binary_dilation(result)
    return result[1:-1,1:-1,:]


def _gaps(mask, n):
    pad = 2 * n if n else 1
    result = np.pad(mask, ((pad,pad),(pad,pad),(0,0)))
    for _ in range(n):
        result = binary_dilation(result)
    result = _largest(result, background=True)
    for _ in range(n):
        result = binary_erosion(result)
    return _largest(result[pad:-pad,pad:-pad,:], background=True)


def _iterative(mask):
    for n in range(6):
        mask = _gaps(_islands(mask, n), n)
    return _gaps(mask, 15)


def postprocess(image, cort, trab):
    trab = _iterative(trab)
    expanded = trab.copy()
    for _ in range(8):
        expanded = binary_dilation(expanded)
    bone = median(image >= 0, footprint=np.ones((3,3,1)))
    bone = _gaps(_islands(bone, 3), 15)
    full = _iterative(trab | cort | expanded | bone)
    return full & ~trab, trab
