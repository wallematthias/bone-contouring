"""Default morphology from Nathan Neeteson's utils/postprocessing.py (GPL).

All arrays here are XYZ, as in the original implementation. Scientific
defaults are internal, not user-tunable. VTK/plotting alternatives are omitted.
"""
import numpy as np
from scipy.ndimage import generate_binary_structure
from skimage.filters import median
from skimage.measure import label
from skimage.morphology import binary_dilation, binary_erosion


_FOOTPRINT = generate_binary_structure(3, 1)


def _erode(mask, n):
    # A footprint sequence sends all iterations to SciPy in one call, allowing
    # its changed-voxel tracking instead of allocating/scanning the volume in
    # Python n times. Keep skimage's default erosion border=True (mode=ignore).
    # Never send zero iterations: SciPy interprets that as until-convergence.
    return binary_erosion(mask, footprint=[(_FOOTPRINT, n)]) if n else mask


def _dilate(mask, n):
    # Same cross-shaped 3D neighbourhood and default border=False as the
    # published repeated single-step dilation, including unpadded Z faces.
    return binary_dilation(mask, footprint=[(_FOOTPRINT, n)]) if n else mask


def _largest(mask, background=False):
    labels = label(~mask if background else mask, background=0)
    counts = np.bincount(labels.ravel())[1:]
    # Degenerate intermediate erosion can remove everything; do not interpret
    # a zero label as foreground or silently produce a full-volume mask.
    kept = labels == (int(np.argmax(counts)) + 1) if counts.size else np.zeros_like(mask)
    return ~kept if background else kept


def _islands(mask, n):
    result = np.pad(mask, ((1,1),(1,1),(0,0)))
    result = _erode(result, n)
    result = _largest(result)
    result = _dilate(result, n)
    return result[1:-1,1:-1,:]


def _gaps(mask, n):
    pad = 2 * n if n else 1
    result = np.pad(mask, ((pad,pad),(pad,pad),(0,0)))
    result = _dilate(result, n)
    result = _largest(result, background=True)
    result = _erode(result, n)
    return _largest(result[pad:-pad,pad:-pad,:], background=True)


def _iterative(mask):
    for n in range(6):
        mask = _gaps(_islands(mask, n), n)
    return _gaps(mask, 15)


def postprocess(image, cort, trab, *, progress=None):
    report = progress or (lambda message: None)
    report("Post-processing 1/4: refining trabecular compartment...")
    trab = _iterative(trab)
    report("Post-processing 2/4: expanding trabecular support...")
    expanded = _dilate(trab, 8)
    report("Post-processing 3/4: extracting bone support...")
    bone = median(image >= 0, footprint=np.ones((3,3,1)))
    bone = _gaps(_islands(bone, 3), 15)
    report("Post-processing 4/4: refining full envelope...")
    full = _iterative(trab | cort | expanded | bone)
    return full & ~trab, trab
