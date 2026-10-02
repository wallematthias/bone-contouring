"""Read-only timing/parity check against the published morphology fixture.

An optional NPZ contains calibrated density and initial cort/trab arrays in ZYX
order. This benchmarks morphology only, not model inference or contour accuracy.
No scans, masks, or derivatives are written.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from skimage.filters import median
from skimage.measure import label
from skimage.morphology import binary_dilation, binary_erosion


ROOT = Path(__file__).resolve().parents[1]


def published_postprocess():
    # Load only the scientific functions, excluding legacy Bonelab/VTK imports.
    tree = ast.parse((ROOT / "tests/fixtures/unet_legacy/postprocessing.py").read_text())
    names = {"keep_largest_connected_component_skimage", "remove_islands_from_mask",
             "fill_in_gaps_in_mask", "dilate_and_subtract", "extract_bone",
             "iterative_filter", "postprocess_masks_iterative"}
    namespace = dict(np=np, binary_dilation=binary_dilation, binary_erosion=binary_erosion,
                     sklabel=label, median=median)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(functions, type_ignores=[]), "published_morphology", "exec"), namespace)
    return namespace["postprocess_masks_iterative"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="NPZ: density (mg HA/cm³), cort, trab in ZYX")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--slices", type=int, default=32)
    args = parser.parse_args(argv)
    if args.start < 0 or args.slices <= 0:
        parser.error("start must be nonnegative and slices must be positive")
    if args.snapshot:
        selection = slice(args.start, args.start + args.slices)
        with np.load(args.snapshot, allow_pickle=False) as snapshot:
            density = snapshot["density"][selection]
            if not density.size or not np.isfinite(density).all():
                parser.error("selected density must be nonempty and finite")
            image = ((2*np.clip(density.astype(np.float64), -400, 1400)-1000)/1800).transpose(2, 1, 0)
            cort = snapshot["cort"][selection].transpose(2, 1, 0).astype(bool)
            trab = snapshot["trab"][selection].transpose(2, 1, 0).astype(bool)
        if image.shape != cort.shape or image.shape != trab.shape:
            parser.error("density and masks must have the same shape")
    else:
        x, y, z = np.ogrid[:192, :192, :args.slices]
        trab = np.broadcast_to((x-95)**2 + (y-95)**2 < 55**2, (192, 192, args.slices)).copy()
        full = np.broadcast_to((x-95)**2 + (y-95)**2 < 65**2, trab.shape).copy()
        cort = full & ~trab
        image = np.where(cort, .8, -.6)
    spec = importlib.util.spec_from_file_location("optimized_postprocessing", ROOT / "src/bone_contouring/unet/_postprocessing.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(f"Timing published morphology on {image.shape} XYZ ({image.size:,} voxels)...", flush=True)
    started = perf_counter()
    expected = published_postprocess()(image, cort, trab)
    reference_seconds = perf_counter() - started
    print(f"Published: {reference_seconds:.3f} s; timing optimized implementation...", flush=True)
    started = perf_counter()
    actual = module.postprocess(image, cort, trab)
    optimized_seconds = perf_counter() - started
    for result, want in zip(actual, expected):
        np.testing.assert_array_equal(result, want)
    print(json.dumps(dict(shape_xyz=image.shape, reference_seconds=reference_seconds,
                          optimized_seconds=optimized_seconds,
                          speedup=reference_seconds/optimized_seconds, voxel_identical=True)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
