"""Headless AIM file I/O and batch planning using aimio-py only."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import py_aimio

from . import __version__
from .weights import MODEL_LABEL, WEIGHTS_SHA256


def image_stem(path):
    match = re.fullmatch(r"(.+)\.aim(?:;\d+)?", Path(path).name, re.IGNORECASE)
    if not match or re.search(r"(?:_MASK|_SEG|_LABELS?)$", match[1], re.IGNORECASE):
        raise ValueError(f"Expected an AIM image, not a mask or SEG: {path}")
    return match[1]


def output_paths(output, stem):
    root = Path(output)
    prefix = re.sub(r"_xct$", "", stem, flags=re.IGNORECASE)
    paths = {role: root/f"{prefix}_desc-{role}_mask.AIM" for role in ("full", "trab", "cort")}
    paths.update({f"{role}_sidecar": path.with_suffix(".AIM.json") for role,path in list(paths.items())})
    paths["provenance"] = root/f"{prefix}_UNET.json"
    return paths


@dataclass(frozen=True)
class BatchJob:
    source: Path
    output: Path


def plan_batch(inputs, output):
    """Plan everything before reading scans/loading weights; never overwrite."""
    output = Path(output).expanduser().resolve()
    jobs = []
    input_roots = []
    for item in inputs:
        path = Path(item).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            input_roots.append(path)
            for candidate in sorted(path.rglob("*")):
                if not candidate.is_file() or "derivatives" in candidate.relative_to(path).parts:
                    continue
                try:
                    image_stem(candidate)
                except ValueError:
                    continue
                jobs.append(BatchJob(candidate.resolve(), output/candidate.relative_to(path).parent))
        else:
            image_stem(path)
            input_roots.append(path.parent)
            jobs.append(BatchJob(path, output))
    if not jobs:
        raise ValueError("No AIM image inputs were found.")
    jobs = [BatchJob(job.source, job.output.resolve()) for job in jobs]
    raw_roots = [*input_roots, *(job.source.parent for job in jobs)]
    for job in jobs:
        for root in raw_roots:
            if job.output == root or root in job.output.parents:
                raise ValueError("Output must be outside input/raw directories.")
    seen = set()
    for job in jobs:
        for target in output_paths(job.output, image_stem(job.source)).values():
            if target in seen:
                raise ValueError(f"Output names collide: {target}")
            seen.add(target)
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite existing output: {target}")
    return jobs


def write_masks(output, stem, masks, metadata, *, source, device):
    output = Path(output)
    paths = output_paths(output, stem)
    for target in paths.values():
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {target}")
    cort, trab, full = (np.asarray(masks[role]) for role in ("cort","trab","full"))
    if cort.ndim != 3 or cort.shape != trab.shape or cort.shape != full.shape:
        raise ValueError("Compartment masks must have the same 3D shape.")
    if any(not np.isin(m, (0,1)).all() or not m.any() for m in (cort,trab,full)):
        raise ValueError("Expected nonempty binary masks.")
    if np.any(cort & trab) or not np.array_equal(full, cort | trab):
        raise ValueError("Full must equal disjoint cort/trab union.")
    output.mkdir(parents=True, exist_ok=True)
    meta = dict(metadata)
    meta["unit"] = "native"
    log = str(meta.get("processing_log_raw") or meta.get("processing_log") or "")
    meta["processing_log"] = log + f"\nbone-contouring {__version__}; unet; {MODEL_LABEL}; device={device}; binary compartment mask\n"
    with tempfile.TemporaryDirectory(prefix=".unet-", dir=output) as staging:
        for role in ("full","trab","cort"):
            py_aimio.write_aim(str(Path(staging)/paths[role].name),
                              (masks[role]*127).astype(np.int8), meta, unit="native")
        provenance = dict(version=__version__, method="unet", model=MODEL_LABEL, weights_sha256=WEIGHTS_SHA256,
                          source=str(Path(source).resolve()), device=str(device),
                          masks={role:paths[role].name for role in ("full","trab","cort")},
                          postprocessing="Published defaults; no additional toolbox smoothing/peel")
        geometry = {key: list(metadata[key]) for key in ("position", "offset", "element_size") if key in metadata}
        geometry["dimensions"] = list(full.shape[::-1])
        provenance["source_geometry"] = geometry
        for role in ("full", "trab", "cort"):
            sidecar = dict(provenance, short_role=role, content_type="mask",
                           role={"full":"periosteal_mask", "trab":"trabecular_mask", "cort":"cortical_mask"}[role],
                           software={"name":"bone-contouring", "version":__version__})
            (Path(staging)/paths[f"{role}_sidecar"].name).write_text(json.dumps(sidecar, indent=2)+"\n")
        (Path(staging)/paths["provenance"].name).write_text(json.dumps(provenance, indent=2)+"\n")
        # Identify U-Net artifacts before any mask becomes visible. Exclusive
        # same-filesystem links publish complete files atomically; marker last.
        roles = [*(f"{role}_sidecar" for role in ("full", "trab", "cort")),
                 "full", "trab", "cort", "provenance"]
        for role in roles:
            temporary = Path(staging)/paths[role].name
            # Windows fsync/_commit requires a writable handle. Do not truncate
            # the completed staging file before publishing its exclusive link.
            with temporary.open("r+b") as stream:
                os.fsync(stream.fileno())
            os.link(temporary, paths[role])
    return paths


def run_batch(inputs, output, device="auto"):
    from .inference import Segmenter
    jobs = plan_batch(inputs, output)
    segmenter = Segmenter(device)
    results = []
    for index, job in enumerate(jobs, 1):
        print(f"Case {index}/{len(jobs)}: {job.source}", flush=True)
        density, meta = py_aimio.read_aim(str(job.source), density=True)
        spacing = tuple(meta.get("element_size", ()))
        if spacing and not np.allclose(spacing, (.061,.061,.061), atol=.0005):
            print(f"WARNING: voxel size {spacing}; published weights target 0.061 mm radius/tibia.", flush=True)
        masks = segmenter.segment(density, progress=lambda message: print(message, flush=True))
        paths = write_masks(job.output, image_stem(job.source), masks, meta,
                            source=job.source, device=segmenter.device)
        print(f"Completed: {paths['provenance']}", flush=True)
        results.append(paths)
    return results
