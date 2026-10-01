"""Read-only comparison with genuine native FULL_MASK/CORT_MASK pairs.

Requires py_aimio (optional benchmark dependency). Runs each case in a fresh
process so peak RSS is not inherited from previous scans. Never writes scans or
existing derivatives; calls the image API directly, bypassing batch mask reuse.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

import numpy as np
from scipy import ndimage as ndi
import SimpleITK as sitk

from bone_contouring import generate_masks_from_image, resolve_preset


def read(path, *, density=False):
    import py_aimio
    array, metadata = py_aimio.read_aim(str(path), density=density)
    image = sitk.GetImageFromArray(array.astype(np.float32) if density else (array != 0).astype(np.uint8))
    image.SetSpacing(metadata['spacing'])
    image.SetOrigin(metadata['origin'])
    image.SetDirection(metadata.get('direction', image.GetDirection()))
    return image, metadata


def aligned(image, reference):
    return sitk.GetArrayFromImage(sitk.Resample(
        image, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)).astype(bool)


def dice(a, b):
    denominator = int(a.sum()) + int(b.sum())
    return 2 * int((a & b).sum()) / denominator if denominator else 1.


def surface_distances(a, b, spacing):
    distances = []
    for left, right in zip(a, b):
        le = left & ~ndi.binary_erosion(left)
        re = right & ~ndi.binary_erosion(right)
        if not le.any() or not re.any():
            return {'mean_mm': None, 'p95_mm': None}
        distances.extend([ndi.distance_transform_edt(~le, sampling=spacing)[re],
                          ndi.distance_transform_edt(~re, sampling=spacing)[le]])
    d = np.concatenate(distances)
    return {'mean_mm': float(d.mean()), 'p95_mm': float(np.percentile(d, 95))}


def adjacent_surface_distances(mask, spacing):
    """Axial envelope movement; includes anatomy/motion, not just roughness."""
    distances, pair_means = [], []
    previous = previous_dt = None
    for layer in mask:
        filled = ndi.binary_fill_holes(layer)
        edge = filled & ~ndi.binary_erosion(filled)
        dt = ndi.distance_transform_edt(~edge, sampling=spacing) if edge.any() else None
        if previous is not None and previous.any() and edge.any():
            d = np.concatenate((previous_dt[edge], dt[previous]))
            distances.append(d)
            pair_means.append(float(d.mean()))
        previous, previous_dt = edge, dt
    if not distances:
        return {'mean_mm': None, 'p95_mm': None, 'max_pair_mean_mm': None,
                'omitted_pairs': max(0, len(mask)-1)}
    d = np.concatenate(distances)
    return {'mean_mm': float(d.mean()), 'p95_mm': float(np.percentile(d, 95)),
            'max_pair_mean_mm': max(pair_means), 'omitted_pairs': len(mask)-1-len(distances)}


def metrics(full, trab, cort, nf, nt, nc, spacing):
    result = {'full_dice': dice(full, nf), 'trab_dice': dice(trab, nt), 'cort_dice': dice(cort, nc),
              'cort_volume_bias_pct': 100 * (int(cort.sum()) / int(nc.sum()) - 1),
              'full_volume_bias_pct': 100 * (int(full.sum()) / int(nf.sum()) - 1),
              'partition_gap_voxels': int((full & ~(trab | cort)).sum()),
              'partition_overlap_voxels': int((trab & cort).sum()),
              'compartments_outside_full_voxels': int(((trab | cort) & ~full).sum()),
              'full_holes_xy_voxels': sum(int((ndi.binary_fill_holes(s) & ~s).sum()) for s in full),
              'trab_holes_xy_voxels': sum(int((ndi.binary_fill_holes(s) & ~s).sum()) for s in trab)}
    for name, left, right in [('outer', nf, full), ('endosteal', nt, trab)]:
        result.update({name + '_' + key: value for key, value in surface_distances(left, right, spacing).items()})
    for name, mask in [('outer', full), ('endosteal', trab), ('native_outer', nf), ('native_endosteal', nt)]:
        result.update({name + '_adjacent_' + key: value for key, value in adjacent_surface_distances(mask, spacing).items()})
    return result


def pairs(root):
    records = json.loads((root / 'derivatives/ImportedContours/manifest.json').read_text())['records']
    found = {}
    for record in records:
        role = record.get('metadata', {}).get('source_label')
        if role in {'FULL_MASK', 'CORT_MASK'}:
            found.setdefault((record['subject_id'], record['session_id']), {})[role] = record['path']
    return {key: pair for key, pair in sorted(found.items()) if set(pair) == {'FULL_MASK', 'CORT_MASK'}}


def run_case(root, key, pair, args):
    subject, session = key
    case = Path(*Path(pair['FULL_MASK']).parts[-4:-1])
    filename = f'{case.parts[0]}_{case.parts[1]}_voi-tibia'
    saved_path = root / str(pair['FULL_MASK']).replace('/ImportedContours/', '/BoneContours/')
    reference, _ = read(saved_path)
    nf_image, fm = read(root / pair['FULL_MASK'])
    nc_image, cm = read(root / pair['CORT_MASK'])
    for meta in (fm, cm):
        log = meta.get('processing_log_raw', '')
        if 'GobjCreateAimPeel' not in log or 'FFT' in log:
            raise ValueError(f'{subject}: unverified native compartment reference')
    nf, nc = aligned(nf_image, reference), aligned(nc_image, reference)
    if (nc & ~nf).any():
        raise ValueError(f'{subject}: native cortex outside native full mask')
    nt = nf & ~nc
    image, metadata = read(root / case / (filename + '_xct.AIM'), density=True)
    params = resolve_preset(modality='xct2', site='tibia', segmentation='gauss',
                            outer_contour=args.method, inner_contour=args.method)
    if args.outer_threshold is not None:
        params.outer.periosteal_threshold = args.outer_threshold
    if args.inner_threshold is not None:
        params.inner.endosteal_threshold = args.inner_threshold
    params.segmentation.enabled = False
    start = perf_counter()
    generated = generate_masks_from_image(image, params)
    seconds = perf_counter() - start
    # macOS ru_maxrss is bytes, Linux KiB; includes IO + image API, not just filters.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == 'darwin' else 1024)
    full, trab, cort = (aligned(im, reference) for im in (generated.full, generated.trab, generated.cort))
    spacing_yx = reference.GetSpacing()[:2][::-1]
    compared = metrics(full, trab, cort, nf, nt, nc, spacing_yx)
    old = [sitk.GetArrayFromImage(reference).astype(bool)]
    for role in ('trab', 'cort'):
        im, _ = read(saved_path.with_name(saved_path.name.replace('desc-full', f'desc-{role}')))
        old.append(aligned(im, reference))
    baseline = metrics(*old, nf, nt, nc, spacing_yx)
    provenance = json.loads(Path(str(saved_path) + '.json').read_text())
    return {'subject': subject, 'session': session, 'shape_xyz': list(image.GetSize()),
            'spacing_xyz_mm': metadata['spacing'], 'seconds_image_api': seconds,
            'peak_rss_mib_through_image_api': peak, 'parameters': asdict(params),
            'baseline_parameters': provenance['parameters'], args.method: compared, 'saved_standard': baseline,
            'algorithm_metadata': generated.metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--aimio-path', type=Path)
    parser.add_argument('--case', help='Manifest worker selector, e.g. SAMPLE526/004')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--method', choices=('standard', 'buie', 'ipl_match', 'stable_3d'), default='buie')
    parser.add_argument('--outer-threshold', type=float, help='Override the selected preset threshold')
    parser.add_argument('--inner-threshold', type=float, help='Override the selected preset threshold')
    args = parser.parse_args()
    if args.aimio_path:
        sys.path.insert(0, str(args.aimio_path))
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    included = pairs(args.dataset_root)
    if args.case:
        key = tuple(args.case.split('/'))
        print(json.dumps(run_case(args.dataset_root, key, included[key], args)), flush=True)
        return
    if args.output_dir is None:
        parser.error('--output-dir is required except in worker mode')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for key in list(included)[:args.limit]:
        command = [sys.executable, str(Path(__file__).resolve()), '--dataset-root', str(args.dataset_root),
                   '--case', '/'.join(key), '--method', args.method]
        if args.outer_threshold is not None:
            command += ['--outer-threshold', str(args.outer_threshold)]
        if args.inner_threshold is not None:
            command += ['--inner-threshold', str(args.inner_threshold)]
        if args.aimio_path:
            command += ['--aimio-path', str(args.aimio_path)]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"Case {'/'.join(key)} failed:\n{completed.stderr}")
        row = json.loads(completed.stdout)
        results.append(row)
        print(json.dumps({k: row[k] for k in ('subject', 'seconds_image_api', 'peak_rss_mib_through_image_api', args.method)}), flush=True)
        (args.output_dir / f'{args.method}_benchmark.json').write_text(json.dumps(results, indent=2))
    flattened = []
    for r in results:
        flat = {k: r[k] for k in ('subject', 'session', 'seconds_image_api', 'peak_rss_mib_through_image_api')}
        for method in (args.method, 'saved_standard'):
            flat.update({method + '_' + k: v for k, v in r[method].items()})
        flattened.append(flat)
    with (args.output_dir / f'{args.method}_benchmark.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)


if __name__ == '__main__':
    main()
