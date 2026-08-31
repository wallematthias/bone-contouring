# Bone Contouring Design

## Purpose

`bone-contouring` extracts the reusable mask-generation methods from the Timelapsed HR-pQCT workflow into a small standalone Python library. The package is SimpleITK-first: callers provide a three-dimensional `SimpleITK.Image` and receive geometry-preserving binary `SimpleITK.Image` masks.

## Public API

The `bone_contouring` package root will export:

- `ContourParameters`, `SegmentationParameters`, `OuterContourParameters`, and `InnerContourParameters` dataclasses.
- `GeneratedMasks`, containing `seg`, `full`, `trab`, `cort`, `mask_provenance`, and `metadata`.
- `generate_masks_from_image(image, parameters=None, *, segmentation_image=None)`.
- `generate_bone_segmentation(image, parameters=None, *, full_mask=None, trab_mask=None, cort_mask=None)`.
- `load_preset(...)` and `resolve_preset(...)` for composable modality, site, segmentation, outer-contour, and inner-contour choices.

`numpy_xyz` conversion helpers and array-level segmentation functions will live in a documented secondary module for tests and future integrations. They are not the primary v1 contract.

## Parameters And Presets

The parameter object nests outer, inner, and segmentation configuration. Presets use explicit dimensions:

- modality: `xct1` or `xct2`
- site: `radius`, `tibia`, or `knee`
- segmentation: `laplace_hamming`, `gauss`, or `adaptive`
- outer contour: `standard` or `geodesic`
- inner contour: `standard` or `none`

`resolve_preset` composes these dimensions over stable defaults and returns a fresh `ContourParameters`; user code can alter that result without affecting later calls. `load_preset` accepts a named preset and delegates to the same composition path.

## Algorithms And Safeguards

The implementation retains the Timelapsed methods that are portable without workflow/config dependencies:

- Standard outer contouring uses smoothing/thresholding, largest-component cleanup, closing/opening, terminal-slice restoration, and per-slice hole filling.
- Standard inner contouring derives trabecular and cortical masks inside `full`, including site-specific trabecular closing defaults.
- Segmentation supports Gaussian thresholds, adaptive thresholds, and Laplace-Hamming filtering with component cleaning.
- Outputs obey `seg <= full`, `trab <= full`, `cort <= full`, and avoid overlap between trabecular and cortical compartments.
- Metadata records selection and fallback decisions. If an endosteal result is implausible (empty trabecular mask, invalid partition, or excessively small trabecular fraction), the package falls back to `trab=full`, `cort=empty` and records the reason. An empty segmentation support image falls back to image-based contouring and records the event.

The optional geodesic implementation depends on `hrpqct-geodesic-contour`; selecting it without that extra installed raises a clear `RuntimeError`.

## Package Shape

- `src/bone_contouring/parameters.py`: public dataclasses and validation.
- `src/bone_contouring/presets.py`: composable defaults and preset resolution.
- `src/bone_contouring/laplace_hamming.py`: frequency-domain filter and binarizer.
- `src/bone_contouring/_arrays.py`: array/SimpleITK conversion, morphology, thresholds, and contour algorithms.
- `src/bone_contouring/api.py`: SimpleITK public entry points and output metadata.
- `src/bone_contouring/__init__.py`: stable exports only.

## Testing

Focused `pytest` tests use small synthetic cylindrical images. They verify package import/API exports, preset composability, SimpleITK geometry preservation, hole filling, component cleanup, segmentation methods, inner-contour none behavior, fallback metadata, and the optional geodesic adapter through a test double.

## Constraints

- Write only inside `/Users/matthias.walle/Documents/14_GitHub/active/bone-contouring`.
- Python 3.11+, NumPy, SciPy, and SimpleITK are required runtime dependencies.
- `hrpqct-geodesic-contour` is optional.
- Package metadata uses the MIT license.
- No network, push, or edits to TimelapsedHRpQCT or SlicerBoneImagingToolbox.
