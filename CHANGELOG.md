# Changelog

## 0.3.0 — Unreleased

- Add fixed-default published HR-pQCT U-Net inference as optional `bone-contouring[unet]`, with CPU/CUDA/MPS, verified weights, native AIM batch CLI and a scene worker. Preserve Neeteson et al.'s model/morphology and attribution; no Bonelab/vtkbone dependency.
- Publish complete AIM files exclusively and atomically, sidecars first and completion marker last. Depend on bone-imaging-derivatives 0.1.6 for completion-gated discovery.
- Change distribution license to GPL-3.0-only to include the GPL scientific backend; preserve the former MIT notice and prior grants. Standard contouring behavior is unchanged.

## 0.2.1 — 2026-10-01

- Restore the default 3-voxel axial minimum cortical compartment peel in standard/stable-3D contouring, applied after smoothing and hole filling. Preserve Z end slices and explicit `inner.peel=0` overrides.
- Record the effective peel radius and advance standard algorithm provenance/settings hashes to `topology_first_v2`.
- Add regressions for low-density full/trab collapse, all scanner/site defaults, peel overrides/validation, and empty peeled ROIs without fallback.

## 0.2.0 — 2026-10-01

- Replace standard contouring across XCTI/XCTII radius, tibia, and knee with the shared topology-first implementation. Preserve site-specific density and tissue-segmentation settings; no legacy standard selector remains.
- Remove opening before envelope filling and forced cortical peeling; fill axial envelope holes before and after modest physical signed-distance regularization.
- Keep tissue segmentation independent of contour-envelope support, including native-unit Laplace–Hamming inputs.
- Record the standard algorithm revision, endosteal footprint, smoothing parameters, and advisory contour QA in provenance and settings hashes.
- Add experimental Buie, IPL-matching, and stable-3D stages, with read-only comparison benchmarks and regression tests.
- Handle short stacks using a physical-unit discrete Gaussian when recursive filtering cannot process the input dimensions.

Migration: `standard` intentionally changes generated masks. Existing saved masks are not regenerated automatically. Native-mask comparisons cover XCTII radius/tibia only; XCTI/knee native accuracy and multi-bone handling are not established. Largest-component selection assumes one target bone. This is not a literal Buie implementation or a voxel-identical Scanco/IPL replacement.
