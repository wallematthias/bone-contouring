# Changelog

## 0.2.0 — 2026-10-01

- Replace standard contouring across XCTI/XCTII radius, tibia, and knee with the shared topology-first implementation. Preserve site-specific density and tissue-segmentation settings; no legacy standard selector remains.
- Remove opening before envelope filling and forced cortical peeling; fill axial envelope holes before and after modest physical signed-distance regularization.
- Keep tissue segmentation independent of contour-envelope support, including native-unit Laplace–Hamming inputs.
- Record the standard algorithm revision, endosteal footprint, smoothing parameters, and advisory contour QA in provenance and settings hashes.
- Add experimental Buie, IPL-matching, and stable-3D stages, with read-only comparison benchmarks and regression tests.
- Handle short stacks using a physical-unit discrete Gaussian when recursive filtering cannot process the input dimensions.

Migration: `standard` intentionally changes generated masks. Existing saved masks are not regenerated automatically. Native-mask comparisons cover XCTII radius/tibia only; XCTI/knee native accuracy and multi-bone handling are not established. Largest-component selection assumes one target bone. This is not a literal Buie implementation or a voxel-identical Scanco/IPL replacement.
