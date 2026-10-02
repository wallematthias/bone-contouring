# Approved: one GPL scientific package

Bundle the tested, fixed-default Nathan Neeteson et al. U-Net inference runtime
under `bone_contouring.unet`, retaining the published model and morphology.
Use GPL-3.0-only for bone-contouring and the original inference draft, preserving
upstream attribution and the existing standard contouring interface.

1. Add regression tests for the bundled API, optional dependencies, and interrupted
   publication. Retain frozen upstream scientific oracles in test fixtures.
2. Copy the minimal inference runtime, add the optional `unet` extra and CLI,
   and publish complete native AIM files exclusively and atomically, marker last.
3. Make shared manifest replacement atomic; require a complete U-Net case before
   downstream discovery. Keep standard contour discovery unchanged.
4. Point Slicer scene, batch and Setup at bone-contouring; report manifest failures
   honestly and allow manifest-only recovery without rerunning inference.
5. Verify base and optional tests, scientific parity, native handoff and wheel
   metadata; review, commit and merge locally. No push, tag or PyPI upload.

Weights download at module first use to the existing shared cache, never in Setup
or distribution artifacts. Device selection is the only inference control.
