# Bone Contouring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a SimpleITK-first standalone package that generates full, trabecular, cortical, and bone segmentation masks.

**Architecture:** The public facade owns SimpleITK inputs/outputs and metadata. Parameter and preset modules provide an independent stable configuration surface; array algorithms use `x, y, z` NumPy arrays internally and are isolated from package orchestration.

**Tech Stack:** Python 3.11+, setuptools, NumPy, SciPy, SimpleITK, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-31-bone-contouring-design.md`

## Global Constraints

- Write only in this repository.
- Expose the v1 root API from the design document.
- Input/output images are three-dimensional SimpleITK images with preserved geometry.
- Runtime dependencies are NumPy, SciPy, and SimpleITK; geodesic contouring is optional.
- Use small synthetic test data and no network access.

---

### Task 1: Package Contract And Presets

**Files:**
- Create: `pyproject.toml`, `README.md`, `LICENSE`, `src/bone_contouring/__init__.py`, `src/bone_contouring/parameters.py`, `src/bone_contouring/presets.py`, `tests/test_presets.py`

**Interfaces:**
- Produces `ContourParameters`, nested parameter dataclasses, and `resolve_preset` / `load_preset`.

- [x] **Step 1: Write failing tests** for root exports and a composed `xct1/radius/laplace_hamming/geodesic/none` preset.
- [x] **Step 2: Run `pytest tests/test_presets.py -q`** and verify collection fails because the package is absent.
- [x] **Step 3: Implement minimal packaging, dataclasses, and composable presets.**
- [x] **Step 4: Run `pytest tests/test_presets.py -q`** and verify it passes.

### Task 2: Array Segmentation Core

**Files:**
- Create: `src/bone_contouring/laplace_hamming.py`, `src/bone_contouring/_arrays.py`, `tests/test_segmentation.py`

**Interfaces:**
- Consumes `SegmentationParameters`.
- Produces `segment_bone_xyz`, `adaptive_threshold_xyz`, and `laplace_hamming_binarize_xyz` boolean arrays.

- [x] **Step 1: Write failing tests** for component cleaning, full-mask restriction, Gaussian segmentation, and adaptive validation.
- [x] **Step 2: Run `pytest tests/test_segmentation.py -q`** and verify failure from missing functions.
- [x] **Step 3: Implement NumPy/SimpleITK conversion and segmentation helpers.**
- [x] **Step 4: Run `pytest tests/test_segmentation.py -q`** and verify it passes.

### Task 3: Contour And Public API

**Files:**
- Create: `src/bone_contouring/api.py`, `tests/test_api.py`
- Modify: `src/bone_contouring/_arrays.py`, `src/bone_contouring/__init__.py`

**Interfaces:**
- Consumes `SimpleITK.Image`, `ContourParameters`, and optional segmentation support image.
- Produces `GeneratedMasks`, `generate_masks_from_image`, and `generate_bone_segmentation`.

- [x] **Step 1: Write failing tests** for geometry, hole filling, `inner=none`, fallback metadata, and geodesic delegation.
- [x] **Step 2: Run `pytest tests/test_api.py -q`** and verify failure from missing public functions.
- [x] **Step 3: Implement contour generation and API facade.**
- [x] **Step 4: Run `pytest tests/test_api.py -q`** and verify it passes.

### Task 4: Quality Gate

**Files:**
- Modify as required by test or lint findings only.

- [x] **Step 1: Run `pytest -q`** and resolve any failures.
- [ ] **Step 2: Run `ruff check .`** and resolve lint findings. `ruff` is not installed locally.
- [ ] **Step 3: Run `python -m build`** and verify the distribution builds without errors. The local Python 3.13 environment has no installed `setuptools` backend; an isolated build would need to obtain the declared build dependency, which is outside the no-network constraint.
