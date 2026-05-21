# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (Layer 3 — PromptStore validators)

- `sam_manager/core/prompt_store.py` (NEW) — module-level `validate_references(refs, masks)` + typed exceptions `ReferenceCountMismatchError(STATUS_CODE = 7)` and `ReferenceSizeMismatchError(STATUS_CODE = 6)`. Per architecture v4 §4 + drawio Page 4, Layer 3 PromptStore emits status 6 REFERENCE_SIZE_MISMATCH (refs / masks shape / dtype / pair alignment) and status 7 REFERENCE_COUNT_MISMATCH (`len(refs) != len(masks)`). Both exceptions subclass `ValueError` so Layer 4 `ErrorHandlingBackend` propagates them unchanged (mirroring `InvalidImageError` pattern).
- `MockBackend._validate` now reduces to two delegation calls — `validate_target(target)` (Layer 1) + `validate_references(refs, masks)` (Layer 3). Inline refs/masks shape/dtype/pair-alignment loop deleted (was ~15 lines). Future SegGPTBackend wrapper does not need to re-implement the same checks.
- `test/test_prompt_store.py` (NEW) — 26 unit cases covering exception contracts (STATUS_CODE / ValueError subclass), happy path (empty / single / multi-pair), count mismatch (status 7), ref shape+dtype rejection, mask shape+dtype rejection, pair alignment, iteration order (first-bad-wins; count check before per-pair). `status 5 PROMPT_NOT_FOUND` deferred until the full PromptStore class lands with named / inline mode lookup.

### Changed (Layer 4 test infra — boom_class fixture)

- `test/conftest.py` (NEW) exposes a `boom_class` pytest fixture returning a deterministic error-raising `BackendInterface`. Promotes the previously-inline `class BoomBackend(MockBackend)` definitions in `test_error_handler.py` to a shared fixture so adding a new exception type to the wrapper coverage matrix no longer requires redefining the backend per case.
- `test_error_handler.py` parametrizes non-ValueError exception coverage across `RuntimeError / KeyError / OSError / ZeroDivisionError / ConnectionError` (5 types x 2 invariants = 10 pytest instances). Two prior inline-class single-case tests (`test_runtime_error_wrapped_as_backend_error`, `test_generic_exception_wrapped_as_backend_error`) deleted — fully subsumed. Message preservation + original-instance invariants stay covered by the surrounding `BackendError` constant / state tests.
- Net: ErrorHandlingBackend Decorator seam moves from a 2-case spot-check to a 5-type matrix while shrinking the test module (`-20` lines after the fixture migration).

### Added (Layer 1 — RequestValidator)

- `sam_manager/core/request_validator.py` — `validate_target(target)` module-level function + `InvalidImageError(ValueError)` exception with `STATUS_CODE = 3`. Per architecture v4 §4 + drawio Page 4 status_code emit table, Layer 1 RequestValidator owns status_code 3 INVALID_IMAGE: rejects target images that fail the (H, W, 3) uint8 RGB non-empty contract. ROS-agnostic core — the 4 Frontend Adapters (ROS 2 Service / Topic / FastAPI / Python API) will all call `validate_target` and map `InvalidImageError` to their wire representation. `InvalidImageError` subclasses `ValueError` so the Layer 4 `ErrorHandlingBackend` `except ValueError: raise` branch propagates it unchanged rather than remapping to status_code 11 BACKEND_ERROR.
- `MockBackend._validate` now delegates target shape / dtype / non-empty checks to `validate_target`. Refs / masks pair-alignment + count checks remain in MockBackend pending the Layer 3 PromptStore PR that owns status 6 / 7 emit.
- `test/test_request_validator.py` — 18 unit cases covering: STATUS_CODE / ValueError subclass / message carriage; happy path (valid RGB uint8 + 1×1 edge case); shape rejection (2D grayscale, RGBA, pseudo-3D single channel, 1D, 4D); dtype rejection (float32, uint16, bool); empty rejection (zero height / width / both); error diagnostics (shape + dtype surfaced in message).

### Changed (Layer 4 — ROS-agnostic core seam)

- Moved Layer 4 backend modules into a `core/` subpackage to establish the ROS-agnostic seam ahead of Frontend Adapter PRs (ROS 2 Service handler / FastAPI handler will live in `sam_manager.adapters.<frontend>`, not in `core/`):
  - `sam_manager.backend` → `sam_manager.core.backend`
  - `sam_manager.error_handler` → `sam_manager.core.error_handler`
  - `sam_manager.backends.mock` → `sam_manager.core.backends.mock`
  - structlog logger names updated to match new paths (`sam_manager.core.error_handler`, `sam_manager.core.backends.mock`).
- Pure refactor — no public behavior change beyond the import-path rename.

### Added

- Initial repository scaffold: `README.md` (English) + `doc/README.{zh-TW,zh-CN,ja}.md` (4-language sync), `LICENSE` (Apache-2.0, aligned with `ycpss91255-docker/base`), `.gitignore`, and this `CHANGELOG.md`.
- `sam_manager_msgs` ROS 2 interface package with the wire contract for the segmentation service:
  - `msg/MaskRLE.msg` — COCO-style `counts` + `size [H,W]`, no category label (sam_manager does not classify).
  - `srv/SegmentFromReference.srv` — request carries `Header` + target `Image` + `camera_tf` (caller-supplied, no tf2 lookup) + either `reference_name` (named mode) or `reference_images[]` + `reference_masks[]` (inline mode). Response carries `Header` + `MaskRLE` + `RegionOfInterest` bbox + `confidence` + `has_mask` + `has_bbox` + `status_code` (0..11, reserved numbering) + `error_message`.
  - 12 status codes documented inline in the .srv comment and again in `README.md`. Emit-site mapping (Layer 1 RequestValidator / Layer 2 QueueCore + TimeoutMonitor / Layer 3 PromptStore / Layer 4 ErrorHandler / Layer 5 ConfidenceGate) preserved from the CLAUDE.md `ros2-msg-design` skill spec.
  - Fields intentionally absent from the wire: `capability_level`, `backend_name`, `inference_latency_ms` (startup config / metrics only — capability degradation is expressed implicitly through `has_mask` + `has_bbox`); `depth`, `class_label`, `category`, `text_prompt`, `mm_per_pixel`, `distance`, `angle`, `aligned` (application-layer responsibilities).
- `package.xml` + `CMakeLists.txt` for `sam_manager_msgs` (ament_cmake + rosidl_default_generators; depends on std_msgs, sensor_msgs, geometry_msgs).

### Added (Layer 4 — backend abstraction)

- `src/sam_manager/` ROS 2 `ament_python` package with the Layer 4 backend abstraction:
  - `sam_manager/backend.py` — `BackendInterface` ABC + `InferResult` dataclass. `infer()` is the only abstract method; `warmup()` and `health_check()` ship with safe defaults (no-op / always healthy) so trivial backends do not need to override them.
  - `sam_manager/backends/mock.py` — `MockBackend`, a deterministic synthetic backend that returns a centered rectangular mask covering a configurable fraction (`mask_coverage`, default 0.25) of the target. Used by downstream Layer 1 / 2 / 5 tests so they need neither GPU nor a real `seggpt` install. Validates input shape / dtype and `len(refs) == len(masks)`, raising `ValueError` on mismatch (mirroring `seggpt.api.SegGPTBackend.infer()` line 97-100).
  - `InferResult` carries `target` by reference (no deep copy) so downstream layers can pair masks with the input image for local storage / log metadata without paying the cost of duplicating a multi-MB image per call. `latency_ms` and `gpu_mem_mb` are optional telemetry fields the backend self-reports (`None` for Mock).
- Structured logging via `structlog` with OTel-aligned event names (`backend_infer_started`, `backend_infer_completed`). The full structlog pipeline configuration (JSON renderer / sinks / trace_id binding) lands in the Layer 1 PR; this PR just emits the events through the default logger.
- pytest test suite (`test/test_backend.py` 5 cases + `test/test_mock.py` 11 cases) + ament_copyright / ament_flake8 / ament_pep257 lint stubs (the copyright stub is `@pytest.mark.skip`'d, mirroring `core_sam_bridge` until a per-file header policy is decided).

### Added (Layer 4 backfill — strict TDD, this PR)

- `sam_manager/error_handler.py` — `BackendError` (status_code=11 class constant) + `ErrorHandlingBackend(BackendInterface)` decorator. Per CLAUDE.md / `ros2-msg-design` SKILL.md status_code emit table, Layer 4 ErrorHandler maps backend runtime failures (GPU OOM, CUDA error, model load) to wire `status_code 11 BACKEND_ERROR`. `ValueError` from inner backend propagates unchanged so the upstream caller (Layer 1 RequestValidator) emits the appropriate code from 3 / 4 / 6 / 7 / 8. Implementation: Decorator pattern around `BackendInterface`; `warmup()` + `health_check()` forward to inner; emits `backend_infer_failed` structlog event before re-raising. 11 unit tests.
- `MockBackend._validate` now asserts **pair alignment** — `refs[i].shape[:2] == masks[i].shape` — mirroring the seggpt Layer 2 contract and catching mismatched (image, mask) pairs that previously slipped through. Mirrors the kind of validation we want SegGPTBackend to inherit when it lands.
- `MockBackend` geometry is now aspect-ratio-preserving: previously `min(H, W)` was used for both rectangle sides, biasing non-square targets (800×600 with `coverage=0.25` produced 18.75%). The new formula scales `H` and `W` independently by `sqrt(coverage)`, so coverage matches the parameter regardless of aspect ratio. Square targets unaffected.
- Test suite expanded to 47 unit cases (was 19) with 100% line coverage across `backend.py` + `backends/mock.py` + `error_handler.py`:
  - `test_mock.py` 11 → 29 cases: input validation (pair alignment + RGBA + pseudo-3D + dtype variants), output invariants (`InferResult` instance, mask dtype, binary-values-only, centered geometry, exact class_ids, target buffer untouched), boundary geometry (coverage 0.0 / 1.0 / non-square), structlog OTel emit (events + attributes + no emit on early raise).
  - `test_backend.py` 5 → 7 cases: ABC override of `warmup()` / `health_check()`.
  - `test_error_handler.py` NEW 11 cases.
- `.github/workflows/main.yaml` gains `pytest coverage (>= 80%)` step running `pytest --cov=sam_manager --cov-report=term-missing --cov-fail-under=80`. CI installs `pytest-cov` via pip alongside structlog.
- `doc/test/TEST.md` — single source of truth for test counts + 4-category coverage table (Smoke / Unit / Integration / Lint per CLAUDE.md「TDD 測試分類」).
- `setup.py` `extras_require['test']` adds `pytest-cov`.

### Added (CI)

- `.github/workflows/main.yaml` — single `build` job runs `colcon build --packages-select sam_manager_msgs` inside `ros:humble-ros-base` on `ubuntu-latest`, then `ros2 interface show` smoke-checks `srv/SegmentFromReference` and `msg/MaskRLE`. Job name `build` is kept stable so `main` branch protection can require it by name (added via `gh api PATCH .../branches/main/protection` after the workflow's first green run). Workflow triggers on push to `main`, tag push (`v*`), pull requests, and manual dispatch.

### Notes

- Package naming: the upstream CoreSAM design docs (`coreSAM_ws/CLAUDE.md`, `.claude/skills/ros2-msg-design/SKILL.md`) use the placeholder name `coresam_msgs`; the actual package ships as `sam_manager_msgs` to reflect the "manages SAM-family backends" role. The harness CLAUDE.md will be updated in a follow-up to match.
- Topic-pair messages (`SegmentRequest.msg` / `SegmentResult.msg` for the streaming `/coresam/segment_request` + `/coresam/segment_result` topics) are deferred to a follow-up PR — the srv is sufficient to lock the wire contract.
- Docker scaffold (`docker/.base/` subtree of `ycpss91255-docker/base`, mirroring `ycpss91255-docker/seggpt`) is deferred to a follow-up PR. Build / lint for this PR uses CI-side ROS 2 humble runners directly.
