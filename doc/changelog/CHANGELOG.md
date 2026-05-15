# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Added (CI)

- `.github/workflows/main.yaml` — single `build` job runs `colcon build --packages-select sam_manager_msgs` inside `ros:humble-ros-base` on `ubuntu-latest`, then `ros2 interface show` smoke-checks `srv/SegmentFromReference` and `msg/MaskRLE`. Job name `build` is kept stable so `main` branch protection can require it by name (added via `gh api PATCH .../branches/main/protection` after the workflow's first green run). Workflow triggers on push to `main`, tag push (`v*`), pull requests, and manual dispatch.

### Notes

- Package naming: the upstream CoreSAM design docs (`coreSAM_ws/CLAUDE.md`, `.claude/skills/ros2-msg-design/SKILL.md`) use the placeholder name `coresam_msgs`; the actual package ships as `sam_manager_msgs` to reflect the "manages SAM-family backends" role. The harness CLAUDE.md will be updated in a follow-up to match.
- Topic-pair messages (`SegmentRequest.msg` / `SegmentResult.msg` for the streaming `/coresam/segment_request` + `/coresam/segment_result` topics) are deferred to a follow-up PR — the srv is sufficient to lock the wire contract.
- Docker scaffold (`docker/.base/` subtree of `ycpss91255-docker/base`, mirroring `ycpss91255-docker/seggpt`) is deferred to a follow-up PR. Build / lint for this PR uses CI-side ROS 2 humble runners directly.
