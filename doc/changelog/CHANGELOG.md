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

### Notes

- Package naming: the upstream CoreSAM design docs (`coreSAM_ws/CLAUDE.md`, `.claude/skills/ros2-msg-design/SKILL.md`) use the placeholder name `coresam_msgs`; the actual package ships as `sam_manager_msgs` to reflect the "manages SAM-family backends" role. The harness CLAUDE.md will be updated in a follow-up to match.
- Topic-pair messages (`SegmentRequest.msg` / `SegmentResult.msg` for the streaming `/coresam/segment_request` + `/coresam/segment_result` topics) are deferred to a follow-up PR — the srv is sufficient to lock the wire contract.
- Docker scaffold (`docker/.base/` subtree of `ycpss91255-docker/base`, mirroring `ycpss91255-docker/seggpt`) is deferred to a follow-up PR. Build / lint for this PR uses CI-side ROS 2 humble runners directly.
