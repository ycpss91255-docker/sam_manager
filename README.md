# sam_manager

**[English](README.md)** | **[繁體中文](doc/README.zh-TW.md)** | **[简体中文](doc/README.zh-CN.md)** | **[日本語](doc/README.ja.md)**

ROS 2 service that wraps SAM-family promptable segmentation backends behind a stable interface. Given a target image plus a reference (image + mask) — either by name from a runtime prompt store or inline in the request — returns a binary mask describing the same object class the reference depicts.

The default backend is SegGPT (via `ycpss91255-docker/seggpt`'s Layer 2 Python API). The interface is designed so a future backend swap (VRP-SAM, PerSAM-F, SAM 2 Tiny, …) requires no consumer-side change — backend selection is a node startup parameter, not part of the wire contract.

## Scope

`sam_manager` **does**:

- Accept visual prompts (image + reference mask) and return a pixel-level binary mask.
- Express backend capability implicitly through two booleans on the response (`has_mask`, `has_bbox`) — never exposes internal level names (`PIXEL_MASK` / `COARSE_BBOX` / `ROI_ONLY`) or backend identity on the wire.
- Report `confidence ∈ [0,1]`, a `status_code` (0=OK, 1..11 errors), and an `error_message`.
- Run multiple backends pluggably (SegGPT, VRP-SAM, …) selected at node startup via the `backend_name` parameter. Runtime switching is intentionally not supported (Rule of Three).
- Manage a prompt store on disk (`prompts_dir/<name>/{image.png, mask.png}`) so named-mode requests resolve to a pre-curated reference pair.

`sam_manager` does **not** (these are application-layer responsibilities):

- Classify objects. Only returns a mask; never returns "this is a pallet" or "this is a beam".
- Process depth. RGB only.
- Convert pixels to millimetres (px-to-mm reverse-engineering is fully deprecated; the standard path is `mask × depth → 3D` via `tf` projection or RANSAC plane fit in the application layer).
- Compute geometry (distance, angle, plane fit).
- Decide alignment / drive forks / navigate.
- Accept text prompts.
- Look up `camera_tf` from the ROS 2 tf2 tree. The caller carries `camera_tf` in the request; a test-only `default_camera_tf` parameter provides a fallback.

## Architecture (planned)

```
sam_manager/                      # repo root (this directory)
├── src/
│   ├── sam_manager_msgs/         # ROS 2 IDL — service / message definitions
│   ├── sam_manager/              # node — LifecycleNode, request queue, prompt store (future PR)
│   └── sam_manager_bringup/      # launch files + config (future PR)
├── doc/
│   ├── README.zh-TW.md
│   ├── README.zh-CN.md
│   ├── README.ja.md
│   └── changelog/
│       └── CHANGELOG.md
├── LICENSE
└── README.md
```

The runtime layering inside `sam_manager` (planned, not yet implemented):

```
Layer 1 — InterfaceAdapter   ROS 2 srv handler, FastAPI handler, Python API (the only layer that imports rclpy)
Layer 2 — QueueCore          FIFO request queue, 1 GPU worker, timeout monitor
Layer 3 — PromptStore        named-mode: scan prompts_dir at startup, O(1) by-name lookup at runtime
Layer 4 — BackendInterface   Protocol that all backends implement; BackendSelector picks one at startup
Layer 5 — ConfidenceGate     mask-quality and confidence thresholding; sets has_mask + status_code
```

Layers 2–5 are ROS-agnostic — they accept and return plain Python objects so the same core code drives the ROS 2 srv, the FastAPI test endpoint, and unit tests without spinning up ROS.

## Status

| Component | State |
|---|---|
| Repo scaffold (this README, LICENSE, CHANGELOG, .gitignore) | bootstrap PR |
| `sam_manager_msgs` IDL (srv / msg definitions) | first PR |
| `sam_manager` node + backend interface + SegGPT wrapper | future PR |
| `sam_manager_bringup` launch files + Docker | future PR |

The IDL package alone is enough to lock the wire contract; downstream packages can compile against it before the runtime node exists.

## Interface contract (locked by `sam_manager_msgs`)

### Service `sam_manager_msgs/srv/SegmentFromReference`

Request: `std_msgs/Header header` · `sensor_msgs/Image image` · `geometry_msgs/TransformStamped camera_tf` · then either `string reference_name` (named mode) or `sensor_msgs/Image[] reference_images` + `sensor_msgs/Image[] reference_masks` (inline mode). Mode is fixed at node startup.

Response: `std_msgs/Header header` · `sam_manager_msgs/MaskRLE mask_rle` (COCO-style `counts` + `size [H,W]`) · `sensor_msgs/RegionOfInterest bbox` · `float32 confidence` · `bool has_mask` · `bool has_bbox` · `uint8 status_code` · `string error_message`.

Fields intentionally **absent** from the wire: `capability_level`, `backend_name`, `inference_latency_ms` (startup-config / metrics-only), `depth`, `class_label`, `category`, `text_prompt`, `mm_per_pixel`, `distance`, `angle`, `aligned`.

### Status codes

| Code | Name | Reason | Emit site |
|---|---|---|---|
| 0 | `OK` | Success | (no emit) |
| 1 | `EMPTY_MASK` | Inference returned an all-zero mask | Layer 5 ConfidenceGate |
| 2 | `LOW_CONFIDENCE` | `confidence < startup threshold` | Layer 5 ConfidenceGate |
| 3 | `INVALID_IMAGE` | Target image decode failure / bad dims | Layer 1 RequestValidator |
| 4 | `INVALID_HEADER` | `frame_id` missing / timestamp anomaly | Layer 1 RequestValidator |
| 5 | `PROMPT_NOT_FOUND` | Named mode: `reference_name` not in store | Layer 3 PromptStore |
| 6 | `REFERENCE_SIZE_MISMATCH` | Inline: `image[i].size != mask[i].size` | Layer 3 PromptStore |
| 7 | `REFERENCE_COUNT_MISMATCH` | Inline: `len(images) != len(masks)` | Layer 3 PromptStore |
| 8 | `INVALID_TF` | `camera_tf` malformed / NaN / inf / absent without default | Layer 1 RequestValidator |
| 9 | `QUEUE_FULL` | Request queue at capacity | Layer 2 QueueCore |
| 10 | `TIMEOUT` | Queue wait + inference exceeded node timeout | Layer 2 TimeoutMonitor |
| 11 | `BACKEND_ERROR` | Backend raised (GPU OOM / CUDA error / …) | Layer 4 ErrorHandler |

Numbering is reserved — removed codes never get reassigned.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Companion repos

- [`ycpss91255-docker/seggpt`](https://github.com/ycpss91255-docker/seggpt) — SegGPT backend, the first concrete implementation of `BackendInterface`. `sam_manager` imports its Layer 2 Python API (`from seggpt.api import SegGPTBackend`); it never imports backend code in the reverse direction.
- [`ycpss91255-docker/base`](https://github.com/ycpss91255-docker/base) — Docker workspace template used by both repos.
