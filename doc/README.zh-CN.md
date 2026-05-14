# sam_manager

**[English](../README.md)** | **[繁體中文](README.zh-TW.md)** | **[简体中文](README.zh-CN.md)** | **[日本語](README.ja.md)**

ROS 2 service,将 SAM 家族的 promptable segmentation backend 包在稳定接口后面。给定目标图像 + reference(image + mask)— 由 runtime prompt store 用名称查,或在 request 中 inline 带入 — 返回描述同类物件的二值 mask。

默认 backend 是 SegGPT(通过 `ycpss91255-docker/seggpt` 的 Layer 2 Python API)。接口设计为未来换 backend(VRP-SAM、PerSAM-F、SAM 2 Tiny…)时 consumer 端零修改 — backend 选择是 node 启动参数,不在 wire contract 内。

## Scope

`sam_manager` **做**:

- 接受 visual prompt(image + reference mask),返回 pixel-level 二值 mask。
- 用 response 上两个 booleans(`has_mask`、`has_bbox`)隐晦表达 backend 能力 — 从不把内部 level 名称(`PIXEL_MASK` / `COARSE_BBOX` / `ROI_ONLY`)或 backend 身份放在 wire 上。
- 回报 `confidence ∈ [0,1]`、`status_code`(0=OK,1..11 各类错误)、`error_message`。
- 多 backend pluggable(SegGPT、VRP-SAM…),node 启动时通过 `backend_name` 参数选定。runtime 不切换(Rule of Three)。
- 管理硬盘上的 prompt store(`prompts_dir/<name>/{image.png, mask.png}`),named-mode request 解析为预先 curate 的 reference pair。

`sam_manager` **不做**(这些是应用层责任):

- 物件分类。只回 mask,从不回「这是栈板」或「这是挡板」。
- 处理 depth。只接受 RGB。
- Pixel↔mm 换算(px↔mm 反推已全面弃用;标准路线是应用层用 `mask × depth → 3D`,经 `tf` 投影或 RANSAC 平面拟合)。
- 几何计算(距离、角度、平面拟合)。
- 对齐判断 / 牙叉控制 / 车辆 navigation。
- 接受 text prompt。
- 从 ROS 2 tf2 tree lookup `camera_tf`。caller 在 request 中带入;测试用 `default_camera_tf` 参数做 fallback。

## Architecture(规划中)

```
sam_manager/                      # repo root(本目录)
├── src/
│   ├── sam_manager_msgs/         # ROS 2 IDL — service / message 定义
│   ├── sam_manager/              # node — LifecycleNode、request queue、prompt store(未来 PR)
│   └── sam_manager_bringup/      # launch files + config(未来 PR)
├── doc/
│   ├── README.zh-TW.md
│   ├── README.zh-CN.md
│   ├── README.ja.md
│   └── changelog/
│       └── CHANGELOG.md
├── LICENSE
└── README.md
```

`sam_manager` 内 runtime 分层(规划中,尚未实作):

```
Layer 1 — InterfaceAdapter   ROS 2 srv handler、FastAPI handler、Python API(唯一 import rclpy 的层)
Layer 2 — QueueCore          FIFO request queue、1 GPU worker、timeout monitor
Layer 3 — PromptStore        named-mode:启动时扫 prompts_dir,runtime O(1) by-name lookup
Layer 4 — BackendInterface   所有 backend 实作的 Protocol;BackendSelector 启动时选一个
Layer 5 — ConfidenceGate     mask 质量与 confidence 门槛;设 has_mask + status_code
```

Layer 2~5 是 ROS-agnostic — 收 / 返回纯 Python 对象,因此同一份 core code 同时撑 ROS 2 srv、FastAPI 测试 endpoint、不启 ROS 的 unit test。

## 状态

| 组件 | 状态 |
|---|---|
| Repo scaffold(本 README、LICENSE、CHANGELOG、.gitignore) | bootstrap PR |
| `sam_manager_msgs` IDL(srv / msg 定义) | first PR |
| `sam_manager` node + backend interface + SegGPT wrapper | future PR |
| `sam_manager_bringup` launch files + Docker | future PR |

光是 IDL package 就足以锁 wire contract;下游 package 可以在 runtime node 还没诞生前先对它编译。

## 接口契约(由 `sam_manager_msgs` 锁定)

### Service `sam_manager_msgs/srv/SegmentFromReference`

Request:`std_msgs/Header header` · `sensor_msgs/Image image` · `geometry_msgs/TransformStamped camera_tf` · 然后二择一: `string reference_name`(named mode)或 `sensor_msgs/Image[] reference_images` + `sensor_msgs/Image[] reference_masks`(inline mode)。模式在 node 启动时锁定。

Response:`std_msgs/Header header` · `sam_manager_msgs/MaskRLE mask_rle`(COCO 风 `counts` + `size [H,W]`)· `sensor_msgs/RegionOfInterest bbox` · `float32 confidence` · `bool has_mask` · `bool has_bbox` · `uint8 status_code` · `string error_message`。

刻意**不放**在 wire 上的字段:`capability_level`、`backend_name`、`inference_latency_ms`(启动 config / metrics only)、`depth`、`class_label`、`category`、`text_prompt`、`mm_per_pixel`、`distance`、`angle`、`aligned`。

### Status code

| Code | 名称 | 原因 | Emit 位置 |
|---|---|---|---|
| 0 | `OK` | 成功 | (no emit) |
| 1 | `EMPTY_MASK` | 推论完成但 mask 全空 | Layer 5 ConfidenceGate |
| 2 | `LOW_CONFIDENCE` | `confidence` 低于启动门槛 | Layer 5 ConfidenceGate |
| 3 | `INVALID_IMAGE` | 目标图像解码失败 / 尺寸不合 | Layer 1 RequestValidator |
| 4 | `INVALID_HEADER` | `frame_id` 缺失 / timestamp 异常 | Layer 1 RequestValidator |
| 5 | `PROMPT_NOT_FOUND` | Named mode:找不到 `reference_name` | Layer 3 PromptStore |
| 6 | `REFERENCE_SIZE_MISMATCH` | Inline:`image[i].size != mask[i].size` | Layer 3 PromptStore |
| 7 | `REFERENCE_COUNT_MISMATCH` | Inline:`len(images) != len(masks)` | Layer 3 PromptStore |
| 8 | `INVALID_TF` | `camera_tf` 格式错 / NaN / inf / 缺值且无 default | Layer 1 RequestValidator |
| 9 | `QUEUE_FULL` | Request queue 满 | Layer 2 QueueCore |
| 10 | `TIMEOUT` | 排队 + 推论超时 | Layer 2 TimeoutMonitor |
| 11 | `BACKEND_ERROR` | Backend 异常(GPU OOM / CUDA error 等) | Layer 4 ErrorHandler |

编号预留 — 移除的 code 编号不会被重新分配。

## License

Apache-2.0。见 [LICENSE](../LICENSE)。

## 相关 repo

- [`ycpss91255-docker/seggpt`](https://github.com/ycpss91255-docker/seggpt) — SegGPT backend,`BackendInterface` 的第一个具体实作。`sam_manager` 通过 Layer 2 Python API(`from seggpt.api import SegGPTBackend`)import;反向 import backend 不允许。
- [`ycpss91255-docker/base`](https://github.com/ycpss91255-docker/base) — 两个 repo 共用的 Docker workspace template。
