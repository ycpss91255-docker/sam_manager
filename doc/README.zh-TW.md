# sam_manager

**[English](../README.md)** | **[繁體中文](README.zh-TW.md)** | **[简体中文](README.zh-CN.md)** | **[日本語](README.ja.md)**

ROS 2 service,將 SAM 家族的 promptable segmentation backend 包在穩定介面後面。給定目標影像 + reference(image + mask)— 由 runtime prompt store 用名稱查,或在 request 中 inline 帶入 — 回傳描述同類物件的二值 mask。

預設 backend 是 SegGPT(透過 `ycpss91255-docker/seggpt` 的 Layer 2 Python API)。介面設計為未來換 backend(VRP-SAM、PerSAM-F、SAM 2 Tiny…)時 consumer 端零修改 — backend 選擇是 node 啟動參數,不在 wire contract 內。

## Scope

`sam_manager` **做**:

- 接受 visual prompt(image + reference mask),回傳 pixel-level 二值 mask。
- 用 response 上兩個 booleans(`has_mask`、`has_bbox`)隱晦表達 backend 能力 — 從不把內部 level 名稱(`PIXEL_MASK` / `COARSE_BBOX` / `ROI_ONLY`)或 backend 身分放在 wire 上。
- 回報 `confidence ∈ [0,1]`、`status_code`(0=OK,1..11 各類錯誤)、`error_message`。
- 多 backend pluggable(SegGPT、VRP-SAM…),node 啟動時透過 `backend_name` 參數選定。runtime 不切換(Rule of Three)。
- 管理硬碟上的 prompt store(`prompts_dir/<name>/{image.png, mask.png}`),named-mode request 解析為預先 curate 的 reference pair。

`sam_manager` **不做**(這些是應用層責任):

- 物件分類。只回 mask,從不回「這是棧板」或「這是擋板」。
- 處理 depth。只接受 RGB。
- Pixel↔mm 換算(px↔mm 反推已全面棄用;標準路線是應用層用 `mask × depth → 3D`,經 `tf` 投影或 RANSAC 平面擬合)。
- 幾何計算(距離、角度、平面擬合)。
- 對齊判斷 / 牙叉控制 / 車輛 navigation。
- 接受 text prompt。
- 從 ROS 2 tf2 tree lookup `camera_tf`。caller 在 request 中帶入;測試用 `default_camera_tf` 參數做 fallback。

## Architecture(規劃中)

```
sam_manager/                      # repo root(本目錄)
├── src/
│   ├── sam_manager_msgs/         # ROS 2 IDL — service / message 定義
│   ├── sam_manager/              # node — LifecycleNode、request queue、prompt store(未來 PR)
│   └── sam_manager_bringup/      # launch files + config(未來 PR)
├── doc/
│   ├── README.zh-TW.md
│   ├── README.zh-CN.md
│   ├── README.ja.md
│   └── changelog/
│       └── CHANGELOG.md
├── LICENSE
└── README.md
```

`sam_manager` 內 runtime 分層(規劃中,尚未實作):

```
Layer 1 — InterfaceAdapter   ROS 2 srv handler、FastAPI handler、Python API(唯一 import rclpy 的層)
Layer 2 — QueueCore          FIFO request queue、1 GPU worker、timeout monitor
Layer 3 — PromptStore        named-mode:啟動時掃 prompts_dir,runtime O(1) by-name lookup
Layer 4 — BackendInterface   所有 backend 實作的 Protocol;BackendSelector 啟動時選一個
Layer 5 — ConfidenceGate     mask 品質與 confidence 門檻;設 has_mask + status_code
```

Layer 2~5 是 ROS-agnostic — 收 / 回傳純 Python 物件,因此同一份 core code 同時撐 ROS 2 srv、FastAPI 測試 endpoint、不啟 ROS 的 unit test。

## 狀態

| 元件 | 狀態 |
|---|---|
| Repo scaffold(本 README、LICENSE、CHANGELOG、.gitignore) | bootstrap PR |
| `sam_manager_msgs` IDL(srv / msg 定義) | first PR |
| `sam_manager` node + backend interface + SegGPT wrapper | future PR |
| `sam_manager_bringup` launch files + Docker | future PR |

光是 IDL package 就足以鎖 wire contract;下游 package 可以在 runtime node 還沒誕生前先對它編譯。

## 介面契約(由 `sam_manager_msgs` 鎖定)

### Service `sam_manager_msgs/srv/SegmentFromReference`

Request:`std_msgs/Header header` · `sensor_msgs/Image image` · `geometry_msgs/TransformStamped camera_tf` · 然後二擇一: `string reference_name`(named mode)或 `sensor_msgs/Image[] reference_images` + `sensor_msgs/Image[] reference_masks`(inline mode)。模式在 node 啟動時鎖定。

Response:`std_msgs/Header header` · `sam_manager_msgs/MaskRLE mask_rle`(COCO 風 `counts` + `size [H,W]`)· `sensor_msgs/RegionOfInterest bbox` · `float32 confidence` · `bool has_mask` · `bool has_bbox` · `uint8 status_code` · `string error_message`。

刻意**不放**在 wire 上的欄位:`capability_level`、`backend_name`、`inference_latency_ms`(啟動 config / metrics only)、`depth`、`class_label`、`category`、`text_prompt`、`mm_per_pixel`、`distance`、`angle`、`aligned`。

### Status code

| Code | 名稱 | 原因 | Emit 位置 |
|---|---|---|---|
| 0 | `OK` | 成功 | (no emit) |
| 1 | `EMPTY_MASK` | 推論完成但 mask 全空 | Layer 5 ConfidenceGate |
| 2 | `LOW_CONFIDENCE` | `confidence` 低於啟動門檻 | Layer 5 ConfidenceGate |
| 3 | `INVALID_IMAGE` | 目標影像解碼失敗 / 尺寸不合 | Layer 1 RequestValidator |
| 4 | `INVALID_HEADER` | `frame_id` 缺失 / timestamp 異常 | Layer 1 RequestValidator |
| 5 | `PROMPT_NOT_FOUND` | Named mode:找不到 `reference_name` | Layer 3 PromptStore |
| 6 | `REFERENCE_SIZE_MISMATCH` | Inline:`image[i].size != mask[i].size` | Layer 3 PromptStore |
| 7 | `REFERENCE_COUNT_MISMATCH` | Inline:`len(images) != len(masks)` | Layer 3 PromptStore |
| 8 | `INVALID_TF` | `camera_tf` 格式錯 / NaN / inf / 缺值且無 default | Layer 1 RequestValidator |
| 9 | `QUEUE_FULL` | Request queue 滿 | Layer 2 QueueCore |
| 10 | `TIMEOUT` | 排隊 + 推論超時 | Layer 2 TimeoutMonitor |
| 11 | `BACKEND_ERROR` | Backend 異常(GPU OOM / CUDA error 等) | Layer 4 ErrorHandler |

編號預留 — 移除的 code 編號不會被重新分配。

## License

Apache-2.0。見 [LICENSE](../LICENSE)。

## 相關 repo

- [`ycpss91255-docker/seggpt`](https://github.com/ycpss91255-docker/seggpt) — SegGPT backend,`BackendInterface` 的第一個具體實作。`sam_manager` 透過 Layer 2 Python API(`from seggpt.api import SegGPTBackend`)import;反向 import backend 不允許。
- [`ycpss91255-docker/base`](https://github.com/ycpss91255-docker/base) — 兩個 repo 共用的 Docker workspace template。
