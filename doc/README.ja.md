# sam_manager

**[English](../README.md)** | **[繁體中文](README.zh-TW.md)** | **[简体中文](README.zh-CN.md)** | **[日本語](README.ja.md)**

SAM ファミリーの promptable segmentation backend を安定したインターフェースの背後にラップする ROS 2 service。ターゲット画像と reference(image + mask)が与えられると — runtime prompt store から名前で参照するか、request 内に inline で渡すか — reference が示すのと同じクラスのオブジェクトを記述する二値 mask を返します。

デフォルト backend は SegGPT(`ycpss91255-docker/seggpt` の Layer 2 Python API 経由)。インターフェースは将来の backend 交換(VRP-SAM、PerSAM-F、SAM 2 Tiny…)時に consumer 側の修正が不要なよう設計されています — backend の選択は node 起動パラメータであり、wire contract には含まれません。

## Scope

`sam_manager` が**やる**こと:

- Visual prompt(image + reference mask)を受け取り、pixel レベルの二値 mask を返す。
- Response 上の 2 つの booleans(`has_mask`、`has_bbox`)で backend 能力を暗黙的に表現 — 内部 level 名(`PIXEL_MASK` / `COARSE_BBOX` / `ROI_ONLY`)や backend identity を wire に乗せることはない。
- `confidence ∈ [0,1]`、`status_code`(0=OK、1..11 各種エラー)、`error_message` を報告。
- 複数 backend を pluggable に(SegGPT、VRP-SAM…)、node 起動時に `backend_name` パラメータで選定。runtime での切替は意図的に非サポート(Rule of Three)。
- ディスク上の prompt store(`prompts_dir/<name>/{image.png, mask.png}`)を管理し、named-mode request を curate 済み reference pair に解決。

`sam_manager` が**やらない**こと(以下はアプリケーション層の責任):

- オブジェクト分類。mask のみを返し、「これはパレットだ」「これはビームだ」とは言わない。
- Depth の処理。RGB のみ。
- Pixel↔mm 換算(px↔mm 逆算は全面廃止;標準ルートはアプリケーション層で `mask × depth → 3D` を `tf` 投影または RANSAC 平面適合で行う)。
- 幾何計算(距離、角度、平面適合)。
- 整列判定 / フォーク制御 / 車両ナビゲーション。
- Text prompt の受付。
- ROS 2 tf2 tree からの `camera_tf` lookup。Caller が request に乗せる;テスト用に `default_camera_tf` パラメータが fallback。

## Architecture(計画中)

```
sam_manager/                      # repo root(本ディレクトリ)
├── src/
│   ├── sam_manager_msgs/         # ROS 2 IDL — service / message 定義
│   ├── sam_manager/              # node — LifecycleNode、request queue、prompt store(future PR)
│   └── sam_manager_bringup/      # launch files + config(future PR)
├── doc/
│   ├── README.zh-TW.md
│   ├── README.zh-CN.md
│   ├── README.ja.md
│   └── changelog/
│       └── CHANGELOG.md
├── LICENSE
└── README.md
```

`sam_manager` 内の runtime 階層化(計画中、未実装):

```
Layer 1 — InterfaceAdapter   ROS 2 srv handler、FastAPI handler、Python API(rclpy を import する唯一の層)
Layer 2 — QueueCore          FIFO request queue、1 GPU worker、timeout monitor
Layer 3 — PromptStore        named-mode:起動時に prompts_dir をスキャン、runtime は O(1) by-name lookup
Layer 4 — BackendInterface   全 backend が実装する Protocol;BackendSelector が起動時に 1 つ選択
Layer 5 — ConfidenceGate     mask 品質と confidence 閾値;has_mask + status_code を設定
```

Layer 2~5 は ROS-agnostic — プレーンな Python オブジェクトを受け渡しするため、同じ core code が ROS 2 srv、FastAPI テスト endpoint、ROS を立ち上げない unit test を同時に駆動できます。

## ステータス

| コンポーネント | 状態 |
|---|---|
| Repo scaffold(本 README、LICENSE、CHANGELOG、.gitignore) | bootstrap PR |
| `sam_manager_msgs` IDL(srv / msg 定義) | first PR |
| `sam_manager` node + backend interface + SegGPT wrapper | future PR |
| `sam_manager_bringup` launch files + Docker | future PR |

IDL package だけで wire contract をロックでき、downstream package は runtime node が誕生する前にそれに対してコンパイルできます。

## インターフェース契約(`sam_manager_msgs` でロック)

### Service `sam_manager_msgs/srv/SegmentFromReference`

Request:`std_msgs/Header header` · `sensor_msgs/Image image` · `geometry_msgs/TransformStamped camera_tf` · 続いて二者択一: `string reference_name`(named mode)または `sensor_msgs/Image[] reference_images` + `sensor_msgs/Image[] reference_masks`(inline mode)。モードは node 起動時に固定。

Response:`std_msgs/Header header` · `sam_manager_msgs/MaskRLE mask_rle`(COCO 風 `counts` + `size [H,W]`)· `sensor_msgs/RegionOfInterest bbox` · `float32 confidence` · `bool has_mask` · `bool has_bbox` · `uint8 status_code` · `string error_message`。

意図的に wire に**乗せない**フィールド:`capability_level`、`backend_name`、`inference_latency_ms`(起動 config / metrics only)、`depth`、`class_label`、`category`、`text_prompt`、`mm_per_pixel`、`distance`、`angle`、`aligned`。

### Status code

| Code | 名前 | 理由 | Emit 位置 |
|---|---|---|---|
| 0 | `OK` | 成功 | (no emit) |
| 1 | `EMPTY_MASK` | 推論完了したが mask が全 0 | Layer 5 ConfidenceGate |
| 2 | `LOW_CONFIDENCE` | `confidence` が起動閾値未満 | Layer 5 ConfidenceGate |
| 3 | `INVALID_IMAGE` | ターゲット画像のデコード失敗 / 寸法不正 | Layer 1 RequestValidator |
| 4 | `INVALID_HEADER` | `frame_id` 欠落 / timestamp 異常 | Layer 1 RequestValidator |
| 5 | `PROMPT_NOT_FOUND` | Named mode:`reference_name` が store にない | Layer 3 PromptStore |
| 6 | `REFERENCE_SIZE_MISMATCH` | Inline:`image[i].size != mask[i].size` | Layer 3 PromptStore |
| 7 | `REFERENCE_COUNT_MISMATCH` | Inline:`len(images) != len(masks)` | Layer 3 PromptStore |
| 8 | `INVALID_TF` | `camera_tf` 不正 / NaN / inf / default なしで欠落 | Layer 1 RequestValidator |
| 9 | `QUEUE_FULL` | Request queue 満杯 | Layer 2 QueueCore |
| 10 | `TIMEOUT` | キュー待ち + 推論が node timeout を超過 | Layer 2 TimeoutMonitor |
| 11 | `BACKEND_ERROR` | Backend 異常(GPU OOM / CUDA error 等) | Layer 4 ErrorHandler |

番号は予約済み — 削除された code 番号は再割り当てされません。

## License

Apache-2.0。[LICENSE](../LICENSE) を参照。

## 関連 repo

- [`ycpss91255-docker/seggpt`](https://github.com/ycpss91255-docker/seggpt) — SegGPT backend、`BackendInterface` の最初の具体実装。`sam_manager` は Layer 2 Python API(`from seggpt.api import SegGPTBackend`)で import;backend からの逆方向 import は禁止。
- [`ycpss91255-docker/base`](https://github.com/ycpss91255-docker/base) — 両 repo が共用する Docker workspace template。
