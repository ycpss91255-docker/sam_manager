# Tests

Single source of truth for sam_manager test counts and layout. CI
hooks compare per-file `def test_*` counts against this document.

All tests run inside Docker via the standard ROS 2 humble image:

```bash
docker run --rm -v "$(pwd):/work" -w /work ros:humble-ros-base bash -c '
  apt-get update -qq
  apt-get install -y --no-install-recommends python3-pip
  rosdep update --rosdistro humble
  rosdep install --from-paths src --ignore-src -r -y
  python3 -m pip install structlog pytest-cov fastapi pillow python-multipart httpx
  . /opt/ros/humble/setup.bash
  colcon build --packages-select sam_manager_msgs sam_manager
  . install/setup.bash
  colcon test --packages-select sam_manager --event-handlers console_direct+
  colcon test-result --verbose
  python3 -m pytest src/sam_manager/test/ \
    --cov=sam_manager --cov-report=term-missing --cov-fail-under=80
'
```

Total: **151 unit tests** + 1 skipped (`test_copyright` — opt-in once
a per-file header policy is decided) plus `ament_flake8` and
`ament_pep257` (both run via `colcon test`). Delta from previous
revision: +16 `test/adapters/fastapi/test_app.py` (integration
cases via `starlette.testclient.TestClient`) and +6
`test/adapters/fastapi/test_models.py` (pydantic schema invariants).

Line coverage: **100%** at this revision; CI gate is **80%** so
the remaining layer (Layer 2 RequestQueue) can be added without
immediately tightening the bar.

## 4-category coverage

Per CLAUDE.md「TDD 測試分類（4 個面向）」:

| # | Category | Where | What it covers |
|---|----------|-------|----------------|
| 1 | Smoke | `test/test_*.py` (any) | Package + module import + ABC instantiability + `colcon build` success |
| 2 | Unit | `test/test_backend.py` / `test/test_mock.py` / `test/test_error_handler.py` / `test/test_request_validator.py` / `test/test_prompt_store.py` / `test/test_confidence_gate.py` | BackendInterface ABC, MockBackend behaviour, ErrorHandling decorator, Layer 1 target validator, Layer 3 reference validators, Layer 5 ConfidenceGate |
| 3 | Integration | `test/adapters/python/test_segment.py` + `test/adapters/fastapi/test_app.py` + `test/adapters/fastapi/test_models.py` | Python API adapter (raw InferResult pass-through); FastAPI adapter (HTTP route -> validate -> infer -> ConfidenceGate -> JSON wire response); pydantic schema contracts |
| 4 | Lint | `test/test_flake8.py` + `test/test_pep257.py` + `test/test_copyright.py` (skipped) | flake8 (ament default) + pep257 with Google-style ignores |

## Test inventory

### test/test_backend.py (7)

| Test | What |
|------|------|
| `test_abc_cannot_instantiate_directly` | `BackendInterface()` raises TypeError |
| `test_mock_subclasses_backend_interface` | `issubclass(MockBackend, BackendInterface)` |
| `test_polymorphism_via_base_type` | Caller holds MockBackend through ABC type |
| `test_default_warmup_is_noop` | `warmup()` returns None, no raise |
| `test_default_health_check_returns_true` | `health_check()` defaults to True |
| `test_subclass_can_override_warmup` | Subclass replacement of warmup fires |
| `test_subclass_can_override_health_check_returning_false` | Subclass may report unhealthy |

### test/test_confidence_gate.py (26)

Output shape:
| Test | What |
|------|------|
| `test_output_is_confidence_gate_output_instance` | Returns ConfidenceGateOutput dataclass |
| `test_output_carries_all_required_fields` | All wire fields present |

Happy path (non-empty mask):
| Test | What |
|------|------|
| `test_non_empty_mask_sets_has_mask_true` | Any positive pixel -> has_mask=True |
| `test_non_empty_mask_sets_has_bbox_true` | has_bbox mirrors has_mask |
| `test_non_empty_mask_emits_status_zero` | High-confidence non-empty -> status 0 |
| `test_confidence_passes_through` | InferResult.confidence verbatim |
| `test_confidence_unreported_defaults_to_one` | None -> 1.0 |

EMPTY_MASK (status 1):
| Test | What |
|------|------|
| `test_empty_mask_sets_has_mask_false` | All-zero -> has_mask=False |
| `test_empty_mask_sets_has_bbox_false` | All-zero -> has_bbox=False |
| `test_empty_mask_emits_status_one` | All-zero -> status 1 |
| `test_empty_mask_rle_counts_is_empty_list` | counts == [] (wire "no mask") |
| `test_empty_mask_rle_size_still_carries_shape` | size still (H, W) |
| `test_empty_mask_bbox_is_zero` | bbox (0, 0, 0, 0) |

LOW_CONFIDENCE (status 2):
| Test | What |
|------|------|
| `test_low_confidence_with_non_empty_mask_emits_status_two` | conf < threshold -> status 2 |
| `test_confidence_at_threshold_passes` | conf == threshold OK |
| `test_default_threshold_is_zero_point_three` | 0.29 fails, 0.30 passes (default 0.3) |

Status priority:
| Test | What |
|------|------|
| `test_empty_mask_priority_over_low_confidence` | EMPTY > LOW when both apply |

bbox geometry:
| Test | What |
|------|------|
| `test_bbox_tight_around_pixels` | Tight rect around positive pixels |
| `test_bbox_single_pixel` | Single pixel -> (x, y, 1, 1) |
| `test_bbox_fully_filled` | Full mask -> full image |

RLE encoding:
| Test | What |
|------|------|
| `test_rle_size_is_h_w_tuple` | size is (H, W) tuple |
| `test_rle_counts_sum_equals_pixel_count` | Sum of counts == H*W |
| `test_rle_starts_with_background_run` | Foreground-first prepends 0 |
| `test_rle_all_background` | Empty mask -> counts is [] |
| `test_rle_alternating_runs_match_mask` | Manually constructed mask matches |

Validation:
| Test | What |
|------|------|
| `test_evaluate_rejects_empty_masks_list` | Empty masks list raises ValueError |

### test/test_mock.py (21)

Constructor:
| Test | What |
|------|------|
| `test_constructor_rejects_out_of_range_coverage` | `mask_coverage` outside [0, 1] raises |

Happy-path output:
| Test | What |
|------|------|
| `test_output_shape_matches_target` | `result.masks[0].shape == target.shape[:2]` |
| `test_output_count_is_one_mask_one_class` | Single mask + single class_id == 0 |
| `test_target_is_reference_not_copy` | `result.target is target` (no buffer copy) |
| `test_mask_coverage_matches_param` | Square 200×200 coverage in [0.45, 0.55] |
| `test_determinism_same_input_same_output` | Two calls → byte-identical mask |
| `test_telemetry_is_none_for_mock` | `latency_ms` / `gpu_mem_mb` are None |

Input validation removed -- MockBackend no longer self-validates.
Coverage moved to: `test/test_request_validator.py` (Layer 1) +
`test/test_prompt_store.py` (Layer 3) +
`test/adapters/python/test_segment.py` (realistic call path).

Output invariants:
| Test | What |
|------|------|
| `test_output_is_inferresult_instance` | Result is InferResult, not dict |
| `test_output_mask_dtype_is_uint8` | mask dtype invariant |
| `test_output_mask_values_only_zero_or_255` | Binary values, no gradient |
| `test_output_mask_geometry_is_centered` | Center == 255, four corners == 0 |
| `test_output_class_ids_exactly_zero` | `class_ids == [0]` exact match |
| `test_output_target_dtype_shape_unchanged` | Caller target not mutated |

Boundary / geometry:
| Test | What |
|------|------|
| `test_mask_coverage_zero_produces_all_zero_mask` | coverage=0.0 → no positive pixels |
| `test_mask_coverage_one_fills_image` | coverage=1.0 → ≥ 95% positive |
| `test_mask_coverage_nonsquare_target` | 800×600 coverage in [0.20, 0.30] |

structlog emit:
| Test | What |
|------|------|
| `test_log_emits_infer_started_with_attributes` | `backend_infer_started` + n_refs / target_shape / backend attrs |
| `test_log_emits_infer_completed_with_attributes` | `backend_infer_completed` + n_masks / backend attrs |

### test/test_request_validator.py (18)

Exception contract:
| Test | What |
|------|------|
| `test_invalid_image_error_status_code_is_three` | `InvalidImageError.STATUS_CODE == 3` |
| `test_invalid_image_error_subclasses_value_error` | IS-A ValueError so Layer 4 wrapper propagates unchanged |
| `test_invalid_image_error_carries_message` | Raised instance preserves diagnostic message |

Happy path:
| Test | What |
|------|------|
| `test_accepts_valid_rgb_uint8` | `(H, W, 3)` uint8 returns None |
| `test_accepts_minimum_size_one_by_one` | 1×1 RGB image is valid |

Shape rejection:
| Test | What |
|------|------|
| `test_rejects_2d_grayscale` | `(H, W)` raises |
| `test_rejects_rgba_four_channels` | `(H, W, 4)` raises |
| `test_rejects_pseudo_3d_single_channel` | `(H, W, 1)` raises |
| `test_rejects_ndim_one` | 1D flat array raises |
| `test_rejects_ndim_four` | 4D batched array raises |

dtype rejection:
| Test | What |
|------|------|
| `test_rejects_float32_dtype` | float32 raises |
| `test_rejects_uint16_dtype` | uint16 raises |
| `test_rejects_bool_dtype` | bool raises |

Empty rejection:
| Test | What |
|------|------|
| `test_rejects_empty_zero_height` | `(0, W, 3)` raises |
| `test_rejects_empty_zero_width` | `(H, 0, 3)` raises |
| `test_rejects_empty_zero_by_zero` | `(0, 0, 3)` raises |

Error diagnostics:
| Test | What |
|------|------|
| `test_error_message_mentions_shape` | Shape surfaced in error message |
| `test_error_message_mentions_dtype` | dtype surfaced in error message |

### test/test_prompt_store.py (26)

Exception contracts:
| Test | What |
|------|------|
| `test_reference_count_mismatch_status_code_is_seven` | `STATUS_CODE == 7` |
| `test_reference_size_mismatch_status_code_is_six` | `STATUS_CODE == 6` |
| `test_count_mismatch_subclasses_value_error` | IS-A ValueError |
| `test_size_mismatch_subclasses_value_error` | IS-A ValueError |

Happy path:
| Test | What |
|------|------|
| `test_accepts_empty_lists` | `validate_references([], [])` OK |
| `test_accepts_single_aligned_pair` | 1 ref + 1 mask matching |
| `test_accepts_multiple_aligned_pairs` | N pairs of varying sizes |

Count mismatch (status 7):
| Test | What |
|------|------|
| `test_rejects_more_refs_than_masks` | 1 ref + 0 masks |
| `test_rejects_more_masks_than_refs` | 0 refs + 1 mask |
| `test_rejects_off_by_one_count` | 2 refs + 3 masks |
| `test_count_mismatch_message_mentions_both_lengths` | Diagnostic surfaces lengths |

Ref shape + dtype (status 6):
| Test | What |
|------|------|
| `test_rejects_grayscale_ref` | refs[i] (H, W) raises |
| `test_rejects_rgba_ref` | refs[i] (H, W, 4) raises |
| `test_rejects_pseudo_3d_ref` | refs[i] (H, W, 1) raises |
| `test_rejects_uint16_ref` | refs[i] uint16 raises |
| `test_rejects_float32_ref` | refs[i] float32 raises |

Mask shape + dtype (status 6):
| Test | What |
|------|------|
| `test_rejects_3channel_mask` | masks[i] (H, W, 3) raises |
| `test_rejects_pseudo_3d_mask` | masks[i] (H, W, 1) raises |
| `test_rejects_bool_mask` | masks[i] bool raises |
| `test_rejects_float32_mask` | masks[i] float32 raises |

Pair alignment (status 6):
| Test | What |
|------|------|
| `test_rejects_pair_height_mismatch` | refs[i] H differs from masks[i] |
| `test_rejects_pair_width_mismatch` | refs[i] W differs from masks[i] |
| `test_rejects_pair_both_dim_mismatch` | Both dims differ |
| `test_size_mismatch_message_identifies_index` | Diagnostic names the index |

Iteration order:
| Test | What |
|------|------|
| `test_first_bad_pair_raises_before_subsequent_bad_pairs` | First-bad-wins |
| `test_count_mismatch_checked_before_pair_checks` | Count check has higher priority |

### test/test_error_handler.py (19 collected = 9 single + 2 parametrized x 5)

| Test | What |
|------|------|
| `test_wraps_successful_infer` | Happy path passthrough |
| `test_subclasses_backend_interface` | `issubclass(ErrorHandlingBackend, BackendInterface)` |
| `test_polymorphism_via_base_type` | Caller holds wrapped through ABC |
| `test_non_value_error_wrapped_as_backend_error` x 5 | Any non-ValueError -> BackendError(11), parametrized over RuntimeError / KeyError / OSError / ZeroDivisionError / ConnectionError |
| `test_non_value_error_preserves_cause_chain` x 5 | `BackendError.__cause__ is original`, parametrized over same 5 exception types |
| `test_value_error_propagates_unchanged` | ValueError NOT wrapped |
| `test_backend_error_constant_is_eleven` | `BackendError.STATUS_CODE == 11` |
| `test_backend_error_holds_original_exception` | `BackendError.original` is original ex |
| `test_backend_error_str_includes_original_type_and_message` | str surfaces type + msg |
| `test_warmup_delegates_to_inner` | warmup forwards |
| `test_health_check_delegates_to_inner` | health_check forwards |

### test/adapters/python/test_segment.py (12)

Happy path:
| Test | What |
|------|------|
| `test_returns_backend_infer_result_unchanged` | Adapter returns backend's InferResult untouched |
| `test_passes_references_through_to_backend` | refs / masks forwarded without mutation |
| `test_polymorphic_backend_accepted` | Decorator-wrapped Mock works through ABC |

Layer 1 propagation (status 3):
| Test | What |
|------|------|
| `test_invalid_image_error_propagates_unchanged` | Grayscale target -> InvalidImageError surfaces |
| `test_invalid_image_rejection_skips_backend_call` | backend.infer never runs on validation fail |

Layer 3 propagation (status 6 / 7):
| Test | What |
|------|------|
| `test_reference_count_mismatch_propagates_unchanged` | len(refs) != len(masks) -> status 7 |
| `test_reference_size_mismatch_propagates_unchanged` | Pair shape mismatch -> status 6 |
| `test_reference_size_rejection_skips_backend_call` | backend.infer never runs on refs fail |

Layer 4 propagation (status 11):
| Test | What |
|------|------|
| `test_backend_error_propagates_unchanged` | ErrorHandlingBackend wrapping a raise -> BackendError(11) |
| `test_value_error_from_unwrapped_backend_propagates` | Bare ValueError from unwrapped backend bubbles up |

Validation order:
| Test | What |
|------|------|
| `test_target_validation_runs_before_reference_validation` | Target check (Layer 1) precedes refs check (Layer 3) |
| `test_reference_validation_runs_before_backend_call` | Refs check (Layer 3) precedes backend.infer |

### test/adapters/fastapi/test_app.py (16)

Happy path (status 0):
| Test | What |
|------|------|
| `test_happy_path_returns_status_zero` | Mock default confidence -> status 0 |
| `test_happy_path_returns_has_mask_true` | Non-empty mask -> has_mask=true |
| `test_happy_path_returns_mask_rle` | RLE counts sum + size match target |
| `test_happy_path_returns_bbox` | Non-zero bbox for non-empty mask |
| `test_happy_path_returns_confidence` | Mock default 1.0 surfaces |
| `test_happy_path_carries_references` | ref + mask pair still status 0 |

Layer 5 statuses (1 / 2):
| Test | What |
|------|------|
| `test_empty_mask_returns_status_one` | Mock(coverage=0.0) -> status 1 + counts=[] |
| `test_low_confidence_returns_status_two` | Mock(confidence=0.1) -> status 2 |

Layer 1 / 3 statuses (3 / 6 / 7):
| Test | What |
|------|------|
| `test_invalid_target_shape_returns_status_three` | RGBA target -> 3 |
| `test_garbage_bytes_target_returns_status_three` | PIL parse fail -> 3 |
| `test_reference_count_mismatch_returns_status_seven` | 2 refs + 1 mask -> 7 |
| `test_reference_size_mismatch_returns_status_six` | pair size differ -> 6 |

Layer 4 status (11):
| Test | What |
|------|------|
| `test_backend_error_returns_status_eleven` | Wrapped BoomBackend -> 11 |

Error response shape:
| Test | What |
|------|------|
| `test_error_message_carried_on_failure` | non-empty error_message |
| `test_error_response_still_carries_full_shape` | every field serialised |

Threshold control:
| Test | What |
|------|------|
| `test_confidence_threshold_form_field_overrides_default` | form field flows in |

### test/adapters/fastapi/test_models.py (6)

| Test | What |
|------|------|
| `test_mask_rle_model_accepts_minimal_inputs` | Empty counts + size valid |
| `test_mask_rle_model_round_trip` | dump -> validate preserves fields |
| `test_bbox_model_carries_four_ints` | All four int fields |
| `test_segment_response_carries_all_required_fields` | Full response instantiates |
| `test_segment_response_round_trip` | dump -> validate preserves SegmentResponse |
| `test_segment_response_rejects_missing_field` | Missing has_bbox -> ValidationError |

### test/conftest.py (0 tests — shared fixtures only)

| Fixture | What |
|---------|------|
| `boom_class` | Returns a `BackendInterface` subclass taking `exc_factory` callable; lets tests build error-raising backends without per-case inline class definitions |

### test/test_copyright.py (1, skipped)

Marked `@pytest.mark.skip` per `core_sam_bridge` convention until a
per-file copyright header policy is decided.

### test/test_flake8.py (1)

Standard `ament_flake8.main_with_errors()` runner.

### test/test_pep257.py (1)

Standard `ament_pep257.main` runner with Google Style docstring
ignores: `D213, D401, D403, D406, D407, D413`.
