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
  python3 -m pip install structlog pytest-cov
  . /opt/ros/humble/setup.bash
  colcon build --packages-select sam_manager_msgs sam_manager
  . install/setup.bash
  colcon test --packages-select sam_manager --event-handlers console_direct+
  colcon test-result --verbose
  python3 -m pytest src/sam_manager/test/ \
    --cov=sam_manager --cov-report=term-missing --cov-fail-under=80
'
```

Total: **65 unit tests** + 1 skipped (`test_copyright` — opt-in once
a per-file header policy is decided) plus `ament_flake8` and
`ament_pep257` (both run via `colcon test`).

Line coverage: **100%** at this revision; CI gate is **80%** so the
remaining layers (2, 3, 5) can be added without immediately tightening
the bar.

## 4-category coverage

Per CLAUDE.md「TDD 測試分類（4 個面向）」:

| # | Category | Where | What it covers |
|---|----------|-------|----------------|
| 1 | Smoke | `test/test_*.py` (any) | Package + module import + ABC instantiability + `colcon build` success |
| 2 | Unit | `test/test_backend.py` / `test/test_mock.py` / `test/test_error_handler.py` / `test/test_request_validator.py` | BackendInterface ABC, MockBackend behaviour, ErrorHandling decorator, Layer 1 target validator |
| 3 | Integration | (placeholder — added when Layer 5 ConfidenceGate or Layer 1 ROS 2 handler lands and we have cross-layer flows to test) | TBD |
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

### test/test_mock.py (29)

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

Input validation (ValueError on malformed input):
| Test | What |
|------|------|
| `test_validation_target_must_be_rgb_uint8` | 2D / float32 / empty target |
| `test_validation_refs_and_masks_length_mismatch` | `len(refs) != len(masks)` |
| `test_validation_refs_must_be_rgb_uint8` | `refs[i]` 2D |
| `test_validation_masks_must_be_2d_uint8` | `masks[i]` 3-channel |
| `test_validation_pair_alignment` | `refs[i].shape[:2] != masks[i].shape` |
| `test_validation_target_rgba_four_channels` | target 4-channel RGBA |
| `test_validation_target_single_channel_pseudo_3d` | target (H, W, 1) |
| `test_validation_refs_dtype_non_uint8` | `refs[i]` dtype uint16 |
| `test_validation_masks_dtype_bool` | `masks[i]` dtype bool |
| `test_validation_masks_pseudo_3d` | `masks[i]` (H, W, 1) |

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
| `test_log_skips_emission_on_validation_failure` | No emit when `_validate` raises |

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

### test/test_error_handler.py (11)

| Test | What |
|------|------|
| `test_wraps_successful_infer` | Happy path passthrough |
| `test_subclasses_backend_interface` | `issubclass(ErrorHandlingBackend, BackendInterface)` |
| `test_polymorphism_via_base_type` | Caller holds wrapped through ABC |
| `test_runtime_error_wrapped_as_backend_error` | RuntimeError → BackendError(11) |
| `test_generic_exception_wrapped_as_backend_error` | KeyError → BackendError(11) |
| `test_value_error_propagates_unchanged` | ValueError NOT wrapped |
| `test_backend_error_constant_is_eleven` | `BackendError.STATUS_CODE == 11` |
| `test_backend_error_holds_original_exception` | `BackendError.original` is original ex |
| `test_backend_error_str_includes_original_type_and_message` | str surfaces type + msg |
| `test_warmup_delegates_to_inner` | warmup forwards |
| `test_health_check_delegates_to_inner` | health_check forwards |

### test/test_copyright.py (1, skipped)

Marked `@pytest.mark.skip` per `core_sam_bridge` convention until a
per-file copyright header policy is decided.

### test/test_flake8.py (1)

Standard `ament_flake8.main_with_errors()` runner.

### test/test_pep257.py (1)

Standard `ament_pep257.main` runner with Google Style docstring
ignores: `D213, D401, D403, D406, D407, D413`.
