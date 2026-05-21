"""Unit tests for sam_manager.backends.mock.MockBackend."""
import numpy as np
import pytest

from sam_manager.core.backend import InferResult
from sam_manager.core.backends.mock import MockBackend


def _rgb(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _gray(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def test_output_shape_matches_target():
    """result.masks[0] has the target's H, W."""
    target = _rgb(80, 120)
    result = MockBackend().infer(target, [], [])
    assert result.masks[0].shape == (80, 120)


def test_output_count_is_one_mask_one_class():
    """Mock always emits a single mask with class_id=0."""
    result = MockBackend().infer(_rgb(40, 40), [], [])
    assert len(result.masks) == 1
    assert len(result.class_ids) == 1
    assert result.class_ids[0] == 0


def test_target_is_reference_not_copy():
    """InferResult.target is the same ndarray object the caller passed in."""
    target = _rgb(20, 30)
    result = MockBackend().infer(target, [], [])
    assert result.target is target


def test_mask_coverage_matches_param():
    """Coverage 0.5 produces ~50% positive pixels (allow +-5% due to rounding)."""
    target = _rgb(200, 200)
    result = MockBackend(mask_coverage=0.5).infer(target, [], [])
    coverage = (result.masks[0] == 255).sum() / result.masks[0].size
    assert 0.45 <= coverage <= 0.55


def test_determinism_same_input_same_output():
    """Two calls with identical input return byte-identical masks."""
    target = _rgb(64, 64)
    backend = MockBackend(mask_coverage=0.3)
    r1 = backend.infer(target, [], [])
    r2 = backend.infer(target, [], [])
    assert np.array_equal(r1.masks[0], r2.masks[0])


def test_telemetry_is_none_for_mock():
    """Mock does not report latency or gpu memory."""
    result = MockBackend().infer(_rgb(8, 8), [], [])
    assert result.latency_ms is None
    assert result.gpu_mem_mb is None


def test_constructor_rejects_out_of_range_coverage():
    """mask_coverage outside [0, 1] raises ValueError."""
    with pytest.raises(ValueError):
        MockBackend(mask_coverage=-0.1)
    with pytest.raises(ValueError):
        MockBackend(mask_coverage=1.5)


# Input validation cases (target / refs / masks shape / dtype / pair
# alignment) were deleted when MockBackend retired its internal
# _validate. The same coverage now lives in:
#   - test/test_request_validator.py (Layer 1, status 3)
#   - test/test_prompt_store.py       (Layer 3, status 6 / 7)
#   - test/adapters/python/test_segment.py (validators run via the
#     Python API Frontend Adapter on the realistic call path)


# ─────────────────────────── Output validation ───────────────────────────


def test_output_is_inferresult_instance():
    """Return type must be InferResult (not raw dict / tuple)."""
    result = MockBackend().infer(_rgb(20, 20), [], [])
    assert isinstance(result, InferResult)


def test_output_mask_dtype_is_uint8():
    """result.masks[0] dtype must be uint8 — downstream Layer 5 assumes uint8."""
    result = MockBackend().infer(_rgb(40, 40), [], [])
    assert result.masks[0].dtype == np.uint8


def test_output_mask_values_only_zero_or_255():
    """result.masks[0] is binary (only 0 / 255), no gradient values."""
    result = MockBackend(mask_coverage=0.4).infer(_rgb(80, 80), [], [])
    unique = set(np.unique(result.masks[0]).tolist())
    assert unique.issubset({0, 255}), f"non-binary values: {unique}"


def test_output_mask_geometry_is_centered():
    """Mask rectangle is centered: center pixel == 255, four corners == 0."""
    target = _rgb(100, 100)
    result = MockBackend(mask_coverage=0.25).infer(target, [], [])
    mask = result.masks[0]
    h, w = mask.shape
    assert mask[h // 2, w // 2] == 255, "center pixel not set"
    assert mask[0, 0] == 0, "top-left corner unexpectedly set"
    assert mask[0, w - 1] == 0, "top-right corner unexpectedly set"
    assert mask[h - 1, 0] == 0, "bottom-left corner unexpectedly set"
    assert mask[h - 1, w - 1] == 0, "bottom-right corner unexpectedly set"


def test_output_class_ids_exactly_zero():
    """class_ids must be [0] exactly — single class, id 0."""
    result = MockBackend().infer(_rgb(20, 20), [], [])
    assert result.class_ids == [0]


def test_output_target_dtype_shape_unchanged():
    """Backend must not mutate caller's target — dtype + shape preserved."""
    target = _rgb(50, 70)
    orig_dtype = target.dtype
    orig_shape = target.shape
    orig_sum = target.sum()
    MockBackend().infer(target, [], [])
    assert target.dtype == orig_dtype
    assert target.shape == orig_shape
    assert target.sum() == orig_sum, "backend mutated target buffer"


# ─────────────────────────── Boundary / geometry ────────────────────────


def test_mask_coverage_zero_produces_all_zero_mask():
    """mask_coverage=0.0 returns a mask with no positive pixels."""
    result = MockBackend(mask_coverage=0.0).infer(_rgb(100, 100), [], [])
    assert (result.masks[0] == 255).sum() == 0


def test_mask_coverage_one_fills_image():
    """mask_coverage=1.0 returns a near-fully-filled mask (>=95%)."""
    result = MockBackend(mask_coverage=1.0).infer(_rgb(100, 100), [], [])
    coverage = (result.masks[0] == 255).sum() / result.masks[0].size
    assert coverage >= 0.95, f"coverage=1.0 produced only {coverage:.2%}"


def test_mask_coverage_nonsquare_target():
    """Coverage must match parameter even for non-square target (e.g. 800x600)."""
    target = _rgb(600, 800)
    result = MockBackend(mask_coverage=0.25).infer(target, [], [])
    coverage = (result.masks[0] == 255).sum() / result.masks[0].size
    assert 0.20 <= coverage <= 0.30, \
        f"non-square 800x600 coverage=0.25 produced {coverage:.2%}"


# ─────────────────────────── structlog emit ───────────────────────────


def test_log_emits_infer_started_with_attributes():
    """backend_infer_started event carries n_refs, target_shape, backend."""
    import structlog
    with structlog.testing.capture_logs() as logs:
        MockBackend().infer(_rgb(30, 40), [], [])
    started = [e for e in logs if e["event"] == "backend_infer_started"]
    assert len(started) == 1, f"expected 1 started event, got {len(started)}"
    e = started[0]
    assert e["n_refs"] == 0
    assert e["target_shape"] == [30, 40, 3]
    assert e["backend"] == "mock"


def test_log_emits_infer_completed_with_attributes():
    """backend_infer_completed event carries n_masks, backend."""
    import structlog
    with structlog.testing.capture_logs() as logs:
        MockBackend().infer(_rgb(20, 20), [], [])
    completed = [e for e in logs if e["event"] == "backend_infer_completed"]
    assert len(completed) == 1
    e = completed[0]
    assert e["n_masks"] == 1
    assert e["backend"] == "mock"


# test_log_skips_emission_on_validation_failure was deleted along
# with Mock's _validate. Mock now trusts the caller has validated;
# unconditional emit of backend_infer_started / completed is the
# documented behavior (the structlog stream is the trace of "backend
# ran", not "request was valid"). Validation-failure observability
# moves to the Frontend Adapter, which logs the typed exception
# before the backend is ever called.
