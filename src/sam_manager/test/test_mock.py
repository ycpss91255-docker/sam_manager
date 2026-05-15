"""Unit tests for sam_manager.backends.mock.MockBackend."""
import numpy as np
import pytest

from sam_manager.backends.mock import MockBackend


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


def test_validation_target_must_be_rgb_uint8():
    """target must be (H, W, 3) uint8."""
    backend = MockBackend()
    with pytest.raises(ValueError):
        backend.infer(_gray(40, 40), [], [])           # 2D grayscale
    with pytest.raises(ValueError):
        backend.infer(_rgb(40, 40).astype(np.float32), [], [])  # wrong dtype
    with pytest.raises(ValueError):
        backend.infer(np.zeros((0, 0, 3), dtype=np.uint8), [], [])  # empty


def test_validation_refs_and_masks_length_mismatch():
    """refs and masks lists must have the same length."""
    target = _rgb(20, 20)
    backend = MockBackend()
    with pytest.raises(ValueError):
        backend.infer(target, [_rgb(20, 20)], [])      # 1 ref, 0 masks
    with pytest.raises(ValueError):
        backend.infer(target, [], [_gray(20, 20)])     # 0 refs, 1 mask


def test_validation_refs_must_be_rgb_uint8():
    """Each ref must be (H, W, 3) uint8."""
    target = _rgb(20, 20)
    backend = MockBackend()
    with pytest.raises(ValueError):
        backend.infer(target, [_gray(20, 20)], [_gray(20, 20)])


def test_validation_masks_must_be_2d_uint8():
    """Each ref mask must be (H, W) uint8 (single-channel)."""
    target = _rgb(20, 20)
    backend = MockBackend()
    with pytest.raises(ValueError):
        backend.infer(target, [_rgb(20, 20)], [_rgb(20, 20)])  # 3-channel mask
