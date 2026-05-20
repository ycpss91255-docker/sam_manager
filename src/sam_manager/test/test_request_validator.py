"""Unit tests for sam_manager.core.request_validator (Layer 1).

Per architecture v4 §4 + drawio Page 4 status_code emit table, Layer 1
RequestValidator owns status_code 3 INVALID_IMAGE for target shape /
dtype / non-empty checks. This module is the ROS-agnostic core
contract -- 4 Frontend Adapters (ROS 2 Service / Topic / FastAPI /
Python API) will all call validate_target before forwarding to the
backend, so the typed exception InvalidImageError lets each adapter
map the failure to its wire representation (e.g. ROS 2 srv response
status_code = 3).

InvalidImageError subclasses ValueError so the existing
ErrorHandlingBackend wrapper (Layer 4) propagates it unchanged --
matches the docstring contract that input-validation errors bypass
status_code 11 BACKEND_ERROR mapping.
"""
import numpy as np
import pytest

from sam_manager.core.request_validator import (
    InvalidImageError,
    validate_target,
)


def _rgb(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ─────────────────────────── Exception contract ───────────────────────


def test_invalid_image_error_status_code_is_three():
    """InvalidImageError.STATUS_CODE == 3 per drawio Page 4 emit table."""
    assert InvalidImageError.STATUS_CODE == 3


def test_invalid_image_error_subclasses_value_error():
    """InvalidImageError IS-A ValueError so ErrorHandlingBackend propagates it unchanged."""
    assert issubclass(InvalidImageError, ValueError)


def test_invalid_image_error_carries_message():
    """Raised instance preserves caller-provided diagnostic message."""
    err = InvalidImageError("custom diagnostic")
    assert "custom diagnostic" in str(err)


# ─────────────────────────── Happy path ───────────────────────────────


def test_accepts_valid_rgb_uint8():
    """Valid (H, W, 3) uint8 RGB returns None and does not raise."""
    assert validate_target(_rgb(40, 60)) is None


def test_accepts_minimum_size_one_by_one():
    """1x1 RGB image is valid (non-empty edge case)."""
    assert validate_target(_rgb(1, 1)) is None


# ─────────────────────────── Shape rejection ─────────────────────────


def test_rejects_2d_grayscale():
    """Shape (H, W) without channel dim raises."""
    gray = np.zeros((40, 40), dtype=np.uint8)
    with pytest.raises(InvalidImageError):
        validate_target(gray)


def test_rejects_rgba_four_channels():
    """Shape (H, W, 4) is rejected -- only RGB accepted."""
    rgba = np.zeros((40, 40, 4), dtype=np.uint8)
    with pytest.raises(InvalidImageError):
        validate_target(rgba)


def test_rejects_pseudo_3d_single_channel():
    """Shape (H, W, 1) is rejected -- not real RGB."""
    pseudo = np.zeros((40, 40, 1), dtype=np.uint8)
    with pytest.raises(InvalidImageError):
        validate_target(pseudo)


def test_rejects_ndim_one():
    """1D array is rejected."""
    flat = np.zeros((100,), dtype=np.uint8)
    with pytest.raises(InvalidImageError):
        validate_target(flat)


def test_rejects_ndim_four():
    """4D array is rejected (e.g. batched (N, H, W, 3))."""
    batched = np.zeros((2, 40, 40, 3), dtype=np.uint8)
    with pytest.raises(InvalidImageError):
        validate_target(batched)


# ─────────────────────────── dtype rejection ─────────────────────────


def test_rejects_float32_dtype():
    """float32 target is rejected -- only uint8 accepted."""
    f32 = np.zeros((40, 40, 3), dtype=np.float32)
    with pytest.raises(InvalidImageError):
        validate_target(f32)


def test_rejects_uint16_dtype():
    """uint16 target is rejected -- only uint8 accepted."""
    u16 = np.zeros((40, 40, 3), dtype=np.uint16)
    with pytest.raises(InvalidImageError):
        validate_target(u16)


def test_rejects_bool_dtype():
    """bool target is rejected -- only uint8 accepted."""
    b = np.zeros((40, 40, 3), dtype=bool)
    with pytest.raises(InvalidImageError):
        validate_target(b)


# ─────────────────────────── Empty rejection ─────────────────────────


def test_rejects_empty_zero_height():
    """Shape (0, W, 3) is rejected as empty."""
    empty = np.zeros((0, 40, 3), dtype=np.uint8)
    with pytest.raises(InvalidImageError):
        validate_target(empty)


def test_rejects_empty_zero_width():
    """Shape (H, 0, 3) is rejected as empty."""
    empty = np.zeros((40, 0, 3), dtype=np.uint8)
    with pytest.raises(InvalidImageError):
        validate_target(empty)


def test_rejects_empty_zero_by_zero():
    """Shape (0, 0, 3) is rejected as empty."""
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    with pytest.raises(InvalidImageError):
        validate_target(empty)


# ─────────────────────────── Error diagnostics ────────────────────────


def test_error_message_mentions_shape():
    """Error message includes the offending shape for debugging."""
    gray = np.zeros((40, 40), dtype=np.uint8)
    with pytest.raises(InvalidImageError) as excinfo:
        validate_target(gray)
    assert "(40, 40)" in str(excinfo.value) or "shape" in str(excinfo.value).lower()


def test_error_message_mentions_dtype():
    """Error message includes the offending dtype for debugging."""
    f32 = np.zeros((40, 40, 3), dtype=np.float32)
    with pytest.raises(InvalidImageError) as excinfo:
        validate_target(f32)
    assert "float32" in str(excinfo.value) or "dtype" in str(excinfo.value).lower()
