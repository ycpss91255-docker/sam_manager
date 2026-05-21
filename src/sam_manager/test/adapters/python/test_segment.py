"""Unit tests for sam_manager.adapters.python.segment.

Python API Frontend Adapter -- the smallest tracer-bullet vertical
through the validators + backend stack. Tests verify:

  - the adapter calls Layer 1 validate_target before anything else
  - the adapter calls Layer 3 validate_references before
    backend.infer
  - typed exceptions from Layer 1 / 3 propagate unchanged (caller
    catches by type and maps to wire status_code at its own layer;
    Python API does not own wire encoding)
  - BackendError from Layer 4 ErrorHandlingBackend propagates
    unchanged
  - happy path returns the backend's InferResult untouched

Why a Python API Frontend Adapter exists per architecture v4 §4:
internal-test only, not on the production wire. Its value here is
forcing the validators out of MockBackend (Mock retires _validate
in the same PR) and pinning the adapter shape that ROS 2 / FastAPI
adapters will mirror.
"""
import numpy as np
import pytest

from sam_manager.adapters.python.segment import segment
from sam_manager.core.backend import InferResult
from sam_manager.core.backends.mock import MockBackend
from sam_manager.core.error_handler import BackendError, ErrorHandlingBackend
from sam_manager.core.prompt_store import (
    ReferenceCountMismatchError,
    ReferenceSizeMismatchError,
)
from sam_manager.core.request_validator import InvalidImageError


def _rgb(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _gray(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


# ─────────────────────────── Happy path ───────────────────────────────


def test_returns_backend_infer_result_unchanged():
    """Adapter returns backend's InferResult unwrapped."""
    backend = MockBackend()
    target = _rgb(40, 40)
    result = segment(backend, target, [], [])
    assert isinstance(result, InferResult)
    assert result.target is target
    assert result.masks[0].shape == (40, 40)


def test_passes_references_through_to_backend():
    """Adapter forwards refs / masks to backend without mutation."""
    backend = MockBackend()
    target = _rgb(40, 40)
    refs = [_rgb(40, 40), _rgb(40, 40)]
    masks = [_gray(40, 40), _gray(40, 40)]
    result = segment(backend, target, refs, masks)
    assert isinstance(result, InferResult)


def test_polymorphic_backend_accepted():
    """Adapter accepts any BackendInterface (Decorator-wrapped Mock works)."""
    wrapped = ErrorHandlingBackend(MockBackend())
    result = segment(wrapped, _rgb(30, 30), [], [])
    assert isinstance(result, InferResult)


# ─────────────────────────── Layer 1 propagation ─────────────────────


def test_invalid_image_error_propagates_unchanged():
    """validate_target failure surfaces as InvalidImageError, not BackendError."""
    backend = MockBackend()
    with pytest.raises(InvalidImageError) as excinfo:
        segment(backend, _gray(40, 40), [], [])
    assert excinfo.value.STATUS_CODE == 3


def test_invalid_image_rejection_skips_backend_call():
    """Backend.infer is NOT called when validate_target raises."""
    sentinel: list = []

    class TraceBackend(MockBackend):
        def infer(self, *_a, **_kw):
            sentinel.append("called")
            return super().infer(*_a, **_kw)

    backend = TraceBackend()
    with pytest.raises(InvalidImageError):
        segment(backend, _gray(40, 40), [], [])
    assert sentinel == [], "backend.infer must not run when target is invalid"


# ─────────────────────────── Layer 3 propagation ─────────────────────


def test_reference_count_mismatch_propagates_unchanged():
    """validate_references count check surfaces ReferenceCountMismatchError."""
    backend = MockBackend()
    with pytest.raises(ReferenceCountMismatchError) as excinfo:
        segment(backend, _rgb(40, 40), [_rgb(40, 40)], [])
    assert excinfo.value.STATUS_CODE == 7


def test_reference_size_mismatch_propagates_unchanged():
    """validate_references pair check surfaces ReferenceSizeMismatchError."""
    backend = MockBackend()
    with pytest.raises(ReferenceSizeMismatchError) as excinfo:
        segment(backend, _rgb(40, 40), [_rgb(40, 40)], [_gray(50, 50)])
    assert excinfo.value.STATUS_CODE == 6


def test_reference_size_rejection_skips_backend_call():
    """Backend.infer is NOT called when validate_references raises."""
    sentinel: list = []

    class TraceBackend(MockBackend):
        def infer(self, *_a, **_kw):
            sentinel.append("called")
            return super().infer(*_a, **_kw)

    backend = TraceBackend()
    with pytest.raises(ReferenceSizeMismatchError):
        segment(backend, _rgb(40, 40), [_rgb(40, 40)], [_gray(50, 50)])
    assert sentinel == [], "backend.infer must not run when refs are invalid"


# ─────────────────────────── Layer 4 propagation ─────────────────────


def test_backend_error_propagates_unchanged(boom_class):
    """BackendError from wrapped backend surfaces unchanged."""
    inner = boom_class(lambda: RuntimeError("boom"))
    wrapped = ErrorHandlingBackend(inner)
    with pytest.raises(BackendError) as excinfo:
        segment(wrapped, _rgb(40, 40), [], [])
    assert excinfo.value.STATUS_CODE == 11


def test_value_error_from_unwrapped_backend_propagates(boom_class):
    """A bare ValueError from the backend bubbles up without remapping."""
    inner = boom_class(lambda: ValueError("manual raise"))
    # Direct adapter call (no ErrorHandlingBackend wrapper)
    with pytest.raises(ValueError):
        segment(inner, _rgb(40, 40), [], [])


# ─────────────────────────── Validation order ────────────────────────


def test_target_validation_runs_before_reference_validation():
    """Layer 1 raises first when both target and refs are invalid."""
    backend = MockBackend()
    # Target is grayscale (status 3) AND refs has count mismatch (status 7).
    # Adapter must surface InvalidImageError, not ReferenceCountMismatchError.
    with pytest.raises(InvalidImageError):
        segment(backend, _gray(40, 40), [_rgb(40, 40)], [])


def test_reference_validation_runs_before_backend_call():
    """Layer 3 raises before backend.infer when refs are invalid (target ok)."""
    sentinel: list = []

    class TraceBackend(MockBackend):
        def infer(self, *_a, **_kw):
            sentinel.append("called")
            return super().infer(*_a, **_kw)

    backend = TraceBackend()
    with pytest.raises(ReferenceCountMismatchError):
        segment(backend, _rgb(40, 40), [_rgb(40, 40)], [])
    assert sentinel == [], "backend.infer must wait until refs validated"
