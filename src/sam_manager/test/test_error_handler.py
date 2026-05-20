"""Unit tests for sam_manager.error_handler (Layer 4 ErrorHandler).

Per CLAUDE.md / SKILL.md status_code emit table, Layer 4 ErrorHandler
maps backend internal failures to status_code 11 BACKEND_ERROR.
Input-validation errors (ValueError) propagate unchanged so the
caller (Layer 1 RequestValidator) emits the appropriate
status_code (3 / 4 / 6 / 7 / 8) instead.
"""
import numpy as np
import pytest

from sam_manager.core.backend import BackendInterface, InferResult
from sam_manager.core.backends.mock import MockBackend
from sam_manager.core.error_handler import BackendError, ErrorHandlingBackend


def _rgb(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ─────────────────────────── Happy path ───────────────────────────────


def test_wraps_successful_infer():
    """A successful infer() passes through unchanged."""
    wrapped = ErrorHandlingBackend(MockBackend())
    result = wrapped.infer(_rgb(40, 40), [], [])
    assert isinstance(result, InferResult)
    assert result.masks[0].shape == (40, 40)


def test_subclasses_backend_interface():
    """ErrorHandlingBackend is itself a BackendInterface (Decorator pattern)."""
    assert issubclass(ErrorHandlingBackend, BackendInterface)


def test_polymorphism_via_base_type():
    """Caller can hold ErrorHandlingBackend through BackendInterface type."""
    wrapped: BackendInterface = ErrorHandlingBackend(MockBackend())
    result = wrapped.infer(_rgb(20, 20), [], [])
    assert isinstance(result, InferResult)


# ─────────────────────────── BackendError mapping ─────────────────────


def test_runtime_error_wrapped_as_backend_error():
    """RuntimeError from inner backend raises BackendError(status_code=11)."""
    class BoomBackend(MockBackend):
        def infer(self, *_args, **_kwargs):
            raise RuntimeError("CUDA out of memory")

    wrapped = ErrorHandlingBackend(BoomBackend())
    with pytest.raises(BackendError) as excinfo:
        wrapped.infer(_rgb(20, 20), [], [])
    assert excinfo.value.status_code == 11
    assert isinstance(excinfo.value.original, RuntimeError)
    assert "CUDA out of memory" in str(excinfo.value.original)


def test_generic_exception_wrapped_as_backend_error():
    """Any non-ValueError exception from inner → BackendError(11)."""
    class BoomBackend(MockBackend):
        def infer(self, *_args, **_kwargs):
            raise KeyError("model_weights")

    wrapped = ErrorHandlingBackend(BoomBackend())
    with pytest.raises(BackendError) as excinfo:
        wrapped.infer(_rgb(20, 20), [], [])
    assert excinfo.value.status_code == 11
    assert isinstance(excinfo.value.original, KeyError)


def test_value_error_propagates_unchanged():
    """ValueError from inner backend propagates (not wrapped) — Layer 1 owns it."""
    wrapped = ErrorHandlingBackend(MockBackend())
    # MockBackend raises ValueError on grayscale target — should bubble up
    # as ValueError, not BackendError.
    with pytest.raises(ValueError):
        wrapped.infer(np.zeros((10, 10), dtype=np.uint8), [], [])


# ─────────────────────────── Parametrized exception coverage ─────────
#
# These cases pull the BoomBackend class from the shared `boom_class`
# pytest fixture (test/conftest.py) so test_error_handler does not
# redefine an inner BoomBackend per exception type. Parametrization
# also makes the seam between ErrorHandlingBackend and any concrete
# raising backend explicit -- the wrapper must behave the same way
# regardless of which non-ValueError exception the inner raises.


@pytest.mark.parametrize("exc_type", [
    RuntimeError,
    KeyError,
    OSError,
    ZeroDivisionError,
    ConnectionError,
])
def test_non_value_error_wrapped_as_backend_error(boom_class, exc_type):
    """Any non-ValueError exception is wrapped with status_code 11."""
    inner = boom_class(lambda: exc_type("boom"))
    wrapped = ErrorHandlingBackend(inner)
    with pytest.raises(BackendError) as excinfo:
        wrapped.infer(_rgb(20, 20), [], [])
    assert excinfo.value.status_code == 11
    assert isinstance(excinfo.value.original, exc_type)


@pytest.mark.parametrize("exc_type", [
    RuntimeError,
    KeyError,
    OSError,
    ZeroDivisionError,
    ConnectionError,
])
def test_non_value_error_preserves_cause_chain(boom_class, exc_type):
    """BackendError preserves the original via raise ... from exc."""
    original = exc_type("specific message")
    inner = boom_class(lambda: original)
    wrapped = ErrorHandlingBackend(inner)
    with pytest.raises(BackendError) as excinfo:
        wrapped.infer(_rgb(20, 20), [], [])
    assert excinfo.value.__cause__ is original
    assert excinfo.value.original is original


def test_backend_error_constant_is_eleven():
    """BackendError.STATUS_CODE class constant is 11 (BACKEND_ERROR per CLAUDE.md)."""
    assert BackendError.STATUS_CODE == 11


def test_backend_error_holds_original_exception():
    """BackendError.original references the underlying exception object."""
    original = RuntimeError("boom")
    err = BackendError(original=original)
    assert err.original is original
    assert err.status_code == 11


def test_backend_error_str_includes_original_type_and_message():
    """str(BackendError) surfaces original exception type + message for debug."""
    original = RuntimeError("CUDA error 700")
    err = BackendError(original=original)
    s = str(err)
    assert "RuntimeError" in s
    assert "CUDA error 700" in s


# ─────────────────────────── Delegation ───────────────────────────────


def test_warmup_delegates_to_inner():
    """warmup() forwards to inner backend."""
    calls: list[str] = []

    class TraceBackend(MockBackend):
        def warmup(self) -> None:
            calls.append("inner-warmup")

    wrapped = ErrorHandlingBackend(TraceBackend())
    wrapped.warmup()
    assert calls == ["inner-warmup"]


def test_health_check_delegates_to_inner():
    """health_check() forwards to inner backend."""
    class UnhealthyBackend(MockBackend):
        def health_check(self) -> bool:
            return False

    wrapped = ErrorHandlingBackend(UnhealthyBackend())
    assert wrapped.health_check() is False
