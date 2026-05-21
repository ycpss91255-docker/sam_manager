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
#
# RuntimeError + KeyError single-case checks were removed -- the
# parametrized cases below (over 5 exception types) cover the same
# wrapping invariant without per-case inline class definitions.
# Message preservation is still asserted by
# test_backend_error_str_includes_original_type_and_message; original
# instance identity by test_backend_error_holds_original_exception.


def test_value_error_propagates_unchanged(boom_class):
    """ValueError from inner backend propagates unchanged (Layer 1 / 3 owns it).

    MockBackend no longer self-validates, so a backend-side ValueError
    now comes from a deterministic raise fixture rather than feeding
    malformed input to Mock. The wrapper invariant is the same:
    ``except ValueError: raise`` lets the exception bubble out without
    remapping to status_code 11 BACKEND_ERROR.
    """
    inner = boom_class(lambda: ValueError("validator-side error"))
    wrapped = ErrorHandlingBackend(inner)
    with pytest.raises(ValueError):
        wrapped.infer(_rgb(20, 20), [], [])


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
