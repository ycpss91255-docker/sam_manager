"""Unit tests for sam_manager.backend (BackendInterface ABC + InferResult)."""
import numpy as np
import pytest

from sam_manager.backend import BackendInterface, InferResult
from sam_manager.backends.mock import MockBackend


def test_abc_cannot_instantiate_directly():
    """BackendInterface is abstract -- TypeError on direct instantiation."""
    with pytest.raises(TypeError):
        BackendInterface()  # type: ignore[abstract]


def test_mock_subclasses_backend_interface():
    """MockBackend explicitly inherits BackendInterface."""
    assert issubclass(MockBackend, BackendInterface)


def test_polymorphism_via_base_type():
    """Caller can hold a MockBackend through the BackendInterface type."""
    target = np.zeros((32, 32, 3), dtype=np.uint8)
    backend: BackendInterface = MockBackend()
    result = backend.infer(target, [], [])
    assert isinstance(result, InferResult)


def test_default_warmup_is_noop():
    """Default warmup() returns None and does not raise."""
    backend = MockBackend()
    assert backend.warmup() is None


def test_default_health_check_returns_true():
    """Default health_check() reports healthy."""
    backend = MockBackend()
    assert backend.health_check() is True
