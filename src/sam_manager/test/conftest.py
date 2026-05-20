"""Shared pytest fixtures for sam_manager tests.

Hosts cross-test helpers that would otherwise be redefined inline
in multiple test modules. Currently exposes:

- ``boom_class``: a configurable error-raising BackendInterface used
  by test_error_handler to parametrize over exception types. Promotes
  the previously-inline ``class BoomBackend(MockBackend)`` definitions
  to a single shared fixture so adding a new exception type to the
  coverage matrix does not require redefining the class.

Future fixtures (e.g. canned target / refs / masks builders) should
follow the same pattern: define the class / factory at module scope,
expose it via a ``@pytest.fixture`` that returns the factory.
"""
from typing import Callable, List

import numpy as np
import pytest

from sam_manager.core.backend import BackendInterface, InferResult


class _BoomBackend(BackendInterface):
    """BackendInterface that raises whatever the factory returns.

    Instances are produced via the ``boom_class`` fixture rather than
    constructed directly so test files do not need to import this
    private class explicitly.
    """

    def __init__(self, exc_factory: Callable[[], BaseException]) -> None:
        """Capture the exception factory.

        Args:
            exc_factory: zero-arg callable returning the Exception
                instance ``infer()`` should raise. Lets tests
                parametrize over exception types without redefining
                the backend per case.
        """
        self._exc_factory = exc_factory

    def infer(
        self,
        target: np.ndarray,
        refs: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> InferResult:
        """Always raise the configured exception; never returns."""
        raise self._exc_factory()


@pytest.fixture
def boom_class():
    """Return the BoomBackend class for parametrized exception tests.

    Returns:
        A BackendInterface subclass taking a single ``exc_factory``
        callable in its constructor. Tests typically call
        ``boom_class(lambda: SomeError("msg"))`` to build an instance
        whose ``infer()`` deterministically raises that exception.
    """
    return _BoomBackend
