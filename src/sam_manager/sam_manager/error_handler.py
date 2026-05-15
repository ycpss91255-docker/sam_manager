"""sam_manager Layer 4 ErrorHandler.

Per the CLAUDE.md / ros2-msg-design SKILL.md status_code emit table,
Layer 4 ErrorHandler is the layer responsible for catching backend
internal failures (GPU OOM, CUDA error, model load error, ...) and
mapping them to wire status_code 11 BACKEND_ERROR.

Input-validation errors (ValueError) are NOT wrapped — they propagate
unchanged so the upstream caller (Layer 1 RequestValidator) emits the
appropriate status_code (3 INVALID_IMAGE / 4 INVALID_HEADER /
6 REFERENCE_SIZE_MISMATCH / 7 REFERENCE_COUNT_MISMATCH /
8 INVALID_TF).

The Layer 4 ErrorHandler is implemented as a decorator around an
inner BackendInterface. This keeps the abstraction composable
(wrapping is opt-in) and lets MockBackend stay free of error-handling
concerns for downstream unit tests.
"""
from typing import List

import numpy as np
import structlog

from sam_manager.backend import BackendInterface, InferResult

_logger = structlog.get_logger("sam_manager.error_handler")


class BackendError(Exception):
    """Wraps a backend's runtime failure with the wire status_code.

    Attributes:
        original: the underlying exception raised by the inner backend.
        status_code: the wire status_code to surface in the ROS 2 srv
            response (11 BACKEND_ERROR by default).
    """

    STATUS_CODE = 11  # BACKEND_ERROR per CLAUDE.md status_code 表

    def __init__(self, original: Exception, status_code: int = STATUS_CODE) -> None:
        """Wrap an underlying backend exception with the wire status_code."""
        self.original = original
        self.status_code = status_code
        super().__init__(
            f"backend raised: {type(original).__name__}: {original}"
        )


class ErrorHandlingBackend(BackendInterface):
    """Layer 4 ErrorHandler — decorator that maps backend exceptions to BackendError.

    Wrapping semantics:
        ValueError       → propagates unchanged (Layer 1 emits 3/4/6/7/8)
        anything else    → BackendError(status_code=11)

    warmup() and health_check() forward to the inner backend so the
    decorator stays transparent for lifecycle calls.
    """

    def __init__(self, inner: BackendInterface) -> None:
        """Wrap an inner BackendInterface with exception-to-status mapping."""
        self._inner = inner

    def infer(
        self,
        target: np.ndarray,
        refs: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> InferResult:
        """Forward to inner backend; convert non-ValueError exceptions to BackendError."""
        try:
            return self._inner.infer(target, refs, masks)
        except ValueError:
            raise
        except Exception as exc:
            _logger.warning(
                "backend_infer_failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise BackendError(original=exc) from exc

    def warmup(self) -> None:
        """Forward to inner backend."""
        self._inner.warmup()

    def health_check(self) -> bool:
        """Forward to inner backend."""
        return self._inner.health_check()
