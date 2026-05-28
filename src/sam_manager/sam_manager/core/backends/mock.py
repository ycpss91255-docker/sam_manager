"""MockBackend -- deterministic synthetic backend for unit testing.

Returns a single mask covering a configurable centered rectangle of the
target. Used by downstream Layer 1 / 2 / 5 tests so they need neither GPU
nor a real `seggpt` install.
"""
from typing import List

import numpy as np
import structlog

from sam_manager.core.backend import BackendInterface, InferResult

_logger = structlog.get_logger("sam_manager.core.backends.mock")


class MockBackend(BackendInterface):
    """Synthetic backend that produces a fixed-coverage centered mask."""

    def __init__(
        self,
        mask_coverage: float = 0.25,
        confidence: float = 1.0,
    ) -> None:
        """Initialize the mock with a fixed mask area + confidence.

        Args:
            mask_coverage: fraction of the target area covered by the
                output mask, in [0, 1]. Default 0.25 (1/4 of the image).
            confidence: backend-reported confidence in the produced
                mask, in [0, 1]. Default 1.0 (max confident). Layer 5
                ConfidenceGate consumes this value to decide status 2
                LOW_CONFIDENCE.

        Raises:
            ValueError: mask_coverage or confidence is outside [0, 1].
        """
        if not 0.0 <= mask_coverage <= 1.0:
            raise ValueError(
                f"mask_coverage must be in [0, 1], got {mask_coverage}")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {confidence}")
        self._coverage = mask_coverage
        self._confidence = confidence

    def infer(
        self,
        target: np.ndarray,
        refs: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> InferResult:
        """Return a centered rectangular mask covering `mask_coverage` of target.

        The backend trusts the caller (Frontend Adapter or test
        fixture) to have already passed inputs through Layer 1
        ``validate_target`` + Layer 3 ``validate_references``. Mock
        no longer self-defends; this aligns its behavior with
        SegGPTBackend and future production backends so unit tests
        catch missing-validator bugs instead of silently masking
        them.
        """
        _logger.info(
            "backend_infer_started",
            n_refs=len(refs),
            target_shape=list(target.shape),
            backend="mock",
        )

        height, width = target.shape[:2]
        cy, cx = height // 2, width // 2

        # Aspect-ratio-preserving rectangle:
        # scale H, W independently by sqrt(coverage). This keeps the
        # mask centered with the same aspect as the target image, and
        # the resulting area / total area equals `coverage` (up to
        # integer-rounding bias on tiny images).
        scale = float(np.sqrt(self._coverage))
        rect_h = int(round(height * scale))
        rect_w = int(round(width * scale))
        half_h = rect_h // 2
        half_w = rect_w // 2

        mask = np.zeros((height, width), dtype=np.uint8)
        if half_h > 0 and half_w > 0:
            mask[cy - half_h:cy + half_h, cx - half_w:cx + half_w] = 255

        result = InferResult(
            target=target,
            masks=[mask],
            class_ids=[0],
            latency_ms=None,
            gpu_mem_mb=None,
            confidence=self._confidence,
        )

        _logger.info(
            "backend_infer_completed",
            n_masks=len(result.masks),
            backend="mock",
        )
        return result
