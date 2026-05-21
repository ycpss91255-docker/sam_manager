"""MockBackend -- deterministic synthetic backend for unit testing.

Returns a single mask covering a configurable centered rectangle of the
target. Used by downstream Layer 1 / 2 / 5 tests so they need neither GPU
nor a real `seggpt` install.
"""
from typing import List

import numpy as np
import structlog

from sam_manager.core.backend import BackendInterface, InferResult
from sam_manager.core.prompt_store import validate_references
from sam_manager.core.request_validator import validate_target

_logger = structlog.get_logger("sam_manager.core.backends.mock")


class MockBackend(BackendInterface):
    """Synthetic backend that produces a fixed-coverage centered mask."""

    def __init__(self, mask_coverage: float = 0.25) -> None:
        """Initialize the mock with a fixed mask area fraction.

        Args:
            mask_coverage: fraction of the target area covered by the
                output mask, in [0, 1]. Default 0.25 (1/4 of the image).

        Raises:
            ValueError: mask_coverage is outside [0, 1].
        """
        if not 0.0 <= mask_coverage <= 1.0:
            raise ValueError(
                f"mask_coverage must be in [0, 1], got {mask_coverage}")
        self._coverage = mask_coverage

    def infer(
        self,
        target: np.ndarray,
        refs: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> InferResult:
        """Return a centered rectangular mask covering `mask_coverage` of target."""
        self._validate(target, refs, masks)

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
        )

        _logger.info(
            "backend_infer_completed",
            n_masks=len(result.masks),
            backend="mock",
        )
        return result

    @staticmethod
    def _validate(
        target: np.ndarray,
        refs: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> None:
        """Validate request inputs via Layer 1 + Layer 3 helpers.

        Target shape / dtype / non-empty (status 3 INVALID_IMAGE)
        delegates to Layer 1 ``validate_target``. Reference count
        (status 7) + per-pair shape / dtype / pair alignment
        (status 6) delegate to Layer 3 ``validate_references``.

        Once Frontend Adapter (ROS 2 srv handler / FastAPI handler)
        lands, this Mock-internal validation becomes redundant -- the
        adapter will call Layer 1 + Layer 3 before forwarding to
        ``infer()``. Kept here so MockBackend is self-contained for
        downstream unit tests in the meantime.
        """
        validate_target(target)
        validate_references(refs, masks)
