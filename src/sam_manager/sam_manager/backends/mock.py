"""MockBackend -- deterministic synthetic backend for unit testing.

Returns a single mask covering a configurable centered rectangle of the
target. Used by downstream Layer 1 / 2 / 5 tests so they need neither GPU
nor a real `seggpt` install.
"""
from typing import List

import numpy as np
import structlog

from sam_manager.backend import BackendInterface, InferResult

_logger = structlog.get_logger("sam_manager.backends.mock")


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
        half = int(np.sqrt(self._coverage) * 0.5 * min(height, width))
        cy, cx = height // 2, width // 2

        mask = np.zeros((height, width), dtype=np.uint8)
        if half > 0:
            mask[cy - half:cy + half, cx - half:cx + half] = 255

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
        """Raise ValueError on malformed input; mirrors seggpt Layer 2 contract."""
        if target.ndim != 3 or target.shape[2] != 3 or target.dtype != np.uint8:
            raise ValueError(
                "target must be (H, W, 3) uint8, "
                f"got shape={target.shape} dtype={target.dtype}")
        if target.size == 0:
            raise ValueError("target must not be empty")
        if len(refs) != len(masks):
            raise ValueError(
                f"refs ({len(refs)}) and masks ({len(masks)}) length mismatch")
        for i, (ref, ref_mask) in enumerate(zip(refs, masks)):
            if ref.ndim != 3 or ref.shape[2] != 3 or ref.dtype != np.uint8:
                raise ValueError(
                    f"refs[{i}] must be (H, W, 3) uint8, "
                    f"got shape={ref.shape} dtype={ref.dtype}")
            if ref_mask.ndim != 2 or ref_mask.dtype != np.uint8:
                raise ValueError(
                    f"masks[{i}] must be (H, W) uint8, "
                    f"got shape={ref_mask.shape} dtype={ref_mask.dtype}")
