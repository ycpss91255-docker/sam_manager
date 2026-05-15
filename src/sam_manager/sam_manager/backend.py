"""sam_manager Layer 4 — backend abstraction contract.

Backends (SegGPT, VRP-SAM, PerSAM-F, ...) implement BackendInterface to
produce raw segmentation output. Confidence scoring, has_mask / has_bbox
flag setting, RLE encoding, and status_code determination all happen in
downstream Layer 5 ConfidenceGate, NOT inside backends.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class InferResult:
    """Result of a single inference call.

    `target` is held by reference (not deep-copied) -- caller already owns
    the buffer; the reference here lets downstream layers pair masks with
    their input image for local storage / log metadata without paying the
    cost of duplicating a multi-MB image per call.
    """

    target: np.ndarray
    masks: List[np.ndarray]
    class_ids: List[int]
    latency_ms: Optional[float] = None
    gpu_mem_mb: Optional[float] = None


class BackendInterface(ABC):
    """Abstract base for sam_manager backends.

    Subclasses must implement `infer`. `warmup` and `health_check` ship
    with safe defaults (no-op / always healthy) so trivial backends do
    not need to override them.
    """

    @abstractmethod
    def infer(
        self,
        target: np.ndarray,
        refs: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> InferResult:
        """Run inference.

        Args:
            target: (H, W, 3) uint8 RGB image.
            refs: list of (H, W, 3) uint8 RGB reference images.
            masks: list of (H, W) uint8 binary reference masks; same
                length as `refs`.

        Returns:
            InferResult carrying `target` (by reference), one or more
            output masks, per-mask class ids, and optional telemetry.

        Raises:
            ValueError: invalid input shape / dtype / length mismatch.
            RuntimeError: backend internal failure (GPU OOM, model load
                error, etc.).
        """

    def warmup(self) -> None:
        """Optional pre-allocation / model load.

        Called once at lifecycle on_configure. Default no-op; backends
        that benefit (e.g. SegGPT CUDA kernel JIT) override.
        """

    def health_check(self) -> bool:
        """Return True iff the backend can serve a request.

        Called at on_activate. Default always healthy; backends that
        require GPU / model files override to introspect state.
        """
        return True
