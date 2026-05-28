"""sam_manager Layer 4 — backend abstraction contract.

Backends (SegGPT, VRP-SAM, PerSAM-F, ...) implement BackendInterface
to produce raw segmentation output: masks + per-mask class ids +
backend-internal telemetry (latency, GPU memory, per-call confidence
value the backend computed from its own internals such as logit
distribution).

has_mask / has_bbox flag setting, RLE encoding, bbox computation, and
status_code determination happen downstream in Layer 5 ConfidenceGate
which consumes the InferResult. The confidence *value* is backend
knowledge (only SegGPT knows what its logits mean), but the threshold
that turns that value into status 2 LOW_CONFIDENCE is Layer 5's call.
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

    ``confidence`` is the backend's self-reported confidence in
    ``masks[0]`` (range 0..1). Backends derive it from internal signals:
    SegGPT computes it from per-pixel logit distribution, MockBackend
    takes it as a constructor parameter for deterministic testing.
    Layer 5 ConfidenceGate applies the threshold to emit status 2
    LOW_CONFIDENCE; the value itself stays in the backend's hands.
    ``None`` means "backend did not report" -- Layer 5 treats unreported
    as ``1.0`` (assume confident) so legacy backends that pre-date this
    field do not start tripping status 2.
    """

    target: np.ndarray
    masks: List[np.ndarray]
    class_ids: List[int]
    latency_ms: Optional[float] = None
    gpu_mem_mb: Optional[float] = None
    confidence: Optional[float] = None


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
