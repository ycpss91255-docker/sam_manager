"""FastAPI Frontend Adapter -- POST /segment HTTP route.

The second concrete Frontend Adapter per architecture v4 §4 (Python
API tracer landed first in #7). HTTP + multipart transport; the same
validate -> backend.infer -> ConfidenceGate.evaluate pipeline the
Python API uses, plus the bytes <-> ndarray conversion and the
typed-exception <-> status_code mapping HTTP needs.

create_app(backend, confidence_threshold=0.3) -> FastAPI

Pattern is a factory function rather than a module-level app:
  - Tests instantiate with MockBackend()
  - Production wires SegGPTBackend() (or any other concrete backend)
  - The adapter never owns backend selection; that is the caller's
    responsibility (matches the Python API tracer in #7).
  - Caller is also responsible for wrapping with ErrorHandlingBackend
    if they want non-ValueError exceptions remapped to status 11.

Convention: HTTP 200 + status_code field for every anticipated
outcome (0, 1, 2, 3, 6, 7, 11). HTTP 4xx / 5xx is reserved for
transport-layer problems FastAPI itself surfaces (missing form
field, malformed multipart, FastAPI internal failure) -- those have
no status_code in our domain table.
"""
import io
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image, UnidentifiedImageError

from sam_manager.adapters.fastapi.models import (
    BBoxModel,
    MaskRLEModel,
    SegmentResponse,
)
from sam_manager.core.backend import BackendInterface
from sam_manager.core.confidence_gate import (
    STATUS_EMPTY_MASK,
    STATUS_LOW_CONFIDENCE,
    STATUS_OK,
    ConfidenceGateOutput,
    evaluate,
)
from sam_manager.core.error_handler import BackendError
from sam_manager.core.prompt_store import (
    ReferenceCountMismatchError,
    ReferenceSizeMismatchError,
    validate_references,
)
from sam_manager.core.request_validator import (
    InvalidImageError,
    validate_target,
)


def create_app(
    backend: BackendInterface,
    confidence_threshold: float = 0.3,
) -> FastAPI:
    """Build a FastAPI app exposing POST /segment.

    Args:
        backend: any BackendInterface. The adapter does not select.
            Wrap with ErrorHandlingBackend if you want non-ValueError
            exceptions remapped to status 11 (BackendError).
        confidence_threshold: default threshold passed to Layer 5
            ConfidenceGate when the request does not override.

    Returns:
        A FastAPI app instance. Caller mounts it under uvicorn or
        plugs into TestClient for unit tests.
    """
    app = FastAPI(title="sam_manager FastAPI Frontend Adapter")

    @app.post("/segment", response_model=SegmentResponse)
    async def segment(
        target: UploadFile = File(...),
        reference_images: List[UploadFile] = File(default_factory=list),
        reference_masks: List[UploadFile] = File(default_factory=list),
        confidence_threshold: Optional[float] = Form(default=None),
    ) -> SegmentResponse:
        """Run inference on the uploaded target with optional references."""
        effective_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else _default_threshold
        )

        try:
            target_arr = await _decode_target(target)
            refs_arr = [await _decode_rgb(r) for r in reference_images]
            masks_arr = [await _decode_mask(m) for m in reference_masks]

            validate_target(target_arr)
            validate_references(refs_arr, masks_arr)
            result = backend.infer(target_arr, refs_arr, masks_arr)
            gate_out = evaluate(
                result,
                confidence_threshold=effective_threshold,
            )
            return _pack_success(gate_out)
        except (
            InvalidImageError,
            ReferenceCountMismatchError,
            ReferenceSizeMismatchError,
            BackendError,
        ) as exc:
            return _pack_failure(status_code=exc.STATUS_CODE, message=str(exc))

    _default_threshold = confidence_threshold
    return app


async def _decode_target(upload: UploadFile) -> np.ndarray:
    """Decode the target image upload. PIL failure -> InvalidImageError."""
    return await _decode_rgb(upload, on_pil_fail=InvalidImageError)


async def _decode_rgb(
    upload: UploadFile,
    on_pil_fail: type = ValueError,
) -> np.ndarray:
    """Decode an upload as an RGB uint8 ndarray.

    PIL.UnidentifiedImageError + any decode failure surface as
    ``on_pil_fail`` so the route can map them onto a wire status
    (target -> InvalidImageError; reference -> ValueError that
    propagates to the validator and emerges as status 6 / 7).
    """
    raw = await upload.read()
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except UnidentifiedImageError as exc:
        raise on_pil_fail(f"upload is not a parseable image: {exc}") from exc
    except Exception as exc:
        raise on_pil_fail(f"upload could not be decoded: {exc}") from exc

    arr = np.asarray(img)
    # Preserve the original mode -- the downstream validator decides
    # whether (H, W, 3) / (H, W, 4) / (H, W) is acceptable. Forcing
    # everything to RGB here would mask channel-count mistakes and
    # turn status 3 INVALID_IMAGE into silent success.
    return arr


async def _decode_mask(upload: UploadFile) -> np.ndarray:
    """Decode a reference mask upload as a (H, W) uint8 ndarray."""
    raw = await upload.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("L")
        img.load()
    except UnidentifiedImageError as exc:
        raise ValueError(f"mask upload is not parseable: {exc}") from exc
    return np.asarray(img)


def _pack_success(gate: ConfidenceGateOutput) -> SegmentResponse:
    return SegmentResponse(
        status_code=gate.status_code,
        has_mask=gate.has_mask,
        has_bbox=gate.has_bbox,
        mask_rle=MaskRLEModel(
            counts=gate.mask_rle_counts,
            size=list(gate.mask_rle_size),
        ),
        bbox=BBoxModel(
            x_offset=gate.bbox[0],
            y_offset=gate.bbox[1],
            width=gate.bbox[2],
            height=gate.bbox[3],
        ),
        confidence=gate.confidence,
        error_message=_default_error_message(gate.status_code),
    )


def _pack_failure(status_code: int, message: str) -> SegmentResponse:
    return SegmentResponse(
        status_code=status_code,
        has_mask=False,
        has_bbox=False,
        mask_rle=MaskRLEModel(counts=[], size=[0, 0]),
        bbox=BBoxModel(x_offset=0, y_offset=0, width=0, height=0),
        confidence=0.0,
        error_message=message,
    )


def _default_error_message(status_code: int) -> str:
    """Human-readable note for the Layer 5 statuses (0 / 1 / 2)."""
    if status_code == STATUS_OK:
        return ""
    if status_code == STATUS_EMPTY_MASK:
        return "backend produced no mask (status 1 EMPTY_MASK)"
    if status_code == STATUS_LOW_CONFIDENCE:
        return "backend confidence below threshold (status 2 LOW_CONFIDENCE)"
    return ""
