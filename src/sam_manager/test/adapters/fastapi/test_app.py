"""Integration tests for sam_manager.adapters.fastapi.app.

The FastAPI Frontend Adapter takes the same role as the Python API
adapter (validate -> backend.infer -> evaluate) but speaks HTTP +
multipart. Tests use starlette.testclient.TestClient (httpx-backed)
so they exercise the full request -> validation -> backend ->
ConfidenceGate -> wire response path without spinning up uvicorn.

Wire response shape mirrors the ROS 2 srv response so the future
ROS 2 Frontend Adapter can lift the same packing logic verbatim.
HTTP 200 + status_code field is the convention for anticipated
errors (3 / 6 / 7 / 11). HTTP 4xx / 5xx is reserved for things the
adapter cannot map (FastAPI internal failures, missing form fields,
malformed multipart -- those are transport-layer problems).
"""
import io
from typing import List

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from sam_manager.adapters.fastapi.app import create_app
from sam_manager.core.backends.mock import MockBackend
from sam_manager.core.error_handler import ErrorHandlingBackend


def _png_bytes(arr: np.ndarray) -> bytes:
    """Encode an ndarray (RGB uint8 or grayscale) as PNG bytes."""
    if arr.ndim == 3:
        img = Image.fromarray(arr, mode="RGB")
    else:
        img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _rgb_png(h: int, w: int) -> bytes:
    return _png_bytes(np.zeros((h, w, 3), dtype=np.uint8))


def _gray_png(h: int, w: int) -> bytes:
    return _png_bytes(np.zeros((h, w), dtype=np.uint8))


def _client(backend=None) -> TestClient:
    """Build a TestClient with a configurable backend (Mock by default)."""
    return TestClient(create_app(backend or MockBackend()))


def _files(
    target: bytes,
    refs: List[bytes] = (),
    masks: List[bytes] = (),
) -> list:
    """Build the multipart payload FastAPI expects."""
    payload = [("target", ("target.png", target, "image/png"))]
    for i, r in enumerate(refs):
        payload.append(("reference_images", (f"ref_{i}.png", r, "image/png")))
    for i, m in enumerate(masks):
        payload.append(("reference_masks", (f"mask_{i}.png", m, "image/png")))
    return payload


# ─────────────────────────── Happy path (status 0) ───────────────────


def test_happy_path_returns_status_zero():
    """Mock with default confidence -> status_code 0 OK."""
    r = _client().post("/segment", files=_files(_rgb_png(40, 40)))
    assert r.status_code == 200
    assert r.json()["status_code"] == 0


def test_happy_path_returns_has_mask_true():
    """Mock with non-empty mask -> has_mask=true."""
    r = _client().post("/segment", files=_files(_rgb_png(40, 40)))
    assert r.json()["has_mask"] is True


def test_happy_path_returns_mask_rle():
    """Response carries COCO RLE counts + size matching target."""
    r = _client().post("/segment", files=_files(_rgb_png(40, 60)))
    body = r.json()
    assert body["mask_rle"]["size"] == [40, 60]
    assert sum(body["mask_rle"]["counts"]) == 40 * 60


def test_happy_path_returns_bbox():
    """Response carries a non-zero bbox for non-empty mask."""
    r = _client().post("/segment", files=_files(_rgb_png(40, 40)))
    bbox = r.json()["bbox"]
    assert bbox["width"] > 0
    assert bbox["height"] > 0


def test_happy_path_returns_confidence():
    """Mock default confidence 1.0 surfaces on response."""
    r = _client().post("/segment", files=_files(_rgb_png(40, 40)))
    assert r.json()["confidence"] == 1.0


def test_happy_path_carries_references():
    """Adapter accepts ref + mask pair and still returns status 0."""
    r = _client().post(
        "/segment",
        files=_files(
            _rgb_png(40, 40),
            refs=[_rgb_png(40, 40)],
            masks=[_gray_png(40, 40)],
        ),
    )
    assert r.json()["status_code"] == 0


# ─────────────────────────── Layer 5 statuses (1, 2) ─────────────────


def test_empty_mask_returns_status_one():
    """Mock with mask_coverage=0.0 -> status 1 EMPTY_MASK."""
    r = _client(MockBackend(mask_coverage=0.0)).post(
        "/segment", files=_files(_rgb_png(40, 40))
    )
    body = r.json()
    assert body["status_code"] == 1
    assert body["has_mask"] is False
    assert body["mask_rle"]["counts"] == []


def test_low_confidence_returns_status_two():
    """Mock with confidence=0.1 -> status 2 LOW_CONFIDENCE."""
    r = _client(MockBackend(confidence=0.1)).post(
        "/segment", files=_files(_rgb_png(40, 40))
    )
    assert r.json()["status_code"] == 2


# ─────────────────────────── Layer 1 / 3 statuses (3, 6, 7) ──────────


def test_invalid_target_shape_returns_status_three():
    """RGBA target -> status 3 INVALID_IMAGE."""
    rgba = np.zeros((40, 40, 4), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    r = _client().post("/segment", files=_files(buf.getvalue()))
    assert r.json()["status_code"] == 3


def test_garbage_bytes_target_returns_status_three():
    """Non-image bytes -> status 3 (PIL cannot parse)."""
    r = _client().post(
        "/segment",
        files=[("target", ("target.png", b"not an image at all", "image/png"))],
    )
    assert r.json()["status_code"] == 3


def test_reference_count_mismatch_returns_status_seven():
    """len(refs) != len(masks) -> status 7."""
    r = _client().post(
        "/segment",
        files=_files(
            _rgb_png(40, 40),
            refs=[_rgb_png(40, 40), _rgb_png(40, 40)],
            masks=[_gray_png(40, 40)],
        ),
    )
    assert r.json()["status_code"] == 7


def test_reference_size_mismatch_returns_status_six():
    """refs[0].shape[:2] != masks[0].shape -> status 6."""
    r = _client().post(
        "/segment",
        files=_files(
            _rgb_png(40, 40),
            refs=[_rgb_png(40, 40)],
            masks=[_gray_png(50, 50)],
        ),
    )
    assert r.json()["status_code"] == 6


# ─────────────────────────── Layer 4 status (11) ─────────────────────


def test_backend_error_returns_status_eleven(boom_class):
    """Wrapped backend raising RuntimeError -> status 11 BACKEND_ERROR."""
    inner = boom_class(lambda: RuntimeError("simulated GPU OOM"))
    wrapped = ErrorHandlingBackend(inner)
    r = _client(wrapped).post("/segment", files=_files(_rgb_png(40, 40)))
    body = r.json()
    assert body["status_code"] == 11
    assert "simulated GPU OOM" in body["error_message"] or body["error_message"]


# ─────────────────────────── Error message + shape ───────────────────


def test_error_message_carried_on_failure():
    """status != 0 carries a non-empty error_message."""
    rgba = np.zeros((40, 40, 4), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    r = _client().post("/segment", files=_files(buf.getvalue()))
    assert r.json()["error_message"]


def test_error_response_still_carries_full_shape():
    """Failure responses still serialise every field (defaults)."""
    r = _client().post(
        "/segment",
        files=[("target", ("target.png", b"garbage", "image/png"))],
    )
    body = r.json()
    for k in (
        "status_code",
        "has_mask",
        "has_bbox",
        "mask_rle",
        "bbox",
        "confidence",
        "error_message",
    ):
        assert k in body, f"missing field {k}"


# ─────────────────────────── confidence_threshold ────────────────────


def test_confidence_threshold_form_field_overrides_default():
    """Form field confidence_threshold flows into ConfidenceGate."""
    r = _client(MockBackend(confidence=0.5)).post(
        "/segment",
        files=_files(_rgb_png(40, 40)),
        data={"confidence_threshold": "0.8"},
    )
    # With threshold 0.8 and confidence 0.5, status 2 fires.
    assert r.json()["status_code"] == 2
