"""Unit tests for sam_manager.adapters.fastapi.models pydantic schemas.

The pydantic models are wire-shape stamps -- their behaviour is
mostly "serialise to JSON, parse from JSON" tested implicitly by
test_app. These tests pin the contract directly: every field name,
type, and required-ness must match the ROS 2 srv response so a
shared client library can read either transport.
"""
import pytest
from pydantic import ValidationError

from sam_manager.adapters.fastapi.models import (
    BBoxModel,
    MaskRLEModel,
    SegmentResponse,
)


def test_mask_rle_model_accepts_minimal_inputs():
    """Empty counts + arbitrary size dim is valid."""
    m = MaskRLEModel(counts=[], size=[0, 0])
    assert m.counts == []
    assert m.size == [0, 0]


def test_mask_rle_model_round_trip():
    """JSON serialise + parse preserves fields."""
    m = MaskRLEModel(counts=[2, 3, 2], size=[40, 60])
    data = m.model_dump()
    again = MaskRLEModel.model_validate(data)
    assert again == m


def test_bbox_model_carries_four_ints():
    """x_offset / y_offset / width / height all int."""
    b = BBoxModel(x_offset=1, y_offset=2, width=3, height=4)
    assert (b.x_offset, b.y_offset, b.width, b.height) == (1, 2, 3, 4)


def test_segment_response_carries_all_required_fields():
    """Every wire field is required (no silent defaults)."""
    r = SegmentResponse(
        status_code=0,
        has_mask=True,
        has_bbox=True,
        mask_rle=MaskRLEModel(counts=[0, 100], size=[10, 10]),
        bbox=BBoxModel(x_offset=0, y_offset=0, width=10, height=10),
        confidence=1.0,
        error_message="",
    )
    assert r.status_code == 0


def test_segment_response_round_trip():
    """Model dump and reparse equals original."""
    original = SegmentResponse(
        status_code=2,
        has_mask=True,
        has_bbox=True,
        mask_rle=MaskRLEModel(counts=[2, 2, 2], size=[2, 3]),
        bbox=BBoxModel(x_offset=1, y_offset=1, width=2, height=1),
        confidence=0.1,
        error_message="LOW_CONFIDENCE",
    )
    assert SegmentResponse.model_validate(original.model_dump()) == original


def test_segment_response_rejects_missing_field():
    """Missing has_bbox raises ValidationError (not silently defaulted)."""
    with pytest.raises(ValidationError):
        SegmentResponse(
            status_code=0,
            has_mask=True,
            # has_bbox omitted
            mask_rle=MaskRLEModel(counts=[], size=[0, 0]),
            bbox=BBoxModel(x_offset=0, y_offset=0, width=0, height=0),
            confidence=1.0,
            error_message="",
        )
