"""Pydantic response models for the FastAPI Frontend Adapter.

The shapes mirror the ROS 2 srv response (sam_manager_msgs/srv/
SegmentFromReference) so the future ROS 2 Adapter can lift the
packing logic without a second round of schema design. Keeping the
two wire shapes congruent also means downstream tooling that talks
to either transport sees the same field set.
"""
from typing import List

from pydantic import BaseModel


class MaskRLEModel(BaseModel):
    """COCO-style RLE mirroring sam_manager_msgs/msg/MaskRLE."""

    counts: List[int]
    size: List[int]


class BBoxModel(BaseModel):
    """Bounding box mirroring sensor_msgs/RegionOfInterest fields."""

    x_offset: int
    y_offset: int
    width: int
    height: int


class SegmentResponse(BaseModel):
    """Wire response for POST /segment.

    Every field is always serialised. On failure (status_code != 0),
    mask_rle.counts is the empty list, bbox is zeroed, confidence
    drops to 0.0, and error_message names the typed exception that
    surfaced. Consumers should branch on has_mask before reading
    mask_rle.counts or bbox.
    """

    status_code: int
    has_mask: bool
    has_bbox: bool
    mask_rle: MaskRLEModel
    bbox: BBoxModel
    confidence: float
    error_message: str
