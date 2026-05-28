"""sam_manager Layer 5 ConfidenceGate -- InferResult -> wire fields.

Per architecture v4 §4 + drawio Page 4 status emit table, Layer 5
ConfidenceGate is the last stop before the Frontend Adapter packs the
wire response. It converts a backend's raw ``InferResult`` into the
fields the wire actually carries:

  - ``has_mask`` / ``has_bbox`` booleans (capability flags downstream
    consumers can branch on without knowing the backend's level)
  - COCO column-major RLE encoding of the binary mask
  - Tight bbox around the positive pixels
  - Confidence pass-through (value is backend knowledge; only the
    threshold check is Layer 5's call)
  - ``status_code`` (0 OK / 1 EMPTY_MASK / 2 LOW_CONFIDENCE)

The confidence threshold is a function parameter, not a module
constant -- so tests can stress different sensitivities without
monkey-patching, and the Frontend Adapter can wire launch config
through to the gate when the lifecycle node lands.

Status priority: EMPTY_MASK before LOW_CONFIDENCE. If the mask has
no positive pixels there is no point grading confidence; the
downstream consumer should see "no mask available" and fall back to
whatever its abort logic is. ADR-0004 records the rationale.
"""
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from sam_manager.core.backend import InferResult


# Wire status codes per drawio Page 4. Module constants rather than an
# Enum so the integers match the ROS 2 srv contract verbatim and the
# Frontend Adapter does not have to translate.
STATUS_OK = 0
STATUS_EMPTY_MASK = 1
STATUS_LOW_CONFIDENCE = 2

# Confidence value Layer 5 substitutes when the backend did not
# report (InferResult.confidence is None). 1.0 keeps legacy backends
# from tripping status 2 the moment this layer lands.
DEFAULT_UNREPORTED_CONFIDENCE = 1.0


@dataclass
class ConfidenceGateOutput:
    """Wire-facing fields derived from one InferResult.

    Frontend Adapter (ROS 2 srv handler / FastAPI handler) reads
    these fields and packs them into its transport's response shape
    (ROS 2 srv message, JSON body, ...). The Adapter never recomputes
    -- everything here is final.

    bbox is (x_offset, y_offset, width, height) in pixel coordinates,
    matching ``sensor_msgs/RegionOfInterest`` field names. (0, 0, 0, 0)
    when the mask is empty (no positive pixels to box).

    mask_rle_counts is column-major (Fortran order) alternating run
    lengths starting with a background run, per
    ``sam_manager_msgs/msg/MaskRLE``. Empty list signals "no mask";
    consumers should branch on has_mask, not on counts emptiness.
    """

    has_mask: bool
    has_bbox: bool
    mask_rle_counts: List[int]
    mask_rle_size: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    confidence: float
    status_code: int
    error_message: str = field(default="")


def evaluate(
    result: InferResult,
    confidence_threshold: float = 0.3,
) -> ConfidenceGateOutput:
    """Compute wire fields + status_code from a backend's InferResult.

    Args:
        result: backend output. Layer 5 only inspects ``masks[0]`` and
            ``confidence`` (multi-mask handling stays in the backend;
            sam_manager wire is single-mask per response).
        confidence_threshold: minimum acceptable confidence
            (inclusive). Below this -> status 2 LOW_CONFIDENCE.
            Default 0.3.

    Returns:
        A ConfidenceGateOutput carrying every wire field the
        Frontend Adapter needs to pack a response.

    Raises:
        ValueError: ``result.masks`` is empty (no mask candidate to
            evaluate at all).
    """
    if not result.masks:
        raise ValueError("InferResult.masks must contain at least one mask")

    mask = result.masks[0]
    height, width = mask.shape

    has_mask = bool(np.any(mask > 0))

    # PIXEL_MASK backends derive bbox from the mask; has_bbox follows
    # has_mask 1:1. COARSE_BBOX backends (future) would override here
    # by carrying their own bbox in InferResult and setting
    # has_bbox=True even when has_mask=False.
    has_bbox = has_mask

    if has_mask:
        bbox = _compute_bbox(mask)
        mask_rle_counts = _encode_rle_coco(mask)
    else:
        bbox = (0, 0, 0, 0)
        # Empty list is the wire convention for "no mask available"
        # (sam_manager_msgs/msg/MaskRLE docstring); consumers should
        # already be branching on has_mask before reading counts.
        mask_rle_counts = []

    confidence = (
        result.confidence
        if result.confidence is not None
        else DEFAULT_UNREPORTED_CONFIDENCE
    )

    status_code = _classify_status(
        has_mask=has_mask,
        confidence=confidence,
        confidence_threshold=confidence_threshold,
    )

    return ConfidenceGateOutput(
        has_mask=has_mask,
        has_bbox=has_bbox,
        mask_rle_counts=mask_rle_counts,
        mask_rle_size=(height, width),
        bbox=bbox,
        confidence=confidence,
        status_code=status_code,
    )


def _classify_status(
    has_mask: bool,
    confidence: float,
    confidence_threshold: float,
) -> int:
    """Pick the status_code emitted by Layer 5.

    EMPTY_MASK wins over LOW_CONFIDENCE -- if there is no mask to
    grade, grading the confidence is meaningless.
    """
    if not has_mask:
        return STATUS_EMPTY_MASK
    if confidence < confidence_threshold:
        return STATUS_LOW_CONFIDENCE
    return STATUS_OK


def _compute_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Tight bbox around positive pixels.

    Returns (x_offset, y_offset, width, height). Caller must have
    verified the mask is non-empty.
    """
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    y_min, y_max = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    x_min, x_max = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
    return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)


def _encode_rle_coco(mask: np.ndarray) -> List[int]:
    """COCO-style column-major alternating run lengths.

    The mask is flattened in Fortran (column-major) order to match the
    pycocotools convention. The first run is always background even if
    the first pixel is foreground -- we prepend a zero in that case so
    the alternation begins with background.
    """
    binary = (mask > 0).astype(np.uint8).flatten(order="F")
    # Locate run boundaries: indices where consecutive values differ.
    diffs = np.diff(binary)
    boundaries = np.flatnonzero(diffs) + 1
    boundaries = np.concatenate(
        ([0], boundaries, [binary.size])
    )
    counts = np.diff(boundaries).tolist()

    # If the very first pixel is foreground, prepend a zero-length
    # background run so the alternation starts with background.
    if binary[0] == 1:
        counts = [0] + counts

    return counts
