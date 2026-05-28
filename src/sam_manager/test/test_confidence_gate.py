"""Unit tests for sam_manager.core.confidence_gate (Layer 5).

Per architecture v4 §4 + drawio Page 4 status emit table, Layer 5
ConfidenceGate consumes the backend's InferResult and produces:

  - has_mask  bool        true iff mask has at least one positive pixel
  - has_bbox  bool        true iff a bounding box could be computed
                          (currently identical to has_mask -- PIXEL_MASK
                          backends only; COARSE_BBOX downgrade is a
                          future-backend story)
  - mask_rle  (counts, size)
                          COCO-style column-major RLE; empty counts list
                          on EMPTY_MASK so downstream can detect "no
                          mask provided"
  - bbox      (x, y, w, h) tight bbox in pixel coords; (0, 0, 0, 0)
                          when mask empty
  - confidence float      pass-through from InferResult.confidence;
                          1.0 if backend did not report (None)
  - status_code int       0 OK / 1 EMPTY_MASK / 2 LOW_CONFIDENCE,
                          EMPTY first if both apply

The threshold for status 2 LOW_CONFIDENCE is a function parameter
(default 0.3) so callers / tests can stress different sensitivities
without monkey-patching a module constant.
"""
import numpy as np
import pytest

from sam_manager.core.backend import InferResult
from sam_manager.core.confidence_gate import (
    ConfidenceGateOutput,
    evaluate,
)


def _rgb(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _result(mask: np.ndarray, confidence=1.0) -> InferResult:
    return InferResult(
        target=_rgb(*mask.shape),
        masks=[mask],
        class_ids=[0],
        confidence=confidence,
    )


def _centered_mask(h: int, w: int, half_h: int, half_w: int) -> np.ndarray:
    """Helper: 1-block centered rectangle of `255`s."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cy, cx = h // 2, w // 2
    mask[cy - half_h:cy + half_h, cx - half_w:cx + half_w] = 255
    return mask


# ─────────────────────────── Output shape ────────────────────────────


def test_output_is_confidence_gate_output_instance():
    """evaluate() returns a ConfidenceGateOutput dataclass."""
    out = evaluate(_result(_centered_mask(40, 40, 5, 5)))
    assert isinstance(out, ConfidenceGateOutput)


def test_output_carries_all_required_fields():
    """Output carries every wire-facing field with expected types."""
    out = evaluate(_result(_centered_mask(40, 40, 5, 5)))
    assert isinstance(out.has_mask, bool)
    assert isinstance(out.has_bbox, bool)
    assert isinstance(out.mask_rle_counts, list)
    assert isinstance(out.mask_rle_size, tuple)
    assert isinstance(out.bbox, tuple)
    assert isinstance(out.confidence, float)
    assert isinstance(out.status_code, int)


# ─────────────────────────── Happy path: non-empty mask ──────────────


def test_non_empty_mask_sets_has_mask_true():
    """Any positive pixel -> has_mask=True."""
    out = evaluate(_result(_centered_mask(40, 40, 5, 5)))
    assert out.has_mask is True


def test_non_empty_mask_sets_has_bbox_true():
    """has_bbox mirrors has_mask for PIXEL_MASK backends."""
    out = evaluate(_result(_centered_mask(40, 40, 5, 5)))
    assert out.has_bbox is True


def test_non_empty_mask_emits_status_zero():
    """High-confidence non-empty mask emits status 0 OK."""
    out = evaluate(_result(_centered_mask(40, 40, 5, 5), confidence=1.0))
    assert out.status_code == 0


def test_confidence_passes_through():
    """InferResult.confidence flows verbatim onto output.confidence."""
    out = evaluate(_result(_centered_mask(40, 40, 5, 5), confidence=0.7))
    assert out.confidence == 0.7


def test_confidence_unreported_defaults_to_one():
    """None confidence -> 1.0 (legacy backends do not trip status 2)."""
    out = evaluate(_result(_centered_mask(40, 40, 5, 5), confidence=None))
    assert out.confidence == 1.0
    assert out.status_code == 0


# ─────────────────────────── EMPTY_MASK (status 1) ───────────────────


def test_empty_mask_sets_has_mask_false():
    """All-zero mask -> has_mask=False."""
    empty = np.zeros((40, 40), dtype=np.uint8)
    out = evaluate(_result(empty))
    assert out.has_mask is False


def test_empty_mask_sets_has_bbox_false():
    """All-zero mask -> has_bbox=False (PIXEL_MASK backend)."""
    empty = np.zeros((40, 40), dtype=np.uint8)
    out = evaluate(_result(empty))
    assert out.has_bbox is False


def test_empty_mask_emits_status_one():
    """All-zero mask -> status_code 1 EMPTY_MASK."""
    empty = np.zeros((40, 40), dtype=np.uint8)
    out = evaluate(_result(empty))
    assert out.status_code == 1


def test_empty_mask_rle_counts_is_empty_list():
    """status 1 -> counts is empty list (wire convention for "no mask")."""
    empty = np.zeros((40, 40), dtype=np.uint8)
    out = evaluate(_result(empty))
    assert out.mask_rle_counts == []


def test_empty_mask_rle_size_still_carries_shape():
    """size is still populated so consumer knows the image dimensions."""
    empty = np.zeros((40, 60), dtype=np.uint8)
    out = evaluate(_result(empty))
    assert out.mask_rle_size == (40, 60)


def test_empty_mask_bbox_is_zero():
    """Empty mask -> bbox (0, 0, 0, 0)."""
    empty = np.zeros((40, 40), dtype=np.uint8)
    out = evaluate(_result(empty))
    assert out.bbox == (0, 0, 0, 0)


# ─────────────────────────── LOW_CONFIDENCE (status 2) ───────────────


def test_low_confidence_with_non_empty_mask_emits_status_two():
    """Confidence below threshold -> status 2 even if mask non-empty."""
    out = evaluate(
        _result(_centered_mask(40, 40, 5, 5), confidence=0.1),
        confidence_threshold=0.3,
    )
    assert out.status_code == 2


def test_confidence_at_threshold_passes():
    """Confidence equal to threshold is acceptable (>=, not >)."""
    out = evaluate(
        _result(_centered_mask(40, 40, 5, 5), confidence=0.3),
        confidence_threshold=0.3,
    )
    assert out.status_code == 0


def test_default_threshold_is_zero_point_three():
    """Function default confidence_threshold is 0.3."""
    # confidence 0.29 should trip status 2 with the default; 0.30 should not.
    out_below = evaluate(_result(_centered_mask(40, 40, 5, 5), confidence=0.29))
    out_at = evaluate(_result(_centered_mask(40, 40, 5, 5), confidence=0.30))
    assert out_below.status_code == 2
    assert out_at.status_code == 0


# ─────────────────────────── Status priority ─────────────────────────


def test_empty_mask_priority_over_low_confidence():
    """When mask is empty AND confidence < threshold, EMPTY_MASK wins."""
    empty = np.zeros((40, 40), dtype=np.uint8)
    out = evaluate(_result(empty, confidence=0.05), confidence_threshold=0.3)
    assert out.status_code == 1


# ─────────────────────────── bbox geometry ───────────────────────────


def test_bbox_tight_around_pixels():
    """bbox is the tight rect (x, y, w, h) around the positive pixels."""
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:20, 15:30] = 255  # rows 10..19, cols 15..29
    out = evaluate(_result(mask))
    # (x_offset=15, y_offset=10, width=15, height=10)
    assert out.bbox == (15, 10, 15, 10)


def test_bbox_single_pixel():
    """Single positive pixel -> bbox (x, y, 1, 1)."""
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[7, 13] = 255
    out = evaluate(_result(mask))
    assert out.bbox == (13, 7, 1, 1)


def test_bbox_fully_filled():
    """All-positive mask -> bbox covers entire image."""
    mask = np.full((40, 60), 255, dtype=np.uint8)
    out = evaluate(_result(mask))
    assert out.bbox == (0, 0, 60, 40)


# ─────────────────────────── RLE encoding ────────────────────────────


def test_rle_size_is_h_w_tuple():
    """mask_rle_size is (H, W) matching the mask shape."""
    out = evaluate(_result(_centered_mask(40, 60, 5, 5)))
    assert out.mask_rle_size == (40, 60)


def test_rle_counts_sum_equals_pixel_count():
    """Sum of RLE counts equals total pixel count H*W."""
    mask = _centered_mask(40, 40, 5, 5)
    out = evaluate(_result(mask))
    assert sum(out.mask_rle_counts) == 40 * 40


def test_rle_starts_with_background_run():
    """COCO RLE alternates starting with a run of zeros (background).

    If the mask's first pixel (column-major: top-left) is foreground,
    the encoder still emits a 0 first so the alternation begins with
    background.
    """
    mask = np.full((10, 10), 255, dtype=np.uint8)
    out = evaluate(_result(mask))
    # All foreground: counts should be [0, 100] (0 bg + 100 fg).
    assert out.mask_rle_counts == [0, 100]


def test_rle_all_background():
    """Empty mask -> counts list is empty (wire "no mask" convention)."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    out = evaluate(_result(mask))
    assert out.mask_rle_counts == []


def test_rle_alternating_runs_match_mask():
    """Manually constructed mask has expected alternating runs (Fortran order)."""
    # 4x4 mask. Fortran (column-major) traversal:
    # col0 = [0, 0, 1, 1], col1 = [0, 0, 1, 1],
    # col2 = [1, 1, 0, 0], col3 = [1, 1, 0, 0]
    # Linearised: 0 0 1 1 | 0 0 1 1 | 1 1 0 0 | 1 1 0 0
    #        =    0 0 1 1 0 0 1 1 1 1 0 0 1 1 0 0
    # Runs (alternating, starting with 0): 2 2 2 4 2 2 2 ... let me recompute
    # 0 0 1 1 0 0 1 1 1 1 0 0 1 1 0 0
    # bg=2, fg=2, bg=2, fg=4, bg=2, fg=2, bg=2 → [2, 2, 2, 4, 2, 2, 2]
    mask = np.array([
        [0, 0, 255, 255],
        [0, 0, 255, 255],
        [255, 255, 0, 0],
        [255, 255, 0, 0],
    ], dtype=np.uint8)
    out = evaluate(_result(mask))
    # Sum still equals 16.
    assert sum(out.mask_rle_counts) == 16
    # Alternating starting with background (>=2 counts of zeros first).
    assert out.mask_rle_counts == [2, 2, 2, 4, 2, 2, 2]


# ─────────────────────────── Validation ──────────────────────────────


def test_evaluate_rejects_empty_masks_list():
    """InferResult.masks must have at least one mask."""
    bad = InferResult(target=_rgb(40, 40), masks=[], class_ids=[])
    with pytest.raises(ValueError):
        evaluate(bad)
