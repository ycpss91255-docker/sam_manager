"""Unit tests for sam_manager.core.prompt_store (Layer 3).

Per architecture v4 §4 + drawio Page 4 status_code emit table, Layer 3
PromptStore owns:
  - status 5 PROMPT_NOT_FOUND (named-mode lookup miss; deferred until
    the full PromptStore class lands with named/inline mode handling)
  - status 6 REFERENCE_SIZE_MISMATCH (refs[i] / masks[i] shape, dtype,
    pair alignment)
  - status 7 REFERENCE_COUNT_MISMATCH (len(refs) != len(masks))

This PR scopes to the validator helpers feeding status 6 / 7. Like
Layer 1 RequestValidator, the typed exceptions subclass ValueError so
Layer 4 ErrorHandlingBackend's `except ValueError: raise` branch
propagates them unchanged rather than remapping to status_code 11
BACKEND_ERROR.
"""
import numpy as np
import pytest

from sam_manager.core.prompt_store import (
    ReferenceCountMismatchError,
    ReferenceSizeMismatchError,
    validate_references,
)


def _rgb(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _gray(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


# ─────────────────────────── Exception contracts ─────────────────────


def test_reference_count_mismatch_status_code_is_seven():
    """ReferenceCountMismatchError.STATUS_CODE == 7."""
    assert ReferenceCountMismatchError.STATUS_CODE == 7


def test_reference_size_mismatch_status_code_is_six():
    """ReferenceSizeMismatchError.STATUS_CODE == 6."""
    assert ReferenceSizeMismatchError.STATUS_CODE == 6


def test_count_mismatch_subclasses_value_error():
    """ReferenceCountMismatchError IS-A ValueError so wrapper propagates it."""
    assert issubclass(ReferenceCountMismatchError, ValueError)


def test_size_mismatch_subclasses_value_error():
    """ReferenceSizeMismatchError IS-A ValueError so wrapper propagates it."""
    assert issubclass(ReferenceSizeMismatchError, ValueError)


# ─────────────────────────── Happy path ───────────────────────────────


def test_accepts_empty_lists():
    """Empty refs + empty masks is valid (no references provided)."""
    assert validate_references([], []) is None


def test_accepts_single_aligned_pair():
    """One matching (ref, mask) pair passes."""
    assert validate_references([_rgb(40, 40)], [_gray(40, 40)]) is None


def test_accepts_multiple_aligned_pairs():
    """N matching pairs of varying sizes all pass."""
    refs = [_rgb(40, 40), _rgb(80, 60), _rgb(100, 100)]
    masks = [_gray(40, 40), _gray(80, 60), _gray(100, 100)]
    assert validate_references(refs, masks) is None


# ─────────────────────────── Count mismatch (status 7) ───────────────


def test_rejects_more_refs_than_masks():
    """1 ref + 0 masks raises count mismatch."""
    with pytest.raises(ReferenceCountMismatchError):
        validate_references([_rgb(20, 20)], [])


def test_rejects_more_masks_than_refs():
    """0 refs + 1 mask raises count mismatch."""
    with pytest.raises(ReferenceCountMismatchError):
        validate_references([], [_gray(20, 20)])


def test_rejects_off_by_one_count():
    """2 refs + 3 masks raises count mismatch."""
    refs = [_rgb(20, 20), _rgb(20, 20)]
    masks = [_gray(20, 20), _gray(20, 20), _gray(20, 20)]
    with pytest.raises(ReferenceCountMismatchError):
        validate_references(refs, masks)


def test_count_mismatch_message_mentions_both_lengths():
    """Diagnostic message surfaces refs / masks lengths."""
    with pytest.raises(ReferenceCountMismatchError) as excinfo:
        validate_references([_rgb(20, 20), _rgb(20, 20)], [_gray(20, 20)])
    msg = str(excinfo.value)
    assert "2" in msg and "1" in msg


# ─────────────────────────── Ref shape / dtype (status 6) ────────────


def test_rejects_grayscale_ref():
    """refs[i] shape (H, W) without channel dim raises size mismatch."""
    with pytest.raises(ReferenceSizeMismatchError):
        validate_references([_gray(40, 40)], [_gray(40, 40)])


def test_rejects_rgba_ref():
    """refs[i] shape (H, W, 4) raises size mismatch."""
    rgba = np.zeros((40, 40, 4), dtype=np.uint8)
    with pytest.raises(ReferenceSizeMismatchError):
        validate_references([rgba], [_gray(40, 40)])


def test_rejects_pseudo_3d_ref():
    """refs[i] shape (H, W, 1) raises size mismatch."""
    pseudo = np.zeros((40, 40, 1), dtype=np.uint8)
    with pytest.raises(ReferenceSizeMismatchError):
        validate_references([pseudo], [_gray(40, 40)])


def test_rejects_uint16_ref():
    """refs[i] dtype != uint8 raises size mismatch."""
    u16 = np.zeros((40, 40, 3), dtype=np.uint16)
    with pytest.raises(ReferenceSizeMismatchError):
        validate_references([u16], [_gray(40, 40)])


def test_rejects_float32_ref():
    """refs[i] dtype float32 raises size mismatch."""
    f32 = np.zeros((40, 40, 3), dtype=np.float32)
    with pytest.raises(ReferenceSizeMismatchError):
        validate_references([f32], [_gray(40, 40)])


# ─────────────────────────── Mask shape / dtype (status 6) ───────────


def test_rejects_3channel_mask():
    """masks[i] shape (H, W, 3) raises size mismatch (mask must be 2D)."""
    rgb_mask = np.zeros((40, 40, 3), dtype=np.uint8)
    with pytest.raises(ReferenceSizeMismatchError):
        validate_references([_rgb(40, 40)], [rgb_mask])


def test_rejects_pseudo_3d_mask():
    """masks[i] shape (H, W, 1) raises size mismatch."""
    pseudo = np.zeros((40, 40, 1), dtype=np.uint8)
    with pytest.raises(ReferenceSizeMismatchError):
        validate_references([_rgb(40, 40)], [pseudo])


def test_rejects_bool_mask():
    """masks[i] dtype bool raises size mismatch (must be uint8)."""
    bool_mask = np.zeros((40, 40), dtype=bool)
    with pytest.raises(ReferenceSizeMismatchError):
        validate_references([_rgb(40, 40)], [bool_mask])


def test_rejects_float32_mask():
    """masks[i] dtype float32 raises size mismatch."""
    f32 = np.zeros((40, 40), dtype=np.float32)
    with pytest.raises(ReferenceSizeMismatchError):
        validate_references([_rgb(40, 40)], [f32])


# ─────────────────────────── Pair alignment (status 6) ────────────────


def test_rejects_pair_height_mismatch():
    """refs[i] H differs from masks[i] H raises size mismatch."""
    with pytest.raises(ReferenceSizeMismatchError):
        validate_references([_rgb(40, 40)], [_gray(50, 40)])


def test_rejects_pair_width_mismatch():
    """refs[i] W differs from masks[i] W raises size mismatch."""
    with pytest.raises(ReferenceSizeMismatchError):
        validate_references([_rgb(40, 40)], [_gray(40, 50)])


def test_rejects_pair_both_dim_mismatch():
    """refs[i] (100, 100) vs masks[i] (50, 50) raises size mismatch."""
    with pytest.raises(ReferenceSizeMismatchError):
        validate_references([_rgb(100, 100)], [_gray(50, 50)])


def test_size_mismatch_message_identifies_index():
    """Diagnostic message identifies which pair index failed."""
    refs = [_rgb(40, 40), _rgb(40, 40)]
    masks = [_gray(40, 40), _gray(50, 50)]  # idx 1 mismatched
    with pytest.raises(ReferenceSizeMismatchError) as excinfo:
        validate_references(refs, masks)
    assert "1" in str(excinfo.value)


# ─────────────────────────── Iteration order (first-bad-wins) ────────


def test_first_bad_pair_raises_before_subsequent_bad_pairs():
    """Validation stops at the first malformed pair (no all-errors collection)."""
    refs = [_rgb(40, 40), _gray(40, 40)]   # idx 1 invalid shape (grayscale ref)
    masks = [_gray(40, 40), _gray(40, 40)]
    with pytest.raises(ReferenceSizeMismatchError) as excinfo:
        validate_references(refs, masks)
    # idx 0 passes; idx 1 fails. Message should reference idx 1.
    assert "1" in str(excinfo.value)


def test_count_mismatch_checked_before_pair_checks():
    """Count mismatch raises before any per-pair shape check."""
    refs = [_rgb(40, 40)]
    masks = []  # count mismatch
    with pytest.raises(ReferenceCountMismatchError):
        validate_references(refs, masks)
