"""sam_manager Layer 3 PromptStore -- reference validation helpers.

Per architecture v4 §4 + drawio Page 4 status_code emit table, Layer 3
PromptStore owns:
  - status 5 PROMPT_NOT_FOUND (named-mode lookup miss; deferred until
    the full PromptStore class lands with named / inline mode handling)
  - status 6 REFERENCE_SIZE_MISMATCH (refs[i] / masks[i] shape, dtype,
    pair alignment)
  - status 7 REFERENCE_COUNT_MISMATCH (len(refs) != len(masks))

This module currently scopes to the validator helpers feeding status
6 / 7. Like Layer 1 RequestValidator, the typed exceptions subclass
ValueError so Layer 4 ErrorHandlingBackend's `except ValueError:
raise` branch propagates them unchanged rather than remapping to
status_code 11 BACKEND_ERROR.

The full ``class PromptStore`` (named lookup against prompts_dir +
inline blob acceptance) lands in a follow-up PR.
"""
from typing import List

import numpy as np


class ReferenceCountMismatchError(ValueError):
    """refs and masks lists have different lengths -- wire status_code 7.

    Subclasses ValueError so Layer 4 ErrorHandlingBackend's
    except-ValueError branch propagates it unchanged; the Frontend
    Adapter catches and maps to status_code 7 REFERENCE_COUNT_MISMATCH
    in the wire response.
    """

    STATUS_CODE = 7


class ReferenceSizeMismatchError(ValueError):
    """refs / masks shape, dtype, or pair alignment violation -- wire status_code 6.

    Subclasses ValueError so Layer 4 ErrorHandlingBackend's
    except-ValueError branch propagates it unchanged; the Frontend
    Adapter catches and maps to status_code 6 REFERENCE_SIZE_MISMATCH
    in the wire response.
    """

    STATUS_CODE = 6


def validate_references(
    refs: List[np.ndarray],
    masks: List[np.ndarray],
) -> None:
    """Validate reference image / mask pairs meet the Layer 3 contract.

    Args:
        refs: list of reference images; each must be (H, W, 3) uint8 RGB.
        masks: list of reference masks; each must be (H, W) uint8
            with the same H, W as the matching ``refs[i]``.

    Raises:
        ReferenceCountMismatchError: ``len(refs) != len(masks)``.
        ReferenceSizeMismatchError: any ``refs[i]`` or ``masks[i]``
            violates the shape / dtype contract, or the (refs[i],
            masks[i]) pair has mismatched H, W. Iteration is
            first-bad-wins; the message names the offending pair
            index for diagnostics.
    """
    if len(refs) != len(masks):
        raise ReferenceCountMismatchError(
            f"refs ({len(refs)}) and masks ({len(masks)}) length mismatch"
        )
    for i, (ref, ref_mask) in enumerate(zip(refs, masks)):
        if ref.ndim != 3 or ref.shape[2] != 3 or ref.dtype != np.uint8:
            raise ReferenceSizeMismatchError(
                f"refs[{i}] must be (H, W, 3) uint8, "
                f"got shape={ref.shape} dtype={ref.dtype}"
            )
        if ref_mask.ndim != 2 or ref_mask.dtype != np.uint8:
            raise ReferenceSizeMismatchError(
                f"masks[{i}] must be (H, W) uint8, "
                f"got shape={ref_mask.shape} dtype={ref_mask.dtype}"
            )
        if ref.shape[:2] != ref_mask.shape:
            raise ReferenceSizeMismatchError(
                f"refs[{i}] and masks[{i}] must share H, W "
                f"(image-mask pair alignment) - "
                f"refs[{i}] HxW={ref.shape[:2]} vs "
                f"masks[{i}] HxW={ref_mask.shape}"
            )
