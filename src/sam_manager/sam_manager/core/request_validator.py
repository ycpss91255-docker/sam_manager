"""sam_manager Layer 1 RequestValidator -- target image validation.

Per architecture v4 §4 + drawio Page 4 status_code emit table, Layer 1
RequestValidator owns status_code 3 INVALID_IMAGE: rejects target
images that fail the (H, W, 3) uint8 RGB non-empty contract.

This module is the ROS-agnostic core. The 4 Frontend Adapters
(ROS 2 Service / Topic / FastAPI / Python API) call validate_target
before forwarding the request to the backend and catch
InvalidImageError to map it onto their wire representation
(e.g. ROS 2 srv response status_code = 3 + error_message).

InvalidImageError subclasses ValueError so the Layer 4
ErrorHandlingBackend wrapper propagates it unchanged rather than
remapping it to status_code 11 BACKEND_ERROR
(see core.error_handler docstring contract).

Scope intentionally limited to target validation. refs / masks
pair-alignment + count checks (status 6 / 7) belong to Layer 3
PromptStore and will land in the Layer 3 PR.
"""
import numpy as np


class InvalidImageError(ValueError):
    """Target image failed Layer 1 validation -- wire status_code 3.

    Subclasses ValueError so Layer 4 ErrorHandlingBackend's
    except-ValueError branch propagates it unchanged; the Frontend
    Adapter catches and maps it to status_code 3 INVALID_IMAGE in
    the wire response.
    """

    STATUS_CODE = 3


def validate_target(target: np.ndarray) -> None:
    """Validate target image meets the (H, W, 3) uint8 RGB non-empty contract.

    Args:
        target: candidate target image.

    Raises:
        InvalidImageError: target violates the shape / dtype /
            non-empty contract. The message names the offending
            attribute (shape or dtype) for diagnostics.
    """
    if target.ndim != 3 or target.shape[2] != 3:
        raise InvalidImageError(
            f"target must be (H, W, 3), got shape={target.shape}"
        )
    if target.dtype != np.uint8:
        raise InvalidImageError(
            f"target must be uint8, got dtype={target.dtype}"
        )
    if target.size == 0:
        raise InvalidImageError(
            f"target must not be empty, got shape={target.shape}"
        )
