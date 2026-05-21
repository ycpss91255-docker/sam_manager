"""sam_manager Python API Frontend Adapter.

The smallest Frontend Adapter shipped per architecture v4 §4: a
single free function that runs Layer 1 + Layer 3 validators then
forwards to a BackendInterface. Intended for internal-test use
(`Python API 不對外`); not on the production wire.

Its job in the architecture is twofold:

1. Pin the validator-first calling convention every Frontend
   Adapter must follow. ROS 2 / FastAPI adapters will mirror this
   shape -- unpack their transport payload to primitives, call the
   same validators, then forward to the backend.
2. Provide the first non-Mock caller of the Layer 1 / Layer 3
   validators so MockBackend can retire its internal _validate
   (per-backend defense-in-depth was a transitional crutch while
   the validator caller did not exist).

Caller catches typed exceptions by class and maps to its own wire
representation. Python API itself does NOT translate exceptions
into status_code integers -- that is the job of the production
Frontend Adapters (ROS 2 srv handler, FastAPI handler). See
ADR-0002 for why typed exceptions subclass ValueError.
"""
from typing import List

import numpy as np

from sam_manager.core.backend import BackendInterface, InferResult
from sam_manager.core.prompt_store import validate_references
from sam_manager.core.request_validator import validate_target


def segment(
    backend: BackendInterface,
    target: np.ndarray,
    refs: List[np.ndarray],
    masks: List[np.ndarray],
) -> InferResult:
    """Validate inputs then run inference through the supplied backend.

    Args:
        backend: any BackendInterface implementation (Mock, future
            SegGPTBackend, ErrorHandlingBackend wrapping either).
            The adapter does not own backend selection.
        target: (H, W, 3) uint8 RGB target image.
        refs: list of (H, W, 3) uint8 RGB reference images.
        masks: list of (H, W) uint8 reference masks; same length as
            ``refs``; ``refs[i].shape[:2] == masks[i].shape``.

    Returns:
        The backend's ``InferResult`` returned unchanged.

    Raises:
        InvalidImageError: target violates Layer 1 contract
            (STATUS_CODE = 3, IS-A ValueError).
        ReferenceCountMismatchError: ``len(refs) != len(masks)``
            (STATUS_CODE = 7, IS-A ValueError).
        ReferenceSizeMismatchError: any per-pair shape / dtype /
            alignment violation (STATUS_CODE = 6, IS-A ValueError).
        BackendError: backend runtime failure when wrapped in
            ErrorHandlingBackend (STATUS_CODE = 11).
        Exception: bare exceptions from an unwrapped backend bubble
            up directly; production Frontend Adapters wrap with
            ErrorHandlingBackend to map them onto status 11.
    """
    validate_target(target)
    validate_references(refs, masks)
    return backend.infer(target, refs, masks)
