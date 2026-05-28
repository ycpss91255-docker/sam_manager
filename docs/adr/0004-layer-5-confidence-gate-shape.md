# Layer 5 ConfidenceGate evaluate() shape + sourcing rules

`sam_manager.core.confidence_gate.evaluate(result, confidence_threshold=0.3)`
takes a backend's raw `InferResult` and returns a `ConfidenceGateOutput`
dataclass carrying every wire-facing field the Frontend Adapter then
packs into its transport response. The Adapter never recomputes; Layer 5
is the final word on `has_mask`, `has_bbox`, `mask_rle`, `bbox`, and
`status_code` (0 OK / 1 EMPTY_MASK / 2 LOW_CONFIDENCE).

## Sourcing rules

- **Confidence value comes from the backend** (`InferResult.confidence`),
  not from Layer 5. Only SegGPT / SAM 2 Tiny / VRP-SAM know how to turn
  their internal logit distributions into a 0..1 number; Layer 5 has no
  view of those internals. Layer 5 *applies* the threshold and decides
  status 2 LOW_CONFIDENCE.
- `None` confidence is treated as `1.0`. Backends pre-dating the field
  would otherwise start tripping status 2 the moment Layer 5 lands.
- **`has_mask` is strict zero**: any positive pixel -> True. Minimum-size
  filtering belongs in the application layer (the user knows what
  "useful" means in their context); Layer 5 should not invent a global
  pixel-count threshold.
- **`has_bbox` mirrors `has_mask`** for PIXEL_MASK backends. The
  architectural case for `has_mask=False / has_bbox=True` is COARSE_BBOX
  tier backends that produce only a coarse box. None of the planned
  backends are in that tier; the differential is preserved as a future
  extension by leaving the two booleans separate on the wire.

## Status priority

EMPTY_MASK (1) before LOW_CONFIDENCE (2). When `has_mask=False`, grading
confidence is meaningless -- the downstream consumer should branch on
"no mask" and run whatever abort logic it has. A wire response that
said `status=2` while also reporting `has_mask=False` would be
misleading. ADR enshrines the priority so future-me does not flip them.

## Threshold as a function parameter, not a module constant

`confidence_threshold` lives on the `evaluate()` signature with a 0.3
default rather than a module-level constant. Reasons:

- Tests can stress different sensitivities without monkey-patching.
- The Frontend Adapter can wire launch config / ROS parameter through
  to the gate when the lifecycle node lands, without core knowing about
  ROS configuration mechanisms.
- 0.3 default is a placeholder; SegGPT's calibrated value will replace
  it in launch config once Phase 0 measurements land.

## RLE encoder is in-house, not pycocotools

The MaskRLE wire format is COCO-style column-major alternating runs.
`pycocotools` would decode this directly, but the producer side is
~15 lines of numpy. Adding pycocotools as a runtime dependency for
those 15 lines is not worth the install footprint (build deps, OpenCV
chain on some platforms). Consumers that want pycocotools to decode
can install it client-side without affecting sam_manager's deployment.

## Considered options

- **Confidence value computed inside Layer 5** -- rejected; Layer 5
  has no access to per-pixel logits, so it would have to fall back to
  mask area / target area, which is geometry not confidence.
- **Status priority LOW > EMPTY** -- rejected; emitting status 2 while
  has_mask=False misleads consumers.
- **Module-constant threshold** -- rejected; less testable, forces
  Frontend Adapter to monkey-patch the module to wire ROS params.
