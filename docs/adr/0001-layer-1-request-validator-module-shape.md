# Layer 1 RequestValidator as a single module of free functions

Architecture v4 §4 assigns status_code 3 INVALID_IMAGE, 4 INVALID_HEADER, and
8 INVALID_TF to Layer 1 RequestValidator. The first shipped check
(`validate_target` for status 3) lives at `sam_manager.core.request_validator`
as a module-level function. We will land `validate_header(header)` and
`validate_tf(tf_matrix)` in the same module rather than splitting per status
or pushing header / TF validation into each Frontend Adapter.

A single module keeps Layer 1's emit-site contract 1:1 with one importable
surface — Frontend Adapters (ROS 2 srv handler, FastAPI handler, Python API)
import three free functions instead of three modules. Header structures are
plain `frame_id: str + stamp: (sec, nanosec)` and TF is a 4x4 numpy matrix,
so neither validator forces the core to depend on rclpy. Splitting per
adapter would duplicate the same validation in four places; splitting per
status file would create three files of < 20 lines each.

## Considered options

- **B: one file per status (header_validator.py / tf_validator.py)** —
  rejected; three thin files for the same architectural layer doesn't pay
  for itself.
- **C: validation owned by each Frontend Adapter** — rejected; four copies
  of the same checks, plus FastAPI / Python API would still need *something*
  resembling a header concept and they'd reinvent it.
