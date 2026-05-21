# sam_manager.core.* is ROS-agnostic; Frontend Adapters live in sam_manager.adapters.*

Architecture v4 §4 requires four Frontend Adapter channels (ROS 2 Service,
ROS 2 Topic, FastAPI, internal Python API) to share one inference core. To
keep that promise enforceable, all modules under `sam_manager.core.*` are
forbidden from importing rclpy, FastAPI, or any framework that ties them
to a specific transport. Concrete transports live under
`sam_manager.adapters.<frontend>/` (not yet shipped) and depend on `core`
one-way.

The `core/` directory itself is the contract. A future PR review that sees
`import rclpy` inside `sam_manager/core/` must reject the change — the
correct home is `sam_manager/adapters/ros2/`. Layer 1 / 3 validator inputs
(target ndarray, header tuple, 4x4 numpy tf matrix) are deliberately framed
in primitives so this rule does not force the validators to grow ROS-shaped
types.

## Consequences

- **Do not** add `rclpy` / `fastapi` / `nicegui` / `aiohttp` to any
  `sam_manager.core.*` module. Pure numpy + structlog + Python stdlib only.
  The `setup.py` `install_requires` of the Python package reflects this.
- **Do not** flatten `core/` back into the top-level `sam_manager/`
  namespace; the subpackage is the visual contract that catches the
  violation in PR review.
- ROS-shaped fields the wire actually carries (`Header`, `Image`,
  `RegionOfInterest`, `MaskRLE`) are unpacked by the ROS 2 Frontend
  Adapter into primitives before calling `core` validators / backends.
  Same pattern for FastAPI's `pydantic` models.
- Tests that exercise core run as pure pytest without `ros:humble` setup;
  cross-layer integration tests that need ROS live under the future
  `sam_manager.adapters.ros2` test tree, not under `test/core/`.
