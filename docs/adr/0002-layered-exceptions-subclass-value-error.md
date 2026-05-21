# Layer 1 / 3 typed exceptions subclass ValueError

Layer 4 `ErrorHandlingBackend` separates "input was malformed" (caller's
fault, surfaces on the wire as status_code 3 / 4 / 6 / 7 / 8) from "backend
crashed" (system fault, status_code 11 BACKEND_ERROR). The wrapper expresses
that distinction with a Python idiom:

```python
except ValueError:
    raise                                 # propagate to Frontend Adapter
except Exception as exc:
    raise BackendError(original=exc) ...  # remap to status 11
```

All Layer 1 + Layer 3 typed exceptions (`InvalidImageError`,
`ReferenceCountMismatchError`, `ReferenceSizeMismatchError`, and future
`InvalidHeaderError` / `InvalidTfError`) therefore subclass `ValueError`,
not `Exception`. Each carries a `STATUS_CODE` class constant so the Frontend
Adapter reads `exc.STATUS_CODE` to populate the wire field without
exception-name pattern matching.

The "obvious" Python pattern would be a dedicated hierarchy
(`class LayerError(Exception): STATUS_CODE: int` with each typed error
subclassing that). We reject it because the ValueError choice is also a
semantic claim — these errors *are* input-value problems, exactly what
ValueError exists to signal — and reusing the standard base lets pytest /
linter / IDE tooling treat them naturally.

## Consequences

- **Do not** change a Layer 1 / 3 exception's base to plain `Exception`
  without also updating `ErrorHandlingBackend.infer` and every Frontend
  Adapter that maps status codes. The split is encoded in inheritance.
- New status_code-bearing exceptions added at Layer 1 / 2 / 3 / 5 must
  follow the same pattern. Layer 4's `BackendError` is the lone exception:
  it subclasses `Exception` because it *is* the system-fault bucket the
  wrapper produces.
