# Project mockturtle patches

This directory groups the dependency changes used jointly by ePMapping and
eGraphMap. The `.patch` files describe the changes for review; `overlay/` holds
the exact replacement headers and their SHA-256 manifest for reproducible builds.
They are two representations of the same existing project modifications.

The preserved patches are:

- `tech-library-function-buckets.patch` groups dominance checks by truth table while preserving entry order within each function.
- `shared-mapped-library.patch` gives transient mapped networks shared ownership of an immutable cell library.
- `cut-prefix-enumeration.patch` prunes infeasible cut-tuple prefixes while preserving surviving tuple order. This patch captures the existing header used by the frozen experiment binary; packaging it does not change the measured implementation.

The current build installs the four fixed headers from
[`overlay/`](overlay/README.md) before
mockturtle consumers compile. `scripts/apply_mockturtle_patches.sh` remains a
compatibility entry point and delegates to the checksum-verified overlay tool.
These three patches remain unchanged as historical provenance. Existing extra
edits cause a conflict rather than being overwritten. Campaigns freeze the
overlay, installer, and exact headers alongside the binary.

Check application, idempotence and conflict preservation with:

```
python3 scripts/experiments/tests/test_mockturtle_patch.py
python3 scripts/experiments/tests/test_library_filter.py
```
