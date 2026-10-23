# ePMapping mockturtle patches

The parent repository preserves the dependency changes used by ePMapping:

- `tech-library-function-buckets.patch` groups dominance checks by truth table while preserving entry order within each function.
- `shared-mapped-library.patch` gives transient mapped networks shared ownership of an immutable cell library.
- `cut-prefix-enumeration.patch` prunes infeasible cut-tuple prefixes while preserving surviving tuple order. This patch captures the existing header used by the frozen experiment binary; packaging it does not change the measured implementation.

The current build installs the four fixed headers from
[`tools/mockturtle-overlay`](../../mockturtle-overlay/README.md) before
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
