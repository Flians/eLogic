# ePMapping mockturtle patches

The parent repository preserves the dependency changes used by ePMapping:

- `tech-library-function-buckets.patch` groups dominance checks by truth table while preserving entry order within each function.
- `shared-mapped-library.patch` gives transient mapped networks shared ownership of an immutable cell library.
- `cut-prefix-enumeration.patch` prunes infeasible cut-tuple prefixes while preserving surviving tuple order. This patch captures the existing header used by the frozen experiment binary; packaging it does not change the measured implementation.

Configuring the ePMapping target runs `bash scripts/apply_mockturtle_patches.sh`.
It applies missing patches, recognizes existing applications, and fails on conflicts while preserving local edits. No dependency commit is required. The base header versions are compatible with mockturtle commits `47d1e70fdf775e1a295016c3c17a1ad206db24c0` and `1c9c3af6a423f650f654ccec340dae9b93995844`; the latter changes the nauty source list. Other submodule edits are not staged by this workflow. Campaigns also freeze the exact header sources used by their native binary.

Check application, idempotence and conflict preservation with:

```
python3 scripts/experiments/tests/test_mockturtle_patch.py
python3 scripts/experiments/tests/test_library_filter.py
```
