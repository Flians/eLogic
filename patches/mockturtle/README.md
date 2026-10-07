# ePMapping mockturtle patch

`tech-library-function-buckets.patch` groups dominance comparisons by truth
table, preserving the original comparison order within each group. The patch
was tested with mockturtle commit `47d1e70fdf775e1a295016c3c17a1ad206db24c0`.

Configuring the ePMapping CMake target runs
`bash scripts/apply_mockturtle_patches.sh`. It applies a clean patch, recognizes
an already applied patch, and fails on conflicts without overwriting existing
edits. No commit in the mockturtle repository is required.

The patch changes only `include/mockturtle/utils/tech_library.hpp`; other local
submodule changes are not captured here. The experiment archive separately
records the source snapshot used to obtain the published measurements.

Check the implementation and application behavior with:

```
python3 scripts/experiments/tests/test_mockturtle_patch.py
python3 scripts/experiments/tests/test_library_filter.py
```
