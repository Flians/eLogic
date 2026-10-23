# Project mockturtle overlay

The four headers here preserve the existing project changes used jointly by
ePMapping and eGraphMap. They are fixed copies, not a second mockturtle checkout.
`manifest.json` records the upstream commit and both upstream/project SHA-256
hashes. The three patches in `tools/patches/mockturtle/` are preserved alongside
the overlay; their contents are unchanged by the move.

| Header | Existing project change |
| --- | --- |
| `algorithms/cut_enumeration.hpp` | Prune impossible fanin-cut prefixes while preserving enumeration order |
| `utils/tech_library.hpp` | Compare supergates within equal-function buckets |
| `views/cell_view.hpp` | Optional shared immutable cell-library ownership |
| `algorithms/emap.hpp` | Pass the shared library into the mapped cell network |

Project CMake configuration and every build verify/apply these files **before**
mockturtle consumers compile. Identical files keep their timestamps. All four
targets are checked before any replacement; incompatible upstream or additional
local changes stop the build without overwriting those changes.

```sh
python3 tools/apply_mockturtle_overlay.py          # apply or verify
python3 tools/apply_mockturtle_overlay.py --check  # read-only verification
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target epmapping_check --parallel 4
```

`MIGBALANCE_APPLY_MOCKTURTLE_OVERLAY=OFF` skips installation; it does not undo an
already installed overlay. A formal unmodified-upstream baseline must use its
recorded clean source and verify that these project modifications are absent.
Do not label a project-overlay build as an unmodified emap baseline. No algorithm
change was introduced when capturing these already existing files.
