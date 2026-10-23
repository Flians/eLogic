# Baseline source repositories

These Git submodules record the sources used to organize baseline
reproduction. Keep baseline tracked source files unchanged. Mapping drivers,
input/output adaptation, timing, verification, and experiment records belong
outside these repositories. Do not optimize baseline rules, extraction,
features, models, or winner selection.

| Method | Source directory | Upstream revision |
| --- | --- | --- |
| ABC `map` | `abc/` | `48498af8189ef321ee876065d8947875cf711294` |
| emap (existing mockturtle) | `mockturtle/` | `434c1ac9899c2fecd450abc89615e375f36e3a24` |
| EqMap | `eqmap/` | `d3144c45d3441530cd8d240c5fc6ea6645b8a2fb` |
| E-Syn | `E-Syn/` | `72f5568ab61bac9b8de224199c45fd975ab2399c` |
| E-morphic | `E-morphic/` | `26c128b5b0453795917474d118b8f89dd7a2ebc9` |

E-morphic's upstream repository is `https://github.com/Gy-Hu/E-Syn2.git`;
the local directory and submodule name use **E-morphic**.
Repository URLs and full revisions are recorded in `baselines.lock.json` and
the parent repository's gitlinks. Initialize the pinned sources from the
project root:

```sh
git submodule update --init tools/abc tools/mockturtle tools/eqmap tools/E-Syn tools/E-morphic
```

Use the existing `tools/mockturtle` directly, as requested; no duplicate
mockturtle baseline submodule is registered. Its current commit and existing
tracked modifications are preserved and recorded in `baselines.lock.json`.
Do not describe this modified working tree as a clean upstream checkout.
The project's four existing modified headers are now preserved in
[`mockturtle-overlay/`](mockturtle-overlay/README.md) and installed automatically
before project compilation. This explicitly named project overlay is separate
from an unmodified-upstream baseline; it is not pushed to the upstream repository.
Earlier stock-emap observations used upstream commit `1c9c3af`; their recorded
provenance is unchanged. The existing `tools/pybind11` submodule is also preserved.

This change registers sources; it does not build or switch the executables in
existing frozen campaigns. The ABC source revision corresponds to the upstream
revision in the installed Debian package version; it is not proof of a
byte-identical rebuild of that package. LFS smudging was skipped during source
registration; fetch upstream LFS assets separately if a build needs them.

Historical E-Syn runs used a `/tmp` source copy with changes exposing search
budgets through environment variables and resolving model paths. Those sources,
binaries, patches, and observations remain as historical evidence. The clean
`E-Syn` submodule does not contain those changes; do not relabel historical
results as runs of unmodified upstream source. New source/build versions need
explicit provenance and fresh experiment checkpoints.

Registration commands, original index snapshot, fixed-source checks, and
preservation checks are in
`results_v2/baseline_submodules_20261023/` relative to the project root.
The authoritative final scope is `final_scope.json`; earlier registration
snapshots retain the subsequently removed duplicate as historical evidence.
