# MIGBalance

Logic synthesis and technology-mapping research with two separately evaluated projects:

| Project | Scope | Entry |
| --- | --- | --- |
| eGraphMap / emapper | AIG-to-standard-cell mapping, currently single-output and delay/area lexicographic optimization | [emapper/README.md](emapper/README.md) |
| ePMapping | Post-mapping area recovery under the initial mapped delay bound | [epmapping/README.md](epmapping/README.md) |
| eGraphMap paper | DAC draft | [emapper_paper/README.md](emapper_paper/README.md) |
| ePMapping paper | DATE paper and talk | [epmapping_paper/README.md](epmapping_paper/README.md) |
| Experiments | Published compact records, failures, provenance and selected CEC artifacts | [data/README.md](data/README.md) |
| Dependencies | Pinned baseline sources and the fixed project mockturtle overlay | [tools/BASELINES.md](tools/BASELINES.md) |

[WORKSPACE.md](WORKSPACE.md) gives the workspace layout.
[Current evidence](data/CURRENT.md) explicitly separates incomplete experiments,
search timeouts, verified incumbents and completed optimizations. Do not combine
these two projects' objectives, budgets or results.

## Build

Initialize the pinned submodules; E-morphic is the local name of upstream E-Syn2:

```sh
git submodule update --init
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target epmapping_check baseline_mapper --parallel 4
cargo build --release --bin bench_rs_egraph_map
```

CMake requires a C++17 compiler, Python interpreter/development module,
Coin-OR CBC dependencies, and Rust/Cargo for the project bridge. Experiment
scripts additionally use Yosys, ABC and the Python packages recorded in their
source/build provenance.

Before consumers compile, CMake verifies/applies four fixed project headers from
[tools/patches/mockturtle/overlay](tools/patches/mockturtle/overlay/README.md). The corresponding
patches are under `tools/patches/mockturtle/`. Identical installed files are not
rewritten; conflicting extra edits stop the build. These are named project
extensions, not an unmodified upstream baseline.

## Verify and inspect

```sh
python3 scripts/maintenance/test_mockturtle_overlay.py
python3 scripts/experiments/tests/test_mockturtle_patch.py
python3 scripts/maintenance/verify_published_data.py --deep
```

Current eGraphMap studies use one600-second mapping-related search window;
independent CEC is excluded. Historical budgets remain attached to their original
cohorts. Baseline source/algorithms are not optimized; historical E-Syn source
patches are disclosed separately from the newly registered clean upstream source.

## Data and local files

The compact data export is versioned under `data/`, with per-file and per-archive
SHA-256 manifests. Large raw experiments and build caches remain local; this is
not a151GB raw-data mirror. The latest legacy database is exported losslessly as
compressed parts; the older root Git LFS object retains its original history.
Shell configuration, credentials and editor state are not publication inputs.

The earlier root documentation remains in
[docs/history/README-through-20261023.md](docs/history/README-through-20261023.md).
