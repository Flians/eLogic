# iwls_runner

A pipeline that drives **mig_egg** on the **IWLS 2026 logic-synthesis
contest** benchmarks.

Input: 100 multi-output truth tables (`benchmarks/ex2NN.truth`).
Output: a contest-shaped zip of binary AIGER files plus per-bench Pareto
points of (delay, area).

This folder is part of branch **`iwls-runner`** in this repository. The
sibling commit on the same branch updates `mig_egg/src/lib.rs` to switch
`rules()` to AIG-only and add an `iwls_run` test entry that this pipeline
drives. Checking out the branch gives you both pieces; you do not need to
apply any external patch.

## What this folder contains

```
iwls_runner/
├── README.md                  this file
├── abc.rc                     ABC startup config (provides resyn2,
│                              compress2rs, &dc2 aliases) — must sit in cwd
│                              for ABC alias commands to work
├── benchmarks/                100 IWLS truth tables (ex200..ex299)
├── scripts/
│   ├── aig_to_prefix.py       binary AIGER -> Lisp prefix expressions
│   ├── prefix_to_aig.py       Lisp prefix -> EQN -> ABC strash -> AIGER
│   ├── run_one_bench.py       one bench end-to-end (truth->AIG->prefix->
│   │                           mig_egg->prefix->AIG->cec)
│   ├── run_all.py             parallel runner over many benches
│   └── finalize.py            cec-verify + per-bench Pareto + contest zip
├── examples/
│   ├── smoke_one.sh           single bench (ex212), ~30s, sanity check
│   └── full_run.sh            all 100 benches + finalize, ~30 min on 16 cores
└── results/                   per-run outputs land here (gitignored)
```

## Quick start

```bash
# 1. build the mig_egg test binary (this branch already has the lib.rs
#    changes the pipeline depends on)
cd ../mig_egg                 # i.e. <repo>/mig_egg
cargo build --release --tests --no-default-features

# 2. point at your repo root and abc binary, then run smoke
export ELOGIC_REPO_DIR="$(cd ../.. && pwd)"   # parent of iwls_runner
export ABC=/abs/path/to/abc                    # default: 'abc' on PATH

cd -                           # back to iwls_runner/
bash examples/smoke_one.sh     # ex212, ~30s

# 3. full run
bash examples/full_run.sh
```

## What the lib.rs sibling commit changes

Single file: `mig_egg/src/lib.rs` (133-line diff, fully revertible with
`git checkout main -- mig_egg/src/lib.rs`).

1. **`rules()` switched to AIG-only.** Majority-language rules (`maj_*`,
   `distri`, `com_associ`, ...) are commented out; the 5 AND-language rules
   already defined in the file (`comm_and`, `comp_and`, `dup_and`,
   `and_true`, `and_false`) are uncommented. `associ_and` stays off
   because its O(N²) match cost blows up `node_limit` on bigger benches.
   Reason: IWLS truth tables strash into pure AIG — running maj rules on
   `(& a b)` form fires zero rewrites.

2. **`iter_limit(1000)` → `iter_limit(10)`** at three sites
   (`simplify_depth`, two `simplify_best` runners). egg's `time_limit`
   is best-effort; tightening `iter_limit` is what actually keeps
   wall-clock bounded on dense ASTs.

3. **New `iwls_run` test entry** at the bottom of the test module. It
   reads `IWLS_PREFIX_FILE` (a path to a prefix file with one PO per line,
   `<idx>\t<expr>`) and, for each line, calls `simplify_best` and prints
   `iwls case <idx> best_expr[0] = <expr>` to stdout. The Python wrapper
   greps for that line. `IWLS_FIRST_DEPTH=true|false` chooses delay-first
   vs area-first cost.

## How a bench gets through

```
benchmarks/ex212.truth
     │  abc:  read_truth -xf ... ; strash ; print_stats ; write_aiger
     ▼
init.aig  (binary AIGER)
     │  scripts/aig_to_prefix.py
     ▼
init.prefix     (one PO per line, e.g.  "0\t(& (~ a) (& b c))")
     │  $ELOGIC_REPO_DIR/target/release/deps/mig_egg-<hash> iwls_run
     │  (env IWLS_PREFIX_FILE=init.prefix)
     ▼
optimized.prefix  (best_expr per PO from simplify_best)
     │  scripts/prefix_to_aig.py  (writes EQN, ABC strash + resyn2)
     ▼
final.aig
     │  abc:  read_truth -xf ... ; cec -n final.aig
     ▼
results/<run>/aigs/ex212_<lev>.aig    (only if cec equivalent)
```

`finalize.py` then walks `results/<run>/aigs/`, dedupes by (lev, and),
applies per-bench Pareto, optionally cec-verifies again, and emits a flat
zip ready for the IWLS contest submission form.

## Useful knobs

| flag / env | default | what it does |
| --- | --- | --- |
| `--workers N` (run_all) | 16 | concurrent benches; e-graph is RAM-heavy on big inputs (~1-15 GB / worker peak), drop to 4 if you OOM |
| `--simplify-timeout S` (run_one / run_all) | 900 | wall cap on the cargo test invocation per bench |
| `--cec-timeout S` | 1800 | wall cap on `cec -n` |
| `--first-depth true\|false` | true | forwarded to `IWLS_FIRST_DEPTH` (delay-first vs area-first cost) |
| `--resume` (run_all) | off | skip benches whose `results/<bench>.json` is `ok=true` |
| `ABC` env | `abc` | path to abc binary |
| `ELOGIC_REPO_DIR` env | (required) | abs path to the repo root that contains `mig_egg/`; the parent of this folder if you keep `iwls_runner/` inside the repo |

## Notes

- All ABC commands run with `cwd=<this folder>` so abc.rc aliases
  (`resyn2`, `compress2rs`, `&dc2`) work. If you replace abc.rc, keep
  those aliases.
- Truth tables are read with `-xf` (binary, file). Each line is one output;
  line length is 2^k where k = #inputs.
- The contest scorer wants `exNNN_LLL.aig` (3-digit padded level) at the
  zip root. `finalize.py` produces exactly that.
- Big benches (ex299, ex230, etc.) hit egg's `node_limit(5000)` ceiling and
  fail simplify. That's a real ceiling of the AIG ruleset on densely shared
  graphs, not a timeout — bumping simplify-timeout doesn't help.
