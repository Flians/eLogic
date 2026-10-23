# epmapping

## Current goal: minimum area with no output-delay regression

The executable defaults to `EPMAPPING_OBJECTIVE=area_delay_bound` when the
objective is omitted. This is constrained area minimization: minimize cell area
subject to worst output arrival not exceeding the initial mapped delay, within
the existing 0.001 ps timing comparison tolerance. Delay breaks area ties.
It does not prioritize timing improvements at the expense of area.

This mode fixes the output-delay budget to the initial mapping, uses live backward
required times and net-area-gain ranking, and enables a transactional whole-network
`(area, delay)` guard. A round violating the delay budget or worsening the area-first
pair is restored. Local and global area-growth allowances are zero. Structural
depth is not constrained; candidate timing is checked in the library pin-delay
model. Noncritical local arrivals can increase within their output timing slack.
Whole-window synthesis is forced off. Every cut oracle uses the area-first goal.

The mode supersedes an inherited `EPMAPPING_PATH_OBJECTIVE=1`. It can also be
selected by `EPMAPPING_AREA_UNDER_DELAY=1`. Numeric objectives and the previously
named unbounded objectives remain explicit historical controls; the defaults of
the reusable C++ parameter struct remain backward compatible.

```sh
python3 scripts/experiments/explore_cut_only.py --root results_area_bound --phase all --profiles bounded
python3 scripts/experiments/explore_cut_only.py --root results_area_bound --phase converged --profiles bounded --rounds 6 --jobs 6
python3 scripts/experiments/report_cut_search.py --root results_area_bound
```

This is bounded greedy search, not a proof of a global minimum area.

## Runtime and recoverable checkpoints

Native SOP/DSD and cofactor memos now replace the least recently used entry when
their configured capacity is full, instead of permanently refusing new entries.
`EPMAPPING_CACHE_REPLACE=0` restores the historical insertion policy. The default
capacity remains `EPMAPPING_SYN_CACHE=8192` per memo, and zero disables storage.
These are process-local structural caches; they are not part of the persistent
e-graph pool and restart in a warm-cache process too.

Cofactor memo entries own immutable mapped DAGs. `EPMAPPING_COFAC_DEDUP=1` (default)
suppresses exact duplicate DAG/pin/polarity structures within a cut before copying
and live pricing. `EPMAPPING_COFAC_PROOF_CACHE=1` (default) proves the full ordered
cut function once when building a structural entry. This proof is valid only in
the fixed-library optimizer that owns the memo. Every use still recomputes live
area/timing, and accepted replacements retain their functional/cycle checks.
The proof-cache switch can restore per-use full-function checking for comparison.
Logs expose duplicate and structural-proof counts.

`EPMAPPING_COFAC_PIVOTS=N` keeps N pivots per decomposition family,
ranked by cofactor and Boolean-difference support sizes. The current default is
two; zero restores all pivots. This is a heuristic search restriction and can
lose area gains. Earlier unrestricted-pivot campaigns and their runtime pilots
retain their recorded settings; the current campaign includes an `allPIVOTS`
control to measure the tradeoff on unchanged inputs.

`EPMAPPING_CHECKPOINT=1` (default) saves the initial mapping and strictly improving,
globally legal canonical snapshots. Round boundaries are checked, and bounded-area
search also checks progress inside a round every 30 seconds after a replacement.
`EPMAPPING_CHECKPOINT_INTERVAL` controls that interval; zero disables only in-round
snapshots. Snapshots are deep independent networks and obey the original global
area/delay bounds. On normal completion, bounded-area search returns its best
checked snapshot, using the same quantized objective as round acceptance.

The manifest `epmapping_checkpoint.json` is published atomically after an immutable
netlist file has been written. Writes and snapshot checks count toward the native
post-mapping budget. No signal handler writes files; SIGALRM remains a timeout.
After a deadline, `scripts/experiments/recover_checkpoint.py` independently audits
the selected physical netlist and runs ABC CEC, recording hashes and timeout status
in `checkpoint_selection.json`. Validation is outside the search budget. Recovery
must not be reported as optimizer completion. Historical paper experiments retain
their original initial-fallback policy until explicitly rerun with the new policy.
For a new three-part campaign, select `run_three_parts.py --timeout-selection
checkpoint`; the policy is frozen in its manifest. The selection preparer and
auditors preserve the raw timeout row, separately verify the recovered netlist,
and keep it classified as a deadline. The default `initial` policy preserves
historical reporting. Reusing a campaign directory with a different policy is
rejected.

## 当前数据与论文入口

本项目是给定已映射网络上的后映射优化；与 `emapper/` 的从AIG进行工艺映射不同，不混用目标、预算或实验结果。

- 当前论文/报告使用 `results_v28_three_parts_query_budget/`；冷/热缓存和消融分别保留，固定初始延迟界限，post-mapping预算1000秒。
- 结果入口：`results_v28_three_parts_query_budget/RESULTS.txt`、`QUERY_BUDGET_ANALYSIS.txt`、`delivery_manifest.json`。
- DATE论文：`epmapping_paper/DATE27-ePMapping-sub/DATE27-ePMapping.tex`。
- 报告/讲稿：`epmapping_paper/talks/epmapping_30min/`。
- `db/` 是运行用数据库，不是可随意丢弃的历史日志；保留源码、库、冻结设置、输出网表和CEC证据。

## 历史记录

旧README的全部实验过程和可选策略说明已原样归档至 [历史说明](docs/history/README-through-20261021.md)，包括旧版本、失败、回退和实验性operator/cache配置。归档文件中的相对路径按原README位置解释；不要将历史默认值当作当前设置。

整个仓库的职责划分、当前产物及清理规则见 [WORKSPACE.md](../WORKSPACE.md)。
