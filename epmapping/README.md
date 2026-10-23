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

## 当前配置与复核（2026-10-23）

论文 v28 使用冻结的完整配置，不能用可执行文件的裸默认值替代：
`EPMAPPING_SYN_CACHE=262144`、`EPMAPPING_OPERATOR_POLICY=3`、
`EPMAPPING_BUILD_REUSE=2`、`EPMAPPING_SHARED_MAPPED_LIBRARY=1`、
`EPMAPPING_LIVE_QUERY_CACHE=9` 和每族两个 pivot。完整设置位于
`results_v28_three_parts_query_budget/cold/manifest.json` 的 `goal`。
尤其是裸默认缓存容量仍为 8192，裸运行不设置 post-mapping 截止；论文运行明确设置 1000 秒。

只读复核现有数据、冻结输入、既有 CEC 证据、独立物理 PPA 与论文均值：

```sh
python3 scripts/experiments/audit_epmapping_state.py --out /tmp/epmapping-audit.json
```

此命令需要本地完整 v28 档案，不运行新优化或新 CEC，不覆盖冻结实验、表格或论文 PDF。
仅核验 GitHub 紧凑归档时，使用 `python3 scripts/maintenance/verify_published_data.py --deep`。
检查点回归测试采用 v28 完整配置及其 `usb_phy` 参考点（21.39 / 158.68），
与旧候选生成试验的 21.38 阈值分开。正常完成和真正的 SIGALRM 截止分别检查，
检查点通过 CEC 不等于优化正常完成。完整配置的 128/128 完成，也不包括弱化控制中的 7 次截止。

新建实验时使用新目录，按配对驱动冻结配置，而不是覆盖原结果：

```sh
python3 scripts/experiments/run_three_parts_paired.py --root results_epmapping_new --jobs 6
```

本次复核及进一步优化建议见 [代码、数据和论文复核](docs/review-20261023.md)。

## 历史记录

旧README的全部实验过程和可选策略说明已原样归档至 [历史说明](docs/history/README-through-20261021.md)，包括旧版本、失败、回退和实验性operator/cache配置。归档文件中的相对路径按原README位置解释；不要将历史默认值当作当前设置。

整个仓库的职责划分、当前产物及清理规则见 [WORKSPACE.md](../WORKSPACE.md)。
