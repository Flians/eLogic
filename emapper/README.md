# eGraphMap / emapper

单输出标准单元工艺映射：在 e-graph 上构造 cuts、匹配相位和物理引脚，并按完整共享覆盖计算面积与延迟。

## 当前入口

- `src/egraph_map.rs`：主版本映射和覆盖恢复。
- `src/egraph_cover_exchange.rs`：绑定来源、完整覆盖保护和 cut 交换。
- `src/egraph_budget.rs`：累计节点准入；重建和状态退役不退额度。
- `src/egraph_regions.rs`：区域边界和输入表示。
- [BASELINES.md](BASELINES.md)：当前协议、数据位置与覆盖状态。
- [RECOVERY.md](RECOVERY.md)：恢复流程的实现说明。

较新的完整单输出库实现仍在隔离实验源码中，不能把实验版当成已经合入的主版本。请从下面的数据入口读取实际源码及二进制指纹。

## 当前实验

主测试为 EPFL 全20个 arithmetic/control 设计；每个设计只运行一次 `resyn2`，所有方法复用同一输入。
当前研究以 `(D,A)` 字典序为主：延迟先于面积；严格降低延迟但增加面积仍是字典序改善，不能称为 Pareto 支配。
映射相关搜索总时限600秒，包含候选生成、重写/搜索和恢复；共享resyn2、纯格式转换、独立CEC与指标报告不计入。
已取消外加节点、cuts、候选池、迭代、阶段与原生子进程硬上限；旧300秒/2秒/20000节点结果仅作历史证据。

| 资料 | 位置（相对仓库根） |
|---|---|
| 公共输入及预处理 CEC | `results_v2/epfl_resyn2_subjects_20261012/report.json` |
| 完整47单输出库 baseline | `results_v2/single_output_wide_20261021/primary_baseline120_checkpoint.json` |
| Baseline A/D/T与失败状态 | `results_v2/single_output_wide_20261021/full_library_baseline_A_D_T/` |
| 当前流式匹配/增量delay-flow研究 | `results_v2/cooperative_mapping_20261022/continue_checkpoint.json` |
| V9同源消融与完整缺失范围 | `results_v2/cooperative_mapping_20261022/v9_progress/` |
| 当前600秒baseline快照 | `results_v2/cooperative_mapping_20261022/baseline43_scope/` |
| GitHub紧凑数据索引 | `data/README.md` |
| DAC初稿 | `emapper_paper/DAC-eGraphMap-draft/main.tex` |
| 发布PDF | `output/pdf/DAC-eGraphMap-draft.pdf` |

baseline 原算法和源码不再修改；[来源清单](../tools/BASELINES.md)区分干净上游与历史补丁版本。
V9两组各3/20、共6个独立CEC输出，尚未完成全EPFL或重复验证；div有同延迟面积回退，不能称为已确认改善或已合入。
完整尝试不等于所有结果成功。基础覆盖的 CEC 通过也不等于 eGraph 优化成功。

## 历史资料与清理

- [旧映射器说明](docs/history/legacy-mapper-notes.md)
- [截至2026-10-21的完整实验流水](docs/history/BASELINES-through-20261021.md)

历史文件是原文快照，其中的相对路径按原文件位置解释。旧版本、失败和消融结果不与当前版本按设计混选。
构建缓存可以重新生成；不要删除源输入、genlib、数据库、网表、CEC证明、报告和指纹。
整个工作区的入口及清理记录见 [WORKSPACE.md](../WORKSPACE.md)。
