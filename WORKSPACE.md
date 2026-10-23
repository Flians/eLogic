# MIGBalance 工作区导航

| 项目 | 职责 | 当前说明 |
|---|---|---|
| `emapper/` | 从AIG进行标准单元工艺映射；当前研究以单输出、(D,A)为主 | [README](emapper/README.md)、[比较协议](emapper/BASELINES.md) |
| `epmapping/` | 已映射网络上的后映射优化：固定初始延迟界限，降低面积 | [README](epmapping/README.md) |
| `emapper_paper/DAC-eGraphMap-draft/` | eGraphMap DAC初稿 | `main.tex`；发布件 `output/pdf/DAC-eGraphMap-draft.pdf` |
| `epmapping_paper/DATE27-ePMapping-sub/` | ePMapping DATE论文 | [论文/讲稿入口](epmapping_paper/README.md) |
| `results_v2/cooperative_mapping_20261022/` | 最新eGraphMap600秒流式/增量研究 | `continue_checkpoint.json`、`v9_progress/`；部分数据，未合入 |
| `data/` | GitHub数据目录与紧凑实验归档 | 原件本地保留；同步范围、遗漏与SHA明确列出 |
| `tools/` | 固定baseline源码及项目mockturtle覆盖层 | [来源说明](tools/BASELINES.md)、[覆盖层](tools/patches/mockturtle/overlay/README.md) |
| `results_v2/single_output_wide_20261021/` | 完整47单输出库baseline、旧对照与原始证据 | `primary_baseline120_checkpoint.json`、`full_library_baseline_A_D_T/` |
| `results_v28_three_parts_query_budget/` | ePMapping当前论文数据 | `RESULTS.txt`、`delivery_manifest.json` |
| `scripts/experiments/campaigns/` | 可审计的实验、计时与报告入口 | 使用各脚本`--help`；恢复以冻结manifest为准 |
| `maintenance/cleanup_20261021/` | 本次整理清单与核验 | [整理报告](maintenance/cleanup_20261021/README.md)、归档/清理/去重清单 |

`emig/`、`mig_egg/`、`map_egg/`、`tools/` 等仍提供依赖/历史算法，不能因不在主线就删除。不同项目的时间预算和优化目标不混用。

## 保留与清理边界

- 保留源代码及未提交修改、库/数据库、原始输入、已发布PDF、实验报告/命令/参数/哈希、失败与CEC证据。
- 历史流水说明进入各项目`docs/history/`；当前README只放职责、运行入口和最新证据位置。
- 优先删除可再生成的编译增量、依赖缓存与LaTeX辅助文件；不删除被manifest直接引用或仍在运行的文件。
- 旧实验的完全相同数据可以经完整SHA核验后去重，原路径和内容须保留。只读共享历史数据不能在原目录覆盖重跑；新实验必须使用新目录。
- 常规清理不操作Git元数据；本次用户明确授权的子模块注册、提交和GitHub同步单独记录。不把清理历史当作删除不利实验结果。
