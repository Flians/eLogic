# MIGBalance 整理记录（2026-10-21）

本轮整理已完成，工作区入口为 [WORKSPACE.md](../../WORKSPACE.md)。没有启动新实验或修改映射算法。

- 删除 15,940 个可再生成缓存/辅助文件：Cargo依赖与增量、CMake/Python缓存、7个已归档的TeX辅助文件。最终可执行文件、动态库及已解析manifest引用保留。
- 对 15,997 个重复实验数据路径使用硬链接去重；3,043组、19,040 个原路径全部通过完整内容SHA核验。保留所有文件名、内容、失败报告和CEC证据。
- 两次操作期间文件系统可用空间净增加 **19.45 GiB**。按原文件allocated blocks统计删除/替换 20.33 GiB；共享区块及外部活动会影响可用空间，两个数字不能等同。
- emapper和epmapping文档改为当前入口；历史原文逐字归档，三份原文SHA核验通过。
- 核验50个源码/论文/数据库/输入等受保护文件，以及四份最新隔离源码指纹（123个条目）；全部通过。源码、库、数据库、论文正文/PDF、冻结原始输入保留；不操作任何 `.git`。

## 清单

- `documentation_archives.json`：历史文档位置及原SHA。
- `cache_plan.json`、`cache_removed.json`：缓存逐文件清理及保留清单。
- `latex_auxiliary.tar.gz`：删除前的7个TeX辅助文件；可从仓库根目录用 `tar -xzf maintenance/cleanup_20261021/latex_auxiliary.tar.gz` 恢复。
- `data_dedup_plan.json`、`data_deduplicated.actions.jsonl`：各重复文件的原路径、权限、时间、SHA及共享目标。
- `data_dedup_validation.json`、`protected_validation.json`：内容及冻结证据核验。
- `status.json`：机器可读结果。

## 后续使用

去重历史文件共享同一inode并设为只读，不能在旧目录原地重跑或直接 chmod 后编辑；新实验使用新目录。若需要编辑某个历史副本，先将它复制到临时文件，再原子替换原路径，最后只对这个新副本恢复写权限；原权限与mtime见actions清单，其他共享路径不受影响。此操作需要相应磁盘空间。

可再生成缓存没有完整备份，下次编译可能重建较慢。没有删除旧实验失败或按指标挑选结果；历史中间数据的空间优化仅限内容完全相同的副本。没有执行 Git GC 或删除 `.git`。

最新实验有限批次已结束：20个设计、18原生优化覆盖、20独立CEC；log2/hyp仍是基础覆盖。恢复位置为 `results_v2/search_resources_20261021/cleanup_resume_checkpoint.json`，重复验证和合入仍待后续研究，清理不改变其结论。
