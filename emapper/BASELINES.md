# eGraphMap 当前比较协议与实验索引

> 最新600秒范围（2026-10-23）：实验入口与冻结源码以 `results_v2/cooperative_mapping_20261022/continue_checkpoint.json` 为准。当前策略只有600秒总映射相关搜索时限，CEC/共享resyn2/纯转换/指标报告排除；node/cut/pool/iteration/stage/RSS硬上限均无。旧20000节点/2秒及300秒实验为历史，不是当前policy。
>
> V6同一配置已14/20独立CEC；V8同源cadence1/16各4/20，不能混成一个mapper。V8 log2 A/D=6094.36/3857.75，对照6324.35/3989.77；但div同D面积9596.88→9598.65回退，拒绝采用简单cadence16。baseline最近完整快照43/120组，106合格observations/11失败/143pending；原算法保持。V9完整评分前置筛选已冻结，282项相关回归通过，同源开关各3/20、共6个CEC输出。adder A/D相同；div同D面积+0.20；log2 D4009.14→3989.77、A6349.19→6324.35，仅一次配对，仍差于更强历史结果，不支持合入。一般延迟物化和完整可暂停共享面积恢复尚未实现，主版本/论文最终表未据部分数据合入。完整报告 `v6_progress14/`、`v7_progress/`、`v8_progress4/`、`baseline43_scope/`；所有完整scope gates仍false。


> **当前新实验策略（2026-10-22，用户最新指定）**：仅每设计/方法600秒总映射时间，CEC另计；取消节点/额外搜索比例、候选池/cut数量、迭代和阶段时钟的硬限制。入口 `scripts/experiments/campaigns/run_time_only600.py`，配置 `emapper/EXPERIMENT_POLICY.json`。基线原算法不改。下方300/60/5/20/2秒及rho/100000等均为保留的历史协议，不能静默套用到新策略，也不能把旧baseline数据改称600秒结果。


本文件保留当前可操作信息。完整历史见 [原始实验流水](docs/history/BASELINES-through-20261021.md)；原文与SHA存档，未删掉失败或负结果。

## 最新推进与九个超时审计（2026-10-22，覆盖下方旧计数）

- 正式600秒全搜索baseline矩阵现在36/120组：30组明确采用，6组新尝试；102个合格方法观测、2失败、156待测槽，剩余84组。新adder两目标EqMap共6CEC；新原版E-Syn/adder面积目标600秒超时，未将partial升格。所有源码/模型/产物哈希复核通过，runner锁空闲；原算法未调优。当前data在total600_progress36_scope，三目标fullscope gate均false。
- 新只读诊断入口analyze_time600_timeouts.py重放原20例，SHA核验后稳定返回11优化/9超时。hyp/log2/multiplier/sin/sqrt/square/voter均在cuts结束前终止；div/mem_ctrl进入恢复，分别保留10,588,394/16,312,958实例。全部9例导入完整、native saturation<=1.25秒，不是node-budget rejection。sin导入只有5,085 e-nodes，也仍超时。
- div cuts464.79秒+小cut生成123.47秒；恢复在import_done后597.04秒开始，留给恢复的时间上界约2.96秒。mem_ctrl对应54.98/231.64秒、恢复在311.02秒后开始，不能以同一种“只剩几秒”解释。尚无RSS/逐轮cut数/恢复内部profile，不能声称OOM或锁定唯一热点。
- 静态审计发现：每轮整cuts clone与全局functional index；全池witness初始化排序/依赖建表没有deadline参数；time-only保留全池并不携带generation frontier；recover复用入口采样的旧remaining duration，外层600秒仍有效；恢复trace仅Drop时落盘，SIGKILL丢失内部里程碑。详细证据/限制/可证伪后续实验见results_v2/time_only_600_20261022/timeout_analysis_replay/report.md。尚未修复或宣称PPA收益。
- 后续只优化eGraphMap：增量cut/match/cover、精确去重和delta索引、可取消/可复用恢复setup、绝对deadline与经过验证的incumbent原子发布。保持唯一600秒总钟，不给baseline做阶段预算调优，不重跑无变化的9个失败。研究kernel未合并默认；九个失败全部保留。
- DAC正文按上述36组和诊断更新；5正文+1参考、标准字号边距，变更4/5/6页视觉QA，1/2/3像素与旧QA一致。正式baseline全覆盖和新版本修复仍未完成。

## 当前执行口径：baseline也仅600秒总映射额度（最新用户指令）

此节覆盖下面所有旧口径。总时间包括候选生成、映射相关重写/搜索、SA和实际LUT/单元映射；父级搜索与嵌套ABC使用区间并集，CEC单独核验且不报告CEC运行时间。禁止优化baseline算法、规则、模型、目标或作者原生赢家选择；作者固有方法步骤原样保留，不把它们包装成额外实验预算。

- 当前入口 `scripts/experiments/campaigns/run_baseline_total600.py`，矩阵 `results_v2/time_only_600_20261022/total600_original_baselines_epfl`；仅采用30组已证明600内完成的同依赖core/原生SA观测。EqMap/E-Syn旧2s/20k/16适配限制会改变搜索，因此即使旧CEC成功也不采用，全部重新尝试。
- EqMap使用原生time-only接口（-t600，无-s/-n）；E-Syn通过既有环境控制设置600秒、非约束节点/迭代值；外层剩余总时间同时覆盖后端映射，不能每阶段重置600。原版源码、二进制、模型保持；这只是预算口径更正，不是baseline算法调优。
- 最新新口径矩阵33/120组（30组明确采用、3组新尝试），96合格方法观测、1失败、163待测槽。dec两目标EqMap共6CEC通过；原E-Syn/dec面积目标实际600秒搜索超时，区间关闭且未使用partial结果。全部runner锁空闲；续跑入口/剩余87组见continue_checkpoint。发布PDF更新至此快照，5正文+1参考逐页QA通过。
- 4项新计时/转发检查通过。首个真实dec EqMap面积目标组三方法全部公共CEC通过，原EqMap组合A44.04/D94.34，搜索+LUT+后端T14.2022s；该数值不代表eGraphMap改善或baseline优化。CEC运行时间字段不进入新公开报告。
- 之前78组旧mapping-calls-only cohort与74组search-inclusive但保留旧适配预算cohort均封存，原partial和失败保留，操作切换不追记为超时。准确旧计数以sealed/campaign为准；新scope pending与旧failure不能混为成功。
- eGraphMap全20的600秒搜索进程已有11优化/9超时、20独立CEC；hyp/log2仍为仅源覆盖。三目标全范围gate仍false。原生研究kernel未合并主默认；全baseline复测尚未完成。

## 最新600秒计时范围澄清（2026-10-22，覆盖下方旧口径）

用户明确确认：候选生成、面向映射的重写/结构搜索、SA及单元/LUT映射均计入同一个600秒额度；一次resyn2、纯格式转换、独立CEC和指标报告排除。作者原算法/模型/native参数不改。父级搜索与嵌套ABC按active interval union计算，不重复扣时。

- 新入口 `scripts/experiments/campaigns/run_baseline_search600.py`，新矩阵 `results_v2/time_only_600_20261022/search600_baselines_epfl`。72组已有观测通过来源、CEC和新增搜索时间保守上界核验采用（188方法观测），不是新跑600秒数据；48组待新口径尝试。三个真实计时检查通过：搜索可执行文件识别、父子区间去重、搜索超时与CEC排除。
- 原 `original_baselines_epfl` 封存于78/120组。E-morphic/div旧口径只计ABC，候选生成/SA长时间未调用ABC；因用户更改口径停止，自有进程树、原日志/临时产物/SHA保留在 `search_scope_transition/`。这是操作中断，不能追记为旧600秒映射超时。新矩阵从独立目录重新尝试。
- eGraphMap已完成20/20的600秒进程本来包含其全部搜索，故本次计时澄清不改变该20观测：11优化覆盖、9超时，仅源覆盖不能算优化成功。hyp/log2尚未获得真正优化覆盖。全scope DA/AD/Pareto均false；研究kernel尚未并入主默认。
- 原始失败/回退及旧300/600计时数据全部保留。新进度以新campaign和continue_checkpoint为准，不能续跑旧fingerprint或把新增搜索T当旧精确mappingT。

## 600秒全20首轮与原版baseline续测（以下为口径澄清前快照）

- eGraphMap新协议已20/20：11完整优化覆盖、9映射超时、20最终网表独立named-port CEC；9个超时分别是div/hyp/log2/multiplier/sin/sqrt/square/mem_ctrl/voter。超时后的源网表只作合法基础覆盖，不计优化成功。
- `results_v2/time_only_600_20261022/full20_candidate_audit.json` 核验820项源码/二进制/产物哈希、实际无global count截断、600秒映射钟、全部CEC。完成图仍未胜全部baseline，不合并研究kernel到主默认。
- 正式原版baseline入口：`scripts/experiments/campaigns/run_baselines600.py`，目录 **`results_v2/time_only_600_20261022/original_baselines_epfl`**。原69组完整成功记录经来源/T/CEC审计显式采用，保持旧300秒标签；其余51组按600秒实际映射累计限时顺序复测。当前有限51job batch正在执行，先读campaign.json/锁/continue_checkpoint.json，禁止并发启动。
- 已完成的真实EqMap+E-morphic计时probe都通过独立CEC；EqMap初始LUT ABC实际0.2155秒并计入相应组合，原生final-SA70次反馈映射14.1697秒、70候选CEC通过。仅计时/正确性probe，不替换主表里任何原生赢家，不据此宣称加速。
- 新harness记录适配后实际argv，Yosys的临时ABC脚本、输入BLIF/lutdefs/genlib/输出及哈希完整保留；原算法/源码/模型/原搜索参数不变。4项真实子进程计时/转发/超时/归档测试通过。
- 截至`progress_after_div_area`，baseline76/120组、188合格方法观测、6报告失败、66待测槽；完整范围DA/AD/Pareto gate均false，MtM未接入。后续实时进度以正式campaign为准，不能把快照计数当最终完成。
- Hyp四个普通baseline均获得合格物理数据：前三个新独立CEC；最后一个emap delay的当前Verilog字节、语义JSON、original/library/model及旧独立GIA证明哈希全部相同，显式复用证明后只结束冗余checker。原始exit-15验证中断与临时验证目录已保留；不改mapping输出/参数、不把复用说成新CEC。
- 复用证据与操作审计在`hyp_identical_delay_proof/`。**所有后续report命令需附** `--proof-reuse results_v2/time_only_600_20261022/hyp_identical_delay_proof/proof_reuse.json`，报告会同时保留原失败字段。不可直接改原report为成功。
- `stage_diagnosis_full20.json`：hyp/log2在cut扩展阶段耗尽，div/mem_ctrl已进恢复仍无完整导出。外层600秒实限有效；恢复内部较早的remaining-time估计及最终导出交接仍需单独改进和新版本验证，不据静态分析宣布已修复。
- 47条库记录实际为45个物理cell类型+2个零面积常量；E-morphic原生反馈Liberty的45个物理名字与之匹配。反馈loaded模型与公共unloaded delay仍不同，不能声称同一内部cost模型或端到端equal compute。
- 论文已改成该47记录单主表、横向A/D/T、明确F/P和历史消融；当前发布快照为全20 candidate、baseline69组，5正文+1参考逐页QA。完整baseline数据齐后再刷新，不把当前草稿称作最终全套比较。

`baselines_checkpoint`/`baselines600_checkpoint`仅是各69组旧观测采用的初始化快照，零新benchmark job；源码均精确归档在`baseline_preflight_history.json`，不可续跑旧fingerprint或重复计数。所有原失败、旧论文、数据和源代码快照保留。

## 600秒主协议推进（2026-10-22，完整对照仍在运行）

- `time_only_600_20261022/epfl_checkpoint/campaign.json`：截至 `progress_after_log2` 完成6/20；adder/bar/max返回完整优化覆盖，div/hyp/log2各600秒超时，三者保留的源覆盖均通过独立CEC，不能算优化成功。cuts逐轮扩张是hyp/log2已观测的主要瓶颈，不能仅取消导入限制便宣称解决。
- `time_only_600_20261022/baselines600_checkpoint/campaign.json`：69/120组完整成功的旧观测，经依赖、实际T/保守上界、CEC及来源核验明确采用（182方法观测），保持旧300秒标签；另51组不完整任务待新600计时复测，旧失败全部保留。不把已采用记录称为新跑600秒结果。
- 新计时harness `run_baseline600_budget.py` / `meter_yosys_abc600.py` 精确接入EqMap最初Yosys内ABC调用；向每个相关方法计一次共享LUT映射前缀，统一Clock合并并行SA反馈，转换/CEC/非映射SA排除。4项真实子进程测试通过；正式EqMap/SA新harness探针尚待当前mapper顺序批结束后执行，不能把合成测试当EPFL CEC。
- 作者原始源码/commit/模型与算法参数不变：3仓库13354个tracked文件、250项选定实际依赖及旧有ABC/Yosys哈希核验通过。旧Yosys初始ABC没有单独二进制哈希，保守上界与旧CEC原样保留，新的精确计时另行固定该可执行文件；限制见 `baseline_system_tools.json`。
- 600版去掉可配置全局node/pool/cut/iteration/helper硬上限与分阶段时钟；小cut/定向宽匹配及功能提案、安全交换的算法内结构/事务批次规则仍在，不宣称穷尽全部候选。
- 最新完整范围CSV/缺失/失败和3目标gate：`progress_after_log2/`，DA/AD/Pareto均false；MtM未接入。论文现为同一进度快照的47库单主表、横向A/D/T，历史消融明确区分；5正文+1参考已逐6页QA。整个600秒全EPFL对照、基线复测和重复验证尚未完成。

本节路径均在 `results_v2/` 下，除显式脚本路径。`baselines_checkpoint`是仅69组采用的初始计时依赖审计快照，未运行新baseline；系统工具哈希补录后的正式入口为`baselines600_checkpoint`，不重复计数。所有旧文稿、数据、失败及哈希保留。

## 历史评价协议（以下300秒及比例预算均不是当前600版）

- EPFL arithmetic/control 全20设计，复用 `results_v2/epfl_resyn2_subjects_20261012/report.json` 的一次 `resyn2` 输入。
- 当前提供的47个单输出单元全部可加载；小 cuts 常规匹配，七个7–9输入单元走定向匹配。不是完整 foundry 库，也不保证穷尽全部宽匹配。
- 同输入、同库、公共物理实例面积与 unloaded block pin delay，独立 named-port CEC。
- 主目标为 `(D,A)` 字典序；`(A,D)` 与 Pareto 检查分开。面积增加不能否定严格延迟改善。
- 映射最多300秒，CEC/适配/预处理/指标排除；原生60秒（EqMap90秒）及作者原搜索参数分别保留，不称端到端 equal compute。
- 本轮统一ρ=1/2，`B=2*(PI+AND)+2`、`G=B+ceil(B/2)`，累计准入不退款；native retained≤100000、完整源覆盖加搜索的全局保留上界B+100000、cut/class≤16、SAT≤16、全局恢复≤2秒。原生地址空间上限16 GiB。
- 定向宽边界不冒充物化 cuts≤16；共用总匹配访问额度、全局保留池和时间。保留上限不等于生成暂存内存上限。

## Baseline 范围

ABC map/map-a、原版独立 emap、EqMap LUT+同库amap及staging controls、E-Syn冻结模型有界适配、E-morphic最终原生SA。原算法/模型/搜索参数不变；未接入MtM。
完整47库集合为120个已尝试任务、260条方法观测：189合格、71失败/未证明/未测。EqMap的T包含明确标注的 enclosing LUT 保守上界；不与精确mapping计时混称。失败不被其他版本或优化探针替换。

| 证据 | 仓库根目录下的位置 |
|---|---|
| Baseline来源、任务与原始失败 | `results_v2/single_output_wide_20261021/primary_baseline120_checkpoint.json` |
| Baseline A/D/T、CEC状态和来源 | `results_v2/single_output_wide_20261021/full_library_baseline_A_D_T/` |
| 原版emap清洁源码及宽度适配 | `results_v2/single_output_wide_20261021/stock_emap9_cut9_reference/` |
| 元数据共享全20首轮 | `results_v2/search_resources_20261021/metadata_epfl/campaign.json` |
| 元数据共享预算核验 | `results_v2/search_resources_20261021/metadata_full20_runtime_audit/validation.json` |
| 排序复用/阶段调度全20首轮 | `results_v2/search_resources_20261021/ranked_epfl/campaign.json` |
| 该版本预算核验与baseline对照 | `results_v2/search_resources_20261021/ranked_full20_runtime_audit/`、`ranked_vs_baseline120_scope/` |
| 最新物理键复用全20（18原生优化覆盖） | `results_v2/search_resources_20261021/small_key_epfl/campaign.json` |

元数据共享版完成20/20，17原生优化覆盖；调度版完成20/20，18原生优化覆盖，div/mem_ctrl正常返回并CEC通过。两版相对上一统一配置的首轮 `(D,A)` 均为9好/11平/0差；不是全套baseline胜出或重复稳定性证明。log2/hyp在调度版仍失败并回退。最新物理键复用版也已完成20/20、18原生优化覆盖、20独立CEC及预算核验；log2/hyp仍为基础覆盖。清理期间未再启动实验，后续从 `cleanup_resume_checkpoint.json` 恢复对照与重复验证。主版本尚未采用这些隔离源码；完整旧版本、负结果、全部指纹保留。

## 整理约定

当前输入、数据、失败、CEC、论文及源码快照不按好坏删除。编译依赖/增量目录属于可再生成缓存；删除或去重须记录清单，并保留被manifest引用的文件及运行中依赖。清理说明见 [WORKSPACE.md](../WORKSPACE.md)。

## 单向库规则独立对照（2026-10-21起，未接入主版本）

研究入口为 `results_v2/library_rules_20261021/README.md`。最新冻结版本 `source_signed` 在有界AIG饱和后执行单向库规则，覆盖恢复/提取代码与当前对照源码相同；支持补出的负相位逻辑类，并修正TT算子补零编码。Cell与Not helper共同预检和计入累计G、独立S；与Boolean阶段共享16轮。三组同二进制、47库、rho=1/2、300秒映射/60秒原生/2秒恢复。小cut算法和16槽上限相同，新增逻辑类仍可能改变实际cut银行；组合组将固定匹配额度分给两条路线，内部计数单位不等同机器操作数。

全20三组60次与五设计30次额外重复均已完成，90最终CEC通过，81次选用原生优化覆盖；其余明确为失败后的基础覆盖，不算原生优化成功。全套实际预算审计通过，所有失败和回退保留。

|相对同二进制控制组|原生覆盖/20|全20 (D,A) 好/平/差|仅D 好/平/差|
|---|---:|---:|---:|
|控制：小cuts＋原定向宽匹配|18|—|—|
|替换：小cuts＋库规则|17|5/3/12|1/14/5|
|组合：小cuts＋两种匹配|16|7/4/9|0/16/4|

五个预定设计sin/sqrt/adder/bar/multiplier的两次额外重复中，替换组10个配对全部差，组合4好/6差。Sqrt、Bar组合组同延迟面积收益可重复，Sin/Multiplier回退也可重复；Adder首轮面积收益在额外重复中翻转。Mem_ctrl首轮唯一严格D改善来自替换组，后验定向重复单独保留于 `signed_mem_ctrl_repeats/`，不冒充全20重复。

- A/D/T、CEC与原生/基础覆盖区分：`signed_full_report/`、`signed_repeats_report/`。
- 实际预算、原生配置及新节点联合计费：`signed_full_budget_audit/`、`signed_repeats_budget_audit/`。
- 全20及未成功baseline的范围保留：`signed_baseline_scope/`，九个lex/Pareto gate均false，baseline算法/模型/参数未改。
- 关键路径和候选覆盖：`signed_coverage/coverage.json`、`signed_key_path_summary.json`。全部60个输出网表的物理关键路径与公共测量复算一致；七个7–9输入单元在40个库阶段均记录0匹配/0新增独立条目。生成计数位于公共截断之前，门型统计不证明同一个原AIG边界或绑定来源。
- 恢复器复用证明：`recovery_reuse_proof.json`；新源码/二进制：`signed-source-freeze.json`、`signed-vs-control.patch`。

以上相对目录均在 `results_v2/library_rules_20261021/`。本次尚无可采用的统一优势，**未合入主版本**，不能按设计混选配置或重复中的最好值。V2严格正相位版与V3复合模式版、最初G-only超S试运行均独立保留；旧编码/相位覆盖限制和负结果不被删除，也不能与V4拼接宣称胜出。

Mem_ctrl后验额外6次现已完成：替换组两次均复现1059.69→1056.51（D约-0.30%），A代价约+0.22%（首轮约+2.41%）；组合两次继续原生超时。该研究当前共96最终CEC、85原生优化覆盖，所有预算和来源核验通过，三个campaign及全局锁均空闲，未接入主版本。最终完整报告为 `results_v2/library_rules_20261021/signed_final_report.md`。


## 紧凑表示与互补引脚匹配审计（2026-10-22，研究版未合入）

证据目录 `results_v2/critical_compact_20261022/`。sqrt/voter/multiplier 的基线关键路径抽样通过 SAT 证明边界对应，再区分生成、保留和选中阶段。sqrt 55个抽样中14个尚无完整已证class边界；voter27和multiplier40均有完整对应。发现非一一引脚匹配缺口：4/5个物理引脚使用2/3个逻辑变量及其互补相位。

同二进制全EPFL三组60/60完成、60最终独立CEC、全部实际预算核验通过。profile0为旧控制；1仅抑制不可能获胜的同目标已就绪队列事件；3在1上增加精确互补引脚匹配。三组原生优化覆盖17/18/16，其余均为基础覆盖。所有baseline原算法保持，主源码未改。相关242测试通过/6跳过/16过滤；另有2个显式匹配测试及3个独立CEC。

profile3相对1首轮：sqrt D50815.83→50399.33、voter738.17→623.21、multiplier2726.51→2625.62，均有面积代价；sin1736.24→1707.86且面积下降。Bar同D面积+2.49；Div、Mem_ctrl新增匹配组原生超时，log2/hyp三组仍基础覆盖。不能按design混选，不能据正例合入。重复计划尚未执行：完整配置已出现合入否决项，且用户随后要求重新评估方法探索阶段的硬限制；不把首轮改善说成重复稳定收益。

新匹配不增加enodes/cut槽，但sqrt生成暂存新增约110万实例，100000 retained cap不代表生成内存上限。Log2旧队列在5秒frontier子限额失败，新队列已越过该步骤但仍触及60秒native限时；Hyp停在原有生成阶段，新增匹配尚未执行。后续宽松诊断与此冻结同预算证据分开，先区分资源截断与结构/匹配缺失，不能把人为窗口当成理论限制。

全套数据和原生/基础覆盖：`full_validation/`；baseline全范围与失败：`baseline_scope/`，各gate仍false；实际来源和说明：`source_freeze.json`、`configuration_erratum.json`、`README.md`。原版本BASELINES原字节已存`BASELINES.before.md`并记录旧manifest哈希。文稿与主版本不据这一轮未重复且有回退的配置更新为最终方案。

## 仅600秒资源策略接入验证

证据 `results_v2/time_only_600_20261022/validation.json`。245相关测试通过、6跳过、16过滤；三个真实进程测试覆盖总时钟传播、子进程组清理、超时源覆盖及日志保护。最终EPFL dec验证原生优化/独立named-port CEC成功，实际保留446760候选、cuts/class21，超过旧100000/16上限，证明非仅文档改名。600秒包括源覆盖构造、导入/饱和/匹配/恢复/导出，各阶段不能重新领取600秒；不再设置16GiB地址空间限制。数量与实际内存仍记录。

当前为新资源执行入口，不冒充主算法PPA合入。小cuts K6、库驱动7–9输入定向匹配、原有规则/匹配族和正确性检查仍是算法选择，不代表穷尽搜索。新600秒全EPFL与baseline对照尚未完成；旧结果完整保留。文稿不能用此单设计策略验证替换主表。

600秒正式新矩阵 `results_v2/time_only_600_20261022/epfl_checkpoint/campaign.json` 已完成首个有限批次1/20（adder原生优化及独立CEC成功），源/库/输入/计时harness均冻结。实际候选108208、cuts/class297、恢复约3.36秒，均已超过旧限制；A87.06/D2569.15、映射约9.20秒。仍未完成全20或重复验证，无后台活跃进程。进度证据 `epfl_progress.json`，不得将单例改善称整体优势。

## 增量 cuts／匹配与早期合法覆盖（2026-10-22，隔离原型）

证据目录 `results_v2/online_mapping_20261022/`，主算法与 baseline 原算法均未改。V1/V2 源码、二进制、harness 和结果分别冻结，不能按设计混选。一个600秒总映射搜索时钟，包括源覆盖构造、候选生成、匹配、恢复和导出；CEC、一次resyn2、纯转换及公共指标排除，无额外数量/阶段上限。

V2 仅组合新增子 cuts，按 delta 匹配并合并同一物理池；mandatory cuts 后即尝试完整延迟覆盖。覆盖经 binding/care/phase/pin、driver、cycle、全图物理指标验证，原子提交后由父进程在原期限内验收 receipt。worker 超时只能保留这种已验收完整覆盖，不能采用任意 partial 文件，仍需独立 named-port CEC。搜索未结束与优化覆盖已可用分别记录，不将旧失败重命名为成功。

252项相关 Rust 测试通过、6忽略、16过滤；另4项 receipt/真实子进程测试通过。64个含共享/反相/紧凑真值表节点的图，增量 cuts 与完整遍历的最终集合相同；固定延迟下实际共享面积恢复的功能和完整覆盖检查通过。这些是有限正确性检查，不证明候选穷尽或最优。

同二进制首两例开关对照及一次额外 candidate 重复均已完成：adder/bar 共6个V2输出全部CEC，A/D逐值复现；另V1两输出CEC。面积恢复关闭/开启分别为 adder160.30/96.03、bar290.36/271.09，D保持2569.15/168.29；映射T首轮约9.43/21.55秒、8.08/15.01秒。旧完整恢复面积仍更低（87.06、262.77），这些回退保留，**不满足合入条件**。详见 `v2_first2_validation/report.md` 和完整20设计missing记录。

当前实现只在 cuts 闭包后进行“输入锥已选中”的实际共享面积 sweep；未选输入锥的联合恢复、延迟物化、全20对照及完整重复尚未完成。V2仍为每批 delta 全量构造物理匹配实例，不能把只导出 live 实例称为延迟物化。下一隔离版本优先复用现有完整覆盖恢复器，以早期完整 incumbent 热启动，并固定原延迟/面积界限；验证前不更新主版本及论文最终主表。baseline36/120检查点哈希及中断目录已审计一致，原版实验继续顺序推进，所有失败保留。

后续V3已在独立冻结源码 `online_mapping_20261022/source_v3/` 复用原 `slack_area_pass`，以完整早期覆盖热启动，使用全局 witness 激活未选输入锥、required-time和实际共享面积检查；固定初始D/A界限，不重新领取时间，保留旧完整覆盖。253相关Rust测试通过、6忽略；新热启动恢复的独立功能/完整性测试通过。

V3同二进制首轮加两次额外重复、配置顺序交替，共6合法pairs/12独立CEC。每次 adder A87.10/D2569.15，bar A248.36/D168.29；相对本版关闭面积恢复的160.30/290.36均减少面积且D不变。相对历史旧完整流程，adder仍约+0.04面积，bar减少约14.41。详见 `online_mapping_20261022/v3_paired_repeats_validation/`。18设计在每个full20矩阵仍missing，不能当完整EPFL收益或全面baseline优势；延迟物化仍未实现，不合入主版本。

原版baseline600最新37/120组、102成功method observations、3失败、155pending；新E-Syn DA adder真实600搜索失败完整保留。`online_mapping_20261022/baseline37_scope/` 的DA/AD/Pareto三gate仍false；Hyp emap-delay旧named-port证明仅按原精确哈希证明复用view保留，非新CEC。baseline原算法/模型保持。

定向Log2 V3已结束并通过独立named-port CEC：600秒搜索确实超时（原exit-9/timeout保留），但约24.85秒父进程已验收完整生成覆盖receipt；最终A4793.55/D4059.30，区别于旧A2919.34/D9710.57的源结构保障覆盖。公共T约600.10秒包含进程终止收尾，采用的覆盖在原600截止前验收；CEC不计时。不能将其写成“搜索完成”。D仍落后同库ABC map3984.94、emap3864.80，面积更大；本轮不是baseline胜出。

最后日志停在 `online_wide_start`：原宽匹配仍整批递归收集结果、缺少增量提交，面积恢复尚未执行。这与旧版Log2 cuts闭包超时是不同执行路径，不能套用旧stage分类器。下一阶段需可中断/分批宽匹配及延迟物化，保持总时间不变。原始报告不修改，额外搜索状态view检查前端/原生exit和deadline各自状态，4项回归通过。证据 `online_mapping_20261022/log2_final_validation.json`。

Log2定向cohort与V3 full20的实际依赖hash全部为包含关系、配置仅设计列表不同，复用资格核验在 `log2_full_scope_dependency_proof.json`；canonical checkpoint仍2/20，未静默修改。联合只读全20范围view现3已测/17missing（`v3_full20_scope.json`），完整对照和所有设计重复未齐，不更新主算法或论文最终主表。

本轮有限批次最终：原版baseline38/120组、104成功observations/4失败/152pending。新EqMap AD bar总600超时，两项原有staging controls成功CEC；失败原样保留，三gate仍false（`online_mapping_20261022/baseline38_scope/`）。Log2已经在全局锁空闲后按 `log2_adoption/proof.json` 显式接入V3 canonical检查点，保留接入前CP快照和原targeted CP；canonical3/20，17missing，不重复计数、不重跑原timeout。最新范围 `v3_full20_scope_after_adoption.json`。所有有限批次结束，当前无后台活跃实验；后续入口与冻结/新版本规则见 `online_mapping_20261022/continue_checkpoint.json`。主算法及论文未因这些部分结果变更。

## 可暂停宽匹配与恢复调度（2026-10-22，隔离版未合入）

新证据 `results_v2/streaming_mapping_20261022/`。显式DFS栈保留宽匹配未访问分支，每批返回；实际面积恢复使用持久游标并在宽匹配未完成时运行。一个600秒总映射时钟不变，CEC排除，baseline原算法不改。4096匹配操作/16恢复候选只是可续调度粒度，不丢弃后续任务、不增加阶段限时。

旧Log2回放为red：wide开始后不再运行恢复/cuts；新回放为green，出现wide yield、恢复和下一轮cuts，已到第4轮。普通匹配/合并批次仍有约120秒、130秒耗时，全池延迟种子/验证也仍耗时，不能称600搜索超时已解决。新Log2原始搜索exit-9/timeout保留，已提交优化输出独立CEC通过：A6100.60/D3855.47，较前版D4059.30降低，较同库emap D3864.80约低0.24%，但面积明显更大。所选网表无>6输入单元，不能把收益归为宽门被选中。

首个冻结source有258相关回归/6忽略/16过滤及4receipt进程测试。全部7个当前宽单元的小例，暂停续跑与旧递归器的物理key及所有target集合一致。首两设计同二进制2x2共8CEC，只有stream+scheduled组合有小面积回退：adder87.10->87.14、bar248.36->248.75，同D；额外两次组合重复再4CEC，逐值复现，全部保留。

后续独立source_protected保存纯delay完整种子，即使它被局部面积incumbent压过仍保留；同一全局截止时间下分别从旧种子/当前覆盖做固定时序界限恢复，不借用历史网表。259相关测试通过；首两设计保护开关及三轮保护重复共8CEC，恢复87.10/248.36，同D不变。这里只消除本轮调度交互引入的回退；更早adder87.06仍更低，不声称全EPFL无回退。

一般物理候选延迟物化、完整增量delay-flow、完整area-flow可暂停化及全EPFL消融/大图重复尚未完成。主版本与论文最终表未更新。最新原版baseline仍38/120；所有失败/原始命令/hash保留。protected版本的Log2正在独立同预算验证，进度见新目录continue_checkpoint；不得把不同源码按设计择优混选。

本轮最终有限批次已结束，当前无后台活跃实验。带保护版本Log2也独立CEC通过，A6100.60/D3855.47，与前一流式版本相同，原600搜索timeout保留；不是已完成搜索。两版各自按完整依赖证明显式接入其canonical CP，均3/20，不能相加成6设计；保护版仍17missing。小设计重复已齐，本版大图重复/全EPFL消融仍缺。

原版baseline最新39/120组、104成功observations/5失败/151pending；新E-Syn ADbar总600失败保留（不是套用旧参数下exit-6）。对照结果及全部失败见 `streaming_mapping_20261022/baseline39_scope/`；当前新方案范围见 `final_stage1/`，DA/AD/Pareto gate均false。Hyp旧named-port证明仅在相同哈希的显式reuse view使用，未声称新CEC。

最终审计 `streaming_mapping_20261022/final_audit.json`：新实验跨各版本/消融/重复共22个CEC网表，但仅3个独立设计；259相关回归通过，6忽略，16过滤，另4receipt进程测试。主native SHA未变，主版本和论文不据不完整数据更新。延迟物化、普通匹配可暂停化、增量时序以及完整area-flow调度尚未实现，不能把小批物化误称为一般延迟物化。后续严格从 `streaming_mapping_20261022/continue_checkpoint.json` 继续；冻结source/source_protected不改，不按设计混选版本。

## 普通匹配游标与增量delay-flow（2026-10-22，隔离v1）

新目录 `results_v2/cooperative_mapping_20261022/`。单一600秒总映射搜索时钟/CEC排除，无新增节点、cuts、pool、iteration或阶段硬上限；baseline原算法不改。普通匹配显式保存boundary/target/variant、SDC subset/permutation/phase位置，2048操作后可续；dirty delay队列4096访问后可续，处理新增实例和旧实例新增target，非依赖引脚采用旧frontier保守readiness。零延迟潜在环做显式已选图检查。每个delta仍保留一次batch frontier路径，不称全流程已无全池扫描。所有局部标签必须经过完整绑定/无环/care/phase/pin验证才能提交。一般物理候选延迟物化、完整area-flow可暂停化未实现。

264项相关回归通过、6忽略/16过滤；早期测试缺失外部library fixture的3个失败完整保留，提供独立fixture目录后全部通过。小例普通匹配quantum1/7/256/4096与旧批处理物理key、全部target和sources相同，SDC条件绑定差分通过；64个DAG及零延迟环/迟到候选通过。首两设计同二进制2x2加组合两次额外重复，12独立CEC，A/D逐次复现。

|配置|adder A/D|bar A/D|
|---|---:|---:|
|control|87.10/2569.15|248.36/168.29|
|仅普通匹配cursor|87.10/2569.15|248.21/168.29|
|仅dirty delay|87.10/2569.15|247.27/168.29|
|组合|87.10/2569.15|248.73/168.29|

组合的bar同D面积回退在3次运行全部复现，拒绝合入。cursor-only也有T代价（adder约14.56→30.42秒、bar29.14→73.89秒）；不能把可暂停性当性能改善。组合首次T约26.35/23.41秒，全部真实T/重复保留在 `first2_v1/all_scope.csv`，不是统计runtime优势。旧log2 trace的普通匹配124/142秒无恢复回放为red，新真实小例出现ordinary yield及恢复回调。v1全EPFL仅2/20，18missing，各lex/Pareto gate false；没有新log2或大图实测，不能据v1宣称log2已解决。

v2正在独立源码中探索batch frontier predecessor与incremental pure-delay predecessor分开保护，并记录各自恢复结果，仍非完整旧恢复轨迹保证；源在 `/tmp/emapper-cooperative-batch600-20261022`，尚未通过构建/真实验证。当前“路径被覆盖”是待检验假设，不当作已证因果。原版baseline从39/120检查点审计后续跑下一EqMap DA bar，不能在批次结束前更新完成计数。main native与论文保持不变。


## 普通匹配/增量delay后续审计（2026-10-23，隔离V2–V6）

以下覆盖上一段“V2未验证”的旧状态。所有源码分别冻结在 `results_v2/cooperative_mapping_20261022/source_v2` 至 `source_v6`，不按设计拼接不同版本。V2将batch frontier与incremental delay完整种子分开，消除了首两设计组合引入的bar面积回退；但log2同预算D3852.82（旧在线control）→4059.30（组合），真实负结果保留。V3独立调度cadence、V4根锥证书缓存、V5局部绑定证明memo依次测试；V5同一配置12/20公共CEC成功，但log2仍为A6386.77/D4010.95，600秒搜索超时；不是搜索完成，仍差于旧在线D3852.82及同库emap3864.80。大规模缺失/回退不能被小例消融掩盖。

V6修正在线入口遗漏MetricEpochGuard的接线，实际调用点red/green回归通过；另以独立开关接入完整旧共享提取器的固定D/A界限尾部提案。275相关测试通过，6ignored/16filtered。尾部仍非可暂停恢复，仅在匹配闭包后、同一全局截止时间内运行；候选物理Instance仍提前构造，不能称一般延迟物化已完成。

V6同binary首两设计2×2控制和两次额外candidate重复均CEC通过，3次candidate A/D完全相同。adder87.10/2569.15不变；bar248.36→248.25，D168.29不变，但映射T约22→213–218秒，不是runtime改善。尾部主要四轮局部搜索，各约47秒。当前canonical `v6_epfl/campaign.json` 已依赖证明采用log2及固定9设计全部结果，12/20独立设计CEC成功（跨首两消融/重复共22个CEC输出，不是22设计）；log2与arbiter保留600搜索timeout及截止前合法优化覆盖receipt。max仍A224.75/D2074.58，旧完整流程同D面积209.68，不能称共享面积回退已消除。cavlc/ctrl/i2c/int2float/priority的尾部恢复降低面积；完整旧共享覆盖轨迹仍未受保证。

完整范围与失败/缺失见 `v6_progress/`，DA/AD/Pareto三gate均false；完整EPFL、全20单因素消融与大设计重复尚未完成。后续剩余设计从同一冻结V6检查点顺序补测，已完成job核SHA跳过；不得把局部失败缓存假设当已证性能收益。log2的约8591次flow证书尝试仅21次更好覆盖提交，失败原因和重复结构需独立诊断；失败证书缓存若实施，必须在相关绑定、source evidence或cut epoch变化后失效，不提升未经证明的覆盖。

原版baseline正式600策略最新40/120组，106合格observations/6失败/148pending；新EqMap DA bar600失败及两个staging control成功均保留，原算法/models/搜索参数不变。ABC map两目标和emap两目标已各20设计；EqMap/E-Syn/E-morphic整体仍未齐，MtM未接。`baseline40_scope/`候选半边是更早完整time600版本，不能当作V6数据。主源码SHA仍为f2bb5f70041163e8dcfb26f89b070ebdd91a05552d535a7d32bcb8987e105bda，主版本和论文最终表均未据部分结果改写。


2026-10-23恢复后的有限批次：V6 canonical14/20，新增div/hyp两例均保留600秒搜索超时、截止前优化覆盖receipt并通过独立CEC。div A9598.44/D43058.72，较同库ABCmap D43089.85和emap D43095.86略低，按DA单例胜出但面积约为emap3.16倍；仅一次运行，非Pareto或重复稳定性结论。hyp A24307.30/D357103.09，明显落后emap D195229.48。缺multiplier/sin/sqrt/square/mem_ctrl/voter六设计；证据 `div_hyp_v6_validation.json`。div开头与只读report有约数秒重叠，保留原run并明确不作隔离runtime改善宣称，后续需隔离重复。

V7负证书缓存是独立假设实验，不是已证log2修复。构造的未证明根锥回归真实复现旧重复检查；新增证据/已选非关键依赖/cut-bank增长必须唤醒重验。277相关测试通过，6ignored/16filtered。最初共享target/debug目录冲突使测试未启动，该setup失败原日志保留；换独立/tmp构建后取得有效red/green。adder同binary开关CEC均通过，A87.10/D2569.15相同；未触发负缓存。

log2同binary负缓存off/on两例CEC均通过，A6386.77/D4010.95完全相同，均600搜索timeout。off约7738次flow证书尝试、on约7625次，均无rejected_*或negative-cache-hit事件，不能把“没有变好”当作“证明失败”。负缓存性能假设在此不受支持，不合入；`v7-log2-final-diagnosis.json`保留真实负结果。观察更支持“合法但全局不改善的覆盖重复整图评分/验证”这一瓶颈，仍需独立定量验证。

后续V8仅是待构建的证书调度草案：首个合法覆盖不延迟，后续可延后dirty根锥验证，在delta结束及最终面积恢复前、原deadline内强制flush。它保留物理绑定和已验证incumbent，但会合并未验证的中间Flow赋值，不能保证另一次历史运行的质量或不遗漏中间更好状态；`v8-proposal-semantics.json`明确此限制。一般延迟物化及完整可暂停共享面积恢复仍未实现。baseline实际冻结队列当前下一项为原E-Syn DA bar（非从中间历史清单推断的EqMap max），仅原算法/600策略，结果完成前不增加计数。主版本/论文保持未合入状态。


## 上游源码管理（2026-10-23，用户最新要求）

禁止优化或修改有源码的baseline。新增tools/abc、tools/eqmap、tools/E-Syn、tools/E-morphic固定提交子模块；E-morphic上游仍为Gy-Hu/E-Syn2。直接复用原tools/mockturtle，撤销重复mockturtle-baseline；当前提交434c1ac及已有脏修改均保留，不称干净上游。来源与最终范围见tools/BASELINES.md、tools/baselines.lock.json及baseline_submodules_20261023/final_scope.json。未构建或切换既有实验二进制，未提交/推送。

历史/tmp E-Syn原生源码已有环境预算入口和模型路径补丁，不能将旧结果改称干净上游子模块的结果。新子模块未应用这些补丁。输入适配器1MB表达式guard亦已记录为协议缺口；新增div E-Syn AD失败发生在作者搜索前，不是600秒搜索超时。移除guard仅有审计/计划，用户新要求后未实施任何baseline适配器或源码修改。旧完整数据及坏例原样保留。

V9新报告cooperative_mapping_20261022/v9_progress保留全部20设计缺失/失败范围，三个gate均false；原版baseline实际43/120组仍未齐。两目标/各配置不得按设计拼赢家。有限批次已结束，当前无后台实验。
