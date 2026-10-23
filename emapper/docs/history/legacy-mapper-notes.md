任务目标
输入：AIGER 电路 + asap7.genlib
输出：assign-free 的门级 Verilog（标准单元实例）
优化目标：参考 emap 的三阶段思路
- delay flow
- area flow
- exact area flow
要求：
- 支持输入/输出反相（inverter 作为映射空间一部分）
- 支持 egg 作为 TT miss 的 fallback
- 用 ABC CEC 做功能等价验证（nec 字段）

当前流程（代码实现）
1.读入与预处理
- 读 AIGER（ASCII/Binary），清理图。
- 读 genlib 并构建 TechLib（含 CellVariant 索引）。

2.节点映射主循环
- 拓扑序遍历节点，做 cut 枚举（受 max_cuts/max_inputs 限制）。
-对每个 cut 调 score_cut：
  -- 先 TT direct match（best_for_tt）
  -- TT miss 且启用 egg（`EMAPPER_USE_EGG` 默认 1；未设置即开启，设为 0/false 关闭）时，走 egg 构造 CoverAig
- 绑定类型：
  -- Direct { cell }
  -- CoverAig { expr }

3.多阶段优化流程
- Stage1: Delay pass（DelayFirst）
- 可选 mixed（关键路径 DelayFirst，非关键路径 AreaFirst + required-time 过滤）
- Stage2: area-flow passes
- Stage3: exact-area passes（已接入递归 ref/deref + MFFC 范围替换框架）

4. 回溯收集最终绑定
从 PO 驱动反向收集必要绑定节点（最近已改为按“实际绑定依赖”回溯，而非 cut.inputs）。

5. 网表打印
- 输出模块端口按 AIGER 顺序
- 常量/反相/连接均用单元实例（无 assign 目标）
- Direct 与 CoverAig 两条实例化路径

6.验证与统计
- benchmark JSON 输出 area/delay/depth/runtime
- profile 输出 cut/egg/tt/tech_match 与命中计数
- 调用 ABC CEC（cec -n）给出 nec

当前状态结论
- 优化流程框架基本齐全（delay/area/exact + required-time + egg fallback + CEC接入）。
- 当前流程在基线测试上已可得到 nec=1（例如 c17 / adder 在 direct-only 配置）。

工程分层（命名已清理）
- 表示层（`emapper::aig`）
  - `aig_graph.rs`：图结构与 AIGER 读入
- 映射层（`emapper::mapping`）
  - `emap.rs`：主映射流程（`emap_with_params`）
  - `emap_cut.rs`：cut 与基础映射逻辑
  - `emap_egg.rs`：egg 相关逻辑
  - `techlib.rs`：工艺库与 CellVariant

直接在 e-graph 上映射的新入口（2026-10-07）

`egraph_map.rs::map` / `bench_rs_egraph_map` 是独立的研究原型；C++ 的
`emapper` 可执行文件仍走 `map_egg` 的局部 cut 路线。

新流程：AIG → 精确等价 saturation → e-class 多分解 cut → 物理实例候选
→ 实例/输出绑定联合 ILP → Verilog → Yosys 展开单元模型 → ABC CEC。
不会在匹配之前把 e-graph 提取为一个固定 AIG。

表示与优化：
- 信号标识为 `(dense e-class, phase)`，反相器也是可共享的独立实例。
- 一个实例可以提供任意数量的输出；实例选择变量 `x` 计一次面积，输出绑定变量
  `z` 决定实际使用哪些输出。未使用的输出 pin 留空；同一物理输出支持多个等价信号别名。
- 同名、不同输出 pin 的 genlib 记录按一个物理单元分组。所有输出必须声明相同面积。
  输入集合取各输出 support 的并集，输出使用各自的 pin-delay 向量。
- 输入 pin 的对称置换只有在全部输出函数和全部时序弧相同时才合并。
- 每个被绑定的输出都有无环约束；时序只计该输出实际依赖的输入。
- cut 池保留局部结构作为有限回退，并按固定排名更新其他候选，避免早期浅层 cut
  挤掉后来发现的共同 PI 边界。ILP 建模前删除所有候选都无法从 PO 访问的实例。
- CBC 超时、非整数结果或无效覆盖会回退到经检查的有限 DAG 覆盖；硬延迟约束
  无法满足时返回错误。`proven_optimal` 仅表示在已生成的有限候选集合中得到最优解。
- 有限覆盖先用树代价和 area-flow 生成起点，再按实际共享 DAG 的面积恢复。
  多输出候选既可联合替换，也可只绑定一个输出；每次接受前检查完整性、无环性、
  硬延迟和实际目标值。恢复结果也作为 CBC 的起点，不将启发式结果标为最优。
- 原有恢复之后增加一个逻辑锥排序阶段：按实际引用次数估计可回收的整段物理面积，
  其他消费者或其他输出仍在使用的单元不会被提前计入回收收益。随后可用实际覆盖
  的引用次数重估完整 cone flow。每步仍以真实 DAG 目标验收。

SDC：
- `EGM_DC_SUPPORT>0` 开启固定边界的 reachable-image 枚举。以有限等价代表求
  边界叶子的联合 PI support，在枚举预算内计算 care mask。
- 候选合法条件为 `((F xor G) & care) == 0`；允许匹配输入更少的单元或常量，
  因此可以真正消去只在 DC 点上影响函数的输入。
- SDC 只添加外部实例/输出候选，不修改 e-graph 的全局等价关系。所有选择均保持
  原始 PI 下的信号语义；结合有限 DAG 的归纳，多个 SDC 候选可以同时使用。
- 超过 PI-support/cone 预算时使用全 care，不把未枚举的点视为 DC。
- 当前未实现 ODC、外部环境假设或有状态电路。ODC 必须使用联合输出关系，并在
  固定上下文中验证整个替换，不能复用这里的逐输出 SDC 条件。

运行：
```bash
cargo build --release --bin bench_rs_egraph_map
EGM_DC_SUPPORT=10 EGM_TIME=30 target/release/bench_rs_egraph_map \
  tools/mockturtle/experiments/cell_libraries/asap7.genlib \
  tools/mockturtle/experiments/benchmarks/c17.aig /tmp/c17-mapped.v
```

主要参数：
- `EGM_K=6`：cut 输入上限，当前支持 2–6。
- `EGM_CUTS=32` / `EGM_CUT_ROUNDS=32`：每类 cut 预算及传播轮数。
- `EGM_ITER=16` / `EGM_NODES=20000`：saturation 预算。
- `EGM_INSTANCES=100000`：可选实例候选预算；为保证回退，精确候选允许超过该值。
  `instance_limit_hit` / `cut_limit_hit` 会报告截断。
- `EGM_MULTIOUTPUT=1`：多输出匹配开关。
- `EGM_MULTI_CUMULATIVE=1`：在有界 partner 窗口中累积边界，可生成三个及更多
  不相交输出 support 的共同边界；边界大小受实际多输出物理输入数与 K 限制。
  设为 0 可运行 pair-only 消融。该有界枚举仍不保证覆盖所有输出组合。
- `EGM_DC_SUPPORT=0` / `EGM_DC_CONE=1024`：默认关闭 SDC；PI-support 最大允许 20。
- `EGM_TIME=30`：CBC 时间预算；设为 0 只运行启发式有限覆盖。
- `EGM_REFINE_ROUNDS=4` / `EGM_REFINE_TRIALS=2000`：共享面积恢复轮数与每轮尝试上限。
- `EGM_REFINE_TIME=2`：恢复阶段的独立秒预算；设为 0 保留原始有限覆盖起点。
  此预算约束可选搜索，不包含必须完成的起点构造与最终验证。
  JSON 的 `initial_area`、`heuristic_area`、`heuristic_secs` 和
  `refinement_trials` / `refinement_accepted` 可用于检查恢复收益与开销。
- `EGM_CONE_RECOVERY=1` / `EGM_LIVE_REFS=1`：逻辑锥排序和实际引用次数重估开关。
  锥排序增加一轮最多 `EGM_REFINE_TRIALS` 次尝试，共用恢复阶段时间预算；引用重估
  最多运行 `EGM_REFINE_ROUNDS` 轮。接受计数包含引用重估接受的完整覆盖。
- `EGM_BOOLEAN=0`：0 为原规则，1 增加 OR 因式分解，2 另加 AND 结合律。
  结合律会扩大 e-graph 和候选池，在有限预算下可能降低映射质量，默认保持关闭。
- `EGM_FLOW_DELAY_TIES=0`：可选设为 1，在面积流成本相同时先比较到达时间。
  可能改变全图共享结构，部分电路面积回退，故默认保持关闭。实验驱动的方法名为
  `egraph_delay_ties`；可用 `--cuts` / `--refine-seconds` 显式记录更大搜索预算。
- `EGM_STRUCTURAL_REFS=0`：额外使用有限 witness 的物理扇出作为面积流种子，
  同一共享实例的输入只计一次；开启后的实现仍按完整面积/延迟与 DAG 检查接受。
  七例中仅 c17 面积改善但延迟变大，故默认关闭。
- `EGM_DELAY_WEIGHT=0`：目标 `area + weight * worst_output_delay`。
- `EGM_OBJECTIVE=weighted`：保留已有加权目标。设为 `area-delay` 即 (面积,延迟)
  字典序；设为 `delay-area` 即 (延迟,面积) 字典序。字典序模式忽略
  `EGM_DELAY_WEIGHT`，不是用极大/极小权重近似。CBC 分两阶段，在共享求解预算
  内先优化主指标，再限制最好已知主指标优化次指标；只有两阶段均证明最优且
  cover 验证通过，才标记字典序最优。CBC=0 时仍按该字典序接受完整启发式覆盖。
- `EGM_MAX_DELAY`：可选硬延迟上限，单位同 genlib 的 pin delay。
- `EGM_NOCEC=1`：显式跳过验证。默认验证需要 `yosys` 和 `abc`；可通过
  `EGM_YOSYS` / `BENCH_RS_EMAP_ABC` 指定可执行文件。

验证使用完整的多输出 Verilog 单元模型，生成公开的规范端口名并按名称做 CEC，
避免 Yosys 重排端口或省略私有符号导致错误的接口对应。未知验证结果返回错误。
网表原型允许端口连接使用 `assign`；面积是选中物理实例的总和，不是 ABC 为
输出连接额外插入 buffer 后的面积。延迟是无负载 pin-delay 模型，未纳入 slew/capacitance。

专项回归：
```bash
cargo test -p emapper egraph_map::tests -- --include-ignored
```
包括 full adder 单实例、三输出单元、不同 support、部分输出悬空、不同输出时序、
SDC 输入消去、SDC/多输出联合随机穷举、重复表达式导入、硬延迟/有限回退以及
真实 CEC 的正反例。CEC 测试默认 ignored，以上命令在工具可用时显式启用。
另包含完全不相交 support 的双 AND 单元回归：启发式和 CBC 都选中一个面积 1.5
的双输出实例，并穷举 16 种输入。边界配对同时考虑共享叶与有界邻近窗口；仍是
有限候选搜索，不保证发现任意远距离、多输出组合。

公共 evaluator 还支持独立 Verilog 模型（例如 EqMap 参数化 LUT）的严格 CEC：
```bash
target/release/bench_rs_netlist_eval --cec-models luts.v golden.aig mapped.v
```

`tests/fixtures/full_adder.aag` + `egraph_map.genlib` 应映射成一个 FA（面积 2，延迟 7）。
`tests/fixtures/sdc.aag` 在 `EGM_K=2` 时，关闭 SDC 的面积为 5.8；设置
`EGM_DC_SUPPORT=3` 后面积为 2.1。两者均应通过 CEC。这些是正确性/机制夹具，
不是对生产库或大型基准的 PPA 改善承诺。

共享面积恢复的可复现实验（同一二进制，关闭 CBC，强制每次 CEC）：
```bash
python3 scripts/experiments/bench_egraph_recovery.py \
  --output /tmp/egraph-recovery.json
```
固定 ASAP7、k=3、cut=8、SDC-support=4，对比恢复预算 0 与 2 秒。
这里的基线是本原型的原始有限覆盖，不是 ABC 或原 `emap` 的结果。
2026-10-07 实测：c17 面积 0.62 → 0.37，c432 25.07 → 12.84，
c499 50.83 → 23.95；六次运行均通过 CEC。完整 JSON 存在
`../results_v2/egraph_recovery_20261007.json`，其中包含对应延迟、恢复耗时和搜索计数。

联网收集的新基线、统一测量口径、复现命令与最新对比见 [BASELINES.md](BASELINES.md)。
