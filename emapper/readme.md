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
  -- TT miss 且开 EMAP_USE_EGG=1 时，走 egg 构造 CoverAig
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