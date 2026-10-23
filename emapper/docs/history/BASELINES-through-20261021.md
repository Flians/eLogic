## 2026-10-21 搜索资源诊断续轮（独立版本，尚未合入）

证据在 `results_v2/search_resources_20261021/`。按（D,A）字典序评价；area增加不否定严格delay改善，Pareto检查另列。原版47单输出库baseline沿用已审计120job/260观测（189合格、71失败/未测），算法/model/原参数不改。`source_exact_work` 的不变排序键缓存最小RED：512实例重复6776次；GREEN229相关test，但div/mem_ctrl四原生对照仍全部native60超时，四条基础覆盖CEC不称优化成功，`cap-prefix-negative.json`保留；不推广此排序缓存。

`source_shared_metadata` 为独立新版本：真实库pin名称及f32 delay向量按完整值intern，写时复制保护已保存覆盖；delay bit级区分±0和非对称pin顺序，不混函数/相位。小支持集union先用栈空间判宽度，精确保留旧集合结果；没有删候选或增任何全局预算。实际RED两项、245回归通过/5ignored/3既有synth过滤，7个真实wide cell fixture CEC。首次新目录缺库/类型准备失败日志保留、不算算法RED。`metadata_epfl`为预先固定全20配置，现20/20已结束、17原生选中/3基础覆盖/20 CEC；完整runtime audit通过。对上一完整47库统一profile1首轮DA9好/11平/0差、D1好/19平/0差、A9好/11平/0差；不挑run。三gate仍false，复测尚未完成。以下为执行中前缀历史：`metadata_prefix_runtime_audit`已核前8job/5native选中：mem_ctrl首次正常返回并fresh-original-named-port CEC，A3660.72/D1059.69/T59.40，native池100000/cuts16/recovery1.8563；对同库emap delay2348.96/1077.61，DA严格优胜但area较大，不称Pareto支配；先前基础4290.74/2731.23只作fallback。div仍native60；log2已越过原allocation failure并完成生成、后timeout；hyp仍cuts60，均不称已解决。main/default/论文暂未切换，后续完整数据及重复再核。

细分cap trace显示div witness2.71s、frontier8.42s、optional fill4.70s，54.82s后尚有大量被剪候选销毁；mem_ctrl本版在54.15s进入恢复并成功。准备中的 `source_heap_witness` 独立替换等价current/next readiness树为heap、复用同一不可变池的旧/新保留排序，并给optional cuts统一20s窗口（原native60/全局300不延长；最多32cut rounds、16SAT、16cut/class、同G/B+100k/2s恢复不变）。窗口在class边界检查，允许部分同步round后停止并明确truncated，不称硬抢占或完整搜索；尚未编译验证/PPA，不当作改善。所有旧失败、源码/hash/命令保留，禁止各case挑不同版本/参数赢家。

## 2026-10-21 完整47单元库 baseline 全范围已尝试，生成索引共享进入隔离回归

`primary_baseline120_checkpoint.json` 已统一收齐20 stock-library9/Cut9 core组与100原版noncore组，全部原始产物逐SHA检查；12个早期noncore明确依赖采用，不重复计数。`full_library_baseline_A_D_T`保留260条方法观测：189条独立CEC合格、71条失败或未测得（其中12条来自4个无report的初始LUT90s退出，另1条hyp ABCdelay补验600s仍unproved）。EqMap的T标注enclosing LUT转换保守上界，其他按实际mapping/反馈active intervals；失败指标不填有效A/D/T，原native参数/模型/算法保持。完整scope两profile的DA/AD/Pareto六gate全部false；AD是后验检查，不能称area-first搜索或全面支配。

新的 `source_shared_generator_keys` 仅在隔离副本接入 `EGM_SHARE_GENERATOR_KEYS`（默认0），原版baseline已退出才编译。运行RED1项确实0共享键vs期望1，源码/日志/patch留档；GREEN最终241项通过/5 ignored/3既有synth过滤，覆盖实际Arc所有权、强制哈希碰撞及九输入高位真值不同、care/来源完整合并、无证据fallback与24图完整生成顺序/函数/pin/proofs oracle。索引复用现有不可变Assignment的完整InstanceKey，Hash/Eq仍比较原完整键，非摘要相等代替证明；不改变candidate/cut/pin筛选及预算。release与源码已冻结，`shared_key_epfl`完整40/40已完成并独立CEC40（32原生选中/8基础）；两组area-flow均固定0.65，仅改变索引键所有权。D20全平，A5好/12平/3差，T15短/5长；multiplier+0.15、max+0.06、arbiter+0.46保留。sin首pair54.67→43.11s，sqrt43.13→37.65s、multiplier34.04→29.45s，均同D；只是首pair，暂不报稳定总体加速。实际索引entry80→16bytes，div3532593、mem_ctrl3713922个键复用原证据，完整key相等性不变。div/mem_ctrl共享组已进cap_done/recover_start但60s外限前未正常返回，仍仅基础CEC；log2仍分配失败（新申请872415232 bytes），hyp仍cuts超时，不能称四大图已解决。`shared_key_full40_runtime_audit`核验ρ独立额外额度与G/no refund、100k/native+全局源池、cut16与宽隐含边界区别、16GiB/60s/300s/recovery2s及0.65真实trace；全部已记录边界通过。`shared_key_vs_baseline120_scope`六gate仍false。`shared_key_repeats2`已64/64结束、64原生选中及fresh CEC；连同首轮，16个设计三对运行48pair D全部持平，A14好/26平/8差，T45短/3长，原生可优化16设计的时间几何均值约减少11.6%（不是含四个失败设计的整体加速声明，n=3不是统计显著性保证）。本版累计104次独立CEC=96原生结果+8基础结果，仍只有20个不同benchmark，另外四个未解决基础覆盖不伪装重复成功。`shared_key_repeated_validation.json`逐run留全部差值及范围，首轮统一profile1不由最佳复测替代。所有有限batch已结束、锁idle、7套源码/bin冻结SHA一致，`final_idle_validation.json`记录；main/default与论文/PDF未切换。已备`paper_evidence_pending_adoption.json`事实段落，未插入主表或改标旧限制库。旧phase-time完整40及重点16复测保持独立版本，主mapper/default及发布PDF仍未更新。

## 2026-10-21 宽匹配后续：消除重复规范化，定位面积流时限断崖；全套首轮通过，尚未合入

证据仍在 `results_v2/single_output_wide_20261021/`，所有旧源码、原始失败与报告保留。新增隔离源码 `source_wide_reuse` 在全池实际正常形、别名幂等及流式唯一性成立时复用整个 small/wide 候选池，保留非流式/非规范的完整旧路径。恢复未过滤索引原有 ExactSizeIterator collect 路径，过滤路径仍只跳过单元名不可能相同的旧键。RED 捕获宽候选池第二次物理键重建；GREEN 与普通/宽候选顺序、函数、相位、引脚延迟和 binding proofs oracle 通过，236 tests/5 ignored/3既有synth过滤。`wide_reuse_epfl` 4/40 是明确前缀，非完整套件：sin 同binary旧pass native60失败，复用后53.74s返回D1736.24且CEC；只是恢复小单元旧延迟水平，不报新延迟SOTA或对删失失败计算普通加速比。sqrt仍有面积代价。

`source_wide_guard` 的完整小单元保留池 bootstrap/严格root-delay准入也已实际试验：238回归通过，`wide_guard_epfl` 六例均CEC；sin A611.65→607.02/D持平，sqrt A2122.77→2122.56仍未消除约100面积损失，multiplier A2287.87→2296.41且D2726.51→2728.48回退。该方案不采用，既不推广为默认，也不按design挑它的好值；完整保留候选池不等于免费获得旧流程完整2秒终态。

只读 `selected_area_audit.json` 与公共面积严格一致：sqrt最初42宽门合计面积6.72，而总面积回退103.70、实例增加649，主要来自小单元覆盖变化，不能归因于宽门自身代价。运行trace进一步定位固定时序界限area DAG的全有或全无行为：成功约0.110s，失败窗口仅约0.091–0.094s。`source_area_reserve` 新增默认不变的 `EGM_WIDE_AREA_FLOW_FRACTION`（0.30），独立比较0.30/0.65；只分配同一个2秒剩余search窗口，不加总时限、不重新执行旧终态，保留250ms返回预留和所有已访问完整incumbent。添加 `required_area_dag_outcome` 区分deadline、不可行producer及完整结束。236回归通过；原版baseline算法未改。

`wide_area_epfl` 已40/40（完整20设计×两配置，同binary/47库/统一ρ=1/2），`wide_area_full40_runtime_audit`核验40 CEC、32原生优化选中/8基础覆盖、累计不退款节点准入/源+搜索保留池/native100k/300s映射/native60s/recovery2s及实际phase比例。两profile公共D全部20持平，A为7好/11平/2差；sqrt2122.77→2020.51，multiplier2287.84→2168.99，均保留D50815.83/2726.51；sin、square各+0.04真实回退保留。sin是多1个INV，square为共享AND项未复用而用AO22替代AO21，见 `tiny_area_regression_audit.json`、`square_selected_binding_differences.json`。两个旧配置area DAG因topological flow deadline退出，新配置均完成。T为8更短/12更长，不宣称总体加速。`wide_area_repeats2`已完成四重点设计两次额外配对复测16/16，16原生选中且fresh CEC，交替次序：连同首轮，sqrt三次同D面积约减102.2，multiplier三次约减118.8；sin六次原生完成，但两次额外pair面积+0.06/−0.04，square+0.04/−82.35，波动不删，不挑最佳run。复测的required-area DAG仍有5次topological deadline；并非全20重复或消除全部回退，未集成主版本。div/hyp/log2/mem_ctrl两配置均基础覆盖，仍未解决原生搜索失败。

实际16限制适用于物化的普通cut bank；定向宽匹配不生成普通Cut条目，其隐含边界不能声称也≤16/class，而是共用全局物理候选保留上限、时间及单一1,600,000访问额度，不增加egraph节点。生成暂存行仍可能数百万，不把retained上限当RAM上限。`wide_area_metadata_correction.json`披露旧runner继承的profile_actual_values标签0/1应为0.30/0.65；原矩阵不修改，冻结wrapper及实际fixed_area_window trace确定真实设置；后续复测使用独立修正元数据runner。

完整库原版baseline先推进到 `original_baselines` 15/120，26成功summary/8失败，无无report退出；`baseline15_checkpoint.json`不可变。进一步审计发现原适配器不但库容量6，emap默认CutSize也6；不能用它代表原版宽门能力。保持旧数据为容量敏感/历史，另用本地上游commit `1c9c3af6a423f650f654ccec340dae9b93995844`干净header快照构建 `stock_emap9_cut9_reference`：适配器仅将库容量及emap结构CutSize均按当前库最大输入数设9，原版hybrid/3轮area-flow/2轮exact-area/16cut/原时限不改；未使用工作区已有header优化。前一个library9/CutSize6预检及漏normalization JSON的无效fixture评估均原样保存，不算算法失败或宽能力证明。正确预检七例全部独立named-port CEC，其中五例选中宽门，结构覆盖非穷尽。新的 `stock9_cut9_core_epfl` 已20/20尝试，原版ABC map/map-a与该宽容量emap两目标共76条CEC成功、4条hyp验证超时；hyp已映射但公共验证60s超时，不称mapping timeout。`stock9_hyp_verification`只复用原网表补验、未调用mapper：三个独立GIA CEC成功（emap两模式与ABCarea），ABCdelay在额外600s CEC guard内仍未证明；原四60s失败全部保留，不把未证明写成不等价。core现79合法/1验证未决。`original_noncore_epfl`只继续原版EqMap/E-Syn/E-morphic，明确依赖证明采用12旧noncore任务，排除所有旧K6 core，剩余有限88job顺序执行，未改算法/模型/参数。当前raw失败区分Amap深度断言、pre-mapping原生失败和CEC超时：div的Amap_ManCreateAnd nLevelMax<4094是网络深度限制，不能误写宽库不兼容。最终全库baseline仍未齐，不称SOTA。source+pool全局预算、baseline算法/模型和原native搜索时限保持；CEC不计300s映射。

`full_library_selected_path_summary.json` 只读重建sqrt/sin/bar六个已CEC覆盖，公共A/D严格一致；sqrt自身2267级、stock emap2049级，两者关键路径均无>6输入单元，AO31数量1306/1667。不能把延迟差距单归因宽门缺失，也不由选中路径推断候选不存在。`wide_matching_work_audit_v2.json`按实际attempt路径并SHA核验17份诊断：sin/sqrt/multiplier用满160万访问，sqrt AO333/OA333零候选而额度已耗尽；这不证明无匹配。先前frontend诊断路径查询0记录的无效probe保留并明确纠正，不能当覆盖证据。`candidate_source_only_failure_audit.json`区分div/hyp native60超时与log2/mem_ctrl 16GiB保护下内存分配失败；基础CEC不等于优化成功。`hyp_dirty_dependency_audit.json`确认当前已有dirty依赖过滤，后续应研究完整快照/功能索引重建和扫描的增量化，尚无计时因果证明或实现收益。

主Rust/default、主二进制与DAC PDF仍未更新；完整证据、失败/回退、命令和来源哈希留档后继续复测与全库baseline补齐。主文稿仍不得把旧限制库表改称全47库比较。

## 2026-10-21 完整单输出库：隔离实现与全 EPFL 首轮对照，尚未合入

新证据目录为 `results_v2/single_output_wide_20261021/`。主 mapper 源码/二进制及论文未切换；baseline 原算法、模型和参数未改。当前提供的 ASAP7 genlib 的47个单输出单元可由新入口加载（不声称涵盖整套foundry库）：<=6输入沿用普通 cut/索引，7个7–9输入单元走库表达式与实际支持集 AND/OR 分解的定向匹配，不枚举全部宽 cuts/NPN排列。XOR/TT3/TT4 已计数功能节点在匹配中按精确函数解释；中间匹配项不能作为未命名物理输入，不增加/union egraph节点。物理函数、依赖引脚、去重键、绑定证据及导出支持宽单元。此匹配有结构/工作额度限制，不能称穷尽覆盖。七个真实宽单元全部逐赋值和独立 named-port CEC 通过；功能节点反例 RED→GREEN、kernel回归217通过/5ignored。

`kernel_epfl_campaign/campaign.json` 首轮40/40完成（两profile、全20设计、同binary、同47单元库），最终40网表有CEC证明。统一ρ=1/2，累计 B+ceil(B/2) 不退款；native retained100000、source+pool全局上限、16cuts/class、SAT16轮、native60秒、全流程mapping300秒、recovery2秒和返回预留.25均保持。CEC/转换/指标不计mapping。关闭宽匹配16个设计原生优化成功，开启15个；9条基础覆盖不能算优化成功。两profile公共D为1改善/18持平/1回退；multiplier A2190.67/D2728.48→2168.81/2726.51（选中38个宽单元），仅首轮，未作稳定优势结论。sqrt D50815.83持平而A2018.76→2122.46；sin开启后native60秒失败，仅基础覆盖 D4772.19，关闭时优化成功 D1736.24。adder/bar/max/i2c/voter等面积回退全部保留。`kernel_full40_report/A_D_T.csv`、`validation.json`保留完整A/D/T、候选/选中宽单元数、缺失与基础覆盖标识；尚不满足合入或全面支配声明。

最初基于main的12/40原型在 `prototype12_checkpoint.json` 独立封存。其源码不实现前端请求的force-compact内核选项；`source_policy_audit.json`区分请求metadata与实际import记录。后续完整40组以已验证f865 K6内核为统一底座，不混版本，不用原型max单pair A改善替换正式结果。

完整47单元库的原版baseline新cohort为 `original_baselines/`，目前1/120 job（adder四个core网表均CEC）。ABC日志加载47单元；独立emap的原生 `tech_library<6>`明确忽略7个宽单元，未改编译参数。`baseline_library_coverage.json`披露实际可用范围，不能声称所有工具均有47单元候选能力。旧40单元库baseline120/120（190合法/70失败）仍完整保留，但不得改名为完整47单元库结果。新库baseline、重复验证和完整比较尚未完成。

后续精确索引修复在 `source_index_filter/` 独立进行：宽匹配只需索引相同宽单元名的旧候选，避免给数百万小单元重复计算物理键。RED1001键→目标1键，完整候选顺序/函数/相位/pin时序/target/binding证据与旧实现逐项等价；232回归通过/5ignored/3个未接入mapper的旧synth测试排除（首次未过滤运行因旧synth耗时中断，原日志保留）。未提高任何预算，先测sin/sqrt；不把该新版本结果混进已完成40组。下一重点是消除sin超时和sqrt共享面积回退，再做统一全套及重复验证，确认后才更新主版本/论文。

完整库原始表达式独立校验：`wide_raw_function_oracle/validation.json` 用原genlib文本直接生成Verilog，不调用mapper表达式parser；与导出模型进行九输入SAT对照，七个宽单元全部等价（覆盖512输入赋值），错误输出负对照被拒绝。`full_library_scope40_vs_core1` 已运行两profile的(D,A)/(A,D)/Pareto六个gate，全部false；(A,D)仅后验检查，未进行area-first搜索，新库baseline缺失仍显式保留。

索引过滤targeted首轮 `index_filter_probe` 已4/4并全部独立CEC：sin两模式仍native60失败，均只基础覆盖；sqrt两模式优化成功、D50815.83持平，A2122.77/2122.43。键重建实际由sin2662221/sqrt1507367降至0，但仍不能称解决sin或sqrt面积代价。sqrt同binary T56.44→49.16仅单pair；control重构引入filter迭代器，可能改变collect预留分配，与原V2的allocation路径须先重新对齐，不能据此声称对原V2的普遍7秒提升。所有target数据与初版40组分开；没有合入主mapper或修改论文。有限批次/测试已退出，`idle_validation.json`证明所有campaign锁idle、三套source冻结哈希及旧PDF哈希一致。下一轮优先保留control原collect路径、复用流式候选已规范化后的完整池，审计sqrt共享恢复/retention；随后统一全20及重复，并继续新库原版baseline。

## 2026-10-20 K6阶段结论：baseline全量尝试完成；新内核未确认最终延迟收益

`execution.json`、`idle_validation.json`记录所有有限batch已退出、锁idle，无后台实验活跃。K6原版baseline120/120组已尝试，补验overlay后260条方法观测190合法/70失败；`baseline_A_D_T/baseline_A_D_T.csv`给出横向A/D/T和来源，190合法条目T均可审计。EqMap是包含enclosing LUT转换的保守上界，非精确mapping时间；其他为实际mapping进程/反馈active interval wall，CEC/转换等不计mapping policy。旧v1仅provenance措辞不够精确，CSV数值相同，留档 `baseline_A_D_T_v1_metadata`，现provenance明确区分上界。全120组15082 raw文件已SHA归档，原算法/参数/model不变。

新默认关闭的隔离内核，均未合main：
- `wire_pass_research` ce6125...：流式生成已规范且alias幂等时，验证实际正常形后复用完整候选池；否则完整旧pass。24图完整候选/顺序/相位/pin成本/binding proofs oracle、224回归通过/5ignored/3既有synth过滤。sqrt同binary统一ρ0.5四格4freshCEC：helper0两模式A2018.65/2018.57、D50815.83不变，单pairT44.13→39.06s；helper1从原生成后HashMap重建OOM推进到native60 timeout，仅基础覆盖，不算优化成功。raw1,507,367/3,115,110候选是生成暂存，retained上限100000不是RAM上限。
- `retention_cache_research` faacee...：保留原排序/round-robin/keep集合，guard/final两次相同producer排序复用；实际RED2→1，64图双排序模式/多mandatory sets及32真实retention oracle、227回归通过。sqrt四格4freshCEC/2优化2基础：helper0 A2018.65/D50815.83均相同，T39.70→38.52s仅单pair；helper1两模式仍native60timeout。rank文件显示builds2→1、retained100000，但未到cap_done。未完成全80配置或重复，12全scope gate false；不挑profile赢家，不以T小样本声称普遍加速。

重要诊断：额外helper7075的raw全池root cover内部D46799.83（old capped pool62493.47），但没有在native60内完成retention、导出和独立CEC，因此绝不填成最终PPA/胜过emap数据。`retention_cache_research/result4.json`保留所有失败。下一步按 `next_experiment.json` 分解rank-cache写完至cap_done间的筛选/析构/诊断/缓存释放成本，再探索精确共享/线性compaction；不能提高预算、接受timeout partial或免费使用历史完整2s终态。hyp cuts和log2/mem_ctrl生成阶段并未解决，不无改动重跑。

论文已按真实证据更新，6正文+1参考；统一最终K6配置的全20/重复及同最终binary K3/K4敏感性仍未齐，主表仍为明确标识的旧限制库研究。main/default和baseline算法均不变，不称全面优势或完成。

## 2026-10-20 K6原版baseline完整120组尝试，四个hyp网表补验合法

canonical `single_output_k6_20261020/resumed_baselines_python_command/campaign.json` 已120/120，有限批次退出，无后台baseline活跃。不可变 `baseline120_checkpoint.json` 原报告186条CEC成功/62条报告失败/4个无report退出job（对应12条未测method观测）。四个缺python环境失败已排除重跑；原算法/model/feature/index/native参数、300秒mapping预算未改。原120组及采用的外部产物15082文件已逐成员SHA归档 `archives/k6-original-baselines-full120-20261020.verified.json`，原字节6.742GB，未删除raw。

hyp四种ABC/emap已有网表均通过独立原输入named-port补验；新 `hyp_gia_delay_verification` 仅emap delay原网表，CEC203.709s、A15021.05995/D195574.25571，未重跑mapping/改变原方法，旧60s CEC timeout仍保留。补验overlay后完整260条方法观测190合法/70失败（含12未测），不是260成功；失败列表 `baseline120_failure_audit.json` 保留62报告失败+4无report原argv/产物。ABC map两目标和独立emap两目标现在各20合法；EqMap/E-Syn/E-morphic失败不删。`scope_allocation_mixedrho40_baseline120` 六gate全false，AD仅posthoc，MtM未接入；该旧candidate仍为逐设计混合rho，不是统一最终配置。

新的统一ρ=1/2同f865 binary输入表示初试 `literal_source_research/prefix4_audit.json`：sqrt/bar×2共4原输入CEC，实际branch/argv/settings/G均核验。sqrt compact A2018.65/D50815.83，literal native在16GiB guard内分配339738640字节失败（非60s timeout），仅基础A1955.33/D126168.75；bar两模式相同A257.31/D168.29。该路线不采用，也不把基础CEC称优化成功；未将初试冒充完整20设计。

隔离 `root_frontier_research` 新默认0的ROOT_FRONTIER_STOP只在post-exchange delay flow中所有输出根确定后停止，不改候选、预算或baseline。实际RED129事件vs1，GREEN4项（128图根覆盖/bindings/metrics含cycle/alias/unused pin、不可达/过期限拒绝、真实dispatcher连通）通过，232回归/5ignored/3既有synth过滤；源/patch/bin/日志冻结。新binary70c545...、两模式明确统一rho1/2，正在有限sqrt/sin初试，尚无延迟改善结论，不合入main。

## 2026-10-20 容量配置审计更正：旧K6配对不是统一ρ=0.5

`single_output_k6_20261020/capacity_schedule_audit.json`逐项读取原argv：allocation40、TT3-support40及representation模板在设计间轮流使用ρ=0.5/1/2（sqrt为2、bar为1、hyp为0.5），每设计的两个profile相同。旧记录中“统一ρ=0.5”不正确；原始结果和配对预算没有更改，但不能把它们宣称为统一最终配置/同统一ρ的全EPFL证据，也不据此选赢家。原累计准入上限仍按各case实际G核验，不由本次发现推断超预算。此前摘要的rho字段不能覆盖实际argv。

尚未运行的显式literal新对照已在argv固定所有设计ρ=1/2；它是新的明确容量配置，不能声称与旧sqrt ρ=2试验相同预算。两profile仍同binary/同库/同G/同300s/60s/2s，其实际导入分支另核验。`literal_source_research/plan-v1-preflight.json`保留更正前计划，plan.json为更正后计划；尚未得到PPA结果，不合入main。

## 2026-10-20 baseline环境纠正：四个E-Syn失败不可算算法失败

恢复后原版Rust `Command::new("python")` 的隐式依赖未正确恢复：PATH中只有python3，没有python。已停止旧续批，保留77组检查点及中断的Ctrl SA（报告JSON不完整且无完整clock audit，不采用）。四个新E-Syn cavlc/ctrl×两目标的exit101来自该环境问题，全部单独保留并排除算法比较；其余73组输出/命令/来源SHA核验后显式采用。`resumed_baselines_verified`现在为封存历史，不再宣称该runner活跃。

新的canonical为 `results_v2/single_output_k6_20261020/resumed_baselines_python_command/campaign.json`，启动器 `run_k6_original_python_environment.py` 只提供原Miniconda python命令；python3/abc/yosys/git/dot解析不变。源码、模型、原feature/index逻辑、64进程、所有native参数/300秒mapping时限均不改。`python_command_policy.json`、`python_environment_adoption.json`记录环境证明与四条排除记录；old坏环境产物不删除。修复后cavlc两目标已CEC通过，ctrl补测和后续原版配置正在有限批次中顺序执行，状态以execution和canonical为准。

## 2026-10-20 K6 exact-support 全20对照：未带来延迟收益，不合入main

持久证据根 `results_v2/single_output_k6_20261020`。`tt3_support_research/full_summary.json` 是同新binary 49753617...、相同300s/native60/recovery2/G/cut16/pool约束的20设计×2模式：40/40 fresh original named-port CEC，32 native优化、8 source基础覆盖。快路径相对reference：D=0好/20平/0差，A=5好/13平/2差；adder +2.51、arbiter +0.53同D回退。mapping T=10更短/0平/10更长，几何比1.00354，不能宣称总体runtime改善或延迟优势，也不按设计择优。完整scope六gate仍false；未做同最终binary K3/K4全套敏感性或本轮重复验证。原始40尝试已逐成员SHA归档 `archives/k6-tt3-support-full40-20261020.verified.json`，不删除失败/回退。

新核仅把TT3 replacement的支持集树并集换成等价有序并集，保留所有原支持≤K的候选、次序、相位和绑定证明；默认0、main未改。RED实际1次reference tree、GREEN0；全支持集/无序fallback/24张含TT3图的完整cuts与binding/proof oracle通过；完整228 tests通过/5ignored/3旧synth过滤，freeze/source/patch/bin/log均存。hyp有145938 classes，其中17711有该替换site；单次进入cut round4约46.65→44.32秒，但仍60秒timeout且仅基础cover。div仍60秒timeout，末stage为cone_prune_done；log2/mem_ctrl仍候选生成分配679477264字节失败，非输入节点拒绝。retained cap不是生成暂存RAM上限。

`adder_incumbent_audit.json` SHA核验：两模式保护的reference均A116.51/D2569.15，scope为本轮pre-exchange完整cover；最终91.14/93.65均未劣于它。没有已见更好incumbent被丢的证据，但这不保证历史完整2秒终态。相同exchange_trials21455、phase_trials2000/accepted77下online area updates169/138不同，不能简单将回退归因于cuts删减或一个bug。优先审计共享面积恢复的调度与实际工作量，不能免费重放历史完整预算cover。

较早word-quotient单因素仍只6例：225回归通过，6CEC/4优化2基础、无D收益，sqrt面积差0.02不作稳定结论；不扩展为最终配置。强制紧凑/自动表示初试sqrt两例均CEC，自动模式native timeout后基础D126168.75，对照D50815.83；不采用。启动前执行位缺失的两个失败另存 `representation_study`，不混入有效表示对照 `representation_study_exec`。

原版baseline持久canonical `resumed_baselines_verified/campaign.json` 已核验54/120组（61原报告CEC、44报告fail、4无report退出job），快照 `baseline54_checkpoint.json`；hyp CEC补验仍独立覆盖，不把补验叫重跑mapping。54组原始/外部adopted产物3710成员全部SHA归档。当前剩余66组有限批次正在顺序运行，状态见execution；算法/model/native参数未改变，不能把所有失败叫mapping timeout。历史107计数的71组原件缺失已逐项列在 `resume_recovery/lost_artifacts_inventory.json`，未静默采用。Git元数据也已持久归档，main .git未改。

DAC已更新K6时间分配负结果/失败范围，`paper_resume/qa.json` 为6正文+1参考，page6视觉检查、1--5/7像素不变，主表旧数据不变；完整K6 baseline/重复未齐，不冒充最终主表。所有lex/AD-posthoc/Pareto gate为false，MtM未接入。

## 2026-10-20 恢复审计：暂停计数107与可核验归档前缀36分开

本次恢复时 `/tmp` 临时实验和原版baseline Git元数据均已不存在。`user_pause.json` 保留107/120已完成的历史观察，但后71组原始产物未归档，**不能据计数复用为当前有效数据**。已逐成员SHA恢复17,344个文件；36组baseline、40组allocation及hyp补验原件可核验。全部原版tracked源码/二进制指纹与冻结状态一致；从真实上游补回原提交Git元数据，没有改main `.git`、baseline算法或参数。E-Syn恢复原已核验BFD二进制，不采用优化probe。

正式新续跑输出直接持久保存于 `results_v2/single_output_k6_20261020/resumed_baselines_verified`；采用原36组，其余为新尝试，不能混称重现丢失的非确定性SA赢家。最初恢复尝试因缺Git元数据失败，保留在 `resumed_baselines_persistent*`，只作为环境失败诊断，不进入方法优劣比较。完整记录见 `resume_recovery/formal_resumption.json`、`git_metadata_verification.json` 与 `restore.json`。300秒映射、60/90秒进程、2秒恢复及累计/候选预算不变。输入表示单因素的持久前端已重新准备，验证只增加元数据记录，native f865...不变；尚待映射和CEC，不能宣称改善。main/default未变。

## 2026-10-20 最新：K6全20时间分配对照已完成，比例不采用；原版baseline继续

以 `results_v2/single_output_k6_20261020/execution.json` 和不可变检查点快照为准。新主库方向为K6的40/47单输出门；7个7--9输入门未支持。K3/K4尚未做同最终binary敏感性，不把旧表改标。main native及默认设置未更改。

- `allocation_full40.json`：同binary f865ae56...、全部20设计、exchange .70/.40两设置40/40 **fresh original named-port CEC**，32次native优化覆盖、8次source基础覆盖。完整20pairs .40相对.70：D=0好/20平/0差，A=3好/12平/5差；未选择.40，不能挑每设计赢家。sqrt仅A2018.91→2018.89、D50815.83不变；multiplier A2296.59→2190.18同D2728.48，但adder90.32→92.76、sin607.02→607.48、voter1092.84→1093.70回退全保留。尚未完成这一研究的重复验证。
- 两设置的hyp/div均native60s timeout；log2/mem_ctrl均16GiB地址空间guard内分配失败，8例仅基础覆盖，不称优化成功/节点导入失败。生成暂存不受retained100000直接约束，未提高任何预算。完整scope六gate全部false，baseline仍未齐，MtM未接入。
- 只读 `sqrt_k6_selected_paths.json` 重现公共A/D：emap D46826.0614/A1540.9900；eGraphMap D50815.8309/A2018.9100。路径2049/2267门，AO31为1742/1306。这仅是已选路径描述，不证明缺失在哪一阶段。`sqrt_pin_symmetry_relaxation.json` 的ABC、emap、两eGraphMap共4网表在固定拓扑、同门同函数的全部pin对称置换后，block D收益均0；不能据此排除跨结构/跨相位组合。
- `selected_physical_duplicates.json`：sqrt/bar/adder两配置及emap共9网表，完全相同cell/参数/命名引脚物理输入的可去重门均0；4项传递重复、参数、非对称pin与cycle单测通过。不能靠简单导出去重消除这些面积差距，必须研究不同共享cover/输入锥。两项只读分析均不导出新网表、不修改baseline、不计作PPA改善。
- 修正阶段解读：protected路径明确设置 `area.terminal_area=false`，但仍执行required-time flow及 `slack_area_pass` 实际唯一实例面积恢复；terminal字段false不表示没有面积恢复。`joint_attempts`也只统计原witness填充失败后转alternate的次数，不是所有联合输入锥检查次数；不能用0推断功能未执行。
- 隔离不可变physical-key共享 `/tmp/emapper-single-output-k6-shared-key-20261020` 默认0、main未合入。RED实际1项失败，GREEN完整224通过/5ignored/3旧synth过滤；24图完整生成顺序/相位/SDC/绑定证据逐项相同，非对称pin不丢。`shared_key_research/log2_pair.json` 同binary两实际案例仍分配失败；679477264字节索引扩容失败变成872415232字节实例数组扩容失败，均只有基础覆盖。单次RSS约7.87/7.78GB不是稳定内存或延迟收益，不盲目扩展此未奏效原型、不合入main。
- 原版baseline canonical仍 `/tmp/emapper-single-output-k6-baselines-bfd-20261020/campaign.json`，已有不可变 `baseline36_checkpoint.json`：36/120组、44原报告CEC、25报告失败、4无report退出失败job。当前从36起的有限84组续批在执行，实际即时计数见execution，不依赖跨turn后台存活。算法/models/native参数未优化。hyp E-morphic本次pre-mapping recursion失败；multiplier部分EqMap输出已产生，但LUT CEC90s timeout，不能误称native搜索/300s映射超时，更不能把verification失败当算法劣势。
- hyp core映射完成而旧checker60s失败；原ABC两例补验通过，emap area旧checker600s timeout仍保留，同一物理网表改独立GIA检查179.325s通过（A13100.01984/D245212.49653/103830门）。不重跑或更改mapper，emap delay仍待补验；proof包含严格canonical PI/PO顺序与错误PO/交换PI负控制。`scope_frozen_allocation40_baseline36` 用不可变检查点及verification history，gate全false。
- 持久归档已包括full40、baseline prefix36及其外部adopt原始文件、key prefix2、旧hyp CEC prefix3和GIA prefix1；逐成员SHA核验、原/tmp与所有坏例不删除。prefix不冒称完整，见 `results_v2/archives/k6-*.verified.json`。
- 下一表示单因素已准备 `source_representation_research/preparation.json`、`run_k6_representation_campaign.py`：同f865 native，保持G不变，仅改变强制紧凑开关；原生import-plan披露实际分支。staged frontend仅报告元数据变更，构造/比较/时钟AST逐项相同，300秒变更的负控制可检出。尚未运行，不宣称原始AIG边界恢复或sqrt改善；待原版baseline批次结束后再测，不并发。
- DAC仅更新已核验的K6初步诊断与未完成限制，旧K3/K4主表逐字保留，不填K6未测数。`paper_progress/qa.json`：6正文+1参考，标准字号/边距；page6重审，1--5/7像素与前QA一致；首8页版本保留，发布PDF已更新。

mapping-only300、native60/EqMap90、inclusive recovery2/return.25、rho1/2累计不退款、cut16/SAT16/cut32、native retained100000/global B+100000及16GiB guard全不变。后续优先有界暂存与合法基础cover保护、关键AO31边界/截断审计、固定D共享面积恢复；不保证胜出，不提前称完成。

## 2026-10-20 最新：K6原版baseline23组；等价cuts内核221回归，开始检验面积恢复窗口

最新状态 `results_v2/single_output_k6_20261020/execution.json`；当前仍未完成K6全EPFL/重复，未合入main，未把旧K3/K4主表改标成新敏感性数据。

- 原版canonical `/tmp/emapper-single-output-k6-baselines-bfd-20261020/campaign.json` 23/120组、26原报告CEC、21报告失败。新增加一个**失败复用**：hyp A目标已经触发输入表达式guard，native未运行、mapping=0；原driver只在报告元数据和native后选择处读取objective（审计行190/273），故D目标显式采用同一失败。`adopt_k6_esyn_guard.py` 核验实际依赖、源记录和零映射，再保留目标声明与原执行argv/SHA；不是新native/runtime样本，不复用成功点，不修改作者算法。首8组BFD无关依赖采用仍单独留存。原环境失败cohort和全部timeout/exit-6保持。
- hyp core四个映射实际完成，失败位于旧公共checker的60秒guard；不能记成mapping timeout。`k6_hyp_verification_plan.json` 仅包含这四个后端/normalize已完成且mapping未耗尽的网表，独立补验不重跑mapper、不采用映射超时的partial incumbent。`/tmp/emapper-k6-hyp-verification-20261020/campaign.json` 当前ABCarea/delay两例补验通过（约45s/例），emap area仍独立600s guard内运行。原失败保留，公共面积/依赖引脚延迟按同一measure_json；对照scope仅通过带原报告/原audit/原模型与输入哈希的verification overlay采用。首2补验纳入 `scope_admission4_baseline23_verification2`，六gate均false。
- 准入修复首同版4例保持：hyp SAT24.592→0.550s、累计179014相同，但cuts仍60s失败，只是原始named-port CEC通过的基础cover；sqrt两例A2018.650002/D50815.830851同点。不是hyp优化成功。
- 另隔离位并行支持化简已RED384赋值扫描/GREEN0，四输入全真值及六输入随机/冗余支持对照、完整cuts/物理池/绑定证明序列通过；217项回归通过（先前3缺夹具失败保留，恢复原夹具后通过）。hyp native同binary诊断两设置仍timeout；原标量进入第4轮约37.37s，新化简约33.47s、第5轮56.43s。只是开销改善，无合法优化cover/PPA结论。
- 新有界有序支持合并/借用排序键仅改变等价实现，不删非对称pin配置、不改候选顺序/截断准则。旧新支持union（含越宽拒绝及乱序/重复fallback）、排序关系、完整候选/绑定序列通过。最终221项通过/5ignored/3旧synth过滤；新完整共享覆盖恢复用例对fraction0/.4/.7/1实际运行1项通过，先前错误`--exact`模块名导致0项日志不算通过。源码 `/tmp/emapper-single-output-k6-kernels-20261020`；main不变，OPS默认0。
- 已实际接通`MapParams.exchange_time_fraction`到 `egraph_cover_exchange::recover` 的同global search deadline，默认仍.70；不是仅添加未使用字段。sqrt trace显示global fixed-bound flow后约.2s剩给local、terminal-area未尝试；下一正式同binary全20比较.70/.40，内核/word/准入都固定1，保留回退，不能把阶段deadline当preemptive硬cap或重置2秒。完整“旧覆盖”仍指本次预算内访问的pre-exchange seed，**不是**历史完整2秒终态免费重放。
- 内核native诊断（profile binary a564...）hyp第5轮从58.51s提前至45.66s，但两例仍native60s timeout；不宣称解决。增加合成测试改变了release元数据/SHA，旧profile源码/bin已逐字重建并永久保留；正式最终native f865ae56c9608f9102affd4516dc6665ce3f17b30cee0a579fa63b466dc6f820 另存，221回归。两次重建SHA证据 `kernels_two_builds_validation.json`；诊断不混成该final binary的正式PPA。
- hyp仅测量的emap面积网表103830物理实例，ABCarea129805，均超过native retained100000；这不是最小下界、不证明更小cover不可能，却提示全局cover/池准入必须继续审计。该测量尚不能代替对应CEC。全局源+搜索B+100000仍不变，生成暂存不算retained RAM上限，不把累计节点改成驻留峰值。

所有方法复用一次resyn2；K6固定40/47单输出门，7个7–9输入未接入；K3/K4保留同最终binary的后续敏感性。mapping-only300、native60、EqMap90、inclusive recovery2/.25返回预留、rho1/2累计准入不退款、cuts16/SAT16/cut32/native池100000与global B+100000不变。先顺序结束有限批次、核查锁/partial/hash，再推进，不依赖跨turn后台存活。

## 2026-10-20 续作：K6准入计数同版对照首4例，hyp仍未完成优化

最新证据仍以 `results_v2/single_output_k6_20261020/execution.json` 为准。每个隔离源码现在使用独立 `isolated-target`：V5独立重建native与先前测量binary SHA完全相同；完整word cuts/候选/绑定来源oracle实际运行1项通过；准入修复214项通过/5ignored/3旧synth过滤。旧0-test产物保留但不算通过。详见 `build_isolation_validation.json`。

- 真实hyp影子诊断至少32768次RHS局部memo增量与原resident增量全部一致，差异0；SAT后累计准入179014<=634761，不退还累计账本。该诊断刻意在SAT后停止，非mapped/CEC/PPA成功，详见 `admission_shadow_validation.json`。
- 冻结准入源码及binary分别保存于 `source_admission_delta/`、`bench_rs_egraph_k6_admission`；新同binary矩阵 `/tmp/emapper-single-output-k6-admission-full-20261020/campaign.json` 当前4/40，WORD固定1、SHADOW关闭，仅切换 `EGM_FAST_ADMISSION_DELTA`。hyp SAT24.592s→0.550s，两次累计准入179014一致，但均在cuts阶段native60s timeout，仅source基础覆盖通过原始named-port CEC；不宣称hyp搜索或延迟已改善。sqrt两设置均native优化成功，A2018.650002/D50815.830851，CEC通过；仅1pair，不能报稳定改善。
- 原版BFD baseline续批运行中，之前10/120并非全套；bar两E-Syn新失败为native exit-6，保留原算法/models/feature/index/预算，不用历史优化probe替换。实际进度取canonical campaign，不依赖跨turn后台存活。
- 只读scope脚本现在按真实profile参数命名，验证实际native stderr的WORD/FAST/SHADOW配置及累计节点/全局retained/300s规则，基础覆盖与优化选择分开；首4case对baseline10的完整20scope六gate均false，missing/fail全留。
- 下一隔离 `EGM_WORD_REDUCE` 默认0原型仅准备源码：以位运算计算必要支持集，对照原标量参考；尚未编译/未运行红绿及整体pool oracle/无PPA。不得当作已接入或hyp解决。其他两个待验证开销假设为支持集合分配和排序Cut克隆，尚未实现，不丢非对称pin候选。

K6仅40/47原始单输出门，七个7–9输入门未接入；K3/K4仍需同最终binary敏感性。mapping-only300s/native60s/EqMap90s/inclusive recovery2s/rho1/2/cuts16/SAT16/cut32/native retained100k与global B+100k均不变；main及正式论文K3/K4表未改标，未完成K6完整实验及重复，不提前宣称主版本采用或优势。

## 当前续作：2026-10-20，K6扩库进行中；发现并修复构建环境，定位准入计数热点

以 `results_v2/single_output_k6_20261020/execution.json` 为最新执行状态；下面同日早期计数/源码状态是历史前缀，不可直接当最终全套数据。

- K6为40/47个原始单输出单元，7个7至9输入单元尚未接入64位cuts；K3/K4保留敏感性。尚未完成K6全EPFL及重复，不改标旧表、不宣布优势。
- 正式baseline续作现在是 `/tmp/emapper-single-output-k6-baselines-bfd-20261020/campaign.json`，10/120组：8组经实际依赖证明显式采用，另2组重新执行；22条CEC、2条原生feature阶段timeout。旧K6首12组的4条E-Syn exit101是LLD未初始化legacy C++objective registry导致的环境失败，保留在旧cohort，不能作为算法失败。原模型/源码/参数不变，仅最终crate使用GNU bfd重新链接。最小Rust LLD/BFD红绿对照及同一XGBoost动态库预测比对已保存；原xgboost DMatrix零值视作missing的行为也未改变。修复后原feature流程仍60秒timeout，不使用此前优化probe。
- 局部word-lift8例、扩展word-math6例均有fresh原始named-port CEC；hyp仍source基本覆盖。扩展word-math的sqrt一对同D50815.83，A2122.56→2019.61，但未重复；控制组晚期交换生成60个additional候选，新组无交换，初始生成池均1507367，不能把时间分配耦合误写成纯数学改写必然带来PPA收益。main未合入。
- 评分向量去重假设被实际库否定：55440次原向量评分对应55008次不同向量，主要由于非对称引脚时序。逐项评分值一致，但2x减少工作断言失败，原型未跑native PPA/不启用/不删物理引脚配置。
- 60秒hyp诊断确认：65536次准入节点快照，至少23.4489秒用于 `total_number_of_nodes()` 全图遍历，几乎占满约24秒SAT阶段。内核perf权限不足，未改系统设置；使用单独计时原型，非PPA结果。每RHS无rebuild前后改读O(1)memo增量的隔离修复已红转绿，64组重建/别名/满额序列与原累计账本一致；不把memo驻留量当累计额度、不退还已准入节点。所有上限不变。
- 新修复源码 `/tmp/emapper-single-output-k6-admission-20261020`，`EGM_FAST_ADMISSION_DELTA`默认0；此前共享target中214项相关测试通过/5ignored/3旧synth过滤，但发现不同Cargo源码目录共用target会命中错误测试产物（whole-pool oracle实际0项）。现在每版本改用独立 `isolated-target`，重新核验完整回归、必须非零且包含指定测试，并对旧word版本作独立重建SHA核对。0项不记作通过；现有CEC数据保留，未完成源码/bin重核前不采用为新主配置。

300秒仅计映射，native60/EqMap90/inclusive recovery2/G=B+ceil(B/2)/cuts16/SAT16/cut32/retained100k及全局B+100k规则保持。原版baseline算法/模型/参数不优化。下一步是隔离构建核验后冻结计数修复完整20设计两profile对照，同时按检查点顺序补齐BFD baseline；全部CEC、累计账本和重复验证完成后才合入main并更新正式论文表。

## 最新工作：2026-10-20，扩展单输出库与搜索开销诊断（进行中）

用户指定更完整单输出库为主线，K3/K4保留为敏感性实验。已逐门清点原genlib：47个单输出单元，K3/K4/K6分别为19/30/40门；7个7至9输入单元暂不在64位cut匹配表示的覆盖范围内。新主评估先使用K6的40门，**不称完整47门库或完整PDK**。清点、排除单元及原库SHA见 `results_v2/single_output_k6_20261020/library_plan.json`。K6数据尚未收齐，旧主表仍是已审计的K3/K4历史结果，不能改标成K6。

- 新实验继续复用全部20个一次resyn2/CEC输入；mapping-only300s、native guard60s（EqMap90s）、inclusive recovery2s、rho=1/2累计准入及retained全局记账规则保持。baseline原算法、models、native参数未变。两个原driver的K3/K4 argparse范围由新nested适配器允许K6，保留原文件/调用栈及E-morphic内部ABC计时；原driver文件未改。原参数解析拒绝K6的日志和适配后解析通过的日志均保存。
- 三个公开baseline源码已从锁定commit恢复，13,354个tracked文件全部匹配历史SHA（E-Syn仅复用历史路径/环境预算适配补丁，没有feature/index优化）。gRPC独立协议文件SHA也匹配。构建失败日志保留；用现有GCC11的libstdc++链接路径修复环境缺失后构建成功，不修改作者代码。EqMap binary逐字相同，E-Syn和E-morphic重建binary不同，记录新SHA，不冒称旧binary，也不据此采用旧K6运行结果。
- 隔离word-lift-only版本 `/tmp/emapper-single-output-k6-20261020`，main未改。208项相关库测试通过/5ignored/3旧synth过滤；新增两项穷举/随机真值展开对照在内。首次全测缺少旧数据库夹具导致2失败，补齐同一夹具后通过，原失败保留。真值展开独立操作计时0.667s与0.075s，只有微基准意义，不能当hyp完成或电路级改善。
- 新K6同binary `word0/word1` 全20对照矩阵当前8/40：adder/bar/div/hyp各两设置，8个fresh original named-port CEC通过；其中4个优化cover，4个仅基础cover。bar两设置均A257.31/D168.29；adder同D2569.15但A89.74→90.32，回退保留。div两次native60s均未完成，source A4154.88/D99097.36；hyp两次同类失败，source A22414.78/D667115.20，均不能称优化成功。一个bar/word1初测与约3秒只读源码SHA审计重叠，记录io_event，保留主点并安排无额外I/O的重复，不据此称隔离加速。
- 下一隔离word-compose版本 `/tmp/emapper-single-output-k6-compose-20261020` 已实现功能cut替换的位并行cofactor/mux，4项穷举/随机专门测试、210项相关库测试通过/5ignored/3旧synth过滤。500,000次独立操作0.815s→0.050s且checksum一致；尚无该版本完整电路CEC/PPA证据，不接入main、不宣布hyp解决。不同版本/库的好点不能择设计拼接。

新baseline K6矩阵将保存120组/260预期状态槽；目前尚未运行完成。待原版baseline、全EPFL同版对照、sqrt/bar/hyp重复、全局预算与失败覆盖核验后再更新正式主表和论文结论。新证据目录 `results_v2/single_output_k6_20261020`，下面各条“当前”均为各自历史检查点。

## 当前检查点：2026-10-19，统一主表与同版独立消融已完成

本轮用户要求已经处理：主表只保留一个事先固定的 eGraphMap `full` 配置，去掉 B/X；每库一张完整宽表，五个原版 baseline 加 eGraphMap，每个方法横向 A/D/T。新增表 III 独立呈现消融及重复结果。主表 eGraphMap 全部来自本次新 full40，不取重复最好值、不按设计挑 source/config。原600个baseline指标单元逐字不变，新full120个指标单元及消融计数均已核验。

- 当前发布 `output/pdf/DAC-eGraphMap-draft.pdf` SHA `6021292760e0a0792e4d0c4464e86b0b2f43f472aee0bc21ec49907ea0be3691`；TeX SHA `bbc54a657efdc101db8b91cd7512970f64a3d2666b9c836036c4b4a2e7c47aa4`。6正文+1参考，正文原10pt/边距不变，主表8pt、消融表footnotesize；7页逐页视觉检查，最后改动4/5/6重新检查、1/2/3/7像素一致，0 overfull/0未解引用，4个非致命underfull保留。证据 `results_v2/independent_components_20261019/paper_qa.json`。
- 统一native SHA `ca0600a071ca57b9a316438b2eb1b3da1b68ce0c61412c7ef6ba7dae62d22c0f`。full保持V46完整算法及参数；新增三个专用移除开关均默认false，通用默认策略不变。201库测试通过/5ignored/3旧无关synth明确过滤；三新开关测试通过，初次错误断言日志单独保留。baseline原算法/models/native参数未改，E-Syn优化探针仍不作为正式baseline。
- 同一binary的全20 EPFL×两库×8设置=320尝试：254公共original named-port CEC成功，66失败（48累计节点准入拒绝、18物理覆盖失败）。full自身34/40；div/hyp/mem_ctrl双库6次仍累计拒绝。所有方法复用一次resyn2的20个AIG；累计20k/cut16/retained100k/SAT16/cut32/area4×2000/inclusive recovery2s/native60/mapping-only300均未提高。累计不是驻留上限，不退役返还；生成暂存另计。阶段fraction在原子trial间检查，不是preemptive硬cap。
- 七项独立移除分别为功能quotient cuts、可选结构cuts的单witness限制、输入相位匹配、canonical引脚分配、SDC、后段共享面积恢复、种子指标保护模块。具体边界见 `plan.json` 和论文：canonical限制不等于只关闭pin-delay lookup；共享面积移除保留共同seed初始化；保护模块包括envelope/seed-area ceiling及bound-feasible DA cache，不是仅关闭最后cap，也不关闭cycle/source/pinned-binding正确性检查。保留相同阶段fraction，未新增替代搜索阶段，但较早结束会改变现有阶段的剩余时间。
- PPA计数符号均为“移除项相对full更低/相同/更高”，只对双方合法的pair统计，覆盖单列。功能cuts移除D=1/25/8、A=16/6/12；one-witness D=0/34/0。移除input-phase只剩22/40 CEC（合法D=0/2/20），canonical-pins28/40（D=0/0/28）。SDC移除D=0/34/0；full最终bank有9个仅SDC证据的selected target bindings，但这不证明独立延迟收益。无后段面积恢复A=0/2/32、D=0/34/0；无种子指标保护A=3/22/9、D=0/34/0，且本次两个设置均未越各自seed envelope，不能据差异声称最终guard实际binding。
- sqrt/sin/priority两库、全部8设置各3轮=144尝试、132 CEC；full18/18，no-input-phase与canonical-pins各12/18，其他各18/18。轮次/配置顺序轮转，不挑最好run。full sqrt D三轮恒为55862.56/47031.68，A范围1941.86–1941.93/2247.91–2248.65；仍有相对emap的面积代价。full四输入sin D1806.55、A516.91，移除功能cuts变1793.92/527.98，反例如实保留。这里是三个目标设计的重复，不冒充全20多轮重复。
- 原版baseline仍为240组/520槽，372CEC/124reported failures/24unmeasured，未重跑或优化其算法。公共physical instance area/unloaded dependent-pin block delay/独立original named-port CEC不变；EqMap T仍是保守上界、E-morphic仍native unseeded赢家。新full的两库DA、AD-posthoc、Pareto共6个全scope gate均false，MtM未接入，不称全面优势或全项目完成。
- 计时中一次cache-copy与原multiplier/no-area恢复重叠，事先按timestamp选定两库全8设置额外16案例敏感性复验、12CEC；原始主点不替换。`storage_event.json`、所有原始失败与中断日志、校验ledger-schema错误及汇总变量覆盖错误、修复前输出与初次8页布局均保留。只修复只读汇总，不据此重跑native或提高预算。
- 证据目录现在是工作区真实目录，不再只靠/tmp：`results_v2/independent_components_20261019`。六份全成员SHA核验压缩归档已永久复制 `results_v2/archives`，原/tmp数据与失败未删除；来源快照、完整命令、输入/库/源码/bin/hash、所有配对/range及baseline gates均存。`archives_permanent.json`、`persistence.json`、`paper_data_validation.json`、`evidence/qualification.json` 可审计。

本轮全部有限进程和所持runner锁已空闲。后续研究仍包括三个累计超限设计的安全压缩、sqrt面积代价、sin功能cuts的时序回退、SDC关键路径利用和历史完整共享终态质量保护；不得按设计拼配置、把字典序胜出叫Pareto支配或将本轮论文更新叫项目全面完成。以下所有较早PDF SHA与“当前”状态均是历史记录。

## 论文完整宽表：2026-10-19

按用户授权缩小表格字号，各库合成一张不拆面板的完整宽表，七种方法全部横向排列，每方法A/D/T三个子列；表格7pt、正文及边距不变。840个指标单元逐项不变，表格外TeX逐字不变，无实验或算法变动。PDF当前为5正文+1参考，满足最多6+1；第5页目视检查，1–4页像素与前版相同，当前第6页参考与前版第7页像素相同，无编译警告/超宽。证据 `results_v2/single_panel_tables_20261019/paper_qa.json`；最新PDF SHA `f298af6c343c0459b3172f7b9d357f0477c9a1cd1e6a2e988a85897006775e53`。以下分面板布局及其哈希为历史版本。

## 论文表格横向调整：2026-10-19

按用户要求，表I/II改为每个方法横向并列A、D、T子列；每库仍为一个表，分上下两个方法面板以保留9pt表格字号和可读性。840个指标单元逐项核验不变，计时定义及≤上界标识不变，不重跑实验。仍6正文+1参考，5/6页目视检查，1/2/3/4/7与前版QA像素相同，无overfull或引用警告。证据 `results_v2/horizontal_runtime_tables_20261019/paper_qa.json`；当前PDF SHA `f829e4d4b328125b31a12599895250f912d79e431d4797bdaa0802315de47595`。以下纵向三行布局及其PDF哈希仅为上一版历史记录。

## 论文表格更新：2026-10-19，Area/Delay/Runtime 合表

原表I/II合为三输入库表I，原III/IV合为四输入库表II；每设计按A/D/T三行展示，T单位秒。全部280组Area/Delay对逐项与旧稿相同，204组有映射计时；76个失败/节点拒绝条目的runtime保留缺失标识。EqMap采用既有初始LUT阶段外层时间加直接mapping的保守上界，B/X采用含内部报告的mapper进程上界，均向上取整到0.01秒并标注≤；不将这些上界解读为隔离算法加速。外部准备、CEC及公共指标报告不计入T，E-morphic仍用嵌套mapping活动区间并集。未重跑或改变baseline算法/参数/数据。重复的汇总表转为正文，保留分库统计。证据为 `results_v2/combined_runtime_tables_20261019/runtime_sources.json` 与 `paper_qa.json`。

当前发布PDF SHA `7a99a4baf3ff8d95c8a658b13833581f650e5787ec1dccea251ba58bc686f16b`；仍6页正文+1页参考，原字体/边距不变，4/5/6页目视检查、1/2/3/7页与旧稿像素相同。下述83c旧PDF为上轮历史版本；原有实验结论与限制不变。

## 当前检查点：2026-10-19，V46 稀疏覆盖交易（默认仍关闭）

最终论文已发布 `output/pdf/DAC-eGraphMap-draft.pdf`，SHA `83c506222d9ed7890ba4dcfccc9322c473577501e6b98279db3a9f586724bc04`；6正文+1参考、7页目视QA、原字号边距与baseline五列未改，证据 `results_v2/protected_recovery_paper_20261019/paper_qa.json`。所有本轮有限试验/归档已结束，当前没有后台任务。下一步按完整共享repair witness诊断推进，不能将“本轮验证结束”当作全项目或全baseline支配已完成。

- 主源码/CLI已接通实验路径，release SHA `14c9c3a587c2b7efef8fccf1a819735491277ba2d195f8eb8ac008151891a1fb`；无Git写入，保留脏工作。完整报告 `results_v2/sparse_cover_transaction_20261019/round_report.json`、来源/补丁/RED/GREEN/测试均在同目录。
- 当前同版固定单输出(D,A)两库全20：40尝试，34公共named-port CEC，6为div/hyp/mem_ctrl双库累计导入拒绝。sqrt/sin/adder双库三轮18/18全CEC、实际预算通过；中轮反转库及设计顺序。不挑最好run；六初探CEC另计，不混成设计数。
- 对完整冻结F42参考：34合法对A23好1同10差、D1好33同0差；其中9项同D面积回退。全6个DA/AD-posthoc/Pareto全scope gate仍false，MtM未接入。不能称SOTA、默认采用或全部目标完成。
- √重复D两库均55862.56/47031.68；K3 A1942.19–2018.87仍有波动，K4 A2248.83–2249.98仍高于旧2215.87。K4 adder A99.42/D2569.47较F的95.34/2571.81有面积代价。sin重复A：K3 457.17–457.33，K4 515.34–515.72；D仍2051.62/1806.55，K4比旧structural慢12.63，不能藏掉。
- 来源记录cut/care/原始与规范相位/引脚/物理身份；保护本次完整已访问PO覆盖含非关键共享依赖，固定Dref传required-time及实际唯一实例面积恢复。稀疏交易借用不可变key，只复制变更记录，2048 touched/fresh上限，所有权移动与释放计入原2秒；32小图与旧交易的去重/裁剪/完整覆盖相同。198测试通过、5ignored、3既有耗时synth明确未运行。
- 严格累计实际20k、cut16、retained100k、SAT16、cut32、area4x2000、inclusive2s/native60/mapping-only300不变；不是resident上限，不退役返还，不每region重置。retained不是RAM/生成暂存上限。摘要只省略可选代理报告，不省计数/完整selected binding证明/CEC，不使用只读诊断frontier当免费seed。
- 原baseline240组/520槽：372CEC/124报告失败/24未测量，原算法/models/native参数不动；四小binary SHA重新核对，旧13403文件全源审计仅引用。旧E-Syn优化probe仍只存档。
- 本轮之前V42全120/102CEC18节点失败、54目标CEC但K4sqrt首次D85011.84；V44全80/68CEC12节点失败、36重复26CEC10预算失败；V45只有4目标CEC而非full20。全部保留，不能改成V46成功。另13个raw目录已逐成员SHA永久归档（加之前8个），见archives.json；不删除原/tmp失败目录。
- 仍缺旧完整历史终态的质量保证。真实trace显示legacy extract会以tree/fanout flow覆盖并验证全信号repair witness，当前新种子省略该全共享参考；旧K4sqrt deferred A2222.65→最终2215.87，新2312.11→2250.21。还有refresh/时间分配差异，不能只凭trace宣称唯一因果。下一实验 `next_shared_witness_experiment.json`：在原2秒种子窗口构造旧式完整sharing参考并保护其可用绑定，和PO实际面积分开，不加载历史网表或在预算外重跑旧2秒。跨结构6-support临时展开、源压缩仅计划。
- 论文更新工作在 `results_v2/protected_recovery_paper_20261019`；主四表B/X各整套固定版本，B为F42、X整套V46，不按case拼版本。标准字号边距6正文+1参考；以该目录最终paper_qa.json确认发布。后文历史记录按时间保留，旧“正在执行”计数不代表后台仍活跃。

# 工艺映射基线与实验记录（2026-10-07）

最新增量（2026-10-19，累计准入修复与全套复验结束）：当前main release SHA49886ba636840a824890cd93fb3e385329ca0dac4eea47206521b6ed20470b76，替代上一5390。单一 NodeAdmission 跨五条strict导入路径及strict SAT共用，union/rebuild不返还额度；restructure/map_with_cells也在新增前检查。非strict旧兼容行为不属于本次严格协议。177库测试通过/5ignored；20 CLI边界尝试为6独立CEC、14预期准入拒绝，原普通6限额先构图16的RED已修复。cumulative_admission_full_20261019完成80/80，68公共CEC/12累计导入失败(div/hyp/mem_ctrl)，每profile34/40合法；34pairs D7改善25相同2回退，A15改善7相同12回退。sqrt/sin/adder三轮交替顺序重复36/36CEC且累计预算审计通过；sqrt两库收益3/3保持，四输入sin D+12.63、adder D+2.02与A+5.03均3/3复现。相对固定emap-delay：K3合法17/20，D2优12平3劣、A2优1平14劣，条件GM D1.003713/A1.119544；K4合法17/20，D6优6平5劣、A3优1平13劣，GM D1.006580/A1.122537。失败全留，12字典序/Pareto gates全false，AD仅后验检查。baseline算法/参数/数值预算未改，4二进制新hash匹配，先前13403文件源码核验引用而未重跑。完整绑定来源、跨pool旧共享cover保护、固定Dref共享面积扩展尚未实现，next_binding_stage.json明确下一阶段。四份raw all-member-SHA压缩归档永久保存；论文更新新版本独立证据，6正文+1参考，字号/边距/preamble不变，2/4/5页视觉QA，1/3/6/7像素与先前QA相同。所有有限任务结束、181唯一runner锁空闲；不声称PPA提升或目标完成。详见 cumulative_admission_20261019/report.txt、final_state.json、implementation.patch、paper_qa.json。

最新状态索引（2026-10-19，用户明确要求先合并）：主源码/二进制已接入近期恢复、功能cuts及独立源规划分支，旧8f6归档，新main SHA539009198b428235976206cc6beb26351b2ae30939b2fdfeaf0df4ce9a360d6d。172库测试通过/5ignored，新主双库sqrt/sin/adder12CEC通过，另3组合边界测试和1普通入口预算探针符合预期。新增研究开关仍默认off，淘汰硬筛选默认off；baseline/既有参数默认/预算不变。新主完整20设计和重复未跑，旧全套不改名。全面分析见 main_integration_20261019/analysis.md。以下先前不合入限制已被本次用户授权覆盖，旧数据原样保留。

当前状态（2026-10-12）：主评估集按用户指示改为 EPFL 的全部
20个算术/控制组合电路，预优化固定一次 `resyn2`。MtM三例未接入，
不计入20例覆盖。`epfl_resyn2_subjects_20261012/report.json` 中20例
均通过preprocessing CEC，所有方法及两库复用同一份AIG；文件
来源为vendored mockturtle EPFL分发，逐文件原始header/SHA固定，
不声称已验证其与当前官方master逐字节一致。
新完整映射实验由 `scripts/experiments/run_epfl_campaign.py` 按单设计
检查点顺序执行，状态见 `results_v2/epfl_checkpoint_20261013/campaign.json`；
完整baseline数据尚未齐。原epfl_campaign_20261012中断批次完整保留。所有job的失败、
超时和缺失覆盖均保留。既有ISCAS数据及论文表是历史pilot，
禁止将这些数字改名为EPFL结果；待新数据核验后再替换主表。
实验优先级仍为area-delay/delay-area，库/预算/公共PPA/独立
named-port CEC要求保持。源码默认EPFL全20，显式--benches可跑
标注的子集，--suite iscas可复现旧套件。
以下旧日期段落为历史记录，原1200成功网表记录属于ISCAS pilot。
算法全面胜出、工业多输出库和ODC仍未完成。



在 emap 之外新增 ABC `map`、`amap`、`&nf` 和 vendored mockturtle 普通 `map`，
并加入 `amap;mfs3 -a` 与显式 `resyn2;&nf` 流程。它们是不同算法/配置和组合流程，
不是九个独立工具。普通 mockturtle 驱动没有经过 `map_egg`，也不调用 emap。

## 联网资料与选择依据

- [ABC 官方仓库](https://github.com/berkeley-abc/abc)：可运行的开源综合、映射和验证工具。
- [ABC amap 源码](https://github.com/berkeley-abc/abc/blob/master/src/map/amap/amapCore.c)：
  标准单元映射，包含 area-flow 和 exact-area 迭代，适合面积基线。
- [ABC &nf 源码](https://github.com/berkeley-abc/abc/blob/master/src/aig/gia/giaNf.c)：
  标准单元 mapper，使用 priority cuts、引用次数和 required time；补充现代映射路径。
- [Versatile SAT-based Remapping for Standard Cells, IWLS 2016](https://people.eecs.berkeley.edu/~alanmi/publications/2016/iwls16_mfs3.pdf)：
  基于 Boolean relation 和 SAT 的后映射优化，作为比逐边界 SDC 更强的 DC 对照。
- [mockturtle 官方 mapper 文档](https://mockturtle.readthedocs.io/en/latest/algorithms/mapper.html)：
  普通 `map` 的面积/延迟控制与 emap 多输出支持。文档中的 `use_dont_cares` 示例用于
  exact-library graph mapping，不能直接解释为当前标准单元 `map` 的 ODC 基线。
- [Technology Mapping Using Multi-Output Library Cells, ICCAD 2023](https://ieeexplore.ieee.org/abstract/document/10323999)：
  多输出对照的研究依据；属于已有 emap 路线，不把它重复计作新算法。
- [Yosys ABC 集成源码](https://github.com/YosysHQ/yosys/blob/main/passes/techmap/abc.cc)：
  说明 Yosys 的该工艺映射路径调用 ABC。当前直接运行 ABC，避免将包装层列成独立算法。

LUT 数量、FPGA dual-output LUT packing、近似映射与允许改变库的单元生成工作，
没有混入标准单元精确等价面积表。商业工具未在此环境实际运行，故没有结果。

## 补充：直接相关的 e-graph 工作与复现状态

前一轮只跑了常规 mapper，遗漏了以下重要相关工作。原有 96 条标准映射记录不包含它们；本轮另完成 EqMap 的受控组合实验。

| 工作 | 目标和方法 | 代码及复现状态 |
|---|---|---|
| [E-Syn, DAC 2024](https://arxiv.org/abs/2403.14242) | Boolean e-graph 重写，工艺相关面积/延迟 cost 引导结构选择 | [官方代码](https://github.com/Gy-Hu/E-Syn)，README 指定论文复现用 `new` 分支；已构建有界冻结模型适配器，完整流程未复现 |
| [E-morphic, DAC 2025](https://arxiv.org/abs/2504.11574) | 常规优化之后用 equality saturation 探索结构，再做工艺映射；剪枝与退火 extraction | [官方代码](https://github.com/Gy-Hu/E-Syn2)，论文复现分支为 `SA-golden-model-delaytuning`；main 是早期随机抽样实现，不能混称论文结果 |
| [EqMap, ICCAD 2025](https://www.csl.cornell.edu/~zhiruz/pdfs/eqmap-iccad2025.pdf) | Rust/egg 驱动的 FPGA LUT remapping；论文使用贪心 extraction | [官方代码](https://github.com/cornell-zhang/eqmap)，当前 main 已有 optional CBC/HiGHS exact extraction；本轮锁定 commit 构建，运行 greedy LUT + 同库 ABC 后端和 staging control |
| [Churchroad, WOSET 2024](https://arxiv.org/abs/2411.11036) | egglog + Lakeroad program synthesis，主要映射 FPGA DSP 原语 | [官方代码](https://github.com/gussmith23/churchroad)；目标资源与标准单元面积不同，不直接进入 ASAP7 面积表 |
| [Nextmap, PLDI 2026](https://zsisco.net/papers/nextmap-pldi26.pdf) | Semantic e-graphs 跨 word/bit 层优化与映射；包含 LUT/DSP 等资源 | 已定位论文，代码/可运行性尚未核实；需要按目标资源另设比较 |

补充材料：[作者对 E-Syn/E-morphic 的说明](https://hongcezh.people.ust.hk/project/egraph/)、
[EqMap 当前 Cargo 配置](https://github.com/cornell-zhang/eqmap/blob/main/Cargo.toml)。
EqMap 当前依赖 `egg 0.11.0`，而 Churchroad 是 `egglog`，应准确区分实现框架。

下一批直接工艺比较优先复现 E-Syn/E-morphic，并给它们接相同 genlib、相同原始 AIG
及公共 evaluator。EqMap 先作为 LUT/重写方法对照；若将其输出再映射为标准单元，
应明确命名为 `EqMap + standard-cell mapper`，同时提供只跑后端 mapper 的对照。
不能用 LUT count/DSP count 冒充标准单元面积，也不能据 current main 的功能判断
论文版具有相同 extraction 能力。本原型的“e-graph + 映射”并不是首创；研究差异需要
进一步核对多输出物理实例、条件匹配和联合上下文替换，而不是仅以使用 egg 为贡献。

## 公共测量口径

1. 从同一原始组合 AIG 出发，按配置使用完全相同的 ASAP7 genlib 子集。
   3 输入与 4 输入子集分别统计，不合并比较。
2. ABC 映射前执行 `strash`。`resyn2` 重构流程单独命名；`mfs3` 从 `amap` 网表开始。
3. Yosys 以库 blackbox 读取实际网表；统一测量每个物理实例面积，包括输出 buffer。
   `assign` 视为连线，不插入额外 buffer。
4. 从每个输出实际函数使用的输入计算 `max(rise_block,fall_block)` 路径延迟。
   没有 slew、负载、线延迟或物理设计数据，不能视为 signoff STA 或物理 PPA。
5. 所有有效记录必须通过完整库模型展开后的 named-port CEC。
   ABC `&get -n` 保留接口名字；不使用按位置猜测对应关系。
6. 超时/工具错误/未知等价结果记为失败，不能进入有效比较；保留命令、日志、网表、
   输入/库/binary hash、工具版本。新报告还保存相关源码 hash 与逐方法参数。

此环境使用 ABC 1.01（编译时间 2022-01-29），并非声称运行最新版 ABC。
具体 Yosys 版本和二进制 hash 在各报告 manifest 中。

配置差异保留在命令里：`&nf` 的 K 与库子集输入上限相同、C=16；mockturtle 普通
`map` 使用本地库的默认 cut size 5、cut limit 49、area-flow/ELA 设置。e-graph
使用 K=3/4、cut=16、CBC=0、SDC-support=4、恢复预算 2 秒。
因此这是原生配置的算法对比，不能称为所有算法拥有相同 cut 枚举/搜索预算。
内部 e-graph CLI 的 CEC 在实验驱动中关闭，外部公共 evaluator 仍强制执行 CEC。

## 3 输入库：最新结果

表中每格为物理面积 / 无负载延迟。列出的基线是每个电路所跑标准方法中面积最低者，
并不代表最快者或一个统一的基线算法。完整表保留所有方法。

| 电路 | 最小面积基线 | 基线面积 / 延迟 | e-graph 面积 / 延迟 |
|---|---|---:|---:|
| c17 | ABC amap | 0.36 / 66.87 | 0.37 / 44.74 |
| c432 | resyn2 + ABC &nf | 10.34 / 394.68 | 12.68 / 545.07 |
| c499 | ABC amap + mfs3 | 23.08 / 373.59 | 23.08 / 330.88 |
| c880 | mockturtle map area | 21.04 / 429.83 | 20.60 / 434.95 |
| c1355 | mockturtle map area | 23.22 / 387.82 | 23.26 / 351.16 |
| c1908 | ABC amap + mfs3 | 17.51 / 450.64 | 19.67 / 526.54 |
| c2670 | resyn2 + ABC &nf | 38.14 / 265.46 | 40.45 / 361.10 |

数据：[3 输入汇总](../results_v2/egraph_comparison_3in_20261007.json)。
5 个电路包含 9 种外部基线配置；c1908/c2670 包含 8 种（未追加与 delay 配置在前
5 例中结果相同的实验性 `&nf -a`）。另有新旧 e-graph 消融，共 75 条有效记录。
这不是对所有方法/全部工业电路的优越性结论：c432、c1908、c2670 的面积和延迟仍有差距。

## 4 输入库：补充验证

| 电路 | 最小面积基线 | 基线面积 / 延迟 | e-graph 面积 / 延迟 |
|---|---|---:|---:|
| c432 | mockturtle map area | 9.65 / 405.22 | 11.26 / 554.93 |
| c499 | mockturtle map area | 23.02 / 391.98 | 23.02 / 330.42 |
| c880 | mockturtle map area | 19.14 / 429.15 | 18.95 / 426.27 |

数据：[4 输入汇总](../results_v2/egraph_comparison_4in_20261007.json)，共 21 条有效记录。
这个生产库子集只有单输出单元，不能据此证明多输出收益；多输出仍以 FA、三输出和
不同 support 等专项夹具验证，完整多输出模型在公共 evaluator 中也按实例计一次面积。

## 本轮优化与消融

- 新增 OR 因式分解和可选结合律。专项穷举验证正确，但结合律在有限预算下会退化：
  c432 面积 12.68→10.99、延迟 545.07→616.35；c499 面积 23.12→26.51。
  因此 `EGM_BOOLEAN=0` 保持默认。
- 实际引用次数的 cone-flow 重估在首批 5 例中未改善结果，不宣称独立收益。
- 增加整锥可回收面积排序，保留原搜索阶段和其有效覆盖；仅在剩余预算内追加搜索。
  物理多输出实例必须等所有所需输出失效后才计入可回收面积。
  3 输入库：c499 23.12/345.16→23.08/330.88；c880 20.64→20.60；
  c1908 19.89→19.67；c2670 40.55/363.11→40.45/361.10。
- 保留硬延迟、无环和完整覆盖检查。默认优化未实现 ODC，不能把 CEC 成功当作 ODC
  搜索已经实现；ODC 下一步需固定上下文、联合输出关系及 SAT/CEC 验收。

## 复现

```bash
cmake -S . -B build
cmake --build build --target baseline_mapper -j2
cargo build --release --bin bench_rs_egraph_map --bin bench_rs_netlist_eval
python3 scripts/experiments/bench_egraph_baselines.py \
  --output /tmp/egm-baselines-new --benches c17 c432 c499 c880 c1355
python3 scripts/experiments/bench_egraph_baselines.py --k 4 \
  --output /tmp/egm-baselines-4in-new --benches c432 c499 c880
python3 scripts/experiments/bench_egraph_baselines.py \
  --output /tmp/egm-recovery-new --methods egraph_previous egraph \
  --benches c17 c432 c499 c880 c1355 c1908 c2670
cargo test -p emapper egraph_map::tests -- --include-ignored
cargo test -p emapper netlist_metrics::tests
```

输出目录必须尚不存在，避免覆盖原始实验。`--timeout` 默认 60 秒，超时终止本次命令
的整个进程组并保留日志。默认基线驱动逐方法顺序执行；跨报告耗时不能自动当作
严格性能比较，尤其多个实验或编译任务同时运行时。

## EqMap：已完成的受控组合实验

官方代码锁定 `d3144c45d3441530cd8d240c5fc6ea6645b8a2fb`（0.10.3），源码未修改。
Rust 1.88 无法编译锁定依赖中的 `if let` guard，使用 Rust 1.99.0 构建成功；
本项目 Rust 二进制和测试仍用独立的 1.88.0 构建。当前 EqMap 还含 ASIC 入口，
但其固定单元规则/面积模型并非一般 genlib 接口，故未当作直接 ASAP7 ASIC baseline。

主实验将 **LUT 上限 6 与最终标准库上限 3 分开**，比较三条流程：

1. 原始 AIG → ABC `amap`。
2. Yosys `abc -lut 6` → LUT 功能模型展开 → ABC `amap`（staging control）。
3. 相同 LUT 输入 → EqMap greedy arc-set，`-k 6 -t 2 -s 20000 -n 16`
   → LUT 展开 → 相同 ABC `amap`。

EqMap 的限制是分区搜索限制，不是全设计两秒。两条 LUT 流程在展开前分别做 CEC；
所有最终网表使用同一 3-input ASAP7 子集、相同公共成本和 named-port CEC。
七个电路的 21 个最终网表全部验证通过；14 个 LUT 阶段也分别通过 CEC。
这些是组合流程实验，不是 EqMap 论文中的 vendor FPGA 结果复现。

| 电路 | 直接 ABC 面积/延迟 | LUT staging 面积/延迟 | EqMap + ABC 面积/延迟 | LUT 数 staging→EqMap |
|---|---:|---:|---:|---:|
| c17 | 0.36/66.87 | 0.79/84.00 | 0.79/84.00 | 2→2 |
| c432 | 13.08/631.23 | 17.05/847.10 | 17.76/825.90 | 45→45 |
| c499 | 24.72/426.09 | 42.67/496.96 | 54.30/540.12 | 96→96 |
| c880 | 21.78/506.64 | 39.55/669.42 | 40.41/721.53 | 77→77 |
| c1355 | 37.24/646.12 | 45.99/626.94 | 54.21/671.35 | 79→76 |
| c1908 | 20.76/543.69 | 36.14/687.88 | 41.29/777.53 | 63→62 |
| c2670 | 43.26/410.52 | 59.25/691.36 | 59.19/638.46 | 105→105 |

LUT count 下降不保证标准单元面积下降：例如 c1355 为 79→76 LUT，但面积
45.99→54.21。这说明跨目标 composition 必须与 staging control 和直接后端同时
比较，不能据此宣布 EqMap 在其原始 FPGA 目标上不好。

另保留 `-k 3` 诊断报告：当前 `src/cost.rs::KLUTCostFn` 对宽 LUT 使用有限
惩罚，并非硬排除，实际输出含 LUT4–LUT6。该报告不能称为合法 3-LUT 资源比较；
最终同库标准单元网表仍通过 CEC。主 `-k 6` 实验检查了实际宽度，全部 ≤6。

复现脚本：`scripts/experiments/bench_eqmap_baseline.py`。输出目录必须是新目录：

```bash
python3 scripts/experiments/bench_eqmap_baseline.py \
  --eqmap-source /tmp/eqmap-baseline --output /tmp/eqmap-repeat \
  --k 3 --lut-k 6 --benches c17 c432 c499 c880 c1355 c1908 c2670
```

[主报告](../results_v2/eqmap_lut6_adapter_3in_20261007/report.json)、
[3-LUT 请求诊断](../results_v2/eqmap_adapter_3in_verified_20261007/report.json)。
报告保存 commit、工具版本/二进制 hash、模型/adapter/script hash、命令、每阶段日志
和网表；与原有比较共用标准库 hash。

已建立独立 [DAC 初稿源码](../epmapping_paper/DAC-eGraphMap-draft/main.tex)，
表格从上述有效报告自动生成，并保存报告 hash 与全方法 CSV。研究差异暂定为
物理多输出实例、保守条件匹配、联合覆盖与共享恢复；真实多输出库、直接 emap
和 E-Syn/E-morphic 论文分支比较仍缺，初稿明确保留这些限制。

## 持续优化进度（2026-10-12）

新增 `EGM_FLOW_DELAY_TIES=1` 搜索分支：在面积流成本相等时先比较实际到达
时间，再比较深度。此操作可能改变 DAG 共享结构，因此默认仍为 0。
顺序反转回归确保快单元不会因候选顺序被漏选；19 项映射专项（含真实 CEC
正反例）通过。

同版本、同库、同预算的七电路对照共 14 个网表全部通过 CEC：

| 电路 | 原默认面积/延迟 | 延迟优先 tie 面积/延迟 |
|---|---:|---:|
| c17 | 0.37/44.74 | 0.37/44.74 |
| c432 | 12.68/545.07 | 12.68/544.27 |
| c499 | 23.08/330.88 | 23.08/327.99 |
| c880 | 20.60/434.95 | 20.60/434.95 |
| c1355 | 23.26/351.16 | 23.62/348.17 |
| c1908 | 19.67/526.54 | 19.68/486.79 |
| c2670 | 40.45/361.10 | 40.16/356.71 |

c2670 两项均改善；c1355、c1908 则存在面积/延迟折中，不能替换默认并宣称
全局改善。报告：`results_v2/egraph_tie_controlled_20261012/report.json`。

新增 `scripts/experiments/check_mapping_dominance.py` 自动门槛：要求每个已测
基线的面积和延迟均不差，且至少一项严格更好（1e-6 相对容差）；同时检查库、
输入、测量口径、CEC、失败记录和缺失设计。相等、CEC 不通过、面积/延迟折中
的合成回归均不能通过；同库明确占优的回归可通过。当前七设计仍均未超过全部
已测基线，`dominance.json` 明确报告失败。未接入的 emap/E-Syn/E-morphic
不能被算作已经超过。

当前优先搜索 c432/c1908：32 cuts、8 秒恢复，OR factoring、延迟 tie 和
加权时序三个分支。预算改变单独记录，不能解释成相同预算下的算法提升。
结果保存到 `results_v2/egraph_weak_design_search_20261012`。下一轮应根据
真实网表表现检查候选结构覆盖、共享修复和时序目标，而非盲目增加规则。

已设置本聊天每小时持续优化（automation `emapper`），保留实现/实验进度并
仅对实质进展、完成、失败或确需处理的问题通知。初稿尚未引用本轮调参结果；
稳定实验、真实多输出与未接入基线补齐后再更新主要论文表格。

上述弱设计搜索的 6 个网表均通过 CEC。c432 的 factoring/tie 面积仍为
12.68；加权时序面积/延迟为 16.19/449.76。c1908 加权时序为
20.01/451.82，仍未同时超过其强基线 17.51/450.64。扩预算未解决
结构性差距，下一轮优先定位候选覆盖与输入结构偏差，而非重复同一预算搜索。

## Heartbeat：三输出边界覆盖与结构扇出种子（2026-10-12）

实现了有界累积 boundary unions，并将 union 上限收紧为实际多输出单元公共
输入数与 K 的较小值。原来的 pair-only 合并无法得到三个独立 support 的共同
边界。新增 `EGM_MULTI_CUMULATIVE=1`（默认）及 pair-only 消融开关。

同二进制、同合成库 `disjoint_triple.genlib` / `disjoint_triple.aag` 的公共测量：

| 配置 | 物理面积 | 无负载延迟 | 实例数 | 独立 CEC |
|---|---:|---:|---:|---|
| pair-only，CBC=0 | 0.6 | 0.2 | 3 | 通过 |
| cumulative，CBC=0 | 0.4 | 3 | 1 | 通过 |
| cumulative，CBC=5 秒 | 0.4 | 3 | 1 | 通过 |

该 synthetic TRIPLE cell 的三个输出为独立反相，时序刻意不同。面积变小但
延迟变大，不能称为双指标优于 INV baseline 或生产库收益。详见
`results_v2/egraph_cumulative_controlled_20261012/report.json`，包含输入/库/
源码/二进制 hash、命令、网表、公共 JSON 测量与 named-port CEC 日志。
单测也穷举 8 种 PI 赋值，并分别验证启发式和 CBC。

加入可选 `EGM_STRUCTURAL_REFS=1`：在局部恢复前用有限 witness 的物理扇出
增加一个面积流种子。同一共享实例只计一次输入负载；修复后的 cover 必须
经完整指标检查。物理引用计数与三输出覆盖回归新增后，映射专项 21 项通过；
库测试 37 项通过、2 项忽略、3 项慢 synth 测试未重跑。

同默认预算的七设计 / 两配置共 14 个网表均通过公共 CEC：只有 c17 改变，
默认 0.37/44.74，新种子 0.36/66.83；其他六个设计完全相同，包括 c432
12.68/545.07 与 c1908 19.67/526.54。新种子仍默认关闭，避免将面积/延迟
折中称为总体改善。见 `results_v2/egraph_structural_refs_20261012/report.json`。
两配置均未通过逐已测基线 dominance 门槛；门槛报告保留在同目录。

另测 c17 硬延迟 42.760001、CBC 配置 5 秒：两配置均取得 0.53/42.76，
各自通过 CEC。但 solver status=Stopped，proven_optimal=false，不能据此
声称该候选空间最优或证明 0.36 面积下不可达到相同延迟。该实验使用不同
solver/约束预算，独立保存于 `egraph_c17_timing_feasibility_20261012`，不与
默认实验混称等预算改善。门槛同时指出其缺少其他六设计的候选输出。

DAC 初稿更新了累积 union 方法、合成机制的真实面积/延迟、扇出种子的负面/
折中结果和 CBC 未证明最优的限制。主表仍冻结于此前同库实验；额外证据的
报告 hash 单独登记于 `table_provenance.json`，不把合成库并入 ASAP7 主表。

下一轮优先：
- 检验原始 AIG 与预优化输入的结构差距，用显式分开的相同前处理对照，
  排查候选覆盖，而不是再次扩大同样 cut/refine 预算。
- 完成 emap 与 E-Syn/E-morphic 的同库原生 baseline 接入。
- 检查仅使用部分多输出 pin 时，未用物理输入能否安全接常量或复用已有信号；
  当前 common boundary 仍要求足够的不同叶，可能漏掉部分绑定机会。
- 累积窗口仍有界，远距离多输出组合与 ODC 仍未解决，不得宣称功能全部完成。

## 显式字典序目标与 DAC 6+1（2026-10-12）

新增 `EGM_OBJECTIVE=area-delay` / `delay-area`，分别按 (面积,延迟) /
(延迟,面积) 优化完整 cover。`weighted` 是兼容旧实验的默认；显式字典序模式
忽略 `EGM_DELAY_WEIGHT`，不是任意 scalar weight 近似。启发式接受使用字典序，
CBC 使用两阶段：优化主指标后限制最好已知主指标，再在剩余配置预算中优化
次指标。两阶段均证明最优且抽取验证通过，才标记 bounded-candidate 字典序
最优；超时/失败保留已验证 incumbent，不宣称最优。数值比较有浮点容差。

安装的 CBC 在改变目标/收紧时序界且带 continuous warm start 时曾返回 fractional
LP 列而报告 Finished，已保留整性检查并在字典序阶段关闭该 warm start；原有效
启发式 seed 仍作为 Rust 端回退。新增回归验证不同主指标、次指标 tie、硬时序
约束及 CBC/无 CBC 路径。22 项映射专项通过。

同 3-input library、cuts=16、refine=2s、CBC=0 的两目标七电路实验，14 个
网表均通过公共 CEC。结果目录分别为 `egraph_area_delay_final_20261012` 和
`egraph_delay_area_20261012`。c432 延迟优先变为 18.87/383.01，面积优先
仍为 12.68/545.07，这是明确的面积/延迟折中。

比较门槛增加 `--objective pareto|area-delay|delay-area`（默认仍是 pareto）。
两种字典序报告为各目录 `lexicographic_comparison.json`。面积优先时 c499、
c880 超过该套件每个已测配置，其他五例未达标；延迟优先七例均未超过全部
已测配置。这不是 Pareto 全面支配，也不包含未接入的 emap/E-Syn/E-morphic。

复现：
```bash
EGM_OBJECTIVE=delay-area target/release/bench_rs_egraph_map library.genlib input.aig mapped.v
EGM_OBJECTIVE=area-delay target/release/bench_rs_egraph_map library.genlib input.aig mapped.v
python3 scripts/experiments/bench_egraph_baselines.py --output /tmp/lex-delay-repeat \
  --k 3 --objective delay-area --methods egraph --benches c17 c432 c499 c880 c1355 c1908 c2670
```

按用户指定的 DAC 正文最多 6 页 + 参考文献另 1 页，初稿强制参考文献新起页；
已编译验证为正文 6 页、参考文献 1 页。方法和目标实验新增到稿件，实验表仍
由有效报告生成，新增报告 hash 进入 provenance。此约束来自用户，未声称
已核实未来 DAC 官方完整 template、匿名规则或截止日期。

## 输入结构对照与 E-Syn 适配（2026-10-12，第二次 heartbeat）

实验驱动增加 `--preprocess resyn2`：显式 ABC balance/rewrite/refactor 序列先
生成带 I/O 名字的共享 subject AIG，预处理本身用 ABC CEC 检查。普通 ABC
mapper、mockturtle 与 e-graph mapper 共用该文件；最终一律对原始 AIG
做公共 named-port CEC。方法标签带 `_resyn2`，保存准备耗时、准备 CEC
耗时/状态、准备 AIG hash、原始输入 hash、命令和网表。原始无预处理
同二进制 objective profile 报告仍作控制，不将组合收益称为纯 mapper 改善。
两次目标流程的 prepared AIG 只在 ABC 写出的时间戳上不同，逻辑和 I/O
数据字节一致；同一报告内所有 mapper 使用完全相同的 prepared 文件。

| 电路/优先级 | 原始输入 e-graph 面积/延迟 | resyn2 + e-graph | 同 resyn2 普通 mapper |
|---|---:|---:|---:|
| c432 面积优先 | 12.68/545.07 | 8.29/502.27 | map-area 7.91/460.52 |
| c1908 面积优先 | 19.67/526.54 | 18.33/466.80 | map-area 18.28/455.14 |
| c432 延迟优先 | 18.87/383.01 | 11.94/378.33 | map-delay 11.42/376.73 |
| c1908 延迟优先 | 27.31/361.53 | 23.96/337.47 | map-delay 22.42/338.33 |

两目标 / 两设计 / 四 mapper 配置共 16 个最终网表均通过 CEC。resyn2 的
额外耗时单独报告，不能称为等端到端预算下的新算法优势。c1908 延迟优先
在已测配置中胜出，但面积更大、仍未 Pareto 胜出。范围仅这两例，门槛
也明确指出缺少其他五设计，不把局部结果当全任务完成。
报告：`egraph_resyn2_area_20261012` / `egraph_resyn2_delay_20261012`。

E-Syn `new` 分支锁定 `72f5568ab61bac9b8de224199c45fd975ab2399c`。
原生 e-rewriter 已用 Rust 1.88 编译；XGBoost 老版本静态库需在 /tmp 为
系统 libstdc++.so.6 建立开发链接名，并对最终 cargo rustc 指定本地 -L。
未写系统目录。模型路径改为 CARGO_MANIFEST_DIR，其余模型权重/规则保持
原样。默认原生 c17 运行在 60 秒外部限制内超时，该失败保留。

加入显式 ESYN_TIME/NODES/ITER 环境预算，未设置时仍是原先 300 秒、
25M nodes、10M iterations；有界试验使用 2 秒/20K nodes/16 iterations。
这里 2 秒只是 runner 配置，不包含抽取、模型加载、Python 图统计和后端，
实际 native wall time 可达约 6–29 秒。冻结上游 full-library 模型，随后接
公共 3-input ASAP7 + ABC amap；从上游模型排出的候选中按实测字典序
选一个。不能称为原论文库/训练/预算的精确复现。

c17 unfolded 有界试验产生 30 个同库 CEC 合格候选，最佳 0.37/66.87。
c432/c1908 unfolded 超过适配器 1M 字符 guard，保留为接入失败，不能
解释成算法不等价或原论文性能失败。保留共享 wire 的 DAG 输入表示另
列为 `_bounded_dag`：两例各 30 个候选均通过 CEC，面积优先最佳分别
12.85/610.82 与 20.16/545.97。这些受表示/预算限制的适配结果不能
替代完整 E-Syn baseline；E-morphic 尚未接入。

适配器 `bench_esyn_baseline.py` 拒绝未知 symbol、丢失/重复输出、循环
赋值和错误 syntax；c17 unfold 与两弱设计 DAG 的格式往返已独立 CEC。
首版适配器未持久化顶层名规范化，导致 30 次工具级 CEC 失败；已修复
且失败日志保留。最终成功候选中没有删去不等价者来捏造算法优势。

移植补丁：`scripts/experiments/patches/esyn-portability-and-budget.patch`。
原模型 SHA、commit、完整 diff、binary SHA、harness SHA、预算、输出
表达式 SHA、每候选日志/网表/指标/CEC 都在报告中。上游 graph_info.py
将 as_completed 的完成顺序作为 CSV 候选索引，可能错配图特征；本轮未
修复该算法行为，保留此复现限制，不将其掩盖成无修改官方运行。

`check_mapping_dominance.py` 现在支持 esyn_ 标签和 --candidate-methods
过滤混合报告中的候选，防止把普通 mapper 控制也当目标实现。包含本轮
E-Syn 适配配置后，c1908 的 delay-area 仍超过全部当地已测配置；area-delay
和 Pareto 两种门槛仍失败。未测设计/完整原论文流程不计入此覆盖。

下一轮优先将原始与预优化结构作为独立、明确验证的候选来源，研究能否
在同一次 e-graph 覆盖中兼顾两者；严格检查 PI/PO 名字对应和条件联合，
不得未经证明 union 任意中间信号。继续完善 E-Syn 的原生预处理/表示与
训练口径，并接入 E-morphic；不要反复重跑已经无改善的 fanout 参数。

## E-Syn2 / E-morphic 官方源接入（2026-10-12，第三次 heartbeat）

用户提供的 E-Syn2 仓库对应 DAC 2025 E-morphic，不是另一个应混入
E-Syn 2024 结果的别名。官方 main 为早期随机采样实现，论文复现分支
为 `SA-golden-model-delaytuning`。本轮固定后者 commit
`26c128b5b0453795917474d118b8f89dd7a2ebc9`。重写器原生构建成功；
抽取器构建失败于缺少 `grpc_communicator/proto/service.proto`。从官方
main commit `1a513f89886408df0c94471de0590df7ab16dcd2` 恢复该文件后
构建成功，最终 tracked 源码 diff 为空，规则、抽取及评分逻辑未改动。
普通沙箱代理解析失败的构建已经升级网络权限重试；失败原因与命令
记录在 `emorphic_source_audit_20261012/report.json`。本轮使用仓库自带
静态 libabc（hash 保存），未重新构建它；随机抽取不调用其评分接口。

新增 `bench_emorphic_baseline.py`：保留 AIG 共享的拓扑 EQN 输入，检查
上游原地预处理后的 OUTORDER，导出随机 choices 为共享 Verilog，再用
公共 ABC amap、3-input ASAP7 与原始输入 named-port CEC。拒绝循环、
缺失选择、错误 eclass、未知 symbol、输出数量改变，并避免内部线名
与真实端口碰撞。每个 sample 的图/choices hash、命令、日志、网表和
CEC 都保存；未通过者不会悄悄删去。

| 设计 | node-sum 抽取，面积优先选择 | node-depth 抽取，延迟优先选择 |
|---|---:|---:|
| c17 | 0.37 / 66.87 | 0.56 / 64.97 |
| c432 | 15.25 / 657.97 | 20.07 / 529.42 |
| c1908 | 30.21 / 604.93 | 40.88 / 516.70 |

两种模式均使用原生 5 iterations、10 samples、random probability 0.8；
runner 原有上限 10 秒、200M nodes，外部每进程 60 秒。60 个最终候选
均通过 CEC。随机 thread_rng 无外部 seed，未做重复运行统计；不能用
单次样本宣称算法稳定性。报告为 `emorphic_random_area_20261012` 与
`emorphic_random_delay_20261012`。这是官方 random extractor + 公共
后端的适配，不是完整论文 SA + golden-delay tuning 流程；原生 SA
还会使用原库和 sizing/stime，需单独解决评分库口径，不能替换名称
声称同库正式复现。普通 mapper 与 emapper 默认策略本轮未改动。

比较门槛新增 emorphic_ 标签，同时修复同名方法不同结果被后报告
覆盖的问题：相同指标/状态观察合并但保存所有来源，不同 cover 或
失败记录全部保留。三个有意义回归测试通过。加入两个 E-morphic
配置后，resyn2 + egraph 的 c432 面积优先胜 16/17、延迟优先 15/17；
c1908 面积优先 14/16、延迟优先 16/16。Pareto 分别为 9/17、4/16，
其他五设计仍缺候选，因此整体三种门槛均不通过；这些数量是配置
观察，不能冒称独立工具数。见 `emorphic_comparison_20261012/`。

下一实验：建立同预算、同 prepared AIG 的 E-morphic random 对照，
再评估保留原结构与 resyn2 结构的安全跨输出候选恢复；完整 SA 流程
需保留原评分库限制，不能将上述局部结果冒充完整 baseline 支配。
DAC 初稿新增此真实范围与限制，主实验表未用随机样本替换。

本轮适配复现入口（先按 audit 固定官方源码和 proto 补丁，Rust 1.88）：
```bash
cargo +1.88.0 build --release --locked --manifest-path /tmp/emapper-emorphic-baseline/e-rewriter/Cargo.toml
cargo +1.88.0 rustc --release --locked --manifest-path /tmp/emapper-emorphic-baseline/extraction-gym/Cargo.toml -- -L native=/tmp/emapper-esyn-link
python3 scripts/experiments/bench_emorphic_baseline.py --source /tmp/emapper-emorphic-baseline --output /tmp/emorphic-area-repeat --benches c17 c432 c1908 --objective area-delay --iterations 5 --samples 10 --random-prob 0.8
```
`emorphic-restore-proto.patch` 仅增加遗漏的官方协议文件；原生阶段预算
不同于 emapper，不据此宣称等端到端预算速度或 QoR 优势。正文 6 页、
参考文献另 1 页已重新编译并逐页目视确认，无字号/边距压缩。

## 用户确认：采用最终 SA 方法，固定 resyn2（2026-10-12）

用户明确指定 E-morphic 只采用最终方法，逻辑优化脚本固定为 resyn2。
当前主对照改为 `emorphic_final_sa`，旧 random-only 结果只作历史审计，
不再作为完成最终 baseline 的替代证据。没有删除或重新命名旧结果。
新增 `bench_emorphic_sa.py` 与透明 ABC 归档启动器
`capture_emorphic_abc.py`，源 SA 搜索/接受/退火逻辑未修改。

公共前处理固定为 balance/rewrite/refactor 的 resyn2 序列。c432/c1908
直接复用 `egraph_resyn2_area_20261012` 的已 CEC 验证 AIG，hash 一致；
两目标报告的 prepared 文件仅 ABC 注释时间戳不同，逻辑/I/O 字节一致。
c17 用相同序列生成 subject 并独立 CEC。输入转换保持共享关系并
记录 EQN/hash；不再采用论文脚本里不同的前处理链搜索。

最终方法保留上游 5 rewriting iterations，10 初始样本、6 轮 SA、
每轮 10 样本、probability 0.8、上游 thread_rng。每个内部候选调用
固定 `dch; map; topo; upsize; dnsize; stime` 评分链；该链是最终方法
的固定内部反馈，不是额外搜索 ABC 脚本。评分 Liberty 从官方库保留
与公共 3-input genlib 同名的 17 个真实单元，常量用连接表达。全局
设置/模板和原 NLDM 表保持，故内部 Liberty loaded score 与公共
genlib 无负载 block delay 模型不同，不能用内部 stime 数字作胜出
依据。也不宣称完整论文库/预算/前处理的逐数值复现。

| 设计 | 最终 SA 所选公共面积 / 延迟 | 内部候选独立 CEC |
|---|---:|---:|
| c17 | 0.60 / 47.86 | 70 / 70 |
| c432 | 20.36 / 365.90 | 70 / 70 |
| c1908 | 37.92 / 309.25 | 70 / 70 |

选中网表严格使用上游 SA 的保留文件；不从 70 个候选中事后挑选公共
指标最好者来替代最终算法。透明 ABC 启动器保留所有命令、原始 EQN、
stdout/stderr、原始网表及 hash，避免原生 cleanup 删除审计证据。
最终所有 210 个候选通过同库网表指标与原始 named-port CEC。首轮
对接因 Yosys 未选中模块而端口重命名失败，修正为 `cd mapped` 后
运行成功；`emorphic_final_sa_20261012` 的失败报告与日志仍保留，
它不是算法等价失败。正式结果目录 `emorphic_final_sa_resyn2_20261012`。

仅使用共同 resyn2 的控制和最终 SA 作本轮活跃比较，历史不同前处理
报告不混入这套新门槛。c432 area-delay 胜 4/5、delay-area 3/5；
c1908 area-delay 胜 4/5、delay-area 4/5；两者 Pareto 均 0/5。
c1908 先前的局部 delay-area 胜出已经被最终 SA 更快的结果推翻：
emapper 是 23.96/337.47，最终 SA 是 37.92/309.25。c432 emapper
11.94/378.33 也比最终 SA 慢。整体门槛仍失败，c17 还缺这套候选，
其他设计未覆盖，不能称为全面胜出。EqMap/E-Syn 后续也需接入相同
prepared subject 才计入固定 resyn2 的新范围，emap 独立对照仍缺。

复现入口：
```bash
python3 scripts/experiments/bench_emorphic_sa.py --source /tmp/emapper-emorphic-baseline --output /tmp/emorphic-final-repeat --prepared-report results_v2/egraph_resyn2_area_20261012/report.json --benches c17 c432 c1908
```
下一轮在固定前处理和预算下改善 critical-path cover，同时恢复共享
降低面积；不再调 ABC 脚本，也不继续用弱随机方法充当最终 baseline。

补充完成 c17 的共同 resyn2 对照：两目标各 ABC amap、普通 map、
egraph 共 6 个网表通过 CEC。area-delay 为 0.37/44.74；delay-area
为 0.53/42.76，与普通 delay mapper 相同，严格字典序不能算胜出。
三个设计的真实报告合并为 `fixed_resyn2_area_delay_20261012` 与
`fixed_resyn2_delay_area_20261012`，均保留来源报告 hash，不产生新测量。
area-delay 对照胜出数 c17 3/4、c432 4/5、c1908 4/5；delay-area
为 3/4、3/5、4/5；严格 Pareto 门槛也失败。当前这一固定前处理
范围覆盖三例，距离原七例套件还缺四例，不以覆盖三例宣称全任务完成。

公共驱动 `bench_egraph_baselines.py` 现在默认 --preprocess resyn2，
默认方法不再包括自带额外 resyn2 的历史 `abc_resyn2_nf`；若显式把
它与共享 resyn2 一起指定，会提前报错，要求改用 abc_nf_delay。
历史原输入报告复现需显式 --preprocess none；没有改动旧记录或
mapper 的默认权重/恢复策略。

## 固定协议的完整既定套件（2026-10-12）

本轮按用户要求完成所有既定设计/库组合：3-input ASAP7 下七设计，
4-input 下 c432/c499/c880 三设计。每个方法复用同一已 CEC 的
resyn2 AIG，两个库分别比较。独立 emap 驱动启用多输出匹配，
没有调用 map_egg；源码/编译 flags/二进制及 vendored commit hash
见 `fixed_complete_20261012/emap_provenance.json`。所有实验保留旧脏
工作和旧报告。

正式报告：
- `fixed_all_area_20261012` / `fixed_all_delay_20261012`：10 个外部
  mapper 配置与 egraph/无 SDC；七例两目标共 98 个成功网表记录。
- `fixed_4in_area_20261012` / `fixed_4in_delay_20261012`：三例共 42。
- `fixed_all_eqmap_20261012` / `fixed_all_eqmap_delay_20261012`，及
  对应 fixed_4in_*：面积与 --min-depth 两个配置，包含 direct ABC
  与 LUT staging control；修复后七例 42、三例 18 个成功记录。
- `fixed_all_esyn_20261012` / `fixed_4in_esyn_20261012`：固定共享
  DAG、原模型、2s/20K/16 iteration 有界 adapter，分别 210/90
  原生候选通过公共测量及 CEC。两种目标从同一个上游已排序候选池
  选择，不增加原生搜索预算，不声称完整原论文复现。
- `fixed_all_emorphic_20261012` / `fixed_4in_emorphic_20261012`：
  最终 SA 方法，固定原生 5 iterations、10 初始 +6*10 邻居、
  probability .8；490/210 候选全部通过 CEC，最终网表保持原生
  SA 选择，不事后用公共指标重选。内部 Liberty score 与公共
  genlib block metric 区分；随机未设外部 seed，非统计优势结论。

合计 1,200 个成功网表测量记录（包含重复控制和内部候选，不能称为
1,200 个电路）。这完成了既定套件的 baseline 数据覆盖；不是对
工业多输出库、ODC 或原论文完整 vendor 流程的完成声明。

保留两类接入失败并无搜索重跑修复：EqMap c2670 --min-depth 会输出
MUX 和 INV，原 LUT-only 模型/计数缺少支持，随后使用官方 simlib.v
及独立 primitive 数量/深度恢复。原生 eqmap.v hash 与修复报告一致；
记录 `fixed_eqmap_delay_c2670_repair_20261012` 的首版 INV pin 错误及
`...repair2...` 的成功复验。E-Syn c2670 30 个候选触发 Python 深度
限制，改用迭代 parser 与 binding 收集；`fixed_esyn_c2670_repair_20261012`
直接复用所有原候选，逐一比对 name/hash。所有旧失败仍在原报告。
collector 仅在原输出 hash 一致且独立 CEC 通过时采用修复记录，
原失败嵌入 repaired_from 并单列 inventory，未丢弃算法坏案例。

`collect_fixed_results.py` 生成 `fixed_complete_20261012/`：全部配置、
来源报告 hash、修复审计、覆盖 inventory、两目标与两套 Pareto 门槛。
面积优先只有 3-input 的 c1355 对全部已连接配置严格字典序胜出；
延迟优先和 Pareto 没有设计对全部配置胜出。4-input 三例均未全胜。
因此 emapper 仍未全面支配 baseline。关闭 SDC 的 c880 从
18.81/347.47 改善到 18.45/334.08，c1355 延迟也改善；如实保留
这些有限预算下的退化，不把 CEC 当作 SDC 一定提高 QoR 的证据。

附加 emap 合成机制控制 `fixed_multi_baselines_20261012` 与实际库分开：
FA area 模式 2/7、一物理实例；delay 模式 13.60/5.00，无 FA；
独立三输出两模式均 0.60/0.20，三个 INV。四网表均 CEC。合成测试
沿用原始夹具，仅证明语义/覆盖，未混入固定 resyn2 的 ASAP7 统计。

DAC 正文主表切换为本轮固定协议结果，新增完整 adapter 表、emap
对照、SDC 退化及失败修复限制；CSV 保留全部配置。当前正式范围
是真实单输出库子集和合成多输出机制，工业多输出收益仍未证实。
并行实验的 wall time 保存，但不据此宣称严格速度优势。

本轮最终编译确认正文 6 页、独立参考文献 1 页，逐页目视检查表格、
分页与公式；无 Overfull 或未定义引用。表数据、CSV 与来源 hash
同步验证完成。

## 可选事件驱动 flow 恢复（2026-10-12，07:23 UTC heartbeat）

新增 `EGM_FLOW_WORKLIST=1`（默认 0），将两处 flow seed/活跃引用
重估改为信号变化触发物理消费者重估的队列。原来最多 32 次全局
扫描，逆序 80 级链仍有根未到达；新队列在相同 deadline 下到达
全部根，并分别遵守面积/延迟字典序。正面积和 depth tie、完整
cover 验证仍保留。候选/规则/SDC 策略未改，恢复预算仍 2 秒。

同一 release 二进制下先检查 c432/c1908 的开关对照，再扩展其他
五设计，全部复用 `fixed_all_area_20261012` 的 resyn2 输入。报告
`worklist_area_20261012` / `worklist_delay_20261012` 含两目标各 4
网表；`worklist_area_expanded_20261012` / `worklist_delay_expanded_20261012`
各含另外五网表。18 个网表均通过公共测量和 named-port CEC。

| 设计/目标 | 默认面积 / 延迟 | 可选 worklist |
|---|---:|---:|
| c499 area-delay | 22.86 / 308.36 | 22.82 / 307.35 |
| c1355 area-delay | 22.78 / 306.85 | 22.78 / 295.60 |
| c432 delay-area | 11.94 / 378.33 | 11.74 / 378.33 |
| c2670 delay-area | 42.45 / 263.14 | 42.22 / 263.14 |
| c880 area-delay | 18.81 / 347.47 | 18.88 / 350.28 |
| c1908 area-delay | 18.33 / 466.80 | 18.34 / 466.19 |
| c1355 delay-area | 30.78 / 273.08 | 30.88 / 273.08 |

`worklist_complete_20261012` 是带原报告 hash 的汇总；七设计覆盖
完整，对 `fixed_complete_20261012/baselines_3in.json` 检查。面积
优先现在 c499 和 c1355 分别胜 17/17 配置；其余五例未全胜。
延迟优先与 Pareto 没有设计全胜。字典序结果不可称双指标全面
支配。由于存在真实回退，worklist 保持可选，不改默认策略，不
把局部更好结果按设计拼接当作新的统一 mapper。

新增深链正反目标回归；格式化后重新构建，mapping 回归包含
之前可选 CEC 测试。实验原始 manifest 的 binary/source hash
保留，后续格式化不重写旧记录。下一步：在原 2 秒恢复预算内
评估保留默认 incumbent、再尝试 worklist 的策略；只能用完整
物理 cover 和实际目标决定接受，不能先运行两次各 2 秒再取最好。

本轮验证最终为 23 项 mapping 回归全部通过（含实际 CEC 正例/
破坏网表负例）。正文仍 6 页，参考文献另 1 页，逐页目视检查通过；
`worklist_complete_20261012/validation.json` 保存日志和最终源码/
二进制 hash。

## 主对照改为 ABC map（2026-10-12，用户指定）

主常规对照统一使用固定 resyn2 后 ABC map -a（面积）/map（延迟）。
独立 emap 两模式及 EqMap、E-Syn、最终 E-morphic SA 保留为主
方法对照；普通 mockturtle map、ABC amap/nf、amap+mfs3 和 LUT
staging/direct-amap 控制进入补充范围。该范围调整来自用户，不
是为了丢掉更强结果；完整原始表/CSV/全配置检查均保留。

`abc_map_primary_20261012` 保存新的主范围 baseline 文件、来源报告
hash，以及默认/可选 worklist 的两目标/Pareto 比较命令和结果。
沿用已 CEC 验证的真实原始 map/map -a 网表，没有把 amap 或 mfs3
结果改名为 map，也没有无依据重新运行相同实验。默认仍只有
3-input c1355 面积优先对全部主配置胜出，worklist 下 c499/c1355
胜出；延迟优先与 Pareto 均未全面胜出。4-input 三例也未全胜。

论文主表同时列出 ABC area/delay、emap area/delay、eGraphMap 两
优先级。amap+mfs3 和普通 map 不再出现在主表。EqMap 的 LUT
组合控制仍如实标注实际 amap 后端；E-Syn 也保留已测 amap 后端，
没有将尚未执行的 map 后端转换冒充既有数据。正文缩减实现调试
过程和投稿计划措辞，改为方法、实验观察、适用范围与讨论；
复现日志、失败修复和完整来源仍留在实验记录中。

公共驱动默认方法同步缩减为 ABC map/map -a、独立 emap 两模式、
egraph 与 no-SDC。所有补充方法仍可通过 --methods 显式选择。
旧全配置报告复现应使用其 manifest 中原始方法列表，不能用新的
默认列表假称已经复现旧的 12 配置实验。

主对照更新稿编译为正文 6 页 + 参考文献 1 页，七页逐页检查
通过，无 Overfull 或未定义引用；来源/主表选择断言和 3 项
比较门槛回归通过。验证记录在 abc_map_primary_20261012/validation.json。

## 单一截止时间的 incumbent 保留（2026-10-12，08:18 UTC heartbeat）

新增可选 EGM_FLOW_COMPARE=1（默认 0）：先运行正常恢复，再把
同一个 Instant deadline 传给 worklist 恢复，绝不为第二阶段
重新开始 2 秒预算。候选集合、饱和预算、库和 resyn2 输入不变；
只有完整合法 cover 按实际配置目标改善时才替换 incumbent。
原 EGM_FLOW_WORKLIST 仍表示直接使用工作队列，两个参数均不
默认启用。CBC 的两阶段字典序证书规则未改动。

主套件七例，两目标，各 egraph/flow_compare 同二进制配对共
28 网表 CEC；四输入三例补充共 12 网表 CEC，总计 40，全部通过。
报告 flow_compare_area_20261012 / flow_compare_delay_20261012 与
flow_compare_4in_area_20261012 / flow_compare_4in_delay_20261012。
加入零预算、过期 deadline 不重启、joint AND/OR 输出共享面积
回归；默认策略不变。

| 案例/目标 | 默认 | compare |
|---|---:|---:|
| 3in c499 area-delay | 22.86/308.36 | 22.82/307.35 |
| 3in c1355 area-delay | 22.78/306.85 | 22.78/295.60 |
| 3in c432 delay-area | 11.94/378.33 | 11.74/378.33 |
| 3in c880 area-delay | 18.81/347.47 | 保留默认 |
| 3in c1908 area-delay | 18.33/466.80 | 保留默认 |
| 3in c2670 area-delay | 34.71/361.56 | 34.69/371.70 |
| 3in c2670 delay-area | 42.45/263.14 | 保留默认 |
| 4in c499 delay-area | 27.74/270.97 | 27.66/270.97 |

全部十个设计/库组合两目标按字典序均不差于本轮同二进制
默认 cover；不等于两个指标均不差。c2670 area-delay 降面积
但增延迟，保留此折中，不用 Pareto 表述。c2670 delay-area
没有在剩余预算内收获原独立 worklist 的 42.22 面积，说明
该策略确实共享时间而不是两次各 2 秒后择优。最大报告恢复
wall time 为约 2.00027 秒，包括截止后的返回/最终指标计算；
搜索使用一个 2 秒绝对 deadline，没有为候选重置预算。

flow_compare_complete_20261012 保存 incumbent 检查、主 ABC-map
范围与原全部配置两种范围的面积/延迟/Pareto 比较及命令。
3in c499/c1355 面积优先均胜全部主配置以及 17/17 原配置；
其余设计未全胜，延迟优先/Pareto 仍没有全部 baseline 胜出。
4in 三设计也未全胜。未扩展到工业多输出库/ODC，不宣称任务
完成。正文新增本轮客观消融，主表仍是默认实现。

下一实验优先改进 c432/c1908 的实际 cut/实例候选覆盖，结合
critical-path 调整而非继续重复搜 fanout 参数；保留同 resyn2、
同库与 2 秒预算。先分析根的最快候选和完整覆盖之间的差距，
避免仅靠不同搜索顺序声称候选 completeness 得到提高。

本轮最终验证：26 项映射回归通过（0 失败、0 忽略），release
构建通过，原有编译警告保留。40 个同输入/同库配对网表全部
通过独立 CEC。论文重新编译并逐页检查，正文 6 页 + 参考文献
1 页，无裁切、重叠、Overfull 或未定义引用；证据和来源哈希
归档于 flow_compare_complete_20261012/validation.json。

## 关键路径与 bounded cut 排序（2026-10-12，10:56 UTC heartbeat）

新增两个可选参数，默认均关闭：EGM_CRITICAL_RECOVERY=1 按当前
最长路径上的局部预测收益排列恢复提案（延迟优先或硬延迟约束
未满足时）；EGM_CUT_DEPTH=1 按最早结构边界深度排列可选 cut。
后者不增加 16-cut、32 rounds、实例上限、饱和或 2 秒恢复预算；
mandatory structural cuts 始终保留，结构深度不当作实际库延迟
下界。每个替换依然经过有限 DAG 验证与完整实例网表目标检查。
关键路径分析保留最长路径并列及各输出的功能依赖弧，忽略
无功能依赖的 pin 弧；新增 joint-output/tied-path 和 tiny-cut
保底回归。全部 28 项映射回归通过，包括独立 CEC 与负例。

新增 initial_delay、seed_delay 诊断；seed_delay 仅为见到的完整
种子最小延迟，不是全候选最快覆盖证明。默认延迟优先 c432
和 c1908 最终延迟等于种子延迟，分别 378.33、337.47，两例
均触及 cut 上限。critical 排序的七设计延迟优先和 c432/c1908
面积优先共 18 网表均 CEC，通过但没有 QoR 收益。

cut-depth 同二进制七设计两目标共 28 网表，CEC 全通过。保存
报告 depth_delay_probe_20261012（c432/c1908）、depth_delay_rest_20261012
（余五例）及 depth_area_20261012；两份延迟报告仅在来源、库、
参数与二进制 hash 相同时合并，不重新运行已测探针。

| 案例/目标 | 原默认面积/延迟 | cut-depth 面积/延迟 |
|---|---:|---:|
| c1908 delay-area | 23.96/337.47 | 26.27/330.48 |
| c2670 delay-area | 42.45/263.14 | 44.50/261.30 |
| c499 delay-area | 29.29/270.97 | 30.06/270.97 |
| c880 delay-area | 23.06/254.78 | 23.08/254.78 |
| c1355 delay-area | 30.78/273.08 | 31.27/273.08 |
| c499 area-delay | 22.86/308.36 | 22.86/307.93 |
| c1355 area-delay | 22.78/306.85 | 22.75/308.67 |
| c2670 area-delay | 34.71/361.56 | 34.78/361.56 |

c432 两目标和 c1908 面积优先均不变。延迟收益伴随面积增加，
面积优先 c1355 收益伴随延迟增加，均不冒充 Pareto 改善；
c2670 面积优先及三个延迟优先坏例完整保留。原默认不变。
critical_depth_complete_20261012 保存 46 个对照网表来源、两目标
和 Pareto 的主范围/全部配置门槛、命令和验证日志；c1355
cut-depth 面积优先仍胜全部主配置，但没有全套基线达标。
本轮只检验 3-input 七例，未扩展 4-input 或工业多输出范围。

下一实验：在固定 cut 配额内保留原排名与深度排名的不同
边界，测试少量配额分配以避免丢失面积友好的边界；首先
分析 c1908 新旧边界交集与实际 critical cone，再决定配额。
不靠提高 cuts 或预算、重新选择坏例、或两次完整预算择优。

论文已纳入无收益和回退的真实消融，主表继续使用默认配置。
最终 PDF 编译两次成功，正文 6 页 + 参考文献 1 页，全部七页
逐页检查通过；无 Overfull 或未定义引用。验证及源/PDF hash
保存于 critical_depth_complete_20261012/validation.json。

2026-10-12 用户要求表 I-IV 数据拆列：面积与延迟分别成列，
EqMap 原 LUT 数/深度箭头也拆为 staging 与 EqMap 的独立列。
数值和实验选择不变，表生成脚本同步修改；源数据一致性与
24 行列数检查通过，七页逐页检查完成，保持正文 6 + 参考 1。
重复消融说明精简，所有收益、坏例和限制保留；格式验证记录
在 table_columns_20261012/validation.json。

## 固定 cut 配额混合排序（2026-10-12，11:58 UTC heartbeat）

新增可选 EGM_CUT_DEPTH_QUOTA=N，默认 0；保留 legacy 排序前
budget-min(N,budget) 项，再按结构深度填满其余独立槽位，排名
重叠不浪费配额。EGM_CUT_DEPTH=1 仍等于全部 optional 槽按深度
排序，若同时设置则全深度优先。mandatory cut 不移除，仍为
16-cut/32-round/100000 optional 实例、2 秒恢复、原饱和预算。
默认和纯深度排名行为不变。新回归验证旧前缀、重叠填槽、
零 budget 和 quota 超限不增加槽数；29 项映射回归全部通过。

新增 cut_depth_swaps，只计最终轮同一 optional 池上相对 legacy
前 budget 项的替换数量。c1908 纯深度/混4/混8 为 248/122/162，
c432 为 3/1/1；这是局部池差异，不是跨迭代全候选包含证明。
各报告保留实际 cuts/instances、命令、库、输入和来源 hash。

主七设计两目标分别评估 mix4/mix8；每次只跑一个配置，不按
设计拼接最优参数。c432/c1908 探针附带纯深度同二进制控制。
cut_mix_delay_probe_20261012、cut_mix_delay_rest_20261012 和
cut_mix_area_20261012 共 44 网表全部公共 named-port CEC 通过。
余五例延迟报告与探针仅在库/来源/二进制/参数一致时合并。

| 案例/目标 | 默认面积/延迟 | mix4 | mix8 |
|---|---:|---:|---:|
| c1355 area-delay | 22.78/306.85 | 22.78/304.64 | 22.78/304.64 |
| c1908 delay-area | 23.96/337.47 | 23.60/337.79 | 23.60/337.79 |
| c499 delay-area | 29.29/270.97 | 31.86/270.97 | 32.00/270.97 |
| c1355 delay-area | 30.78/273.08 | 31.21/273.08 | 32.61/273.08 |
| c2670 delay-area | 42.45/263.14 | 43.14/263.14 | 43.14/263.14 |
| c2670 area-delay | 34.71/361.56 | 34.78/361.56 | 34.78/361.56 |

c1355 面积优先是相对默认的双指标改善，但不等于支配全部
baseline：两参数各只在该例面积优先胜全部主配置 9/9，其
Pareto 仍仅 6/9。全套主范围/原全部配置的两目标及 Pareto
均未达标，结果和命令见 cut_mix_complete_20261012。c432 两
目标、c1908 面积优先及其他未列面积优先均不变；坏例完整
保留。纯深度的 c1908 延迟收益未被混合策略保住；默认不变。
本轮未扩展 4-input、工业多输出库或 ODC，CBC 证书规则不变。

下一实验转向库匹配与 critical cone 诊断：比较纯深度实际
使用的边界、被混合排序淘汰的边界及匹配单元，量化无匹配
cut 占用配额。先找缺失关键候选再实现库感知保留，避免
继续无依据扫描 quota，也不增加预算或挑选设计参数。

本轮最终验证：release 构建通过，29 项回归 0 失败/0 忽略，
44 个公共 CEC 网表通过。论文纳入 mixed 排序收益与坏例，
表 I-IV 拆列格式保持；6 页正文 + 1 页参考文献逐页检查
通过，无 Overfull/未定义引用，验证记录与哈希已归档。

## codex-autoresearch：库感知延迟与受限重关联（2026-10-12）

用户明确要求引入 codex-autoresearch。使用本地 skill 的控制脚本，
在 /tmp/emapper-autoresearch-delay-20261012 初始化独立干净分支
codex/autoresearch-delay，完整保留主仓库脏工作，不向主 .git 写入。
run_id=7d4271c54c784ec5918f14587633f881，前台 numeric target 为
max(D_c432/378.3299961090088, D_c1908/330.4799976348877) < 0.999999。
分母分别是本轮前的 c432 默认与 c1908 已测纯深度最佳延迟，
这是“两例都严格降低延迟”的局部目标，不是全 baseline 胜出。
verify 固定 resyn2/3-input 同库/16 cuts/2 秒恢复；guard 为全部
映射回归（含外部 CEC 和 corrupted-netlist 负例）。控制脚本
负责 trial commit、测量、keep/revert 和事件，未手工修改事件。

初次 init 因 Python 缓存污染及库注释中 symlink 路径不同而失败，
按官方 archive 保留 init-error/log 后修复测量设置再初始化。
后续四轮及源哈希、命令、guard、失败代码/回滚 commit 已归档
在 autoresearch_library_timing_20261012/control、history.tsv、
measurements 与 trials.bundle；主项目适配器初次漏启开关的
probe 也保存为 integration_failures.json，不当作新算法结果。

| 轮次 | 假设 | worst delay ratio | 控制决策 |
|---|---|---:|---|
| baseline | 当前实现 | 1.0211510525 | init |
| 1 | 按实际库匹配和结构 arrival 估计排列 cuts | 1.0000000000 | keep |
| 2 | 用 phase-aware arrival relaxation 替代结构估计 | 1.0000000000 | discard |
| 3 | 不限制辅助类的 AND 重关联 | 1.2139615266 | discard |
| 4 | 重关联只用于导入类，限制类大小/新增节点 | 0.9662199696 | keep，局部 target 达到 |

库检查确认 3-input 子库仅一种 INV，排除漏选更快 INV 型号。
库排序单独令 c1908 变为 24.63/318.43，c432 仍 11.94/378.33；
该诊断的非平凡、非空 cuts 中无直接单元匹配数分别为 c432
9、c1908 355。无直接匹配不意味着 cut 无用，它仍可能参与
后续边界组合/SDC；本实现只是更晚排列，不删除逻辑等价式。
在当前 cover arrivals 下，对输入 arrival 均可用的 producer，
c432 critical signals 未观察到单独替换的延迟收益；未覆盖
输入 arrival 未定义的候选，不据此排除其他局部/联合替换。无限制关联虽使 c432 到
13.12/366.57，却使 c1908 到 60.92/401.19，原始候选约 290 万、
runner 节点软限 overshoot，整轮回滚，不据此声称同硬限胜出。
证据支持 c432 缺少有效 cross-branch 边界，而不是缺少已有
边界上的另一个 INV；不证明全库/全边界最优性。

主项目新增 EGM_LIBRARY_TIMING=1，默认 0；延迟优先时启用
exact-library pin-delay + structural-boundary proxy cut 排序，并
仅在 imported classes 应用精确 AND associativity。Applier 用
原 node_limit 与 cuts_per_class 限制添加，禁止辅助类继续递归
展开。它是 e-graph 内的工艺映射候选探索，外部逻辑输入仍只
用一次固定 resyn2。原 SDC/ODC 和 CBC 证书规则未改动。

集成时新增 hard extraction-pool cap：保留每个有限 witness
binding，再按 target signals 轮流填入静态 pin-delay 排列的
producer；不足容纳 witness 时显式报错，不突破上限。临时
生成池可能超过 100000，进入提取的候选严格不超过 100000，
同时记录 generated_instances 与 instance_budget_pruned。c1908
集成前 143326 个候选，截断为 100000，延迟保留 276.11，面积
由实验仓库 37.31 变为 37.99。不是两次恢复择优；恢复仍共享
单一 2 秒 deadline。新 witness/cap 和有限关联功能回归加入
后，主项目 31 项映射回归通过，0 失败、0 忽略。

主七设计、四输入三设计，各两目标各有默认/新配置配对，总
40 网表，公共物理 evaluator 与 independent named-port CEC
全部通过。报告 library_timing_delay_probe2_20261012（两重点例）、
library_timing_delay_rest_20261012、library_timing_area_20261012、
library_timing_4in_delay_20261012、library_timing_4in_area_20261012。
两份三输入延迟报告仅在参数/源/库/二进制 hash 一致时合并。
所有新配置实际 max cut size <=16、e-nodes<=20000、进入提取
instances<=100000，硬限断言保存在 library_timing_complete_20261012。
仍不声称这些配额代表各不同工具完全相同的运行时或搜索空间。

| 输入库/设计 | 默认 delay-area 面积/延迟 | Library delay-area 面积/延迟 |
|---|---:|---:|
| 3in c17 | 0.53/42.76 | 0.53/42.76 |
| 3in c432 | 11.94/378.33 | 13.64/365.55 |
| 3in c499 | 29.29/270.97 | 46.45/235.90 |
| 3in c880 | 23.06/254.78 | 25.98/234.55 |
| 3in c1355 | 30.78/273.08 | 45.69/236.94 |
| 3in c1908 | 23.96/337.47 | 37.99/276.11 |
| 3in c2670 | 42.45/263.14 | 58.20/249.79 |
| 4in c432 | 10.41/324.45 | 15.81/288.01 |
| 4in c499 | 27.74/270.97 | 74.87/230.26 |
| 4in c880 | 23.72/236.14 | 30.90/203.98 |

面积优先配对全部等于同二进制默认。默认策略不变；新配置
是在 delay-area 优先级下接受面积增大的启发式 cover，不是
Pareto 全胜，也不是 bounded-candidate 最优证明。
check_mapping_dominance.py 分别对两目标和 Pareto，主范围以及
原全部补充配置执行，保留 argv/exit/stderr。三输入 c499、c880、
c1355、c1908、c2670 严格 delay-area 胜全部主/补充观察；c1908
是 8/8 主和 16/16 全观察（相同配置观察按原门槛去重）。
c432 仅 8/9 主、16/17 全，仍慢于 E-morphic 352.37；c17 对
普通 delay cover 持平，未达严格门槛。四输入 c432/c880 胜全部，
c499 8/9 主、16/17 全，仍慢于 emap 226.30。所有全范围
Pareto 检查仍未通过；工业 multioutput 库和 ODC 仍未接入。

论文 Table IV 增加独立列的 Library (D,A) profile，Tables I-II
继续清楚呈现默认配置。正文改用实际比较的 lexicographic
目标，纳入库排序、受限关联、硬候选池、面积代价及未胜出的
对照；运行/适配器调试细节留在复现档案，不进入论文主叙述。
下一步：分析 3in c432 相对 E-morphic 的 13.18 delay gap 和
4in c499 相对 emap 的 3.96 gap，重点是有限 cut/phase 选择与
同延迟面积恢复，而非提高预算或在不同设计间挑参数。

最终核验：硬截断后局部 worst-delay-ratio 仍为 0.9662199696，
两重点例均严格改善；40 CEC、31 回归、全覆盖门槛记录齐全。
论文正文 6 + 参考 1，全部七页逐页检查通过，无裁切/重叠、
Overfull 或未定义引用；Table IV 新 profile 逐值核对保存数据，
Tables I-IV 指标拆列保持。生成池最大 490094，提取池 <=100000，
e-node 最大 7759，实际每类 cut<=16。最大恢复 wall time
2.000153815 秒含截止后返回/最终计算，不重启 2 秒预算。
最终源/PDF hash 与验证见 library_timing_complete_20261012/validation.json。

## Required-time 面积恢复（2026-10-12，14:01 UTC heartbeat）

继续 codex-autoresearch 前台工作流；旧 delay run 经 status 验证
complete 后由官方 archive 保存，再快照主项目 hard-cap 实现，
保留主仓库脏工作。新 run_id=a9e93c756a334402809a03916edf93af，
metric=max(A_c432/13.640000104904175,A_c1908/37.99000019580126)，
target<0.999999；verify 另要求两例延迟分别 <=365.5499973297119
和 <=276.11000061035156（1e-6 tolerance），同 resyn2/同库/16 cuts/
100000 retained instances/单一 2 秒 recovery。guard 为映射回归
和独立 CEC。六轮只有最终共同策略 keep，其余五轮 discard；
局部 metric 达到 0.9743401766，不是全 baseline 成功。

| 轮次 | 假设及观察 | metric / 决策 |
|---|---|---|
| 1 | 空状态 required-time flow，无收益 | 1 / discard |
| 2 | 从 incumbent 初始化 flow，两例重建均不合法 | 1 / discard |
| 3 | 加局部防环，c1908 34.68/276.11，c432 未改善 | 1 / discard |
| 4 | 先做完整物理面积局部排名，仅 c1908 到 36.63 | 1 / discard |
| 5 | witness 支持 inactive phase/helper，c432 13.29，c1908 未改善 | 1 / discard |
| 6 | 先防环 seeded flow，再用剩余时间做局部排名，两例均改善 | 0.9743401766 / keep |

控制事件、所有 source/trial/revert commit、verify/guard 日志、
失败测量和 Git bundle 保存在 autoresearch_slack_20261012。
没有改写控制事件或用每设计最优配置拼接目标。第五/第六轮
观察说明恢复顺序和 residual budget 会影响结果，不证明面积
或延迟全局最优；没有增加时间或候选额度。

新增可选 EGM_SLACK_FLOW=1（默认 0），本轮公共方法
 egraph_library_slack 同时启用 EGM_LIBRARY_TIMING=1。从合法
incumbent 反传各 pin 的 required times，使用其真实 physical
refs/arrival/choice 初始化 area-flow 队列；仅允许满足局部
required time 的候选，并在每个替换前检测是否返回自身。
最终仍执行完整 DAG、真实物理面积/延迟、配置 objective 检查。
局部恢复允许借助已验证 finite witness 的 inactive phase/helper
arrival 估计来预排序，再按完整网表验证；估计不是延迟证明。

所有预评分、队列和局部重新验证共享原来的绝对 deadline。
每轮后续局部验证最多 refine_trials=2000；完整物理面积
pre-scoring 不受此次数约束，但计入相同 2 秒时间，并单独
统计 slack_local_scored。这不是“最多 2000 次所有评分”的
承诺；原常规 cut/flow 评分也与 refinement trial 计数不同。
新增 slack_attempts/valid/accepted、slack_local_scored/trials/
accepted 原始计数；c432 累计 12095 次局部预评分、61 次后续
验证、6 次接受，c1908 2 次合法 flow 尝试、1 次接受，局部
阶段未启动。新的非关键根放慢、cheap quotient cycle 两项
回归加入后，33 项回归全通过（0 失败/0 忽略）。

七个 3-input 设计、三个 4-input 设计，两目标分别对旧 library
profile/新 slack profile 配对，总 40 网表公共 evaluator 和
independent named-port CEC 全通过。报告 slack_flow_delay_probe_20261012、
slack_flow_delay_rest_20261012、slack_flow_area_20261012、
slack_flow_4in_delay_20261012、slack_flow_4in_area_20261012。
源/输入/库/二进制一致时才合并三输入延迟报告。所有 optional
pool<=100000、e-nodes<=20000；预算配置和原始计数留在报告。

| 库/案例 | 旧 Library 面积/延迟 | 新 Slack 面积/延迟 | 归因 |
|---|---:|---:|---|
| 3in c432 | 13.64/365.55 | 13.29/365.55 | 局部恢复接受 6 次 |
| 3in c1908 | 37.99/276.11 | 34.68/276.11 | seeded flow 接受 1 次 |
| 3in c880 | 25.98/234.55 | 24.93/234.55 | 局部恢复接受 17 次 |
| 4in c432 | 15.81/288.01 | 15.66/288.01 | 局部恢复接受 3 次 |
| 3in c2670 | 58.20/249.79 | 57.91/249.79 | 新阶段计数均 0，deadline-sensitive incumbent 波动 |

其余延迟配对和全部面积优先配对不变。c2670 不归因于 slack
机制，也不以此证明跨执行的 incumbent 保证；每次执行内
只接受合法且 objective 改善的覆盖。模式仍默认关闭。
check_mapping_dominance.py 对两库、两目标、Pareto、主/全部
补充范围均执行，命令/exit/stderr 在 slack_flow_complete_20261012。
严格 delay-first 覆盖范围仍是 3in c499/c880/c1355/c1908/c2670，
4in c432/c880；3in c432 仍慢于 E-morphic，c17 持平，4in c499
仍慢于 emap。全范围面积/Pareto 仍未全部通过，真实多输出
工业库、ODC 未完成，CBC 两阶段证书要求未改。

论文 Table IV 的第三组独立 area/delay 列改为最终 Slack
profile，Tables I-II 默认结果不变。方法及消融纳入 required
时间、防环、inactive phase、面积减少和 c2670 波动归因。
正文 6 页 + 参考 1 页已逐页检查，无裁切/重叠/Overfull 或
未定义引用；版式字号/边距未压缩。源/数据/PDF 哈希及计数
验证在 slack_flow_complete_20261012/validation.json。
下一步重点检查 4in c499 raw 490094 ->100000 的 cap 是否
丢失关键边界，以及 3in c432 的 13.18 delay gap；在固定配额
内改进 timing/phase-aware producer 保留，不再盲调恢复顺序。


### 2026-10-12 到达时间感知的 producer 截断（可选，默认不变）

codex-autoresearch 新 run 869e688c2ba2468ba7d2aa1675f1fd57，
一轮 keep，四输入 c499 公共 delay ratio 1.000000017 ->
0.948232449；初始化一次因四输入 adapter/来源路径错误失败，
修复后重新冻结，失败诊断与全部命令保留。
控制器/测量/源码/Git bundle 位于
results_v2/autoresearch_timing_cap_20261012；正式40配对网表位于
results_v2/timing_cap_{3,4}in_{delay-area,area-delay}_20261012，
汇总与16项 gate 位于 timing_cap_complete_20261012。

新增 EGM_TIMING_CAP=1，公开方法 egraph_timing_cap 同时开启
LIBRARY_TIMING、SLACK_FLOW；三个开关默认均关闭。之前每个
producer bucket 按最大固有 pin delay 排序，忽略输入到达时间。
新排序分别对目标 pin，仅累加 functional arcs 的 finite-witness
输入到达时间。保留所有 witness instances，再按各 target 的
producer 轮转填满同一100000池。估计不是下界，也不保证比旧
截断更好；没有条件 egg union、库变更或预算增加。

| 延迟优先案例 | 原 Slack A/D | Timing-cap A/D |
|---|---:|---:|
| 3in c17 | 0.53/42.76 | 0.53/42.76 |
| 3in c432 | 13.29/365.55 | 13.29/365.55 |
| 3in c499 | 46.45/235.90 | 45.23/236.63 |
| 3in c880 | 24.54/234.55 | 24.83/234.55 |
| 3in c1355 | 45.69/236.94 | 52.38/237.99 |
| 3in c1908 | 34.68/276.11 | 36.44/276.11 |
| 3in c2670 | 58.09/249.79 | 56.36/244.63 |
| 4in c432 | 15.66/288.01 | 14.25/291.58 |
| 4in c499 | 74.87/230.26 | 71.46/218.34 |
| 4in c880 | 30.90/203.98 | 30.93/203.98 |

全部面积优先配对不变。四输入 c499 原始生成490094、保留
100000，实际仍删390094；gain来自保留候选的组成，而非增大
pool。该例初始 seed delay218.34，新 slack 阶段计数均0，不能
归因于 slack 恢复。3in c2670也降低delay，但其他案例存在回退，
不可按设计混用结果改名成统一赢家。c880/c2670旧profile面积
波动保留，不把无新机制执行的变化归因给新cap。

40网表独立 named-port CEC通过；34项回归通过（包括新晚输入
快速单元与早输入较慢单元的保留测试），pool<=100000、
e-nodes<=20000、cut配置16和原2秒恢复期限保持。临时候选
生成仍可超过池上限，建池不计为恢复时间，未声称总runtime相同。
新profile在4in三个案例均严格 delay-first胜过每个已连接主/补充
baseline；3in仍五个胜出，c432慢于E-morphic、c17持平。
全范围Pareto未通过，3in c1908面积回退也没有被隐藏。

论文保留Table IV原Slack完整行，新增cap方法与改善/回退消融，
来源进入生成器provenance；6页正文+1页参考逐页视觉检查通过，
没有压缩字号或边距。下一步针对3in c432未截断池中的关键
跨分支边界，分析缺失的库匹配；不能指望cap排序解决无截断
实例。真实工业多输出、ODC及所有baseline Pareto目标未完成。


### EPFL主套件迁移：固定resyn2（2026-10-12）

统一benchmark_suites.py声明20例（10 arithmetic +10 random/control），
四个正式adapter默认改为EPFL，显式旧--benches参数仍兼容。
prepare_epfl_subjects.py将技术无关逻辑输入从库中分离，记录ABC
hash、原始AIG hash/header、命令、耗时、失败和prep CEC；
shared_subject.py允许kind=logic_subjects跨库复用，旧library-tagged
报告仍严格检查库hash，所有输入和AIG/recipe/CEC仍验证。
E-morphic统一调用同一复用函数，提供prepared report时不得因
缺少设计行悄悄重跑resyn2。测试已验证20例唯一覆盖、损坏AIG
拒绝和旧库错配拒绝；全部20例prep CEC通过。

run_epfl_campaign.py单实例文件锁、冻结来源hash、逐job命令/日志/
exit/覆盖记录，两个库两objective核心映射及原三个额外baseline
均列为必需job，失败不丢弃。三个/tmp baseline已重新clone锁定
原commit并构建，原proto patch缺少末尾换行导致git apply失败，
保留此构建恢复错误，改从锁定官方PROTO_COMMIT直接恢复同一文件。
background campaign已启动；更新原heartbeat优先级至EPFL，防止
后续自动继续误用ISCAS作为主评估。不是一轮autoresearch，未改
主库.git，也未更改mapper默认策略或扩大任何预算。

迁移首批：ABC map和eGraphMap在adder/bar均通过独立CEC；独立
emap的raw Verilog将a[0]等AIG symbol作为未转义scalar identifier，
导致Yosys syntax error。这是接口输出失败而非emap功能不等价，
原始文件/normalize.log保留。当前campaign配置已冻结，后续修复
需保留此原始campaign并另存adapter-repair或新campaign，不能
在其执行中修改冻结harness并混装结果。EqMap重构建已完成，
E-morphic rewriter已完成；extractor首版缺stdc++搜索路径失败，
已使用实际gcc11原lib目录重建并保留两次日志。E-Syn native
xgboost依赖构建仍在执行。完整EPFL数据尚未齐，不能胜出宣称。

E-Syn也遇到同一stdc++链接目录缺失，已用gcc11原库目录恢复
（esyn-build-2.log）；此为构建适配，不改变native算法或预算。
论文Protocol已标EPFL主评估正在执行、Tables I--V仍为历史ISCAS，
原论文/PDF完整备份在campaign/historical_iscas_paper。迁移版
依然6+1，逐页检查通过，migration_validation.json保留hash。


### 2026-10-13 EPFL中断恢复及端口转义

心跳发现原后台runner文件锁空闲，首job仅10条记录，没有后台
活跃证据。原campaign标interrupted，不删除目录或已有成功/失败，
不继续宣称完整实验在后台执行。改为同turn完成一批、按单设计/
库/配置checkpoint。完整计划240job，已记录job跳过；未记录的
中断目录拒绝覆盖，需要独立审计恢复。冻结hash与输出单实例锁
保留，原预算不变；不依赖短命sandbox子进程跨turn存活。

新port_identifiers.py保留原名并转义EPFL a[0]等indexed scalar。
mockturtle原raw.v不改，只增加escaped.v及前后hash再normalize。
E-Syn输入/输出使用合法escaped identifier，内部p/o/n symbols
不改；E-morphic保留原有p/o表达式，输出rename使用转义原名。
未改变逻辑、native算法或budget。校验转义幂等，以及E-Syn的
EPFL adder DAG到Verilog/AIG named-port往返CEC，E-morphic eqn
接口接受同一原端口，详见epfl_port_validation_20261013。

epfl_checkpoint_20261013首两job adder/bar，各ABC area/delay、
emap area/delay、egraph area，10网表全部公共测量与独立CEC
通过，先前4个emap语法失败完整保留在原批次。新适配路径首次
检验重测此两设计，未声称native算法改进；egraph bar本次D244.68
而原中断批次242.75是固定2秒恢复的incumbent波动，保留全部
观察，不挑更佳者或归因给端口转换。后续批次继续div/hyp。
这只是2/20设计、1/4核心配置的完成记录；不可声称整体覆盖，
更不能把API/语法失败解释成baseline算法不等价或工艺质量差。
论文仍明确EPFL完整表待验证、ISCAS表历史pilot，6+1PDF不以
不完整的EPFL两设计替换全部主表。原心跳已更新为checkpoint
路径及执行入口，下一步按剩余20设计完整覆盖后做公开gate。

后续div/hyp两个checkpoint已结束：共4job、20条mapping attempts、
14条CEC成功和6条失败。div四个ABC/emap成功，egraph失败
"no finite reachability witness for eclass 3624"；hyp ABC/emap四
配置各60秒timeout，egraph同类有限witness失败。完整错误均在
报告，不能删掉hyp或声称baseline/mapper优势；不是整个算法
等价失败。EPFL完整240job仍未完成，当前checkpoint锁空闲。

根因定位至egraph_sat.rs::cheapest_nodes：展开树的usize成本可
在深度共享DAG上saturating_add到MAX，而 `c < best(MAX)` 没有
接受首个合法候选，误判成没有有限DAG。隔离副本添加75类
重共享链回归，旧代码仅识别64类；修复将MAX视为成本tie而
不是不可达，允许首个可用候选，已有选择仍仅严格降低成本时
替换。新回归通过，34项映射回归全部通过（含外部CEC）；
整crate测试有3个旧synth长测试超过60秒已中断，未报全面通过。
补丁/源码/测试日志在epfl_port_validation_20261013，独立副本
/tmp/emapper-witness-repair-20261013；正在同输入/库/预算验证div。
主frozen源码/二进制未改，不把这个修复悄悄混进现有checkpoint。
后续修复集成必须生成新的明确版本矩阵，保留本次20条成功和
失败及原始中断批次，不依设计择优拼成一个mapper版本。

隔离div修复试验结束：不再作为unreachable立即退出，但映射阶段
仍触发原60秒timeout，报告epfl_witness_div_20261013。尚未生成
合法公共CEC cover，因此只证明深链unit witness回归修复，不能
宣称EPFL div已解决、PPA改善或baseline胜出。修复未集成main。
下一实验以依赖worklist替代cheapest_nodes的整图反复松弛，先
在隔离副本验证相同预算下的真实div cover；同时剖析cuts/generate
是否成为新瓶颈。不要无改变重跑这个60秒失败或提高预算。
已有baseline及失败保留，论文EPFL主表仍待完整数据，不以
14条成功掩盖6失败。没有后台活跃承诺；当前turn所有有限批次
均结束并checkpointed，后续心跳使用新入口明确推进。


### 2026-10-13 两级witness依赖队列与预算审计

旧冻结EPFL版本新增log2/max两检查点，现6/240 job、30条尝试、
23条原始CEC成功、7失败。log2四常规配置通过而egraph有限
witness失败，max五配置原始CEC通过。原始CEC数量不等于
预算合规数量；budget audit在epfl_worklist_diagnostics_20261013。

隔离cheapest_nodes采用父依赖队列，饱和usize回归及34映射测试
通过；div同60秒仍超时。增加阶段计时得到：import0.08s、
saturation0.09s、witness0.17s、cut完成11.62s、generate完成
32.77s。不是无变化重复重跑，定位到后续构造/提取瓶颈。
finite_choices也全池重复扫描；改为area稳定排序的rank队列，
新依赖就绪且rank已过时入下一轮，未过时入当前轮。128个
依赖图逐binding与旧逐轮实现一致，35映射回归全部通过（0忽略）。

新的div队列trial在约47秒产生raw网表，但公共evaluator触发60秒
timeout，最终仍失败，未有independent named-port CEC。实际
保留5375361候选、enodes81802、recovery6.189s，虽配置仍是
100000/20000/2秒，却不满足实际预算，不能宣布同预算改善。
原默认仅在library_timing路径硬cap，正常generate soft cap仍能
积累大量结构fallback；大输入本身也已超e-node allowance。
这些明确列为合规缺口，不将import超限悄悄改称硬20000满足。
验证sidecar审计全部当前原始成功rows，不删掉overbudget网表。

两个修复只在独立/tmp/emapper-witness-repair-20261013，主版本和
冻结campaign未改，源码/diff/report/所有失败计时均归档。
下一实现先在保留有限cover的同时真正落实retained-pool cap和
恢复绝对deadline（包含mandatory setup），必要时明确返回
budget failure；不能延长预算或以source hash不同的设计择优混表。
再剖析cone/metric及公共evaluator性能。DAC仍明确EPFL完整结果
待验证、ISCAS历史pilot；当前失败/预算缺口不支持替换主表或
全面胜出声明。当前有限批次均结束，后续按checkpoint继续。

预算sidecar审计还发现当前adder/bar/max的旧默认候选池也超过
100000，因此23个原始CEC成功不能当作23个预算合规结果，
这些3个egraph默认观察不可纳入同硬预算优胜声明。公共数值
和CEC仍完整保留，合规判定另列，不篡改原始成功/失败。
DAC Protocol已改成nominal generation threshold，Discussion
明确default fallback超cap、setup overrun和import超allowance。
编译6正文+1参考，7页逐页视觉检查，无字号边距压缩；hash
和PDF检查记录进入worklist diagnostics validation.json。


### 2026-10-13 严格预算admission prototype（仍在隔离版本）

新增opt-in strict_budgets/EGM_STRICT_BUDGETS，方法名
 egraph_bounded_resyn2。冻结main默认不变；代码/补丁在
results_v2/epfl_strict_budget_20261013，独立源码仍/tmp/
emapper-witness-repair-20261013。严格模式不暗启LIBRARY_TIMING、
SLACK_FLOW或TIMING_CAP。导入及饱和图均检查实际node数，
不能把import超限改称20000满足；全部候选路径执行同一cap_instances
有限witness保留，见实际instances，而非只看max_instances配置。
恢复原2秒总期限包含mandatory witness/prune/metric/final accounting。
在预算内部预留0.05秒做返回验证，未增加2秒；若任何原子阶段
实际跨总期限则明确budget failure，不提交一个超时incumbent。

新增node-floor/零恢复预算拒绝回归，修正测试构造器Debug编译
问题日志保留；最终36映射回归通过（含独立CEC）。两个目标
各4例：adder/bar/max共6网表实际pool=100000、enodes<20000、
recovery<2秒，独立named-port CEC通过；div导入81802节点，两
目标都立即明确budget failure，未做同预算胜出声明。

| EPFL案例 | Strict (A,D) 面积/延迟 | Strict (D,A) 面积/延迟 |
|---|---:|---:|
| adder | 65.01/3701.61 | 76.83/3016.20 |
| bar | 221.79/252.49 | 298.45/191.85 |
| max | 284.38/3373.74 | 295.43/2735.12 |


这6例仍未胜过全部已测ABC/emap配置，不称PPA改进。20-design
coverage报告保留未测16设计的explicit missing rows，baseline
8检查点加其余missing记录；EqMap/E-Syn/E-morphic尚未测，
required-baselines也明确缺失。四个check_mapping_dominance gate
（两objective+各pareto）全不通过，完整命令/exit/输出保存。
原冻结legacy新增multiplier/sin两个job，各四baseline通过而
默认egraph失败；累计8/240 job、40attempts、31raw CEC、9失败。
raw CEC仍不等于budget合规，旧所有坏例和overrun原样保留。

DAC加入严格profile六个合规CEC、div明确拒绝和全部常规比较
仍未胜出的限制。完整EPFL主矩阵未齐，ISCAS主表仍标历史pilot，
不能以三设计的成功替代全20 coverage。编译6+1且7页逐页QA，
首次8页尝试和修正日志保留，未压字体/边距。下一步优先规划
一个明确严格版本全20矩阵，并研究对大输入分区而不是提高
全局node budget；先估算最小物理witness能否在100000池和2s
恢复内实现，超限继续作为失败。保持现有完整来源及baseline
缺失范围，不逐设计挑源拼赢家。


### 2026-10-13 严格版本80-case矩阵及大图结构估算

新增run_epfl_strict_campaign.py独立矩阵：同一冻结isolated root，
两库*两目标*全20设计=80job，逐设计checkpoint和文件锁。
来源冻结全部emapper Rust、Python helper、mapper/evaluator二进制，
每次入口核验；旧area/delay probe仅source、binary、输入hash、
objective/strict flag匹配时采用，失败观察也采用，不重跑挑优。
状态results_v2/epfl_strict_checkpoint_20261013/campaign.json。
此为真实prototype代码矩阵，不是把旧不同版本case改名拼表。
每批在同一turn等结束，不能依赖后台跨turn存活；默认主库未改。

analyze_epfl_structural_budget.py验证20个prep CEC/hash/recipe后
解析二进制AIGER delta，反向追live cone，计算一个原结构AND2/
NAND2+共享INV witness实例估算，以及拓扑region的boundary/
dual-phase节点上界。delta/极性/分区边界/拒绝自环编码测试通过。
结构报告epfl_structural_budget_20261013，不是工艺映射或CEC
证明，不是最小cover下界。没有重新运行resyn2或实际分区mapper。

div结构witness估算52907个primitive（排除ties），可分为5个
局部peak<=20000节点的region，但累计上界83970，不等于全局
20000预算通过。hyp结构估算296207 primitive，23个region累计
443872，简单结构witness超100000；这不证明逻辑优化后的全局
cover不可能。必须有更紧匹配/共享才可能入池，不能给每region
重置100000池或2秒恢复、16迭代预算以宣称同预算胜出。
完整20例节点/阶段/实例估算逐行保留。下一步先统一全局
候选池、全局恢复deadline及全图iteration accounting，试行
小region的边界/共享保持并最终整图CEC；尚未实现此功能。

本轮old冻结矩阵新增sqrt/square两个job后10/240、50attempts、
40raw CEC、10失败。sqrt默认egraph witness失败，square默认
原始CEC通过但仍需预算审计。严格isolated矩阵10/80已记录，
adder/bar/div/max旧probe在匹配hash时直接采用；新增hyp/log2/
multiplier/sqrt/square明确node budget失败，sin area-first通过
公共CEC，A/D525.00/3271.53、pool100000、nodes15168、
recovery1.9524秒，依旧劣于常规配置。现已知7个严格合规
原始网表（三设计两目标+sin面积优先），没有混入旧overbudget。
四gate以完整20 missing scope重新跑均false，结果
 epfl_strict_progress_region_20261013。

结构估算补上每个局部AIG固定Zero/NotZero import pair；此前
constant-free boundary的估算少2节点。旧report保留，修正版
 epfl_structural_budget_constant_20261013，准确8节点region及
6节点不足测试通过。峰值估算只是必要条件，还需给真实
rewrite留余量和守住累计候选/iteration/shared deadline，未有
实际分区映射或PPA/CEC/最小实例证明，不报大图预算可行已解决。

严格模式已作为显式选项接入main：EGM_STRICT_BUDGETS=1，
Python方法egraph_bounded。默认保持原cheapest_nodes、原
finite_choices、原规则和全关选项；新queued helpers只在严格
模式接入。pool cap对严格所有路径生效，节点及恢复admission
保留，并补上实际mandatory cuts的per-class硬检查。
CLI新增max_cuts_per_class统计，不靠配置16假定实际16。
主36映射回归和饱和成本单项回归通过，release build完成；
根source/binary改变后旧frozen矩阵封存不再在新版本续跑。
备份/sha/改动说明在epfl_strict_main_20261013，未动主.git。

新main严格矩阵epfl_strict_main_checkpoint_20261013单独从头
评估，不采用不同source的isolated7例。首adder/bar两job均
CEC通过，pool100000、actual maxcuts16、nodes3572/9696、
recovery1.4871/1.9509秒；A/D65.01/3701.61及221.79/252.49。
矩阵2/80，仍非完整数据或全baseline胜出。isolated矩阵仍10/80
原版本保存，不选择哪个case更佳来拼统一mapper。
论文更新为7个隔离profile有效初步covers，6+1全部7页QA；
新的main2例不与隔离7例重复累加成9设计或一次矩阵。
下一步优先推进main严格80-case、独立baseline同输入同库完整
覆盖，研究peak-limited region的全局pool/deadline/iteration
accounting及边界共享保持，再做全图named-port CEC。

### 2026-10-13 EPFL 独立基线矩阵与首个 EqMap 结果

新增 scripts/experiments/campaigns/run_epfl_baselines.py，避免改变
main严格矩阵已冻结的root Python helpers。两库全部20设计，每个
设计/库6个检查点（core四配置、EqMap两目标、E-Syn两目标、
E-morphic最终SA），共240job。所有方法复用一次resyn2 subject；
冻结原生tracked源码、commit、模型、修补proto、二进制、公共
evaluator、库源及harness。入口验证fingerprint和完成报告hash，
锁定后顺序运行有限批次，部分中断输出拒绝覆盖。240唯一job和
每设计/库6配置的plan验证通过；未降低或扩展原native预算。
core只跑一次四固定配置，不重复逻辑优化或按objective择版本。

新矩阵 epfl_baseline_checkpoint_20261013 已3/240job：3in adder
core四配置、EqMap area-first三配置共7个公共CEC通过；E-Syn
area-first原生process60秒timeout，失败全保留。

| adder 3in 方法 | 公共面积 | 公共延迟 | 独立CEC |
|---|---:|---:|---:|
| ABC map -a | 57.40 | 3548.84 | 1 |
| ABC map | 80.71 | 2585.48 | 1 |
| independent emap area | 57.38 | 3548.84 | 1 |
| independent emap delay | 90.33 | 2585.46 | 1 |
| ABC amap control | 65.07 | 5212.79 | 1 |
| LUT stage + amap control | 196.61 | 7536.54 | 1 |
| EqMap LUT + amap | 230.62 | 7495.94 | 1 |

EqMap仅首个area-first EPFL设计，跨目标适配不是原生ASIC mapper，
LUT转换的损失已由staging control隔离；不宣称完整baseline齐。
E-Syn日志显示Runner NodeLimit(20640)，饱和仅57.097ms；后续
62份dot导出/原生graph_info.py的64进程pydot特征计算阶段未在
60秒内完成（尚不能精确分解导出与解析各自耗时）。这不是
饱和用完2秒，20k也是原生阈值而非硬上限。原生按as_completed
重编号的既有候选索引问题仍保留；不通过延长预算隐藏失败。

main严格矩阵新增8job后10/80：area-first adder/bar/max/sin
四例公共CEC通过，div/hyp/log2/multiplier/sqrt/square均node
budget failure。max本次285.15/3373.74，不采用isolated旧284.38
作本版本赢家。log2/max batch与新core/EqMap adder batch存在
时间重叠，保留原测量并明确wall-deadline受机器负载影响；不据此
声称隔离运行的恢复/执行时间改进。之后不同campaign也顺序运行。
新汇总 epfl_main_baseline_progress_20261013 对两库/两目标保留
全部20缺失/失败记录，8个lex/Pareto gate均false，原生E-morphic
和EqMap delay尚未接齐，失败不改名为有效结果。

DAC讨论更新为main四例及EPFL EqMap初测/E-Syn阶段性timeout，
旧ISCAS主表继续明确历史pilot；完整EPFL尚未齐，不填主表假数。
保留8页排版尝试，精简文字后恢复标准6正文+1参考，逐7页视觉
QA通过；未压缩字体/边距。下一批先顺序推进baseline matrix
余下adder配置和main控制类，不混合源码或重新跑resyn2。

### 2026-10-13 EPFL 20设计首完整profile及E-Syn特征修复探针

main严格冻结矩阵20/80：3in area-first全部20设计已尝试，
11公共CEC通过、9明确导入node budget拒绝。新增成功cavlc、
ctrl、dec、i2c、int2float、priority、router；arbiter导入24192、
mem_ctrl93638、voter21516超过20000，保持失败，不提高限额。
新汇总 epfl_control_baseline_progress_20261013 审计所有成功的
actual enodes/cuts/pool/recovery，保留其余60case missing；两库
两目标共8gate均false。没有把一个profile完整覆盖说成80case齐。

独立原版baseline矩阵8/240job、18成功summary rows/2失败rows。
adder六配置尝试已齐：EqMap depth+amap222.33/7485.96；
E-Syn两目标均原生process timeout。E-morphic最终SA成功，
70候选全部named-port CEC，原生选中91.49/3536.80，未根据公共
metric改选（ABC map80.71/2585.48）。bar core四配置及EqMap
area-first三配置也CEC通过。控制或失败记录不能计入额外设计。

E-Syn另行修复探针在 esyn_feature_diagnostics_20261013，未改
原/tmp源码、二进制、模型或已冻结baseline矩阵。支持受限egg
DOT格式的串行node/unique directed edge/density读取器，未知
格式明确拒绝；64随机小图与NetworkX完全逐项相同，含重复边、
self-loop。离线62保留DOT约5.802秒解析，输入hash复核未变化。
初始诊断关于完整stdout index及brace-normalization拼接statement
的拒绝版本保留，修正后全部62格式接受；不能据此称所有任意
DOT都受支持。原main.rs使用BTreeMap，已观察58个导出count/key
相同，没有假称HashMap索引错位；真正原生graph_info.py按
as_completed重新编号的特征错配仍需修复。

新探针用原binary/model、同一input.sexpr、原生2s/20k阈值/
16iter及原60s外部timeout，仅换staging graph_info.py，直接
串行计数并用文件index写CSV以修复完成顺序错配。原生运行
34.5728秒正常退出并输出30ranked候选；本次feature阶段7.176秒。
候选全部同库amap+独立CEC通过，证据
 esyn_feature_repair_cec_20261013。同一已验pool选择两目标：
(A,D)72.61/5316.95，(D,A)75.27/5230.40，仍慢于常规ABC。
native threshold实际仍可20640，不声称hard20k或端到端2秒；
34.6秒仅native阶段，不包含30候选后端映射/CEC时间。
这是明确的新feature/index修复适配探针，不是原版本超时突然
成为成功，不混入冻结矩阵，不按设计择版本宣布胜出。来源、
参数、输出/输入hash、physical replay及两目标pool选择保留。

DAC讨论改为11严格cover、9node拒绝、E-morphic原生adder结果
与E-Syn新探针范围；仍ISCAS历史Tables I-V，不假装EPFL全表齐。
6页正文+1页参考全部逐页视觉QA，原字体/边距保持。下一步
推进main delay-first和原baseline未测检查点；把E-Syn修复写成
可重现独立adapter版本全20矩阵，保持原失败和两目标同候选池。
大图region mapping/stitch/global pool/deadline仍未实现，保持
优先级，不把结构估算或特征修复说成emapper PPA胜出。

### 2026-10-13 用户确认的基线冻结规则

用户要求“不要优化baselines，可以设置最大允许时间”。因此停止
E-Syn feature/index修复探针的集成和全矩阵推广；已有探针只保留
独立历史诊断，不能作为正式baseline或按设计混入原版结果。
正式基线维持原方法、模型和native参数，仅保留明确披露的必要
输入/端口正确性适配；算法优化继续集中在emapper自身。

已有单进程限额维持60秒（EqMap90秒）。已询问每设计/每方法
全流程总限额（10分钟、5分钟或1分钟），待具体答复，不推断
已批准某个数值。统一外层限额应覆盖native、后端和CEC，与
原native搜索预算分别记录；策略改变用明确新policy/矩阵，超时
保留partial artifacts并标记case失败，不将部分成功统计成完整
case成功，不通过提高native预算隐藏原超时。

### 2026-10-13 用户最终限时口径：仅工艺映射300秒

用户答复“5分钟，不包含cec”，随后明确“运行时间只限制工艺
映射”。这覆盖此前全流程总时限设想。固定resyn2、输入适配、
转换、独立CEC与公共指标报告不计入；不是所有native程序都
属于工艺映射。基线算法/模型/提取参数继续保持原样，停止基线
优化。mapper进程及ABC map/amap等实际映射调用才消耗额度。

新增nested run_with_mapping_budget.py对direct映射调用累计
300秒，原driver源码未改，不改变native方法或现有进程限额。
超时后保存partial artifacts，成功incumbent降为partial证据，
不能作为完整case成功。合成短额度测试确认CEC耗时不入账、
映射多调用累计超时、清理进程、partial成功拒绝；真实ABC map
adder结果仍57.40/3548.84，审计仅实际映射call计时。
记录 results_v2/mapping_wall_policy_20261013：最早wholeflow
audit为已被用户纠正的历史，仅mapping_only审计符合最新口径。

E-morphic SA内部反馈会调用ABC映射，当前direct-only wrapper
明确拒绝该driver，避免漏计内部映射而宣称规则完全覆盖。
仍需仅在harness/capture层接入nested映射时钟，不改变SA算法，
不把重写/非映射SA工作计入额度。此前冻结矩阵仍历史policy，
新限时须明确新policy矩阵，不能静默混入旧frozen checkpoints。

### 2026-10-13 映射300秒：E-morphic内部反馈计时已接入

新增nested_mapping_clock.py，只在staging capture launcher为
实际ABC后端调用加计时；原capture归档逻辑、ABC完整argv、
native SA算法、模型、重写/提取参数与原生源码/二进制均保持。
并行ABC调用按active时间区间的并集合计wall time，间隔、
重写/非映射SA处理、输入转换、CEC及指标报告不计入。
剩余额度到期由subprocess watchdog返回真实非零退出并保留
stdout/stderr与partial artifacts，不伪造SA反馈cost。
parent watchdog若杀死worker导致interval不完整，case明确
不合规失败，不宣称预算已可靠通过。direct-only拒绝已解除。

并行重叠、idle gap精确interval union测试通过；短额度nested
ABC超时返回124、关闭interval并保存输出类型的测试通过。
真实adder原版E-morphic probe：70actual ABC反馈调用、70候选
全部独立CEC通过，mapping union6.2094秒；native SA整体10.0535
秒，非映射SA工作不计入300秒。原生最终选中91.43/3536.80，
没有按公共指标另选；这是原随机SA另一run，不称PPA改进。
结果 nested_mapping_policy_20261013。计时probe v1 source从
report hash复原并验证归档；v1 raw excluded_other_secs为包含
nested mapping的父进程gross时间53.8597，净排除47.6502秒，
validation明确扣除重叠。当前wrapper只修正audit扣除这部分
重叠，实际mapping clock与基线算法不变。probe v1不冒充更新
wrapper矩阵同版结果，避免来源混合。

新增 run_epfl_mapping_campaign.py 明确300秒mapping-only新矩阵，
两库20设计每设计8config=320job：core四固定配置、strict mapper
两objective、EqMap两objective、原E-Syn两objective、原final SA。
冻结native全部tracked来源/模型/commit/二进制、全部root helpers、
mapper Rust、library source、计时wrapper/clock与新runner；入口
verify已完成report/audit hash，持有新/两个旧runner锁以防并发。
已2/320job（3in adder core、strict area），5summary公共CEC通过，
没有把旧20+新1重复累加成新21设计。旧main20/80和baseline8/240
仍原policy历史保存，不静默新policy续跑，不混合胜出声明。

用户要求不优化baseline继续执行：E-Syn优化probe只历史存档，
不推广、不给正式比较按设计挑版本。后续原E-Syn已知pre-mapping
timeout若能从原frozen source/params/commands/hash证明尚未执行
任何mapping，可另做显式历史失败采用审计（mapping=0），避免
无变化盲重跑；未经证明不能静默跳过、删除或改成有效结果。
这项采用还未实现。优先继续新policy矩阵未测cases，同时实施
emapper自身region mapping/stitch/global-budget功能，不能将
clock工程宣称mapper PPA胜过baselines。

DAC移除E-Syn优化probe在主讨论中的比较，加入mapping-only
300秒及并行union口径、未改baseline和adder70CEC计时验证，
完整EPFL仍未齐/未支配。6正文+1参考：全部7页render核验，
1-5/7与前次已逐页QA像素hash相同，6页重新视觉QA，字体边距
不变。main source与旧native/harness fingerprints复核均未改。

新policy scope单独生成两库两目标全20 coverage，8个lex/Pareto
gate全部false，command/exit/log与missing方法/设计保留于
nested_mapping_policy_20261013；不将旧scope或独立probe充作
新policy完整数据。

### 2026-10-13 区域逻辑拼接验证及原版失败的显式采用

新增 scripts/experiments/prototypes/region_stitch.py：解析已有命名
AIGER，按拓扑划分region，保留跨region共享变量、极性、多个
输出及原primary port名字。global导入上界累加所有boundary和
各region的Zero/NotZero pair，global原结构AND2+共享INV witness
候选上界不按region重置。此为逻辑拼接prototype，未运行egraph、
未生成真实library matches/physical cover、未验证真实global
recovery/iteration accounting，不作为映射预算证书或PPA改进。

region_stitch_smoke_20261013：6个live AND的三区域小例通过独立
named-port CEC；同例输入a[0]跨region保留并通过CEC；破坏一
输出极性后CEC拒绝。global node bound不足、pool bound不足均
明确拒绝，避免局部限额相乘。后续发现候选估算漏掉complemented
PO（尤其PI-only负输出），修正为所有gate-input与PO负极性变量
的共享INV集合+AND+2ties，old v1 source hash验证后存档。
region_stitch_bound_repair_20261013 的四PI负输出上界6、pool5
拒绝测试通过；先前两个三区域输出Verilog逐字不变，CEC证据
仍有效。上界是保守估算，不是最小实例下界或不可行证明。

pre_mapping_failure_adoption_20261013/adopt.py 显式核验原版
E-Syn adder两目标历史failure：old/new native完整tracked来源、
commit/models/binaries及共同harness指纹相同；source/input/
prepared hash/library/representation/native2s/20k/16iter/60s guard
均一致，原report hash正确、0候选、commands无任何mapping调用。
因此mapping时间为0、原native process timeout仍为failure，只
采用已验证历史失败，不新执行、不改方法名、不把失败变成功。
新report/audit保留来源report/argv/hash及采用script hash；未用
E-Syn优化probe替代。此审计不是允许任意失败跨policy直接采用。

正式300s policy矩阵现在8/320，3in adder所有8config已尝试：
13summary公共CEC成功、2原版E-Syn采用failure；正式当前wrapper
SA70候选CEC成功，原生选中91.47/3536.80，不能拿别次91.43/91.49
择优替换。仍迟于ABC map2585.48，两个eGraph目标也未全面胜出。
epfl_mapping_progress_20261013_adder 保留两库全20 missing/fail，
8 lex/Pareto gate均false，参数/来源hash和gate命令/exit/log保存。

DAC加入三区域仅逻辑CEC验证、物理匹配/extraction未实现的限制；
仍明确EPFL全数据未齐、ISCAS表历史。6+1编译，6页重新视觉QA，
1-5/7与上次全部已QA页像素hash完全相同，标准字体/边距不变。
下一批bar原版core及严格两目标，继续同一冻结320矩阵；研究
真实物理witness/library matching及统一pool/deadline/iteration，
不要把逻辑拼接上界说成div/hyp已可在现有预算完成。

### 2026-10-13 真实区域物理见证及全局相位恢复

新增 nested prototype `physical_region_seed.py`，从公共 evaluator 的
library.json 匹配 AND2/INV（恢复时另匹配 NAND2），直接构造真实库
实例；不调用 ABC map，不改 main mapper 或正式冻结来源。负相位
按全局原变量共享：PI 反相只产生一次，内部反相由 producer region
输出，跨区域显式共享。全局实例计数、统一构造 deadline 和候选
模板上限不按 region 重置。节点数仍为保守导入估算，尚无真实
regional egraph/cut/saturation/全图 iteration accounting；不是完整
native strict budget 证书，不免计原输入节点。

physical_region_seed_20261013：普通及 indexed a[0] 三区域网表均
8 实例、面积 0.62、延迟 113.49，独立 named-port CEC 通过；跨区域
PI 负相位共享例 4 实例、面积 0.31、延迟 60.19，确认唯一 PI INV。
错误库绑定 CEC 拒绝、缺少 AND2 匹配和零 deadline 明确失败。
初次 Yosys flatten 的 $scopeinfo 元数据导致 evaluator 拒绝，原目录
完整保留；采用 flatten -noscopeinfo 仅去除元数据。v1 源码存档，
后续 v2 相位恢复来源独立记录，不能混成同版逐设计择优结果。

physical_region_phase_recovery_20261013：四区域 synthetic 小例，
相对自身 AND/INV seed，两字典序均选择 NAND 负相位驱动加正相位
INV。公共面积同为 0.4000000134、5 实例；延迟从 105.5700016 降至
91.0900021（13.716%），三网表独立 CEC 全通过。global retained
模板 7、整体恢复 <2ms；预测面积/延迟与公共 evaluator 在 1e-6
相对误差内一致。初始验证误要求面积也严格下降而 assertion 失败，
保留失败日志，修正为面积不差且延迟严格下降，未删除或重挑实测。
这是自身原型 synthetic 改进，不是 EPFL 改进、baseline 胜出、
最优性证明或多输出库验证。完整命令复现规格、输入/来源/库哈希
及实际日志保留。

正式 mapping-only 300s 矩阵现在 12/320：21 summary 公共 CEC 成功、
3 failure（含两条显式采用原版 E-Syn adder failure，以及本次 bar
area-first 原版 native pre-mapping timeout）。本轮新增 bar core 四
CEC、strict area 一 CEC、EqMap area 三 CEC；原 baseline 算法未改。
epfl_mapping_progress_20261013_bar 核验 report/audit 哈希并保留两库
两目标全20 missing/fail，8 lex/Pareto gate 全 false。三个 runner 锁
空闲，下一未测 bar strict delay-first/EqMap delay-first，再原 E-Syn
与 E-morphic；不能未核验就跨目标采用 bar area failure。

DAC讨论更新为 primitive physical region CEC 及 synthetic 13.7%
延迟改善，明确 regional egraph cut extraction 尚未实现。6正文+1
参考已编译、7页render；1-5/7像素与此前已QA页相同，6页视觉复核
通过，标准字号边距不变。主表仍历史 ISCAS pilot，完整 EPFL 尚未齐。
下一研究接真实跨区域 cut matches/cover 及 egraph/accounting，
不提高预算或把合成小例宣称 div/hyp 已解决。

### 2026-10-13 原生区域匹配/相位恢复接入及 EPFL 全范围自对照

main 已接入 `EGM_REGION_CLASSES`（默认0）和 `EGM_PHASE_RECOVERY`
（默认0），保持旧默认。egraph_regions.rs 在一个全局 egraph 上
分配完整 physical boundary 的 region owner；跨区域 targets 保留
全局 class/signal ID，物理共享和 cover 统一恢复。不是各 region
独立 saturation，不按 region 重置节点、pool、iteration、恢复预算。
按需导入反相节点保留全部 PI/live AND，并按 exact rewrite RHS 上界
预留全局 node 容量；被预算挡住的 rewrite 次数单独输出。成对反相
恢复同时替换 producer 和 reverse INV，利用已经准入的 physical
bindings，检查全图 acyclic cover/unique instance area/公共 pin delay，
不新增 SDC union，不宣称最优。39 映射回归通过、1 ignored；含
未使用输入与负相位保留、共享正输出成对翻转、expired deadline、
原输入 node 拒绝和 rewrite 预留验证。源/旧binary/patch/日志保存在
region_native_integration_20261013，main release 编译完成。

旧 mapping-policy 矩阵在28/320有限批次退出后 sealed，30 summary
CEC、21 reported failure、1个失败进程无report（日志/audit保留）。
bar E-Syn两目标实际为 native exit -6，不是先前进展文字推测的60s
timeout；div/hyp E-Syn明确是 input expression resource guard failure，
不称 native saturation timeout。失败归因以原report/log为准。

正式新自对照 `epfl_region_phase_checkpoint_20261013` 已160/160：
两库*两目标*全20*region-phase/control 两配置，每版44 CEC，合计
88成功、72 node-budget failure；44配对中delay9改善、26持平、9
回退。每版9个大设计全部四setting拒绝，区域ownership未解决大图
node allowance。3in delay-first adder 76.83/3016.20 -> 76.25/2849.22，
两项改善，但仍不能说胜过ABC delay map。4in仍有延迟回退；area-first
也有primary regression，原型自身13.7%不能替代真实对照。
全20 missing/fail/metrics、source/report/audit hash及paired.csv在
`epfl_region_phase_full_20261013`。retained pool <=100000、actual
cuts/class<=16、enodes<=20000、整体 recovery<=2s逐成功case核验；
generated_instances另保留，retained上限不是对生成暂存内存的声明。

另做明确的库时序配置，native source/binary仍同版，默认不变：
`EGM_LIBRARY_TIMING=1 EGM_TIMING_CAP=1`、region256/phase1，只有
delay-area，所有20设计两库固定同配置，不按设计挑选。
`epfl_region_timing_phase_checkpoint_20261013` 已80/80：44 CEC、36
node-budget failure，22配对16delay改善/2持平/4回退。4in max
269.86/3090.61 -> 388.32/2504.82，3in sin 670.17/2806.05 ->
763.09/2580.93，均有明显面积代价；3in adder此配置比control更慢。
`epfl_region_timing_phase_full_20261013`保留所有结果；不能将此配置
4in的好值与前配置3in的好值按设计拼成一个赢家。

新原版 baseline-only 300s 矩阵 `epfl_original_baselines_checkpoint_20261013`
240job，通过依赖审计采用旧28矩阵中的21个baseline job（含failure），
仅未执行Rust candidate源码/binary允许变动，全部实际baseline/native/
models/helpers/ports/clock/evaluator/library/input依赖一致，采用script
与proof完整保存。作者算法/模型/native参数未改，E-Syn优化probe不并入。
首批至29/240，30summary CEC、23reported failure、4 unmeasured失败job；
无report的退出失败保留原argv/log/audit，不能视为成功或默默丢弃。
下一有限12job批在顺序执行；只有该批结束才启动另一campaign。

时钟额外审计：EqMap初始 `yosys abc -lut6` 是嵌套实际LUT mapping，
旧direct wrapper未单独计时。保留精确记录缺口，不把整个Yosys转换
冒称mapping time；对每个EqMap/lut staging成功case使用已存的 enclosing
Yosys elapsed 加 direct mapping elapsed 作保守300s上界，均通过。
该上界不是精确EqMap mapping runtime。未来可仅完善计时capture，
不能改原baseline算法/参数；目前独立case ledger及scope明确披露。
`epfl_region_baseline_scope_20261013`8 lex/Pareto gate false；新时序配置
`epfl_region_timing_baseline_scope_20261013`4 gate false。全部scope含
两库全20与missing/fail，MtM仍未接入，完整baseline尚未齐。

DAC正文新增全局region/atomic phase规则及两组完整EPFL自对照事实，
报告改善/回退和area代价，说明baseline覆盖不全与EqMap nested clock
上界口径。主表仍标记历史ISCAS pilot。6正文+1参考已编译，全7页
render；2-6逐页视觉QA、1/7与旧QA像素相同，标准字体边距不变。
下一优化检查4in/area-first候选损失及phase预算分配；新native修改
必须另建明确版本全矩阵。继续原版baseline全覆盖，不抬预算、不
按case挑赢家；还没有全baseline支配或完整工业多输出/ODC证据。

本轮最后原版baseline有限批次结束，当前41/240、47summary CEC、32
reported failure、4无report失败job。三个新campaign锁idle；下一从sin
core及两目标原版适配继续。最新全scope/gate保存在
`epfl_region_baseline_scope_20261013_41`（8 false）与
`epfl_region_timing_baseline_scope_20261013_41`（4 false），成功EqMap
和LUT-staging记录的direct+enclosing-LUT mapping保守上界均<=300s。
原版代码/model/native参数未改，补齐尚余199job，不称数据已经齐。

### 2026-10-13 phase/local recovery allocation audit

Read-only audit `results_v2/epfl_phase_allocation_diagnostics_20261013` verifies all completed native-study report hashes and retains all 80 pairs including node failures. Of 44 valid pairs, seven phase-enabled cases execute zero subsequent local recovery trials while their controls execute local trials; all seven have higher area and five have higher delay. Native phase acceptance compares against its incoming cover, so it does not ensure superiority over the control that spends its recovery budget differently. The code runs phase recovery before local/cone recovery under the same global deadline. This supports a scheduling-bottleneck hypothesis, not a causal ablation: demanded import and region ordering also change. Next mapper experiment: an isolated explicit version reserves part of the unchanged two-second global recovery deadline for local/cone recovery; evaluate one fixed configuration on the full paired scope, retaining all failures and regressions. No baseline or mapper source was altered by this audit. DAC discussion records the observed counters; PDF remains 6 body pages + 1 references, page 6 visually reviewed and pages 1–5/7 pixel-identical to prior reviewed pages.

### 2026-10-13 original baseline checkpoint 53/240

The next finite 12-job batch completed under the original frozen algorithms and 300-second mapping-only policy. All report/audit hashes were reverified. Current totals are 53/240 jobs, 61 CEC-successful summary rows, 43 reported failed rows and four unmeasured failed jobs (original argv/log/audit retained). All six relevant runner locks are idle; no background campaign is claimed. Sin core and both EqMap configurations succeed; original E-Syn and E-morphic sin fail. Sqrt core succeeds; both original EqMap configurations and E-Syn configurations fail, with native exit/timeout provenance retained. Do not conflate rows with unique designs or failed stages with mapping timeout. Next unattempted job is 3in E-morphic sqrt. Fresh full-20 scopes `epfl_region_baseline_scope_20261013_53` and `epfl_region_timing_baseline_scope_20261013_53` retain missing/failure records; eight and four lex/Pareto gates respectively remain false. Complete baseline coverage and a uniformly better mapper remain unachieved.

### 2026-10-13 phase-time allocation: complete native paired scope

`EGM_PHASE_TIME_FRACTION` is now available, default 1.0; phase recovery itself remains off by default. The experimental setting 0.25 assigns that fraction of the remaining global recovery window to phase recovery, giving local/cone recovery the rest. The stage deadline is checked between atomic trials (not a preemptive hard stage timer); unchanged strict global two-second validation still applies. No node/cut/pool/iteration/mapping budget increase. Tests: 49 egraph-related passed, two ignored. Source patch, snapshots, binaries and logs are in `phase_allocation_implementation_20261013`. Main source has identical implementation but separately built binary, so isolated experiment data are not relabeled as main-run observations.

The complete isolated same-binary study `epfl_phase_allocation_checkpoint_20261013` attempted 160/160 jobs (two libraries, two objectives, all20 designs, both fraction settings), with 88 named-port CEC-successful covers and 72 node-budget failures. Each objective has 22 valid pairs. Delay-first: delay 2 better/20 equal/0 worse; area 6 better/16 equal/0 worse. Area-first: delay 1 better/17 equal/4 worse; area 6 better/15 equal/1 worse. All candidate covers have nonzero local trials; eight fraction1 controls have zero. 4in delay-first max improves A/D323.77/3102.43 to279.93/3082.56; 3in max313.14/2587.75 to251.49/2582.33. Single-run deadline-sensitive observations are not a stability or optimality proof, and the same nine large designs still fail node limits. Complete failed/missing scope, paired.csv, report.md and allocation.json are in `epfl_phase_allocation_full_20261013`. No per-design choice of settings.

The original-baseline continuation is now `epfl_original_baselines_allocation_checkpoint_20261013`, adopting the prior53 jobs with a dependency proof. Only unused Rust candidate mapper sources/binary changed; author/native/helper/model/input/library/clock/evaluator dependencies are byte-identical. Statuses, failed logs and all report/audit hashes are preserved. The prior53-job campaign is historical and cannot continue under its old candidate fingerprints. Baseline algorithms remain unchanged.

### 2026-10-13 main max repetitions and baseline65 checkpoint

`epfl_phase_allocation_max_repeats_20261013` is a targeted stability study on the separately built main binary: max, both libraries, delay-area, three paired repetitions, alternating order and retaining every run. All12 public named-port CEC pass; actual node/cut/pool/global recovery budgets verified. Every paired repetition improves delay and area. Three-input delay deltas are -5.42 in all three runs; area deltas -60.63/-60.17/-60.21. Four-input delay deltas -14.73/-19.87/-19.87, area deltas -35.10/-43.78/-43.78. This small targeted repeat scope does not replace the complete20 scope or establish statistical confidence/global dominance. runs.csv and validation.json retain all results.

The finite original-baseline12job continuation completed at65/240:81 CEC-successful summary rows,49 reported failed rows,four unmeasured failed jobs. Report/audit hashes reverified. Square core/EqMap both objectives pass; original E-Syn/SA fail. Arbiter core/EqMap both objectives pass; original E-Syn fail. Next unattempted is3in E-morphic arbiter. `epfl_phase_allocation_baseline_scope_20261013_65` retains both libraries/two objectives/all20 designs; all eight lex/Pareto gates remain false. Algorithms/models/native parameters unchanged; source-dependency adoption only excludes unused Rust candidate changes. Dataset incomplete, no full-suite victory.

### 2026-10-13 staged-flow exploration and baseline77

The old continuation refused a resume before any new mapping because unrelated plotting/report scripts changed (`plot_paper.py`, new `run_three_parts.py`/`report_three_parts.py`). These dirty files were preserved. New nested `run_epfl_original_dependencies.py` explicitly adopts65 prior jobs after proving all executed dependencies/native sources/models/clocks/libs/inputs/binaries identical and checking the three helpers are not referenced by frozen scripts. The new baseline-only matrix is `epfl_original_baselines_dependency_checkpoint_20261013`, now77/240,106 summary CEC/50reportedfailure/4unmeasured failure. Next3in E-morphic ctrl. Original E-Syn cavlc/ctrl configurations now succeed without optimized probes; cavlc finalSA succeeds, arbiterSA fails. All statuses/report/audit hashes retained. New scope77 eight lex/Pareto gates false, full datasets still incomplete.

`epfl_adder_critical_path_20261013` reconstructs common public metrics exactly from saved netlists/library. ABC delay path128 cells mostly alternatingAOI/OAI; phase-allocated eGraphMap138 cells includes50positiveAO/OA cells. This is cover characterization, not missing-candidate/cause proof.

New isolated `/tmp/emapper-staged-recovery-20261013` implements actual delay-flow -> frozen-bound required-time area-flow -> actual unique-instance area recovery; global(D,A) incumbent retained. Source/patch/bin/logs in `staged_recovery_prototype_20261013`,52egraph-related tests passed/2ignored. Main/default and baselines unchanged. FullD-only two-library all20 prototype/control matrix `epfl_staged_checkpoint_20261013`80/80,44publicCEC/36nodefailures,22 valid pairs:D1better/18equal/3worse. All staged outputs<=reference bound; area flow accepts7/22. Fixed30/20/50remaining-window scheme plus simplified2000-total-trial local scan is not a pure scheduling ablation, and worsens3in adder; retained fullscope/stages/paired.csv in `epfl_staged_full_20261013`. Do not promote default or claim solved timing. Next separate-factor test: event-driven arrival propagation/critical proposal ordering under unchanged budgets and trial cap; preserve all configurations.

### 2026-10-13 staged propagation/order factorial study

New isolated `/tmp/emapper-staged-factors-20261013` separates two factors: dependency-worklist delay flow versus the original fixed32 sweep, and critical-gain local proposal order versus ascending cell area. All four profiles use staged recovery, the same binary/pool, region256/phase0, the same30/20/50 remaining-window schedule, global2s recovery and2000 TOTAL local trials. SAT16/node20000/cuts16/retainedpool100000 unchanged; no baseline algorithm or main/default changes. Source/patch/binary/commands and budget plan are in `staged_factors_implementation_20261013`. 53 egraph-related tests passed, two ignored. A96-cell reverse-ordered cone regression demonstrates that the32-sweep flow fails to reach the output while the dependency worklist reaches the legal96/96 cover under the same deadline; this is an algorithmic propagation test, not an EPFL optimum proof.

`epfl_staged_factors_checkpoint_20261013` completed160/160 (both libraries, delay-area, all20, four fixed profiles):88 public named-port CEC successes,72 explicitly audited import-node-budget failures. Every successful cover satisfies recorded node/cut/pool/global recovery/total-trial limits and its frozen timing bound. `epfl_staged_factors_full_20261013` retains every profile, all five contrasts, all failed logs and source/report/audit hashes. Against the staged sweep/area-order control, event-only delay6better/16equal/0worse, area7better/12equal/3worse over22 legal pairs. Critical order without event flow also improves six delays but increases area in six pairs. Adding critical ordering to event flow improves no delays (22ties), with area2better/18equal/2worse; do not combine per-design profile winners.

Event-only 3in adder A/D84.86/3210.79 ->79.75/2585.48; 3in max280.41/2677.99 ->293.94/2194.54 (area increase retained); 4in max306.05/3108.58 ->294.88/2485.74. Read-only `epfl_staged_event_adder_path_20261013` exactly reconstructs common public metrics, same input/prepared/library hashes and matching upstream counters: adder critical path154 ->128 cells, now63AOI21/63OAI21 plusAND/XOR, equal delay to original ABC map. Identical counts are not a byte-level candidate-pool dump. Single deadline-sensitive runs do not establish statistical stability/global optimality or superiority to all baselines. Main is still the prior phase-budget version; isolated data are not relabeled main results.

Original baselines reached89/240 with130 CEC-successful summary rows,52 reported failed rows and4 unmeasured failures; all retained report/audit hashes verified. Ctrl and dec finalSA pass; dec all original configurations pass, i2c original E-Syn fails. Phase-quota scope89 has eight false gates; event-only scope89 has four false gates. Full20 missing/failures and EqMap conservative nested-LUT timing bounds remain explicit. A further finite seven-job original-baseline batch is running sequentially after the completed mapper study; no parallel campaign or cross-turn background survival is assumed. DAC discussion records the factorial result, area penalties and the corrected adder path; PDF compiled and checked at6body+1reference, standard IEEE font/margins, page6 visual QA clean, pages1–5/7 pixel-identical to prior reviewed pages. Full EPFL baseline coverage remains pending.

Next mapper validation: repeated fixed event-only versus same-binary staged control on adder/max, then a complete fixed event-only versus same-binary coupled recovery comparison before any main integration; preserve every run and area cost. Critical ordering is not adopted merely because some areas improve. Avoid unchanged pre-stage failure reruns unless a full matrix/provenance need justifies them; audit dependency reuse explicitly. Continue original baseline coverage first.

The subsequent seven-job original-baseline batch completed, bringing the active dependency checkpoint to96/240:144 CEC-successful summary rows,52 reported failed rows and4 unmeasured failed jobs. All retained report/audit hashes verified; nine runner locks idle, no active background campaign. I2c finalSA and all six int2float configurations pass using the original native code/models. Next unattempted job is3in core mem_ctrl. Current full20 phase-quota scope96 eight gates false, event-only scope96 four gates false; baseline144 jobs remain. Both reports retain every missing/failed case and conservative EqMap nested-LUT timing proof. Earlier89 evidence remains an independent checkpoint snapshot, not additional cases.

### 2026-10-13 verification transport, repeated queue flow, main opt-in preset

Original baseline dependency matrix reached120/240, then mem_ctrl exposed a verifier transport failure: all four ordinary mappings completed, but a long named-port Yosys `-p` argument exceeded the OS per-argument limit before CEC. The common verifier now writes the exact unchanged command text to a temporary script and calls Yosys `-s`. No port matching, cell models, common metrics or baseline algorithm changes. `cec_scriptfile_transport_20261013` archives the original source/executable, exact production-only patch, build and tests. The1200 indexed-input/output positive and wrong-output negative tests pass, as does the existing multi-output/unnamed-port positive/negative test. `cec_scriptfile_mem_ctrl_replay_20261013` rechecks all four saved netlists without mapping again; all pass independent CEC and all mapped artifact hashes stay identical. Model Verilog/library JSON are byte-identical before/after. Original failures remain in the old matrix/replay provenance.

The explicit CEC-file continuation `epfl_original_baselines_cecfile_checkpoint_20261013` reached142/240,212summaryCEC/80reportedfailures/6unmeasured failed jobs. It adopted unchanged native observations with a transport-source proof, and retried only the two mem_ctrl EqMap jobs previously stopped at LUT CEC before author execution. Their ordinary amap controls now pass; LUT checks still time out under the unchanged process guard, so author EqMap results remain failed. Native mem_ctrl finalSA retains its genuine300s cumulative mapping failure. Do not label CEC timeouts as mapping timeouts or use a transport fix as PPA optimization.

`epfl_event_repeats_20261013` completed24 targeted same-binary runs: adder/max, both libraries, three paired repetitions, alternating order, staged sweep versus dependency queue. All24 pass independent named-port CEC and actual budgets; all12 pairs improve delay, six increase area. All runs retained, not best-run selection. A tuple-key construction error before any mapping is archived. `epfl_event_current_flow_v2_checkpoint_20261013` completed160/160, four fixed same-binary profiles over both libraries/all20 delay-area: coupled phase0, coupled phase-quarter, coupled queue+quarter, staged event. Eight initial successful observations were adopted after an explicitly metadata-only runner repair; all report/audit hashes unchanged.72 original import-node failures are explicitly reused with exact upstream dependency/flag proof, not guessed or silently skipped; new unused analyze_arbiter.py was audited and retained, and the changed CEC backend only affects unchanged model emission before those failures.88 publicCEC successes retained.

Full current-flow comparison: staged event vs coupled phase0 D6better/16equal/0worse, A7better/1equal/14worse; vs coupled quarter D6/16/0, A8better/1equal/13worse. Coupled queue vs coupled quarter D6/16/0, A8better/10equal/4worse. Two of those area increases have equal delay (3in i2c+0.11,4in router+0.09), so even delay-area lexicographic improvement is not uniform. The coupled worklist flag changes seed and refinement propagation, not a pure delay-only factor. Staged2000 TOTAL trials differs from unchanged coupled2000-per-round configured policy; actual trials and common global2s deadline retained. Latest baseline142 gate for coupled queue remains fourfalse, full scope/missing/failures explicit. Do not promote a uniformly better/default mapper claim.

Main CLI now offers opt-in `EGM_RECOVERY_PROFILE=queue-delay`, selecting the validated strict delay-area, region256/phase1/fraction.25/worklist configuration (solver0/cuts16/DCsupport4), with ordinary explicit flag overrides. It packages the existing kernel rather than disguising a new baseline algorithm. Legacy default unchanged; profile rejects non-strict/non-delay-area combinations. `queue_profile_main_integration_20261013` archives source/patch/binary before/after. `main_queue_profile_acceptance_20261013` independently validates main preset vs explicit parameters on3in adder and4in max: fourCEC/budget successes. Adder both72.68/2585.48; max sameD2485.74 but A261.25 vs263.65, deadline-sensitive observations all retained. Isolated full-scope data are not relabeled main measurements.

### 2026-10-13 compact phase-edge import prototype

Isolated `/tmp/emapper-literal-import-20261013` encodes complemented input edges in a two-fanin logical AND operator. Every live original AND and every PI remain stored and globally counted; output Not enodes remain explicit, no region budget resets or input exemptions. Cut truth expansion and bounded reachability decode all phase bits; two exact masked-complement identities are lifted, never SDC unions. Other old negative-pattern rewrites have reduced visibility, so this is a different operator/search-space configuration, not a claim of identical saturation.55 egraph-related tests passed/2ignored, including all four input-phase combinations, duplicate-input constant cases, both output polarities and a41PI/40AND graph whose82 actual compact enodes fit a100 limit while the expanded representation is rejected. Initial compile/undriven-constant failures and fixes retained; prototype not main.

`literal_import_budget_estimate_20261013` is only a conservative storage estimate, not mapping feasibility, PPA or a minimum bound. Targeted3in delay-area trials in `literal_import_implementation_20261013`: arbiter750.46/926.89, voter1491.01/844.37, square2646.59/3365.75 all independently CEC pass with actual enodes<=20000/cuts<=16/pool<=100000/recovery<=2s. Arbiter import12097/postSAT12160, voter import10759/postSAT12128. Sqrt gets past import but fails recovery at2.021050s>2.000000s including setup/final validation; failure retained, no extension. These restore three previously node-rejected targeted covers, not all-library/all-objective coverage or PPA superiority. div/hyp/log2/multiplier/mem_ctrl estimates still exceed20000, no impossibility proof.

After the main CLI change, the142-job baseline matrix was sealed. New `run_epfl_original_queuebridge.py` continuation explicitly proves only unused candidate mapper executable/CLI changed; all executed baseline native/source/commit/models/clock/library/input/evaluator dependencies identical. Original statuses/verification replay artifacts preserved; old adopted no-mapping audits without state arrays are handled only after their source SHA and no-mapping proof validate. `epfl_original_baselines_queuebridge_checkpoint_20261013` is executing the finite remaining98 jobs sequentially; wait for completion before any other mapping campaign, do not depend on cross-turn background survival. Source fingerprint changes still require explicit new versions/proofs.

### 2026-10-13 full EPFL original-baseline completion and source-grounded delay studies

The interrupted 4in EqMap area-first square job was audited: two control rows existed but the native summary and completed clock audit were absent. All 38 artifacts and the old checkpoint are preserved under `epfl_original_baselines_queuebridge_checkpoint_20261013/interrupted_20261013_recovery`; the partial attempt is not counted as a successful case. Its unchanged-argv retry completed with three CEC-successful rows. The old runner then rejected unrelated changes before mapping. Explicit continuation `run_epfl_original_resumed.py` proves all executed native/model/clock/library/input/evaluator dependencies identical, specifically audits newly changed unused reporting/experiment helpers, and adopts the 175 recorded jobs without duplicating them. Dirty user files were retained.

`epfl_original_baselines_resumed_checkpoint_20261013` completed **240/240 grouped jobs**, both inventories and all20 designs, six groups per design/inventory. All report/audit SHA values and frozen dependencies verified. `epfl_original_baselines_complete_v2_20261013` records all520 method/status slots:372 CEC-successful summaries,124 reported failures,24 explicitly unmeasured slots from eight failed jobs without reports (hyp/log2 EqMap, both priorities/inventories). Commands/logs/audits remain available. Attempted coverage is not all-successful PPA coverage. Mem_ctrl conventional four mappings pass; LUT CEC timeouts, original E-Syn resource guards and native SA cumulative300s mapping failures remain. Original baseline algorithms/models/parameters were not optimized. The initial complete report with only one placeholder per unmeasured EqMap job is preserved; v2 explicitly adds both missing controls, not invented measurements.

Fresh full20 gates against all original baselines: queue delay-first/Pareto fourfalse in `epfl_queue_baseline_scope_complete_20261013`; historical phase-quarter both-objective/Pareto eightfalse in `epfl_two_objective_baseline_scope_complete_20261013`. Their candidate versions/profiles remain separate. EqMap initial LUT-stage time remains an enclosing-Yosys conservative upper bound, not exact mapping time. MtM remains unconnected. No full-suite win is claimed.

Read-only `baseline_delay_mechanisms_20261013` records ABC/emap source hashes and their required-time-before-area-flow, mapped-reference, per-phase timing and composed-supergate mechanisms. These concepts informed emapper-only prototypes. `post_flow_delay_audit_20261013` finds all22 queue final native delays equal their delay seeds. This is selected-cover evidence, not global-optimality or missing-candidate proof; next work should prioritize candidate/library coverage rather than assuming more recovery time reduces delay.

Fixed literal-import timing ablation `literal_timing_pairs_20261013`:16 attempts,12 CEC successes/4 recovery-budget failures;6 valid pairs D4better/2worse,A3better/3worse. 3in square A2644.54/D3365.75 ->2579.82/2702.17;4in square A2665.26/D3363.24 ->3114.67/2777.23, with area penalty. Arbiter4in and voter3in delay regressions remain. Both sqrt profiles overrun the unchanged global2s deadline; no failed cover is reported as PPA. `literal_timing_full_20261013` retains every pair/hash. This targeted four-design scope is not full EPFL or a source/config winner mixture.

Separate return reserve prototype `/tmp/emapper-return-reserve-20261013` exposes `EGM_RETURN_RESERVE`, default.05. A.15 reserve reduces search time inside the same2s deadline; node20k/cuts16/pool100k/SAT16/mapping300 remain.55 egraph-related tests pass/2ignored. `return_reserve_pairs_20261013` sqrt both libraries/two alternating-order repetitions:all four.05 controls fail,all four.15 trials pass independent CEC and actual budgets (recovery1.9139–1.9273s). D remains75162.87–75293.69 (3in) and72141.28–72343.00 (4in), slower than conventional mappings. This restores targeted coverage, not delay improvement over an unavailable control or a universal deadline guarantee. Source/executable/patch/build evidence in `return_reserve_implementation_20261013`; main unchanged.

Early required-time prototype `/tmp/emapper-early-required-20261013` allocates20% of remaining recovery window after validated delay seeds, uses physical references and actual(D,A) acceptance. Global2s remains; checks between atomic work are not preemptive stage caps.50 related tests pass/3ignored including a slow secondary-pin rejection. Full80-case paired matrix `early_required_pairs_20261013`:44 CEC/36 import-node failures,22 valid pairs D0better/22equal/0worse;A7better/8equal/7worse. Not promoted. Source/binary/patch/tests in `early_required_prototype_20261013`, full evidence in `early_required_full_20261013`.

Producer-consumer phase prototypes use only admitted opposite-phase producer/qualified-consumer bindings, no new candidates/equalities/SDC unions. Critical arrival only ranks proposals; all acceptance uses full DAG validation and unique physical area/pin delay. Original phase-quarter and global2s/candidate/node/cut/SAT/trial limits stay fixed. Synthetic regression verifies all eight assignments and retains a shared positive output without requiring an initially selected internal INV. Version1 restricts consumer boundary classes and known current/free-witness arrivals; v2 allows different qualified boundaries and validated full-witness arrivals. Each passes50 related tests/3ignored and completes its own full80-case scope:44 CEC/36node failures,22Dties. V1 A3better/17equal/2worse;V2 A2better/18equal/2worse. Neither is promoted. Evidence under `consumer_phase_implementation_20261013`, `consumer_phase_full_20261013`, `consumer_phase_v2_implementation_20261013`, `consumer_phase_v2_full_20261013`. Initial CLI insertion assertion before compilation is retained. Do not blend per-design versions.

DAC now uses complete EPFL original-baseline outcomes in TablesI–IV, separated area/delay and three/four-input inventories. Area-oriented results and all controls/failures remain in the full dataset; ISCAS pilot tables are archived in the prior source/PDF. Main tables explicitly identify the immutable queue study rather than relabeling it as a complete main-binary run. Abstract/protocol/results/discussion include actual coverage, negative stage studies and targeted sqrt coverage limits. Standard IEEE10pt/margins unchanged; text deduplicated and ordinary float placement adjusted to fit6body+1reference. All7 pages visually reviewed, page6 additionally at1800px, no overfull/undefined references, four table bodies exactly match audited data. QA and before/after artifacts in `epfl_paper_update_20261013`; published `output/pdf/DAC-eGraphMap-draft.pdf`. No claim of completed mapper functionality, industrial multi-output/ODC, independent regional saturation or all-baseline dominance.

### 2026-10-13 comprehensive reassessment (read-only PPA study)

`results_v2/comprehensive_reanalysis_20261013/analysis.json` recomputes immutable queue versus conventional delay baselines from the complete original-baseline CSV, using the gates' relative delay tolerance. Against ABC delay map, 3in has 8 ties/3 slower and 4in 1 tie/10 slower among 11 valid cases per inventory; neither has a faster delay. Against emap delay, 3in has 7 ties/4 slower and 4in 1 tie/10 slower. Each inventory also retains nine invalid candidate designs. Every valid queue netlist has zero multi-output instances. Complete attempted baseline coverage remains 240 grouped jobs/520 status slots, not all-successful PPA coverage.

The source audit distinguishes cut ranking/truncation, physical-pool retention, flow propagation, legal-cover repair and recovery. Default optional cuts prioritize support/internal-leaf counts; default pool retention preserves area-ranked full finite witnesses and then fills per-target by intrinsic pin delay. TIMING_CAP uses witness arrivals, not proven minimum-delay arrivals. These are mechanisms and testable hypotheses, not a certified explanation for individual regressions. The 22 final-equals-seed observations and unsuccessful early-required/consumer-phase studies support prioritizing candidate-loss localization over more recovery scheduling trials.

`review.txt` records comparison limitations: same prepared input/common physical scoring and named inventory do not imply identical native timing models, candidate spaces or internal budgets. EqMap/E-Syn are explicitly adapted flows; E-morphic retains original Liberty feedback/native winner. True independent regional mapping/saturation, industrial multi-output evidence and ODC remain incomplete. Next diagnosis should locate baseline critical-cone losses at pre/post-cut, pre/post-cap, flow and legal repair on fixed profiles before changing candidate policies. No baseline algorithm, native budget, main default, paper numerical table or mapper source changed in this reassessment; no new PPA claim or per-design profile mixture.

### 2026-10-13 candidate-loss localization and complete retention factorial

New diagnostic-only source `/tmp/emapper-pool-audit-20261014` records candidate counts and best local estimates before/after cap under one common area-ranked witness. Its test detects loss of an early boundary without mutating candidates; an initial test-only missing Binding PartialEq compile error is preserved and corrected. No main/default/baseline source changed. Snapshots/patch/bin/toolchain/logs are in `pool_loss_localization_20261014`. Four targeted cases pass public named-port CEC and actual budgets.4in adder has95916 candidates with no cap truncation, so its delay regression is not caused by pool cap.4in max504394->100000 loses better local estimates on152 selected-critical signals. Estimates are not minimum-delay bounds or alternative legal covers. Reconstructed3in adder selected truth boundaries show126/128 absent from4in retained cuts; source-independent deterministic import/SAT and matching counts/free/roots support ID correspondence, but no serialized-graph certificate.

An isolated fixed quota8 for <=3-support optional cuts stays within16 total cuts and all existing budgets; prototype `/tmp/emapper-small-cut-quota-20261014`. Initial4in adder A/D91.56/2820.87 ->88.33/2585.23 through different retained cuts, not recovery of the126 reference boundaries. Separate cap-only max/sin probes improve D but have area tradeoffs; neither targeted result is promoted or compared as a renamed main run.

`candidate_factorial_checkpoint_20261014` completed160/160:both inventories/all20 designs/four fixed same-binary profiles (control,arrival-aware cap,small-support quota,combined). Two adder control/quota observations were adopted with exact executed-dependency proof; no duplicate design/cover claim.88 public CEC covers and72 explicit import-node failures; every profile22 valid/18failed. Full data, all failed logs/hashes, actual budget verification, CSV and16 full-scope lex/Pareto gates in `candidate_factorial_full_20261014`. Against control over22 valid pairs:cap D4better/15equal/3worse;quota D6better/16equal/0worse;combined D7better/12equal/3worse. Quota area3better/10equal/9worse and(D,A)8better/10equal/4worse, so it is not uniformly lexicographically better. All16 full-suite gates false. No per-design profile selection; main/default remains unchanged.

Repeated fixed quota0/8 on4in adder/max uses three pairs each, alternating order for two new repeats and preserving factorial r0.12 covers/six pairs all CEC/budget pass; all six delays lower. Adder D-235.640041/A-3.23 each;max D-291.420004 with A+29.43/+16.42/+23.79. All runs retained, no best-run selection/statistical or global-optimum claim. `small_cut_repeats_full_20261014` records scope. Note3in bar/i2c final cut lists are identical between quota/control while tied-delay areas differ slightly; distinguish run variability/retained inventory/recovery from cut-policy causality.

Conditional source-level finding in `multioutput_admission_note.json`: generate() builds/deduplicates single-output base before multi-output/conditional admission; if base_count>=max_instances the first novel latter candidate returns before insertion. This is a code consequence, not observed industrial QoR:EPFL inventories are single-output. Future diversity quotas must consider admission and retention. Original baseline actual/native tracked fingerprints rechecked with no drift.

DAC discussion now records COMPLETE factorial, all regressions and failed gates, and repeat scope; mainTablesI-IV remain the consistent immutable queue comparison rather than per-design prototype winners.6body+1refs/standard fonts/margins; page6 visual QA,1-5/7 pixels identical to prior reviewed pages. QA `candidate_localization_paper_20261014/paper_qa.json`, published stable PDF. All runner locks idle; no background campaign claimed. Next priority:quota tied-D area losses4in ctrl/priority and cap D losses4in bar/dec,3in bar; retain failing large designs and global budgets. Prototype promotion held because full objective regressions remain.


### 2026-10-14 physical phase aliases and candidate-loss repair

Exact selected-boundary diagnosis in `retention_regression_audit_20261014` verifies identical serialized e-graph nodes and PI identities for stock/quota ctrl/priority. Ctrl loses one exact previously selected AOI211 boundary, while priority loses none of its 696 exact selected cut keys; priority regression therefore cannot simply be called missing cuts. Wide-cut area reservation and mixed intrinsic/witness-arrival retention were tested in `balanced_retention_targets_20261014` (25 targeted covers). Wide reservation significantly increases priority area; all bad outcomes remain.

Separate exact-Not canonical cut-bank prototype `canonical_retention_checkpoint_20261014` completes160/160,88 named-port CEC successes/72 import-node failures. Canonical banks plus mapped-reference/required-time recovery give14 better/8 tied delays but one delay-area regression and large max area penalties. This is a distinct fixed profile, not a per-design winner selector or the promoted main configuration.

`canonical_wire_checkpoint_20261014` instead preserves raw e-class cut banks and physical pin interfaces. Exact Not aliases canonicalize physical input wires/output targets before instance deduplication/admission; conditional matches do not establish aliases. All original imported/saturated nodes still count. Within unchanged16 cuts per actual e-class/100000 retained candidates/global2s recovery/SAT16/mapping300s, complete80-case control/trial comparison gives44 named-portCEC successes/36 import-node failures.22 valid pairs:delay9better/13equal/0worse;area15better/2equal/5worse;delay-area20better/2equal/0worse. Five area tradeoffs are3in bar/max/sin and4in max/sin. All eight full-baseline lex/Pareto gates remain false; MtM unconnected.

Five target cases (4in ctrl/priority/bar/dec and3in bar) retain full-study r0 plus two new alternating-order repetitions in `canonical_wire_repeats_full_20261014`:30 legal covers/15 pairs,15delay-area improvements,6delay gains/9ties;12area gains/3area regressions (3in bar). Every run is retained; no statistical/general optimum claim or best-run selection.4in ctrl/priority/dec each repeat saves0.37/7.43/0.36 area with unchanged delay;4in bar saves2.77 delay. The exact-Not optimization does not solve the same nine large-input node failures.

Original baselines remain fully attempted240/240 grouped jobs (372 successful/124reportedfail/24unmeasured slots). Read-only source/commit/actual native model/binary/transport audit checks13366 files with no drift. Baseline algorithms and native budgets were not optimized. Three independent external fixtures pass exact phase-alias full-adder/shared-instance timing, simultaneous SDC/multi-output all-input truth, and correlated complementary physical input pins. Temporary generated instances are not a hard memory bound.

The tested mapper source/CLI are now integrated as opt-in `EGM_CANONICAL_PHASE=1 EGM_CANONICAL_WIRES=1`, with the fixed trial additionally using small-cut quota8,cap diversity1,timing-cap1,mapped refs1/slack flow1. Default flags remain disabled. Main53 related tests pass/3ignored; source equals the tested isolated snapshot, while the root release binary has its own hash. A fresh main-binary full80-case comparison is in progress in `main_wire_checkpoint_20261014`; isolated results are not relabeled as main runs. Source, previous dirty mapper files/binary, patches and tests are saved in `canonical_wire_main_integration_20261014`. Main source/binary must stay frozen until this finite study ends.


Main-binary follow-up is now complete: `main_wire_checkpoint_20261014`80/80,44publicCEC/36import-node failures;22valid pairsD9better/13ties/0worse,A15better/2ties/5worse,DA20better/2ties/0worse. All eight full-baseline gates remain false. Main53tests/3ignored, release source matches isolated tested source; each binary has distinct preserved provenance. This is a complete root-binary fixed profile, not relabeled isolated data.

`main_wire_repeats_full_20261014` uses full-study r0 plus two new alternating-order repeats:30CEC covers/15pairs,all15DA improvements,D6better/9ties,A11better/4worse.4in ctrl/priority/dec consistently save0.37/7.43/0.36 area at unchangedD.4in barD-2.77 each,but one repeatA+4.64;3in barD-0.98 each,A+4.69/+5.52/+4.77. The full-study and every repeat are retained; fixed-wall recovery can yield different area endpoints, so bar area gains are not uniform. No per-design choice or statistical/optimality claim.

DAC TablesI-IV now contain the single fixed root wire-profile result for every EPFL design in both inventories; all five original baseline columns remain unchanged. All80 eGraphMap area/delay table cells match actual main reports or explicitN failures. Methods explain exact Not physical aliases/raw cut banks/pin-interface retention/mixed candidate ranks; discussion includes prior negative profiles, five area tradeoffs, repeat bar area regression and all failed gates. StandardIEEE default10pt/margins,6body+1references,all7pages visually checked; no overfull/undefined refs. `main_wire_paper_update_20261014` contains before/after source/PDF, numeric and rendered QA. Published PDF `output/pdf/DAC-eGraphMap-draft.pdf`. All finite runners ended and source/binary/fingerprints verified; no background campaign claimed.

Next research remains global imported-node capacity and area quality at fixed timing:exact physical aliases leave all raw nodes charged, so no remedy yet for nine large inputs. Any streaming regional SAT/import must preserve global node/pool/iteration/recovery accounting and full-circuit named-portCEC, not reset allowance per region. Preserve max/sin area trades and bar repeat variation; don't infer missing cuts from QoR alone. Baseline algorithms/budgets remain immutable and full results are already collected.


### 2026-10-14 capacity and bounded area recovery ablations

Original EPFL baselines remain fully attempted (240 grouped jobs / 520 slots); no baseline source, algorithm, model or native parameter was optimized. The current root wire mapper remains unchanged while the following explicitly versioned prototypes are tested. Every original PI and live AND remains an actual charged e-node; compact masked-AND operators encode input polarity, rather than granting a separate regional allowance. Exact complement identities preserve Boolean semantics; original pattern visibility can differ. Conditional SDC matches do not become equality aliases.

`capacity_area_checkpoint_20261014` completes 120 fixed-profile attempts:74 independent named-port CEC successes /46 import-node rejections. Reserving35% of the post-seed window for area recovery gives22 delay ties and8 area regressions. The compact polarity profile restores sqrt/square/arbiter/voter in both inventories (8 covers); among22 previously legal pairs, delays11better/11ties and delay-area16better/2ties/4worse. Its area increases in12 pairs. All twelve full-baseline gates remain false; no promotion.

`capacity_area_v2_checkpoint_20261014` completes120 attempts with the same74 CEC/46 rejections. Cheap proposal ordering avoids pre-evaluating every full DAG, but acceptance still requires an exact acyclic cover, lower unique physical area and nonworse delay. Area-only has5 area gains/7ties/10 regressions, all delays tied. Compact profile gives18 delay-area gains/2ties/2 regressions over22 legal pairs, with10 area increases;8 additional covers restored. All twelve gates false; no promotion.

`capacity_area_v3_checkpoint_20261014` completes80 attempts:52 CEC/28 node rejections. A conservative structural import-size rule selects compact polarity only when ordinary import would exceed the global20k allowance; this is not a design-name or QoR winner policy. Exact truth-toggle pin-dependency computation is checked against60000 sampled truth tables. The10% area-window profile gives22 delay ties,6 area gains/9ties/7 regressions, plus8 restored covers. All eight gates false; no promotion. Full source/binary snapshots, build/test logs, report/audit/input/library hashes and actual budget checks are in corresponding implementation/full directories.

Compact coverage reaches15/20 designs per inventory, leaving div/hyp/log2/multiplier/mem_ctrl beyond20k actual imported nodes. For example compact div40989 and hyp211714 nodes still fail; these counts are properties of this representation, not lower bounds on optimized Boolean implementations. Sqrt's recovered cover is slow (4in delay72318.50), so coverage is not a delay win over a failed control or conventional baseline. Pool100k does not cap transient generated instances. Recovery2s/20k nodes/16 cuts/100k retained pool/SAT16/mapping300s remain unchanged.

Fourth prototype is being evaluated with unchanged wire recovery ordering, no explicit area-stage reservation, exact physical-ID Vec sort/dedup area summation and structural compact fallback.60 regressions pass/3 ignored; original native tracked sources (13354 files) and12 actual dependencies were rechecked without drift. Its full comparison and alternating bar/max/sin repeats are pending; no integration or improvement claim yet.


Fourth fixed-version full comparison `capacity_area_v4_checkpoint_20261014` has now ended80/80:52 independent CEC successes/28 import-node rejections. Across22 originally legal pairs, all delays tie; area9better/13equal/0worse (delay-area9gains/13ties). Eight additional legal covers restore sqrt/square/arbiter/voter in both inventories; five designs remain import failures. All eight baseline lex/Pareto gates false. Global actual node/cut/pool/recovery/mapping checks pass; original baseline algorithms unchanged. Source versions are explicit: old root wire binary control versus isolated V4 fixed structural-auto profile, not casewise QoR selection. V4 uses no area-phase time reservation; compact fallback alone raises the in-budget return reserve to.15. The exact sorted/deduplicated physical-ID area sum preserves operation order and single-instance charging. Target alternating repeats are now running; main integration remains conditional. Complete per-case results and hashes in `capacity_area_v4_full_20261014`.


`capacity_area_v4_repeats_full_20261014` is complete:full-study r0 plus two alternating-order repetitions for bar/max/sin in both inventories give36 legal covers/18 pairs. All18 delays tie;17 area improvements and1 tie, no area regressions. No best-run selection or statistical confidence claim; bar benefit magnitude still varies. This supports optional source integration rather than a universal optimum or full-baseline dominance claim.

Source/CLI from the tested V4 snapshot is now integrated in main;60 related tests pass/3ignored and release built. `EGM_LITERAL_AUTO` remains opt-in/default0, area reservation/fast-area scan default0; the tested main profile uses unchanged recovery ordering and structural capacity selection. Exact pin dependencies and sorted unique physical-ID area charging preserve numerical semantics. Main binary7854b020ce97fe23d747e149d47cb2b3ea2fca2fd76f8144faf1b44e37aa234e has separate provenance; isolated data are not relabeled main runs. `capacity_area_main_integration_20261014` archives previous dirty sources/binary, exact source equality and patch/test/build evidence. `capacity_area_main_checkpoint_20261014` adopts only40 old immutable control observations after verifying archived old-version source/bin and unchanged executed dependencies; all40 new-version candidate observations run fresh with main binary. The current finite batch is active; root source/harness are frozen until completion.


Fresh main-binary full scope is now complete:40 newly run candidates give30 public CEC successes/10 import-node rejections, plus40 explicitly adopted immutable old-version controls. The80-observation comparison has52CEC/28 import failures,22 legal pairs with22 delay ties,8 area gains/13ties/1 regression (3in sin+0.72). The eight restored covers are sqrt/square/arbiter/voter in both inventories;five remaining designs still fail import. All eight full-baseline gates false. Source snapshot equals isolated V4, but binary/wall-time endpoints differ; this regression is preserved rather than replaced by the favorable isolated run. Main keeps the path opt-in; no claim of uniformly improved area or all-baseline superiority. Complete data/hash/budget proof in `capacity_area_main_full_20261014`. A fresh24-job alternating target-repeat batch with old-version archived control and new root binary is active; isolated repeat data are not relabeled root runs.


Main target repeats are now complete (`capacity_area_main_repeats_full_20261014`):36 independent CEC covers/18 pairs including full-study r0 and two new alternating-order runs on bar/max/sin in both inventories. All18 delays tie;15 area gains/2ties/1regression. All six bar pairs improve area (3in savings4.45/4.85/4.94;4in7.83/5.10/7.83), while magnitudes still vary.3in max r0/r1 tie andr2 saves.45;3in sin r0 costs.72,r1/r2 save15.04/8.64.4in max saves25.19/15.92/22.04 andsin2.89/2.08/9.23. Every run retained;no best-run selection or statistical confidence claim. All actual budgets and sources/report/audit hashes checked;root source/binary frozen and finite runners ended. No background continuation claimed.

Sin diagnosis (`capacity_area_main_full_20261014/sin_wall_endpoint.json`) additionally verifies exact serialized egraph nodes/final cut banks/input/roots/canonical/free records and retained-pool fingerprints are identical between oldcontrol andnew main. InitialA/D anddelayseed same;phase18trials/1accept each,local481vs490. Thus this particular regression is not explained by missing retained candidates; deadline endpoint sensitivity remains a supported hypothesis, not proof of host-noise causality.

DACTablesI-IV updated to single fixed new rootprofile for all20designs/bothlibs,all80candidate area/delay cellsverified andfive originalbaselinecolumns unchanged. Numericvalues use freshfull-study mainr0,never bestrepeats. Discussion preserves bad earlyarea profiles,restoredbutslow sqrt,5remaining importfails,3in sin+.72/allrootrepeat outcomes,fullbaselinegatesfalse/MtMmissing. StandardIEEEdefault10pt/margins,6body+1refs,all7pagesvisuallyreviewed,nooverfull/undefinedrefs. QA in `capacity_area_main_paper_update_20261014/paper_qa.json`;stableoutputPDFpublished. Optional capacity/metric path integrated;defaultstrategyflagsunchanged.

Nextpriority:certified mapper-native structural reduction for div/hyp/log2/multiplier/mem_ctrl without additionalresyn2 or exempting inputnodes;globalbudget-preserving large-cover candidate/timing quality (especiallyslow sqrt);deterministic or progress-aware local/cone scheduling for3in sin andbar variability. Required-time bounds musthold onallactualpins,sharedphysicalarea chargedonce,no regionalbudgetreset. Baseline240groups remain complete/original;no algorithm/model/nativebudget changes. This is partialprogress,not allbaseline dominance or complete solution ofnine large designs.


### 2026-10-14 delay-flow overflow investigation (current continuation)

Main capacity version and original240 baseline jobs remain unchanged. Allocation-free exact input-cost prototype `flow_cost_checkpoint_20261014` completed80/80:60 public CEC/20 import-node failures,30 legal pairs D2better/28ties;area9better/17ties/4worse. All eight gatesfalse;not promoted. Public sqrt metrics exactly reconstructed against originalABC/emap with source/input/prepared/library hashes.3in sqrt D74631.55->73036.32,4in75876.57->71803.83,still slower than conventional mapping. Four same-delay area regressions retained.61tests/3ignored and8000 bitwise input-cost checks pass;fixture-path failure and correction preserved.

Small whole-PI-support<=6 exact function audit finds div0/hyp2/log2 0/multiplier0/mem_ctrl13 same-function aliases;syntactic storage estimates excluding extra Not aliases40901/211587/29403/24685/46816 remain above20k. This is neither native alias import nor a mapped cover/CEC/feasibility or general impossibility proof;no originalAND exemption or budget reset. Exact-cut-boundary semantic sharing and certified structural reduction remain research needs.

Deep reconvergent delay-flow regression reveals a distinct bug:unfolded tree-area proxy can overflow toInfinity,then input is_finite treats a physically reachable signal as unavailable. The2200-layer finite acyclic fixture fails before repair. `flow_saturation_implementation_20261014` fixes only delay-area worklist tree-proxy overflow from already finite inputs by capping at f64::MAX;unknown-inputInfinity remains unavailable. Actual unique physical area/pin delay are never capped;complete-cover acceptance/budget/CEC unchanged. Weighted/AD proxy behavior is unchanged.63tests/3ignored pass,including known-overflow versus genuinely unavailable inputs. Source/bin/patch/red/green/build evidence saved. This is heuristic proxy saturation,not exact flow-area ranking or an optimality claim.

Selected sqrt physical covers have reference-one unfolded unit-tree bitlength1830–2008 and17–19 outputs with nonfinite additive tree-area estimates;all actual physical areas/delays are finite and reproduced exactly. These diagnostics support relevance of overflow,but are not a direct capture of native labels or a proof of latency cause.4in conventional path2078cells with1739AO31 versus current3137/315 andallocation-free2997/392;selected-path descriptions do not prove candidate absence.

Separate V2 fixed-source full80study in progress `flow_saturation_checkpoint_20261014`;controlcurrentrootcapacity versus bounded-proxy prototype,same flags/library/input/node20k/cuts16/pool100k/SAT16/recovery2s/mapping300.4in sqrt firstD71458.81->68891.86,CEC/budgetpass,bar tied andsinarea improved. All bad/missing cases retained;no main source integration or full-success claim yet. Need complete scope and alternating sqrt/bar/sin repeats before deciding integration. Baseline algorithms/models/nativeparameters unchanged;5import failures not solved by a post-import flow fix.


### 2026-10-14 confirmed proxy overflow and conditional state bound

V2 full `flow_saturation_full_20261014` completed80attempts60CEC/20 import failures,30validpairsD2better/28ties;area7better/20ties/3worse (4voter+.76,3bar+.06,3square+.98). All8gatesfalse. Repeated sqrt/bar/sin bothlibs (fullr0+twoalternating runs)36covers18pairs:6lowerD/12ties,14area gains/3ties/1regression (3bar+.06r0). Every sqrt pair improvesD/A:3inD-1730.00/-1804.79/-1815.85;4in-2566.95/-2696.59/-2575.66. Sin six area gains;4bar three ties;3bar two gains/one tinyregression. Source/binary/budgets/CEC/hashes verified; no per-design winner selection.

An independent AIGER decoder verifies2048 concrete input assignments and20001 pairwise distinct up-to-complement representatives for each remaining design. Witness-based original function-pair lower bounds20715/211561/28399/24684/34942 (div/hyp/log2/multiplier/mem_ctrl) exceed20k. Inequality is exact from differing truth bits;matching samples do not establish equality. The statement applies only to keeping all original scalar functions simultaneously represented in the scalar e-graph (even phase edges);it is NOT a minimum physical cell/area bound or impossibility of different state-retirement/representation algorithms. No random/equality merging was performed. `retained_signal_certificate_verified.json` records input/prepared/source/verifier provenance.

Raw AIG liveness plan shows frontier+allPI+zero peaks626/1758/1318/395/2162,far smaller than all-function counts. This does not include actual region egraphs/rewrite alternatives/interface duplication,matched pool,recovery/stitch/CEC. It is a motivation for certified retirement,not budget feasibility. Global live state/cumulative imported work must be separated and disclosed;no per-region counter or time reset,no opaque original formula exemptions,no same-old-import-total claim. Actual native region matching/stitch/global accounting remains unimplemented.

Tested V2source copied to main egraph_map.rs;63related tests pass/3ignored and releasebuilt. Prior dirty source/bin saved in `flow_saturation_main_integration_20261014`;production source equals tested snapshot. Rootnewbin247ab1119aafbc3f66d094c3db0209a6206a7fa38d689054c13eaaec7375798c;worklist defaultfalse remains,fix appliesonlydelay-area worklist andactualDAGmetricsuncapped. Fresh40rootcandidate casescomplete30CEC/10nodefailure;40immutableold-current-controls adoptedwith exactdependency/source/bin proof. Mainfull30pairsD2better/28ties,A7better/20ties/3worse,allgatesfalse. Rootsqrt3D72757.25/4D68675.32 remain slower thanABC/emap. New root alternating24job repeat is active;isolated results are not relabeled main runs. Five designs still fail;no wholeproject/SOTA completion claim.


Main flow-saturation studies have now ended. `flow_saturation_main_checkpoint_20261014`:40freshrootcandidates30CEC10import-nodefailures plus40 dependency-verified immutableoldcontrols,80observations60CEC20nodefailures.30legalpairsD2better28ties/A7better20ties3worse;all8gatesfalse. Main algorithm source equals the tested isolated V2;binary provenance is distinct. Every row uses one fixed profile,not best prototypes or repetitions. Originalbaseline13354trackedfiles+12actualdependencies recheckednodrift;baseline algorithms/native parameters unchanged.

Freshroot alternating repeats (`flow_saturation_main_repeats_full_20261014`) fullr0+24newjobs gives36CECcovers18pairs:6lowerD/12ties,A15better3ties0worse/DAQ15better3ties. sqrt all6 jointlyimprove:3in D−1700.25/−2841.25/−2602.49,A−30.11/−48.08/−44.30;4in D−2783.49/−1384.05/−1288.72,A−24.80/−29.59/−16.81. Sin all6 lowerarea atsameD (3−11.85/−.61/−18.04;4−5.28/−21.20/−10.54);3bar lowersarea−4.45/−1.42/−1.43,4bar allties. Magnitudevariabilityremains;notstatisticalconfidence/exactoptimality. Fullsuite3area regressions remain andare not replaced by repeats. Main63tests3ignored/release/sourcehashes andactualbudget checks pass;finite runners ended,locks idle.

DAC TablesI-IV now all80candidatearea/delaycellsverified against freshmainfullr0,baselinecolumns unchanged. Discussionincludes proxyoverflow red/greenfixture,uncappedactualmetrics,allrootrepeat ranges,earlierbadcases,30/40coverage/allgatesfalse andconditionaloriginal-function retention certificate.2048exactwitness assignments andindependentAIGERverifier distinguish>20k functionpairs perremainingdesign;this bounds scalar all-originalsimultaneousretentiononly,notoptimizedphysicalPPA/general synthesis feasibility. Rawfrontier liveness plan(395–2162 includingPI/zero) motivatesactualcertifiedstate retirement;no native regionalmatch/stitch/globalbudgetproof has been fabricated. Sixbodypages+oneref,standardIEEEdefault10pt/margins,all7visuallyreviewed/nooverfull/undefinedrefs. Source/PDF/numeric/rendered QA archivedin `flow_saturation_main_paper_update_20261014`;stablePDFpublished.

Nextwork:actual native bounded residency/certified retirement andphysicalstitch,accountallactivegraphs/cachedinterfacesglobally andoriginalAND materialization/cumulativework separately;no perregionnode/time/SAT/cut resets oropaqueformulapacking. Fullwhole-designCEC/commonmetrics andexplicitsame-budget audit beforepromotion. Also remaining sqrt4 gap:ABCpath2078cells/1739AO31vsretained-current3137/315 beforeflowfix;descriptivepathauditdoesnotproveabsentcandidates. Need auditactual retainedproducers/cutinterfaces beforechanging librarymatching/retention. Continue progress-aware/deterministicrecovery study forsin/barvariancewitholdbudgets;allbaselinealgorithmsfrozen. Five designs STILLimportnodefail/currentcoverage30/40,industrialMOT/ODC unassessed;projectnotcomplete/allbaselinewinnotclaimed.


### 2026-10-14 native retirement witness and exact sqrt phase audit

New isolated `research/state_retirement_20261014` constructs a physical structural witness with actual egg materialization of every original live AND. After256ANDs it certifies all live crossing phases against committed global physical producers, drops the previous egraph before rebuilding frontier symbols, and preserves global instance/signal identities. No independent regional saturation or optimized regional cuts:one structural cut/original class, zero SATiterations. One global2-second construction/recovery deadline includes mandatory setup and final global driver/topological checks; committed plus proposed physical candidates consume one100000pool without per-region reset. Actual resident peak and cumulative imported/rebuilt boundary work are separately recorded. Original input is not hidden in an opaque formula and baseline algorithms/models/parameters are unchanged.

Multiplier bothinventories:24556originalANDs materialized,651resident peak,31875committed physical instances,55628cumulative enodes including rebuilt interfaces. Public named-portCEC and all-instance/unloaded-pin metrics agree:area2502.800081,delay5925.149908. These are primitive witnesses, not improved mapping:ABCmap delays2688.12/2662.96. Cumulative>20000 means no prior-import-scope success or same-budget superiority claim. Four meaningful native tests pass:exact indexed/negative-PI/both-output phases across retirements,global pool cannot reset,resident/deadline rejection,and resident/cumulative distinction. An intentionally corrupted multiplier output is rejected by independent named-portCEC.

Fixed primitive fullEPFL20/twoinventory study `retirement_witness_epfl_full_20261014` completed40:37publicCEC,2hyp global proposed+committed pool failures,3in mem_ctrl independent evaluator/CEC60s guard timeout. All37measured delays worse thanABCmap.22CEC records meet the old cumulative-import threshold,15are resident-only diagnostics,not legacy-budget winners. Fullbadcases/commands/elapsed/hash kept. Fullr0 plus twofresh multiplier repetitions/inventory gives6CECcovers with identicalphysicalmetrics;initialpilot2covers are separate and not added tothis count. Currentproduction source/bin unchanged (247ab1119aafbc3f66d094c3db0209a6206a7fa38d689054c13eaaec7375798c),formal maincoverage remains30/40 andallpreviousgatesfalse. No productionpromotion or updatedmainPPA column.

Read-only exact saved-cut phase/pin/input-inversion replay `sqrt_retained_phase_audit_v2_20261014`:4in AO31 has9054unique replayed interfaces on1130criticalsignals,598exact interfaces retained on560signals,570signals without a retained replayed AO31 interface;441criticalsignals selectedAO31.199criticalsignals lose the fastest common-area-witness-estimated AO31 producer directly at cap.3in AOI21/OAI21 retained294/1885 of4236/6158 replayed interfaces;4in AOI21/OAI21 retained295/609 of3430/4862. This distinguishes known matching from retention loss,but is not a globally acyclic substitution/actual timing improvement or proof of all discarded cut absence. Missing transient cone/filter banks remain a limitation.7680 independent exact library-expression checks validate input transformations and both target phases. Initialv1 zero-retained result was invalid because library JSON delays requiref32 expansion before native diagnostic comparison;failedv1 source/report retained and explicitly rejected,not used as evidence.

Baseline native trackedsource/commits and frozen actualdependencies13403files checked:zero drift (`state_retirement_implementation_20261014/baseline_integrity.json`). Old source hashes for unexecuted Rust candidate code are not interpreted as baseline algorithm changes. Production binary itself is unchanged. Newprototype sources/bin/build logs/fullreports/repeats/negativeCEC are archived. Next:real bounded regional cut matching and shared cover/stitch accounting,then samefixedsource fullscope/CEC/repeats before any mainpromotion;sqrt cap diversity ablation must retain witness safety/globalpool100000 and verify wholecover rather than claim a local estimate as benefit. Do not replace main gates or baseline failures with these resident-scope diagnostics.

Final verification:four explicit legacy-scope delay-area/Pareto gates are false (twoinventories);resident-only covers retain their measured diagnostics but are failed for prior-import eligibility. Intentional f[0] corruption independently rejected byCEC. Originalbaseline complete520slots remain unchanged;candidate main columns remain unchanged. DAC discussion updated with37/40 primitive CEC/no ABC delay wins, multiplier651/55628 ledger distinction and sqrt AO31 retention audit. Layout repaired by condensing prose/combining discussion and conclusion, not changing font/margins or tablecells. Sixbody+one references confirmed,changedpages4/5/6 visually reviewed and1/2/3/7pixel-identical topreviousQA;publishedPDF. Every finite mapping/CEC/repeat batch has ended. Next optimized regional matching and fixed-cap coverage ablation are still required;do not claim task/project completion.


### 2026-10-14 regional exact cuts, shared cover and compound retention follow-up

`state_retirement_cuts_20261014` adds real bounded structural cuts (<=16 including trivial) and exact library phase/pin matching. Regional pending physical plans and committed prefix share one100kledger, checked before commit, with oneglobal2s setup/selection/finalverify deadline. Global cell/input/pin identity sharing and output-root traversal prune unused cells from emission. All originalANDs are actual egg nodes;SAT0/16,not an equality-exploration claim. Full40V1 cases:33CEC/7failures,only22eligible under prior cumulative-import interpretation;resident-only11CEC remain diagnostics. Non-best pilot retained separately. Multiplier3in3133.91/3075.63 and4in3700.01/2999.51 improve delay versus primitive2502.80/5925.15 but cost area;no mainpromotion.

V2 `research/state_retirement_cuts_v2_20261014` fixed structural policy:if complete original import upper bound<=20k, keep every originalclass in ONE egg graph,never retire/rebuild logical interfaces. Cut headers reference already-charged originalclasses and all leafIDs checked;header counts reported separately,cut/class<=16. For larger imports use retirement and report cumulative import,never treat resident peak as old-budget eligibility. Globalpool100k including pending plans/proposals,globalrecovery2s,externalmapping300/processguard60,shared prepared AIG/library unchanged.5native tests pass (truth/phase,all cut truth assignments,complete originalgraph across regionboundaries,constant roots,PI-only/deadline). Source/bin/lock/harness in `state_retirement_cuts_v2_implementation_20261014`.

Fixed fullV2 matrix40case completed:33CEC (30prior-import-scope eligible,3resident-only),7failures. Bothdiv andmem_ctrl plus4inlog2 hit independent evaluator/CEC60s guard;bothhyp globalprefix+pendingpool rejected before commit. All failures/commands/logs retained. `retirement_cuts_v2_epfl_report_20261014/paired_main.json` compares one fixedV2 profile to40fresh sameproduction-binary main controls from compound study:30eligiblepairs D11better/7ties/12worse;A3better/0ties/27worse. No per-design profile or source winner mix. Currentmain source/bin unchanged247ab1119aafbc3f66d094c3db0209a6206a7fa38d689054c13eaaec7375798c;V2 notpromoted despite targetgains.

Fullr0 plus twofresh targeted repeats/inventory:12CEC covers for sqrt/multiplier,all metrics identical within each three-run group andglobal pool/time/cuts compliant. sqrt3area2839.329986/D60295.990438;4area2875.460015/D48010.831306,alloriginalclasses retained,zero retirements,cumulative19566/resident<=20k. These are genuine prior-import-budget eligible covers,still slower thanABC60089.98/47071.95. Multiplier3area3133.910004/D3075.630003;4area3700.010007/D2999.510015,stillcumulative55628 andineligible underoldscope. Repetitionstability is not globaloptimality/statisticalconfidence;oldpilotV1 is not added toV2counts.

Separate `/tmp/emapper-cap-compound-20261014` opt-in EGM_CAP_COMPOUND_QUOTA=4096 reserves existing optional slots for one faster4-input compound per witness-criticaltarget;all originalwitness bindings preserved andtotalretainedpool<=100k. Generic compound definition excludes pureAND/OR/NAND/NOR;not benchmark- or AO31-name-special casing.64active tests pass/3ignored,includingquota-vs-roundrobin witness/globalcap fixture. Fixed80freshjobs (twoinventories20designscontrol/compound) complete:29legalpairs,D0better28ties1worse/A4better19ties6worse;4insqrtcandidate process60stimeout aftercontrolsuccess. Cause stage unestablished,no runtime/PPA win claimed.3in no4inputcompound butclock-end variation remains;do not assert its regression proves candidate absence. Source/Cargo/bin/harness andall failures archived incompound_cap_implementation_20261014;notpromoted.

Bothprofiles eight explicitfullscope delay-area/Pareto gatesfalse inretirement_cuts_v2_epfl_report_20261014. Baseline native trackedsource/commits andfrozen actualdependencies13403files recheckedzerodrift;ordinarybaseline240groups/520slots remainoriginalandunchanged. Next:protect validated incumbent andadd required-time-constrained actual unique-area recovery within ONE graph/pool/globaldeadline;not posthoc benchmark-winner merging. Investigate compound4sqrt timeout with explicit stage diagnostics before any newmatrix/repair;do not rerun unchangedtimeout orraise limits. All finitefull/repeatbatchesended. RequiredMOT/ODC andremaininglargelegacy-budget failures unresolved;projectnotcomplete.

Final publication:mainTablesI-IV unchanged;separate discussion reportsV2whole-original-class policy/19566sqrt cumulative nodes,all30eligiblepaired improvements AND regressions,retired multiplier ineligibility,andnegativecompoundreservation.6body+1reference standardfonts/margins;page6 visualQA,1–5/7 pixelidenticaltopreviousQA;publishedPDF. All full/repeat nativeprocesses haveended. No productionpromotion or perdesign/rerun-best mixing.

## 2026-10-14: isolated fixed-bound actual-area recovery, full EPFL repeated

Production source and binary are unchanged (release SHA256 `247ab1119aafbc3f66d094c3db0209a6206a7fa38d689054c13eaaec7375798c`). No experimental result replaces a production table entry or becomes a per-design winner. Baselines remain original: 240 grouped jobs / 520 status slots, including all failures and unmeasured slots. Native sources, commits, models, binaries and actual dependencies were independently rechecked: 13,403 files, zero drift. An initial broad audit flagged the unexecuted Rust mapper CLI; that rejection is archived and the explicit candidate-only dependency exclusion is recorded, not generalized to baseline helpers.

`research/state_retirement_cuts_v3_20261014` first added one minimum-area physical proposal per phase and global actual-area trials. Its eight target attempts are preserved in `actual_area_recovery_targets_20261014`: five CEC passes and three explicit global-deadline failures. No failed attempt is adopted as success. The subsequent isolated V4 uses one descending-arrival required-time pass for positive dependent-pin delays and a 0.15-second return reserve plus measured previous-trial cost estimate, all inside the same global two-second construction/recovery/final-validation window. Six unit tests pass, including slower replacement rejection, cycle rejection and expired-window incumbent preservation. Existing node/cut/pool/iteration/process/external limits are unchanged. Individual trials are not preemptive hard bounds; an actual global overrun remains failure.

V4 retains exact-cut area alternatives in the same global physical pool, including pending dependencies and proposals in its 100,000 allowance. After validating a whole-design cover, its delay becomes the fixed bound. Every trial recomputes unique emitted-instance area and global dependent-pin delay; only strict area reduction within the bound is accepted. Invalid or late trials restore the original binding. This protects the regional initial incumbent, not the production mapper's incumbent, and does not claim global lexicographic or exact-area optimality.

Two independent full matrices, `actual_area_recovery_full_20261014` and `actual_area_recovery_full_repeat_20261014`, use the same frozen source/binary and once-resyn2 inputs for all 20 EPFL designs and both inventories. Combined: 80 observations, 68 CEC passes, 60 legacy-import-budget-eligible covers, 12 failure observations. Each run has six global pending-dependency pool-reservation failures (div, hyp and mem_ctrl in both inventories); all argv/logs/partial artifacts/hashes remain. All successful cases pass the actual fixed-delay/global-budget audit. In the first run, thirty eligible paired V2 controls have delay 0 better / 30 equal / 0 worse and area 29 better / 1 equal / 0 worse. Against the same fresh production controls: delay 11 better / 7 equal / 12 worse; area 7 better / 0 equal / 23 worse. Thus V4 is NOT promoted.

All successful repeated delays agree, but twelve design/inventory cases vary in area with the wall deadline; two runs do not establish statistical confidence and no best repeat is selected. Sqrt 3in delay 60295.990438, area range 2833.169986–2833.979986; 4in delay 48010.831306, area 2867.340015. Both retain every original class in 19,566 cumulative nodes. Bar areas 282.130006/250.820006 agree across repeats, but delays 254.760002/216.030001 still regress versus production. Sin areas vary by 1.14/0.72 while delay remains 2112.149992/1882.000013. Multiplier areas 3128.770004–3130.460004 / 3695.620007–3695.690007 and delays 3075.630003/2999.510015 are diagnostic only: cumulative import is still 55,628, not a legacy-budget success.

Evidence and six false full-scope gates (two inventories × delay-area, area-delay, Pareto) are in `actual_area_recovery_report_20261014`; area-delay is a posthoc gate on this delay-first/fixed-bound area profile, not a separately optimized area-first experiment. Native sources/binaries/drivers/tests/build logs, baseline audit and first failed prototype are preserved in `actual_area_recovery_implementation_20261014`. Complete repeated pairs are retained; no failures dropped or sources mixed.

Next: preserve the production seed inside ONE imported graph/pool/global deadline, then admit regional alternatives without losing its candidates. Address candidate loss across region boundaries and improve constrained actual-area proposals rather than rerunning unchanged failures or selecting outside-budget mapper winners. Multiplier requires certified pre-import elimination or a different explicitly proved representation: state retirement alone cannot reduce the count of already imported original ANDs. The exact separating certificate remains conditional on retaining all original scalar functions, not a lower bound on physical area or all possible mappings. Required industrial multi-output/ODC and independent regional saturation remain incomplete. All finite batches have ended; project is not complete.

Paper publication: new constrained-area results and negative main comparison are separate discussion evidence; all Tables I–IV tabular source unchanged. Standard 10pt/IEEE margins, six body pages plus one reference page. Pages 5–6 visually verified; 1–4/7 pixel-identical to prior verified pages. Rejected eight-page layouts are archived, not fixed by font/margin compression. Published PDF updated; all finite full/repeat processes ended.

## 2026-10-15: delay-focused read-only cause analysis

No new mapping run, changed baseline, production promotion or per-design winner selection. `delay_cause_analysis_20261015` independently reconstructs 120 saved main/regional/ABC/emap netlists for all thirty prior-import-eligible cases, plus twenty original E-morphic author-selected native winners. All all-instance area and unloaded dependent-pin delay values match the existing public evaluator; source input/prepared input/library hashes match. Failed or missing baseline cases remain failed/missing. The fixed production controls are the fresh first-run `compound_cap_epfl_full_20261014` controls, not the earlier publication r0 or a selected best repeat.

All thirty production final delays equal their recorded seed delay. Both sqrt controls have zero local trials. The saved counters do NOT distinguish witness/setup, individual seed computation, proposal preparation and sorting costs; the source's first ordinary local pass can collect/sort the full proposal list before checking the deadline at trial entry. Thus zero trials do not prove that flow alone consumed the window. The fixed clean-environment driver does not enable EGM_CRITICAL_RECOVERY (CLI default 0); proposal priority is estimated area change although whole-cover acceptance uses delay-area. Phase-quarter is a fraction of the remaining window after seeds, not a reservation that guarantees local-delay time.

Selected-path evidence: bar3 main/regional/ABC has 9/12/9 cells and delay190.87/254.76/190.87; mean main/regional arc21.21/21.23. Bar4 has9/10/8 cells, delay188.34/216.03/171.90. Endpoints differ, so extra stages are descriptive association, not a matched-cone causal proof. Sqrt4 paths all end asqrt[0]: main/regional/ABC3061/2130/2078 cells, AO31 counts354/1628/1739, delays73633.69/48010.83/47071.95. Regional retains all original classes within19566 cumulative nodes; remaining delay gap to ABC1.99%. Sin4 regional1882.00 remains slower than ABC1857.54/emap1788.13.

Original E-morphic priority4 native winner tmp_41 has50 cells/D1071.64 versus main102/D2047.24; the same AOI21/OAI21 families dominate. Main priority has2190 e-nodes,37608 producers and no node-budget skips; BOOLEAN0/LIBRARY_TIMING0 installs no explicit association or bounded timing association. This shows structural-representation space, not a lack of faster library cells or an exhausted global cap. Native SA/ABC feedback budgets remain separately disclosed; these paths do not establish equal end-to-end compute or that our two-second recovery can reproduce the winner. No SA candidates were reranked by the public metric.

Pool diagnostics under one common AREA witness show1464/1501 critical-target best-estimate losses for sqrt3/4, but zero such losses for bar, sin and priority. These are local estimates, NOT feasible replacement benefits or lower bounds. Zero loss does not prove optimal cut coverage. V4 clears nontrivial frontier cut banks every256 ANDs and sorts cuts by arity/leaf-ID/TT; this restricts candidates but does not establish the cause of any individual regression. Keep cut, phase, generation/cap and extraction failures distinct.

Priorities: (1) one-import production-incumbent protection plus bounded regional alternatives; (2) explicit stage timing, critical-first top-K trial preparation and reserved local-delay time without raising trial/time caps; (3) preserve exact frontier cuts and both phases inside16 slots/100k global pool, particularly bar/sqrt; (4) targeted exact reassociation/factoring/prefix alternatives for priority/max within20k nodes/16 iterations; (5) fixed required-time actual-area recovery after a stronger delay stage, with joint phase/cone updates and complete-cover validation. Avoid simply increasing AO31 quotas: the prior4096 compound experiment had no delay gain and a timeout. Multiplier cumulative55628 remains unresolved; retirement alone is not an import repair. Detailed limitations, all paths, exact source/report hashes and proposed single-factor/full-suite validation protocol are in the new report. Main source/binary, paper metric tables and original baselines are unchanged.


## 2026-10-15: instrumented delay seed and required-time DAG area flow

Production defaults and baseline algorithms/models/parameters remain unchanged. Three isolated native versions, sources/patches/binaries/tests/build logs archived under `delay_schedule_implementation_20261015`, `delay_frontier_implementation_20261015`, and `delay_area_dag_implementation_20261015`, use one global recovery2s (including witness/setup/seeds/final validation), nodes20000, cuts16, retained pool100000, SAT16, mapping-only300s. Diagnostic serialization is outside recovery and counted conservatively by the process clock; no runtime superiority is claimed. Retained pool remains distinct from transient generated-candidate RAM. No per-region budget reset or exemption for original input nodes.

Schedule pilot: forty CEC covers across five designs/two inventories/four fixed profiles. Critical-first and bounded top-K proposals alone did not consistently improve delay, and the earlier seed-window profile weakened sqrt's starting cover. Stage traces locate sqrt's dominant work in seed propagation, not solely proposal preparation. Scheduling version tests65 pass/3 ignored; finite-frontier version67/3; required-time area-flow version69/3. Initial failed test/build/report attempts remain archived.

The finite-frontier seed releases physical producer/output bindings when all unique physical inputs have finite drivers, keeps repeated dependent-pin arcs, and constructs an acyclic feasible witness. It is not proven globally delay- or lexicographically optimal. Existing event-driven flow is warm-started from that full witness. Full two-profile scope `delay_frontier_full_20261015`: eighty slots, sixty CEC successes, twenty explicitly adopted unchanged pre-recovery import failures. Versus same-binary legacy control, thirty legal pairs D2better/28equal/0worse; A12better/8equal/10worse. No promotion.

Required-time DAG area flow minimizes a sharing proxy subject to backward dependent-pin required times and topological feasibility; actual complete-cover area/delay validation decides acceptance. Invalid/expired trials retain the incumbent; actual unique area accounting is not globally minimum area. This fixed profile replaces the third structural-reference seed and uses35% of the remaining window for its sweep; flags and source are frozen for all designs. Three-profile full scope `delay_area_dag_full_20261015`:120 slots,90 legal CEC covers,30 explicitly adopted pre-recovery import rejections. Sixteen same-version successful pilot slots are reused;74 successful full-campaign slots are new. Do not call all120 fresh mapper runs. All successful actual budget and artifact checks pass; paired retained candidate fingerprints match.

Versus same-binary legacy:thirty legal pairs D2better/28equal/0worse,A24better/2equal/4worse. Sqrt3 control A2921.30/D72496.92 -> A2154.29/D68523.99; sqrt4 A3819.37/D69541.91 -> A2735.82/D66924.64. Bar3/4 A289.64/304.64 ->263.25/282.77 with D190.87/188.34 unchanged. Sin3/4 A615.70/712.58 ->441.88/456.61, max3/4 A284.77/284.64 ->242.40/212.14, all four delays unchanged. Four area regressions remain:3in adder+3.72,3in priority+0.85,4in priority+0.92,4in int2float+0.07. Versus fast-frontier control, Dall30tie,A25better/1equal/4worse. Reports retain all twenty designs and failures; no per-design source/profile mixture.

Targeted full-r0 plus two fresh alternating-order repetitions cover sqrt/bar/sin/max/priority/int2float/ctrl/adder in both inventories. Ninety-eight observations include97CEC and1unmeasured driver failure caused by an operator directory move while a host runner was active but invisible in the sandbox. Directory restoration, failed argv/log and corrected audit are retained; the lock rejected duplicate execution. An independent fresh sin pair is additional evidence, not an overwrite or dropped failure. Forty-eight legal pairs: D6better/42equal/0worse,A36better/0equal/12worse. All six sqrt pairs improve both metrics; candidate delays are68523.99/66924.64 across repeats. Bar candidate areas263.25/282.77 agree across three observations. Sin candidate area ranges441.77–441.95/456.37–457.68; max242.40/212.14–212.20. The same four area regressions persist in all three pairs, with priority4 area53.47–55.73. Small repetitions do not establish statistical confidence or deterministic wall-deadline behavior. All repeats/pool/budget audits in `delay_area_dag_repeat_report_20261015`; no best-run selection.

Original baseline full520 status slots remain unchanged. Actual dependency/native tracked source/commit audit checks13403 files with zero drift. Production binary remains247ab1119aafbc3f66d094c3db0209a6206a7fa38d689054c13eaaec7375798c. All twelve full20 baseline gates (frontier/required-area versions × two inventories × delay-area/area-delay/Pareto) are false; AD is only posthoc on DA covers, MtM unconnected, invalid baseline observations retained. Five original large import failures remain div/hyp/log2/multiplier/mem_ctrl; state retirement does not repair cumulative imports. No SOTA, default promotion or task-completion claim.

Next: protect legacy shared-area incumbents within the SAME candidate pool/global deadline while preserving the stronger delay witness; audit whole-cover gains rather than proxy area alone. Continue exact cut/phase and structural-alternative work for sqrt/priority and certified pre-import reduction for large designs in explicit new versions, with fullscope and repeat validation. Do not increase AO31 quotas or retry unchanged node failures without new evidence. Industrial multi-output/ODC and independent regional saturation remain incomplete.

Paper update: isolated full-suite and targeted-repeat evidence added to discussion; original main Tables I–IV remain byte-identical. Rejected eight-page draft retained; concise prose restores standard10pt/IEEE margins to six body pages plus one reference page. Changed pages4–6 visually reviewed; pages1–3/7 pixel-identical to previously verified pages. No overfull/undefined references. Published PDF and source/render QA stored in `delay_area_dag_paper_20261015`.


## 2026-10-15 shared-incumbent follow-up (in progress; no main promotion)

Three explicit source versions follow the earlier area-DAG study. `shared_incumbent_pilot_20261015` only defers the validated area-DAG result until after phase/local recovery:12CEC,6pairs,Dalltie,A1better2equal3worse; rejects promotion. `shared_seed_full_20261015` additionally restores cold legacy seeds, original tie/witness/structural-reference setup, with a25% remaining-window seed slice. Full80slots:60CEC/20explicit unchanged pre-import-failure adoptions;12 successful same-source pilot jobs reused and48 successful full jobs new. Thirty legal pairs D2better28equal0worse/A19better11equal0worse and matching pool fingerprints/budget/hash checks. However full-r0 plus two alternating-order targeted repeats (`shared_seed_repeat_report_20261015`) yields48CEC/24pairs,D6better18equal0worse,A9better13equal2worse:4in adder areas83.75->84.49 and83.75->83.81. The full first-pass gate passes, repeated gate FAILS, so source remains isolated, not main.

Stage replay of4in adder shows identical first five local-recovery area/trial sequences;~30ms frontier/DAG setup competes with the last slack-area substitutions. This supports a budget-competition mechanism; not merely scalar seed-PPA differences. `progress_incumbent_full_20261015` is a separate70-test version: bounded cold seeds report worklist convergence, frontier runs only after nonconvergence or in naturally unused legacy tail time. Full matrix currently executing finite48new jobs; already observed4in max area278.27->279.75 at equalD, despite settled seeds and no early frontier. Retain as negative evidence and do NOT promote on the successful target pilot. Both sqrt delays still improve in observed covers. Do not mix source versions or pick per-design profiles.

Next isolated source `/tmp/emapper-epoch-incumbent-20261015` is based on shared_seed plus an epoch-invalidated per-thread arrival workspace. It preserves ascending physical-ID area summation, per-pin max-delay arithmetic and all candidate/order/trial rules, seeking lower exact-metric overhead rather than a larger budget. Source is prepared but tests/build/full validation must wait until the finite progress-mapping batch ends. Default production flags and binary remain unchanged; baselines original algorithms/parameters/models remain unchanged. No claims about epoch performance or correctness before tests and independent CEC. Main integration still requires full20/twoinventory nonregression and repeated sqrt/adder/priority/int2float verification, followed by separately recorded root-binary validation.

## 2026-10-15: protected shared incumbent integrated and main-binary verified

The previous “in progress” entry is superseded by completed explicit version studies. Deferred-only V4 left three pilot area regressions; cold-seed V5 passed its full matrix but failed two four-input adder repetitions. Progress-only V6 left two full-suite area regressions. V7 per-cover epoch metrics removed those observed pilot regressions but retained post-recovery diagnostic/evaluator timeouts. All negative sources, binaries, commands, timeouts, and reports remain in `shared_incumbent_*`, `shared_seed_*`, `progress_incumbent_*`, and `epoch_incumbent_*`; no per-design winners are assembled from these versions.

V8 adds exact diagnostic cell-name memoization and one ordered producer index. Golden checks cover all 611 unique `(k, truth-table)` queries from 226,111 sqrt cuts; 64 synthetic candidate pools compare index results, order, duplicate-target and pin behavior to the old scan. Its isolated full study (`epoch_report_full_certified_20261015`) has 59 direct CEC plus one explicitly composed certificate and twenty unchanged pre-recovery rejection adoptions, not eighty fresh native runs. Thirty legal pairs: D2better/28equal/0worse, A19better/11equal/0worse. Its 32 fresh targeted repetitions plus sixteen full observations give 48 verified observations (47 direct, one composed), 24 pairs, D6better/18equal/0worse, A9better/15equal/0worse. Both nonregression gates pass. The original direct timeout is immutable; verification policy/source hashes and negative controls are archived. Eight eligible original-baseline hyp supplementary verifications all fail under the unchanged 60-second verifier guard; canonical baseline statuses are not repaired or removed.

After that validation, the implementation was integrated into `MapParams` and CLI as explicit `fast_delay_seed`, `required_area_dag`, `defer_area_dag`, `preserve_legacy_seeds`, `metric_epochs`, `bounded_proposals`, and `seed_window_fraction` options. The first five are enabled by the tested candidate; bounded proposals remain off. Existing defaults/presets remain unchanged. `emapper/RECOVERY.md` documents the complete campaign settings, not just individual toggles. The typed API has recorded red/green workspace-wiring coverage. Ordinary debug and release egraph tests each pass 75, with four explicitly ignored. The original 2200-layer overflow fixture runs on its own larger test-thread stack; CLI/runtime/node/time/trial settings are unchanged. Old source/binary, new source/patches/binary/fixtures/Cargo locks, logs, and hashes are in `epoch_main_integration_20261015`.

A separate **fresh main-binary** matrix `epoch_report_main_full_20261015` completed all 80 attempts (two inventories × twenty designs × legacy/protected profiles), without adopting prototype mapping results or old import rejections. It yields 60 direct original-input named-port CEC successes and twenty explicit node-budget failures. Thirty legal pairs: **D2better/28equal/0worse; A20better/10equal/0worse**. All successful actual budget checks, paired retained-pool fingerprints and artifact hashes pass. `epoch_report_main_full_certified_20261015/nonregression.json` passes. Main binary SHA256 is `76655d8ba89ba4a80382be6c7ca570e7941d3f1501250d574dc061eef3c31aa6`; public evaluator remains `bb14affbd7bbf79560436850b7122b0d45932d673894b56ea5260e4255462024`.

Main targeted repetitions `epoch_report_main_repeats_20261015` add 32 fresh attempts for adder/priority/int2float/sqrt; all pass direct original-input CEC. Profile order reverses on the second repetition. Including sixteen corresponding full observations gives **48 observations / 24 pairs: D6better/18equal/0worse; A12better/12equal/0worse**. All six sqrt pairs strictly improve delay. No area regression occurs on adder, priority or int2float. Four-input adder candidate area ranges 83.75–83.88, not a best-run selection. Sqrt candidate A/D are 2159.11/68523.99045 (3in) and 2743.80/66924.64062 (4in), identical across the three candidate observations; legacy control varies. Main full R0 control sqrt delays are 72930.20034 and 70473.09060. Small repeats do not establish statistical confidence or load-independent behavior. Full/repeat reports contain every raw observation; no isolated/native results are relabeled as main runs. Main uniquely executes 112 attempts: 92 direct CEC successes and twenty node rejections; the 48-observation repeat summary overlaps sixteen full results and must not be added wholesale to the full count.

Mechanism: preserve the legacy cold seeds/ties/witness/structural-reference working trajectory, save frontier and required-time DAG covers as separately validated alternatives, defer commitment to complete-cover comparison, and reduce repeated exact metric clearing with per-cover epochs. This protects discovered valid incumbents without promising the result of an independent legacy run under arbitrary wall-deadline load. All alternatives share one imported graph, one retained pool and one global recovery window: actual enodes≤20000, cuts/class≤16, retained instances≤100000, SAT≤16, recovery≤2s including setup/final validation, mapping-only≤300s with original process guards. No extra trials, per-region reset or input-node exemption. Retained cap is not a transient generated-memory cap; process timing includes diagnostics conservatively, so no mapping-runtime speedup is claimed.

Original baselines remain complete: 240 groups / 520 status slots, 372 CEC passes / 124 reported failures / 24 unmeasured. Original algorithms, commits, native parameters, models, binaries, common evaluator and actual harness/library/input dependencies were rechecked: **13403 files, zero drift**. Explicit unexecuted Rust candidate exclusions are recorded, not generalized. All 520 baseline statuses (including both objective configurations/staging controls) remain in the new scope. EqMap successes retain conservative enclosing-Yosys-plus-direct mapping upper bounds under 300s; these are not exact mapping runtimes. All six two-inventory full20 `check_mapping_dominance.py` gates (DA, AD, Pareto) remain false; AD is posthoc on these DA covers. MtM is unconnected. See `epoch_main_integration_20261015/baseline_scope` and `baseline_integrity.json`.

Five large designs still fail: div, hyp, log2, multiplier, mem_ctrl in both inventories. Resident retirement does not repair the old cumulative-import allowance, and the new metric metadata does not exempt Boolean functions. Independent regional saturation, industrial multi-output and ODC remain unfinished. Next work is certified pre-import functional reduction/representation and sqrt whole-cover compound/phase quality under explicit fullscope versions, not raising budgets or selecting per-design winners. This recovery configuration is integrated and validated as opt-in, not a default switch, all-baseline win or project-completion claim.

Paper Tables I–IV now use the one fixed fresh main candidate profile; all baseline cells are byte-identical to the prior manuscript. New provenance links every eGraphMap numeric/status cell to its report/hash. Full/repeat outcomes, negative earlier versions, actual budget scope and five remaining failures are disclosed. The document remains standard IEEE10pt with unchanged margins, at most six body pages plus one references page; final compiled/rendered QA is archived in `epoch_main_paper_20261015`. All finite mapping/CEC/repeat batches have ended.

## 2026-10-15 follow-up in progress: import representation and pre-cap delay paths

Main binary/defaults remain unchanged from the validated shared-incumbent version (`76655d8...`). Current finite batch is the isolated `tt3_import_pilot_20261015` (log2/multiplier/sqrt, both inventories, current protected control versus one fixed alternative). It must end and be audited before compilation or another mapping campaign; checkpoint instructions and current session are in `frontier_pool_diagnosis_20261015/progress.json`. No new full-suite or main-promotion claim yet.

Five original compact import failures are actual counts40989(div),211714(hyp),29404(log2),24685(multiplier),47384(mem_ctrl), above20000 before recovery. Exact pre-import XOR recognition, retaining every PI/required root and any independently used interior, produces multiplier actual18896 imported enodes and19998 after saturation. Two inventory pilot maps independently CEC-pass within actual cuts16/pool100000/recovery2s. Their area/delay are2032.34999/3424.83999(3in),2099.43999/3457.95999(4in), slower than ordinary baselines; coverage improvement only. All actual logical operators count globally, no state retirement, regional reset or hidden function cache. Original24556 ANDs are separately reported: this is an explicit richer representation, not retaining every original scalar function or relabeling an expanded-AIG budget. XOR-only intermediate full checkpoint currently16/80; every good/bad observation remains, no claim full comparison complete.

Four explicit pre-cap recovery prototypes each complete four sqrt attempts, all directCEC. Current area-ranked witness and roundrobin cap can discard a complete faster path: a four-instance regression fixture losesD2->10 without root-witness reservation, passes with protection. Native three-input sqrt pre-cap path reservation reachesD60087.260389 (same as original emap), but area2193.159982 exceeds current control2159.109990. Four-input searches do not finish their .5s stage window; no pre-cap optimum or absence claim. Deferring the fast cover to constrained slack recovery, removing per-instance tree allocation (128graphs binding-identical to old readiness), and bounded DAG area sweeps do not remove the measured area tradeoff. All negative variants/timeouts/initial compile errors are preserved; none is selected per design or integrated. Their additional pre-cap work is debited from the original recovery allowance; normal candidate-cap construction is separately retained in mapping time, and no uninterrupted pre-cap-through-cap two-second wall interval is asserted.

Exact two-AND cone contraction into explicit TT3 nodes is a separate prototype. Fixed source-size threshold80% of node allowance controls dispatch, independent of benchmark names/PPA; small former legal inputs retain the old path. Full truth tables and actual children reside in counted nodes. Only unshared interiors are removed; raw AND size is reported. Bounded one-leaf cut substitution avoids an unbounded16^3 Cartesian bank; actual cut cap remains16. 77egraph tests pass/4ignored, including16edge-phase cases, unused inputs and shared interior outputs; actual EPFL mapping/CEC/global budget validation pending. Planning-only upper counts19141log2/15926multiplier/11940sqrt do NOT establish feasibility or PPA. Remainingdiv/hyp/mem_ctrl estimates still exceed20000. Baseline original algorithms/models/parameters and canonical520slots remain unchanged.


## 2026-10-15: bounded exact cone import, full EPFL and alternating repetitions

`cone_guard_implementation_20261015/report.md` is the current complete isolated-study record; its source/binary/patch/test/build hashes are archived alongside. Production Rust/CLI and binary remain unchanged at `76655d8ba89ba4a80382be6c7ca570e7941d3f1501250d574dc061eef3c31aa6`; no experimental value replaces a production table cell.

The fixed90%-capacity rule preserves original import when it fits the initial reserve, otherwise uses exact XOR/TT3 nodes plus up to2048 actual helper functions ranked by structural slack. TT3 holds its full truth table and all real child IDs; source physical certificates only reference imported functions, use the ordinary16-cut bank and100kretained pool, and perform no conditional union. Every actual node counts globally; no retirement/reset or resident/cumulative substitution. This richer operator representation is explicitly **not an unchanged expanded-AND work-unit comparison**. Source AND counts and prototype occurrence-counter semantics remain separate from actual total enodes. The failed80%-threshold study (`tt3_helpers_full_20261015`) retains its3in square delay regression2707.68->2723.04. The new single90%rule also gives up the favorable4in square result; no per-design/library selection.

`cone_guard_full_20261015` completes80fresh attempts:64direct original-input named-port CEC,16explicit node failures. Candidate34/40legal covers vs control30/40. Log2 and multiplier now pass in both libraries with actual imported nodes18103/16630 respectively; sqrt14603, all successful post-SATnodes<=20k, cuts<=16, retainedpool<=100k, ordinary recovery<=2s, mapping<=300s. First-run sqrt3 area/delay2159.11/68523.99->1814.66/63951.92; sqrt4 2743.80/66924.64->2188.56/57161.09. These reduce delay6.67%/14.59% and area15.95%/20.24%, but remain6.43%/21.50% slower than original delay-mode emap. Thirty old-legal pairs: D2better/28equal/0worse, A2better/27equal/1worse. The4in adder area83.75->84.49 is retained; strict full self gate is false.

`cone_guard_repeats_20261015` adds56status slots over seven declared targets and two alternating-order repetitions:48fresh native attempts plus8explicitly adopted unchanged original import failures. Together with the full matrix:128unique native attempts,112directCEC,16native node failures. The report's84-observation target view includes28full observations, not84fresh mappings or a second full-suite repeat. Six sqrt pairs have identical metrics; all12new log2/multiplier candidate observations pass with identical per-design/library metrics. Target view D6better/24equal/0worse; A7better/22equal/1worse, retaining the original adder regression. Four fresh4in adder observations all have area83.75, but never overwrite84.49. Trace choices before the late area stage agree; deadline truncation is a supported hypothesis, not a proved exact interrupted operation. The strict repeated self gate is also false. **Not promoted.**

Actual minimal candidate imports div28644,hyp143890,mem_ctrl29791 exceed20k even with zero optional helpers. These three remain failed; production still has all five import failures because this version is isolated. These counts are not lower bounds or proof of impossibility under another exact representation. Eight saved-netlist path reconstructions reproduce all public metrics: sqrt3path3214->2999cells vs emap2951; sqrt4 2872->2613 vs emap2077. Four-input candidate path has512AO31 vs emap1742. Candidate finite common-witness best-estimate cap losses are zero on its final-critical set; this does not prove retention of all useful candidates or a feasible optimum. Boundary/cut/phase-chain correspondence remains to audit.

Baseline audit in `cone_guard_baseline_scope_20261015` rechecks13403actual dependencies/native tracked files with zero drift. Canonical240groups/520slots remain372CEC/124reported failures/24unmeasured failures. All six full20 baseline gates(two inventories xDA/AD/Pareto) remain false; AD is posthoc on DA covers, not a new AD optimization run. All baseline algorithms/models/native parameters unchanged. Same once-resyn2 inputs, libraries, public metrics, named-port CEC and original process guards; mapping-only300s excludes conversion/prep/CEC/metrics. EqMap nested LUT mapping retains its conservative enclosing-Yosys time bound; no precise runtime claim. MtM unconnected.

Next: preserve validated shared-area progress inside late flow with a real reconvergent fixture; do not loosen gates or repeat until lucky. Audit sqrt baseline AO31-chain functions against admitted boundaries/cuts/phase partners using exact certificates after any signature prefilter. Investigate bounded exact shared-cone representations for the three imports while charging all actual functions. New source/config requires a new full study; no source/version/first-vs-best-repeat mixture. All finite batches ended; project and baseline dominance remain incomplete.

Publication: the DAC discussion now reports the isolated full/repeated cone study, richer-node-unit caveat, remaining three prototype failures and retained adder regression; production TablesI-IV are byte-identical. Six body pages plus one reference page, standard10pt/margins unchanged, no overfull boxes or LaTeX warnings. Pages5-6 visually checked;1-4/7 pixel-identical to previously verified pages. Rejected overflow layouts retained in `cone_guard_paper_20261015`. Published `output/pdf/DAC-eGraphMap-draft.pdf`; no main promotion.


## 2026-10-15: retain already-scored covers, isolated full EPFL; repetitions in progress

The next delay-area-only experiment targets a measured score/application gap, not a new delay gain. `tail_progress_pilot_20261015` records four adder CEC passes with tagged queue observations. All three area-flow rounds finish normally; the earlier full4in adder regression had7958actual-area candidates scored but zero application trials. Its favorable counterpart applied128trials/64acceptances. Thus blaming an unfinished dependency queue is unsupported; the lost opportunity is in the following expensive score-then-apply stage. Raw old failures/hypotheses remain, with trace-field correction in `tail_progress_diagnosis_20261015/trace_field_semantics.json`.

`/tmp/emapper-scored-incumbent-20261015` extracts the original `slack_area_pass` and adds opt-in retention of one already fully validated scored cover. It preserves the original working trajectory and proposal order, performs no extra candidate scoring or actual-metric calls, then compares the saved cover by(D,A) before the original final deferred-incumbent selection. Existing actual global2s/node20k/cut16/pool100k/SAT16/mapping300s limits and baseline algorithms/models/native parameters remain unchanged. Binding-label storage is not an extra physical candidate or exempt Boolean graph. A deterministic real-stage test injects an unavailable application window after scoring: old16/1 vs saved9/1; the joint physical instance is charged once. The recorded red assertion, initial test compilation failure and corrected green tests are retained.79egraph-related tests pass/4ignored; extra checks cover equal scoring counts, disabled retention, expired entry and cheaper-but-slower rejection. Production main Rust/CLI/bin are unchanged; the switch remains experimental, not a documented main API option.

`scored_incumbent_full_20261015` completes80observations, comprising12explicit same-version pilot adoptions and68fresh native attempts. BOTHprofiles use the same prior cone-guard90 import and protected recovery; candidate adds only scored-cover retention. The controls are **not production main**.68direct original-input named-port CEC passes/12unchanged import failures;34common valid pairs D0better/34equal/0worse, A1better/33equal/0worse.4in priority52.55->52.42 atD2047.24, saved-cover eventusedtrue/additional_evaluations0. All paired pool fingerprints, successful actual budgets and hashes pass; paired nonregression gate istrue, all20mappedfalse(div/hyp/mem_ctrl still fail). Main's prior five import failures remain because no promotion occurred.

Eight fixed targets(adder,bar,priority,int2float,sqrt,multiplier,log2,square), both libraries, two alternating-order repetitions are now being run in `scored_incumbent_repeats_20261015`; do not claim completion or select best repeats before its64fresh jobs end. Full and repeat runners are sequential finite batches with shared locks. Currentfirst4in priority repetition reproduces52.55->52.42 with equaldelay. No baseline optimization or changed budgets.

A separate read-only working-state issue guides the next DA experiment: sqrt retains a fast deferred cover while subsequent local work can still use the slower legacy cover. `required_area_dag` eventbest_area/best_delay describe the working incumbent, **not the trial**, so they cannot support a claim that the DAG trial violates its timing bound. The proposed next factor is to activate the best validated deferred cover after the DAG stage only when its delay strictly improves the legacy reference; equal-delay shared-area trajectories remain unchanged. This factor has not yet been implemented or tested. Industrial multi-output/ODC/independent regional saturation and full baseline dominance remain unresolved.

## 2026-10-15: typed cone/scored main integration, fresh full matrix

This supersedes the earlier statement that main still has five import failures. The isolated scored-cover study completed its64 fresh alternating-order target runs; its96-observation view includes32 full-study observations, not96 new runs. Forty-eight legal pairs have no delay or area regression (D0/48/0, A3/45/0). Prototype studies remain separate from main.

After that qualification, exact bounded cone import and saved scored-cover retention were integrated as typed `MapParams` fields, `cone_import` and `scored_incumbent`, with CLI `EGM_CONE_IMPORT` / `EGM_SCORED_INCUMBENT`. Both defaults remain false. Main uses fixed helper quota2048 and a uniform90% initial-capacity threshold; schema2 separates source occurrences from unique actual operator/helper classes. Rich XOR/three-input-table e-nodes change the representation unit compared with expanded ANDs; all actual helpers count and original source AND counts remain explicit. No resident/cumulative substitution or per-region reset is used. The late-area fix saves one already validated scored cover, without extra scoring/metric calls or physical candidates, before the original deferred-incumbent comparison.

`scored_main_integration_20261015` retains before/source/patch/bin/Cargo/logs. Debug and release each pass82 egraph-related tests, four ignored; actual-stage tests cover disabled/expired/slower candidates, shared-instance charging and typed option wiring. Main mapper SHA256 is8f6d45dc7c73eec1f773e8f019d2aae6c97619c8f05a0fd19380c0875f0bc24c; public evaluator remainsbb14affbd7bbf79560436850b7122b0d45932d673894b56ea5260e4255462024.

Fresh main `scored_main_full_20261015` completes80 native attempts, no prototype-PPA adoption:64 direct original-input named-port CEC and16 node-budget failures. Candidate34/40 valid covers versus prior-protected control30/40; log2/multiplier now pass in both libraries. Thirty common legal pairs have D2better/28equal/0worse and A2better/28equal/0worse. All successful actual node<=20k/cut<=16/pool<=100k/SAT<=16/recovery<=2s/mapping<=300s checks and artifact hashes pass. Changed pools are intentional only where representation changes. Strict paired self gate passes; this is not full-baseline dominance or proof that every EPFL circuit maps.

Main sqrt3 A/D2159.11/68523.99 ->1814.66/63951.92; sqrt4 2743.80/66924.64 ->2188.56/57161.09. Main4adder is83.75/2585.23 on both sides. The old prototype84.49 regression remains immutable. Remaining failures are div/hyp/mem_ctrl (28644/143890/29791 minimal actual imported nodes in this representation); these are not impossibility lower bounds. Targeted main repeats are in progress in `scored_main_repeats_20261015`, using eight fixed targets and alternating order, retaining unchanged original control import failures explicitly. Final promotion/publication awaits that audit. Baseline author algorithms, native parameters and models remain unchanged.

## 2026-10-16: main cone/scored qualification closed; data and paper updated

This supersedes the preceding pending-repeat/publication status. `scored_main_repeats_20261015` finished64 status slots:56 fresh native successes plus8 explicitly adopted unchanged same-main control pre-import failures. Its96-observation view includes32 full observations, not96 fresh mappings or all20 repeated designs. Eight targets x two inventories x three paired observations yield36common legal pairs: **D6better/30equal/0worse; A6better/30equal/0worse**. All12 new log2/multiplier candidate observations pass direct original-input named-port CEC. Six sqrt pairs reproduce the full-study metrics; no adder/bar/priority/int2float regression was observed. Main unique totals are136native attempts,120directCEC and16native node failures; adopted and overlapping records are not added again. Both full and repeat self gates pass. The typed opt-in profile is main-binary qualified; defaults/presets remain unchanged.

All successful actual budget checks and recorded source/artifact hashes pass. Typed trace audit finds zero additional scoring evaluations; schema2 ledgers separate source occurrences from actual unique operator/helper classes. Complete prototype source and current main source/bin/patches/before files are retained. Final audit checks1313source/artifact references without errors; three relevant campaign locks are idle. No mapping process is left running by this round.

Original baseline audit rechecks13403actual dependencies/native tracked files:zero drift. Canonical240groups/520slots remain372CEC/124reportedfailures/24unmeasured. Baseline algorithms/models/parameters are unchanged. `scored_main_integration_20261015/baseline_scope` reruns full20DA, AD and Pareto gates for both libraries:all six remain false; AD is posthoc on these DA covers. MtM remains unconnected. Mapping-only300s and all original native guards remain; EqMap nested mapping timing retains conservative enclosing-Yosys bounds, not precise runtime. Source comparison and failure coverage are explicit, not a complete end-to-end equal-compute claim.

Published DAC TablesI–IV now use only the fixed new **main full-R0** candidate rows for all20designs/two libraries. All480numeric/status cells are verified against actual records; baseline cells remain byte-identical. Current EPFL provenance is explicit in `table_provenance.json`; older source lists remain historical. The draft records34/40coverage, sqrt gains and remaining div/hyp/mem_ctrl failures, score/application-gap diagnosis, and full/repeated observation accounting. Actual richer XOR/TT3 node units are distinguished from expanded-AND work units. Standard10pt/margins unchanged;6body+1reference pages, all7visually checked, no LaTeX warnings. Both rejected8page layouts are retained. Final PDF and QA are in `output/pdf/DAC-eGraphMap-draft.pdf` and `scored_main_paper_20261015/paper_qa.json`.

Evidence: `scored_main_integration_20261015/report.md`, `qualification.json`, `main_paired.csv`, `repeated_paired.csv`, full/repeat reports and typed/baseline/final audits. Next isolated DA work: strict-delay-improving deferred working-reference activation while preserving equal-delay shared area; exact AO31 TT3/source-cut retention fixtures; exact shared-cone representations for the remaining three imports with full global accounting. These factors are not yet implemented or claimed. No all-baseline victory, industrial multi-output/ODC/regional-saturation completion or project-complete claim.

## 2026-10-16: next DA round — isolated factor pilots, main unchanged

Current main remains8f6d45dc... and the published fixed main tables remain canonical. `active_deferred_pilot_20261016` runs8fresh sqrt/adder attempts, all directCEC and actual-budget compliant. Both profiles have current cone1/scored1; candidate adds only strict-delay-improving deferred-reference activation, validated all-signal witness graft, and saved old shared cover. Real heuristic red fixture sees local(A1,D10) despite final deferred(A2,D1); green83tests/4ignored. Four paired observations have D0better/4equal/0worse and A2better/2equal/0worse: sqrt3A1814.66->1809.95 and sqrt4A2188.56->2185.61 at unchangedD. This does not establish a new delay gain or qualify main integration without full/repeated evidence.

`hybrid_tt3_pilot_20261016` also runs8fresh allCEC attempts. A synthetic exact4PI AO31 fixture reproduces missing TT3 expansion when a source certificate exists; green83tests/4ignored admits it within the same4-cut bank. But actual sqrt has unchanged cut counts, generated instances, retained physical pools and PPA in both inventories: all4pairs equal. A synthetic mechanism fix is not relabeled as an EPFL timing gain.

`shared_majority_pilot_20261016` has16fresh attempts:12retained node failures and4sqrt directCEC, no budget/hash errors. Exact bounded three-input-majority reconvergence folds globally counted operators, with at most4selected source fringes/child and16parent combinations. Tight-node, shared-original-output, unused-PI and16phase tests pass (84egraph tests/4ignored). Import nodes div28644->28539 and hyp143890->143770 remain over20k; mem_ctrl also remains over. Sqrt4D57161.09->57062.13/A2188.56->2170.59 improves, but sqrt3D63951.92->63961.49 regresses (A1814.66->1814.63). **Not promoted; negative result retained.** Source/counter units are explicit; this is not an unchanged expanded-AND work-unit comparison or a resident/cumulative budget substitution.

`active_critical_pilot_20261016` toggles existing critical-first recovery only, bothprofilesactive1. Four fresh sqrt directCEC, D0better/2equal/0worse, A0better/0equal/2worse, with unchanged pools. It is not promoted. Actual argv/profile fields record the critical flag consistently before mapping.

The next isolated XOR-propagate-majority version has a real import red fixture: `ab+c(a xor b)` currently needs8actual nodes including4PIs/constant instead of6. New exact XOR source fringes remain within4selected fringes/child and16parent combinations; green85tests/4ignored covers both SOP/XOR carry topologies over all16input/output phases and shared XOR outputs. It is building for a separate pilot; no claim yet about EPFL PPA, coverage or large-design success. Native/model/baseline algorithm parameters, mapping300s/native60(EqMap90), globalnodes20k/cuts16/retainedpool100k/SAT16/recovery2s are unchanged. Original baseline520statuses remain intact. All version sources/binaries/logs/negative results are archived separately; no per-design source/config winner mix.


## 2026-10-16: complete library-aware source-plan comparison (isolated V6)

This supersedes the preceding in-progress pilot statements. Production main source/binary/defaults remain unchanged; the earlier main full-R0 primary tables stay canonical. The isolated source versions, failed builds, screens and adverse observations are retained independently, never combined by circuit. V4/V5 screens stopped at12/80 with missing68 explicitly recorded, not full EPFL comparisons.

V6 `/tmp/emapper-library-source-plan-20261016` implements original-library K3/K4-aware exact source-boundary planning, stored TT3/TT4 truth and all operands, and a whole-source reference-count transaction before constructing one e-graph. Source choices old+15, truth queries<=16/node and bounded support joins do not increase the physical cut16/node20000/pool100000/SAT16/recovery2s/mapping300s limits. Optional exact RHS ranking uses structural slack and estimated dependent-pin arrival, explicitly distinct from physical EGM_LIBRARY_TIMING (still0). Successful imports and pre-rebuild peaks are counted; the incomplete20000-node prefix is a failed import, never a mapped incumbent. Richer scalar-function nodes are not expanded-AND work units, and retained-pool limits are not transient-memory limits. No region counter reset, residency substitution or baseline optimization is used. Real AO31 tight-node and all32 input/output-phase mapping fixtures pass;93 egraph-related tests pass/four ignored. Sources/binaries/Cargo/initial failures are archived under library_source_plan_implementation_20261016.

`library_source_plan_full_20261016` completes80 **fresh** attempts:68 direct original named-port CEC passes, twelve node rejections. Both profiles use current protected cone1/scored1; only candidate enables source planning. There are34 common legal pairs: D0better/34equal/0worse, A1better/33equal/0worse. Four-input multiplier area2087.12->2086.32 at unchangedD2954.30 is the sole full-study gain. Div/hyp/mem_ctrl remain failures in both inventories, no new valid coverage. All successful actual budgets and artifact hashes pass. The generic full-study nonregression/any-metric gate is true; this is NOT repeated qualification or evidence of a delay improvement, and main is NOT promoted.

Read-only sqrt diagnostics establish that3in/4in retained cut sets (48966/108759), generated candidates (367347/819195), retained-pool fingerprints and selected physical instances are unchanged. The4in candidate has2139 extra TT4 expressions but no retained-cut gain. Its path still2613 cells/512 AO31 versus unchanged emap2077/1742; this is descriptive, not proof of candidate absence or a causal remedy. Sqrt D remains63951.92/57161.09. Merely admitting more equivalent source expressions did not close this gap.

`library_source_plan_evidence_20261016` reaudits13403 original baseline dependencies with zero drift and reruns all20/two-library DA/AD/Pareto gates:all six false, AD posthoc on DA covers. Original240groups/520slots remain372CEC/124reported failures/24unmeasured; no bad baseline dropped or renamed. Mapping-only300s, original native guards and conservative EqMap nested-mapping upper bounds remain unchanged; MtM unconnected.

Two fresh alternating-order repetitions on eight predeclared targets (sqrt,multiplier,adder,bar,priority,int2float,sin,max) are running in library_source_plan_repeats_20261016. No best-run selection, unchanged-failure retries or incomplete repeat claims. All profiles and source/driver/runner fingerprints are frozen. Publication/integration decision awaits actual repeated data; no new delay gain has been observed in the complete full study.


### V6 final repeated decision (supersedes pending status)

library_source_plan_repeats_20261016 completes64fresh target attempts,63directCEC/one strict recovery failure:4in max old-control2.004620s>2s, retained as failure without rerun replacement or partial success. The96observation target view includes32full observations,47common legal pairsD0/47/0,A4/43/0. Four-input multiplier-0.80 repeats in all3observations; priority-0.27 appears once. Unique full+repeat144native attempts yield131directCEC/13failures and65common legal pairsD0/65/0,A4/61/0; overlapping summaries are not added. Successful actual budgets/hashes pass; the failed recovery is not a new large-design coverage gain.

Canonical main full-R0 comparison is distinct: V6full4in priority52.42 is worse than published main52.15, despite the same-binary experimental control tie. This observed regression is retained; a causal/runtime explanation is not proven. Main source/binary/defaults/primary tables are unchanged and V6 is **not promoted**. Read-only checker labels now distinguish full-study any-metric nonregression from production qualification; initial mislabeled output/source retained, no algorithm/numeric threshold change. Explicit decision/evidence/next research: library_source_plan_evidence_20261016/report.md and qualification.json.

Sqrt retained critical producers offer no faster one-step estimate with all inputs already selected;2359/3309(3in) and1987/2614(4in) critical signals have alternatives requiring unselected input cones. This is descriptive estimate evidence only, not valid whole-cover improvement or proof of candidate absence. Next work targets joint cone/phase trials with old shared-area incumbent protection, and exact pre-admission function/operand canonicalization for near-limit div; neither new algorithm is yet implemented. Hyp/mem_ctrl remain over budget, no residency/cumulative substitution.

Debug/release93egraph tests pass/four ignored. All source/archive/fingerprint checks pass; no native runner remains active. DAC has been updated with negative full/repeated results, main-priority area regression and selected-path limitations; primary four tables byte-identical, standard10pt/margins unchanged,6body+1reference pages all visually checked. Rejected8-page PDF retained; published output/pdf/DAC-eGraphMap-draft.pdf. Baseline520statuses/13403dependencies remain unchanged and all six dominance gates false. Overall task remains unfinished.


## 2026-10-16: joint cone/phase witnesses and batched actual-area recovery (isolated V4)

Main source/defaults/bin remain unchanged (8f6d45dc...). V1 atomic fallback has valid cycle/phase fixtures but no valid new real sqrt/priority repairs or delay improvement; failed builds, negative pilots and four separate source variants are retained. V4's fixed profile instead reserves complete area and fast physical all-signal witnesses before optional100k pool truncation, applies already validated actual-area proposals in bounded batches, and reserves late required-time/area work inside the same2s extraction window.95egraph tests pass in debug and release/four ignored. Fast pre-cap ranking is recorded candidate-generation work inside original native60s/mapping300s guards, not an additional extraction window or equal executed-work claim.

`joint_batch_full_20261016` is complete80fresh attempts:68direct original named-portCEC and12node failures. Both profiles map34/40cases; div/hyp/mem_ctrl still fail both inventories.34legal pairs D14better/20equal/0worse,A6better/16equal/12worse. New full R0 sqrt4 area/delay1980.96/51470.40 versus2188.56/57161.09; sqrt3 1867.26/63200.36 versus1814.66/63951.92 (area regression). Priority3 68.64>68.54 and max4 222.85>221.30 have unchangedD. Priority4 52.21 is below contemporary52.42 but above canonical main52.15: the canonical observation remains authoritative, not erased. The same-binary study has12area regressions; canonical-main comparison has13. **Not promoted.**

The saved legacy shared cover is constructed within the current pool, not every binding of the previous pipeline's final cover. Priority's retained pool is identical to canonical main's, so changed membership does not explain its area regression; the recovery trajectory remains unresolved. Full34new covers attain their frozen-pool raw delay labels within numerical rounding. All-producer inequalities use local1e-6 tolerance; accumulated conservative error<=2*eclasses*1e-6 (sqrt0.029206). This bounds only the admitted physical pool, not global library/cut/area/lex optima. Sqrt4 old capped label57161.09 versus pre-cap/new51470.40 establishes a substantial truncation loss; residual to emap47047.28 needs better candidates within unchanged limits, not merely another extraction order. No particular missing AO31 is proven.

Actual global20knodes/16cuts/100kretained/16SAT/2srecovery plus original guards unchanged. Bounded import now rejects BEFORE adding20001stnode, counting allPIs/PO phases; incomplete prefixes are failures, not mapped coverage or impossibility proofs. Generated transient candidates are not retained pool/RAM units. No retirement/regional reset or residency/cumulative substitution.

Original baselines remain fully attempted240groups/520slots:372CEC,124reported failures,24unmeasured. This wave rechecks13403original dependencies with zero drift; two libraries' full20DA/AD-posthoc/Pareto sixgates allfalse. Original algorithms/models/parameters and unseeded native SA selection remain unchanged; no E-Syn optimization probe is included. Mapping-only300s and conservative EqMap nested-time upper bounds unchanged; MtM unconnected.

Evidence: `joint_batch_evidence_20261016/{protocol.md,report.md,baseline_integrity.json,frozen_pool_delay_brackets.json,baseline_scope/validation.json}`, separate source archives/pilots/full reports. Two fixed alternating-order fresh repeats on eight predeclared targets are being completed; count only actual checkpointed observations. Primary paper tables remain canonical main because area protection and big-design coverage are not solved.

### Joint-witness V4 final repeated decision

All64 declared fresh alternating-order target attempts finish with direct original named-port CEC; actual budgets and source/artifact hashes pass. The96-observation target view includes32full observations, not96new runs:48legal pairs D12better/36equal/0worse,A9better/24equal/15worse. Non-overlapping full+fresh-repeat accounting is144native attempts,132CEC/12node failures,66legal pairs D22better/44equal/0worse,A12better/32equal/22worse. There are16lower-delay/larger-area tradeoffs and6equal-delay/larger-area regressions across these unique pairs.

Sqrt4's three candidate observations all yield1980.96/51470.40; both delay and area improve versus each paired control. Sqrt3 delay63200.36 repeats all three times, but area1867.26/1867.38/1867.59 stays above old1814.66. Priority3's68.64 versus68.54 equal-delay regression and max4's222.85 versus221.30 repeat all three times. Priority4 candidate areas52.21/52.17/52.21 remain above canonical52.15. Bar4's290.78 area versus291.50 repeats without a delay change; bar3 ties. No best observation is chosen. These are small target repetitions, not full20 repeated statistical stability.

The explicit qualification is FALSE. Production main/source/defaults/binary and primary tables are unchanged. Original baseline six full20gates remain false, and no new design becomes legal. Final integrity checks cover both frozen source manifests and every report/audit/pool/trace/frontier reference. This round leaves no native campaign running.

Final V4 publication audit:1064frozen source/artifact references verified without errors; both fresh studies complete and relevant locks idle. Main binary8f6d45dc... unchanged. DAC discussion/abstract updated with positive and adverse V4 evidence, all four canonical-main tables byte-identical. Standard IEEE font/margins retained,6body+1reference pages individually visually checked; rejected8page layout retained. Final PDF published to output/pdf/DAC-eGraphMap-draft.pdf and project main.pdf; QA in joint_batch_paper_20261016/paper_qa.json. No main promotion, no all-baseline victory, no claim that div/hyp/mem_ctrl or industrial multi-output/ODC are solved.

## 2026-10-16: guarded shared-incumbent recovery and helper/phase coverage, isolated studies

Production main source/defaults/bin stay unchanged (8f6d45dc...), with canonical full-R0 primary tables. V5guard-only pilot fixes priority3/max4 but worsens priority4; V6fused evaluator reduces a fixed4000-call kernel measurement260ms->167ms, not a mapper runtime claim. Existing LIBRARY_TIMING also changes SAT rewrites; its pilot has no sqrt delay gain and substantial area costs. Source-operand canonicalization has zero alias savings before div/hyp/mem_ctrl admission failure. All negative factor results, source versions, failed builds and commands are retained separately.

V7 adds a conservative live-sharing area lower bound plus fused full-cover scoring, preserving the old recovery trajectory unless a validated strictly faster reference activates. The minimized five-cell old area trajectory regression and an irrelevant-fast-phase pool truncation regression are locked down; debug/release104egraph tests pass/5ignored. Its full20/two-library compound comparison completes80status slots:68fresh direct named-port CEC plus12explicit failed pre-import adoptions from6prior actual failures, with exact dependency proof. It is not80fresh attempts. Both profiles produce34/40legal covers. Full34pairs D14better/20equal/0worse, A5better/19equal/10worse; each area cost accompanies a strict delay gain.64fresh alternating-order target attempts all pass CEC/actual budgets. Nonoverlapping full+fresh-repeat132native runs allCEC,66legal pairs D22/44/0,A9/41/16. Priority/max restore canonical areas across all three observations, including priority4=52.15 and max4=221.30. This does not prove every previous final shared cover survives a changed candidate pool. No main promotion or Pareto victory. Evidence: guarded_joint_evidence_20261016/report.md,qualification.json and full/repeat budget checks. One derived CSV mistakenly referenced historicalV4; original retained and derivative regenerated from correctV7source reports, without changing experiments or acceptance rules.

Baseline algorithms/models/native parameters are untouched; original240groups/520status slots remain372CEC/124reportedfailures/24unmeasured.13403dependencies reaudited without drift, six all20/two-library DA/AD-posthoc/Pareto gates false; MtM unconnected. Mapping-only300s/native60(EqMap90), same once-resyn2 subjects/libraries/public physical area and dependent-pin block delay unchanged. EqMap nested mapping runtime remains conservative enclosing-Yosys upper bound, not exact runtime.

V8helper allocation pilot has4fresh sqrt CEC. It preserves the exact old2048mandatory source helpers, then fills unused initial global capacity under the same18000initialceiling/20000actualglobalnodes. Old2048is a mandatory subset, not an unchanged cap on actual helpers. Sqrt4 actualhelpers5445/import18000/postsat19944/cuts16/retained100000/recovery1.860243s; A1983.10->2250.28,D51470.40->47047.28. Sqrt3 A1869.48->2155.67,D63200.36->60087.26. Both candidate delays match same-library emap, but area grows substantially. Actual budgets/subset/hashes pass,105debug/release tests pass/5ignored. This pilot is not a full20/repeated or production-qualified improvement. Original libraries/budgets did not increase; generated transient candidate counts are separate from retained/RAM caps.

V9unselected-inverter recovery addresses direct positive/negative producers that have no selectedINV seed. A real minimal RED6area/3delay fixture becomes GREEN2.5/3, with exhaustive truth checks for all four assignments of two PIs, expired stage/zero trial cap and unactivated-legacy gate checks. Debug/release107egraph tests pass/5ignored. New admitted-only proposals share the original phase trial cap, phase0.25window and inclusive global2s; enabled only after validated faster-reference activation. No added candidates or SDC unions. Four fresh sqrt factor attempts allCEC:4in A2249.89->2249.50,D47047.28same;3in A2156.87->2156.15,D60087.26same. Small area gains do not erase helper-allocation area costs. V9source/CLI/Cargo/binary/patch frozen under unselected_phase_implementation_20261016; main unchanged.

A new fixed full20/two-library compound matrix helper_phase_full_20261016 is in progress: both profiles use V7joint flags, candidate adds helper allocation+unselectedINV phase and control disables both. All80declared attempts are fresh, including actual admission-path revalidation for big inputs; no failed/success adoption or designwise configuration choice. Repeats are predeclared on8targets; not yet completed. Div/hyp still fail actual global admission in the initial new slots. Adding helpers cannot reduce the mandatory source forest; no claim of big-design coverage, lower-bound impossibility or resident/cumulative-budget substitution. Full/repeated evidence and canonical-area checks remain prerequisites for integration. Industrial multioutput/ODC/independent regional SAT are unfinished.

Budget terminology clarification for V8/V9:18000 is the optional-helper admission ceiling, not an independent hard cap on mandatory import. Log2 imports18103 mandatory actual nodes in both modes, with no added helpers; this is within the unchanged global20000limit. All actual imported/postSAT nodes remain checked against20000. Frozen raw field initial_node_ceiling and early protocol shorthand are preserved and interpreted by allocator code, not relabeled as actual18000compliance for all designs. The sqrt allocation-specific actual import is18000. No numeric cap, mapper code, failure status or baseline parameter changed.

### Final V9 full/repeated decision and V10 isolated phase-priority repair

This supersedes the preceding pending V9matrix/repeat statements. helper_phase_full_20261016 is COMPLETE80fresh/68directCEC/12actualnodefailures;34legalpairsD4/30/0,A0/30/4. helper_phase_repeats_20261016 completes72fresh directCEC on9targets in two alternating orders (multiplier added after full-screen gain BEFORE repetitions). The108-observation target report includes36full observations, not108new runs. Unique152fresh native attempts yield140CEC/12node failures and70legalpairsD12/58/0,A0/58/12. All source/artifact and actual global-budget checks pass. Compared separately with canonicalmain,34legalpairsD14/20/0,A3/20/11; no tied-delay area regression, but six commonfailed design/librarycases remain failed. Conditional DA nonregression on commonlegal cases is not Pareto/full20baseline victory or universal old-final-incumbent preservation. All six baseline gates false; original baseline520statuses unchanged.

FullR0 sqrt4A2250.39/D47047.28 andsqrt3A2155.93/D60087.26 match original emap delay to reported precision, while emapareas1562.46/1644.87 are much smaller. Raw4in Ddifference about0.000002 is not a meaningful claimed timing gain. Newcriticalpath reconstruction:4in2281->2077cells,AO311338->1743 versusemap1742;3in new/emap2951cells andsamecriticalpath-typecounts. This is descriptive, not proof of one individually missing candidate. Bothinventory multiplierDgains2642.70/2715.38repeat at larger area. priority/max all3observations retain canonical4in52.15/221.30 and3in68.54/249.37;bar4in291.50 stable. None of these values is chosen as a best run across versions.

Import diagnosis:div/hyp/mem_ctrl planned representation uppers28644/143890/29791,zerooptionalhelpers in all12failedslots, actualadmittedprefix20000then nextdistinctoperatorrejectedBEFORESAT/cuts. Estimates are not minimum lowerbounds/impossibilityproofs. No cap change/retirement/reset/input exemption.18000onlyoptional-helper ceiling;mandatorylog2actual18103is globally compliant with20000. See helper_phase_full_report_20261016/failed_import_diagnosis.json and protocol_clarifications.json.

The independent phase factor is NOT qualified:unselected_phase_factor_repeats_20261016 adds8freshCEC to4pilotobservations,6pairsD0/6/0,A2/0/4,identicalpools per pair. A4in pairarea2250.36->2252.28atunchangedD47047.28 is retained; oldphase13trials/1accept versusoptionalfirst5/0. Native traces do not prove every difference causal. A deterministic4cell/cap1fixture independently reproduces optional-cyclic-proposal starvation of an oldatomicflip(D13vs7), without relying on walltime noise. No bad factor result is hidden or relabeled.

Isolated /tmp/emapper-phase-priority-20261016 (V10) fixes this feedback loop by executing LEGACY atomicphase proposals FIRST and spending only remaining originalphasecap/window on extraINVs. RED->GREEN;108debug and108release relatedtests pass/5ignored;releasebuilt. Source/Cargo/bin/red/green/relative_to_v9.patch archived under phase_priority_repair_20261016. This protects legacy PHASE trial priority, not every oldwhole FINALarea flow trajectory across changedpools. V10has no native fullEPFL/repeatedPPA qualification yet and is not substituted for V9data. Productionmaincode/defaults/bin8f6remain unchanged; do not promote the unqualified earlyphase extension or claim allbaseline superiority.

Evidence/decision:helper_phase_evidence_20261016/report.md,qualification.json,canonical_main_pairs.csv,critical_paths.json,final_integrity.json.13403originalbaseline dependencies were reaudited earlier this round withzero drift; newscope references that audit bySHA, not a secondfresh audit. EqMaptimeupperbounds/nativeSAwinner/originalE-Synfailures unchanged;MtM unconnected. Repetition helper sidecars absent from checkpoint keys are explicitly marked first-captured-by-readonly-audit withSHA; smallordinaryimport hasexplicit xor_importfalse andnohelper sidecar. Frozenrawreports/commands/statuses untouched. No claim of generated-transient RAMcap or globaloptimality.

DACdraft now reports full/repeated positiveD and adversearea evidence, negativephasefactor, priority-starvation fixture/unit-onlyrepair, and remainingnodefailures. Primaryfourtables byte-identicalcanonicalmain;standardpreamble/fonts/margins unchanged;6body+1reference pages,all7visuallychecked(finalquota clarification onlypage2changed andvisuallyrechecked,other6pixelidentical). Rejected expanded8pagePDFretained. Published output/pdf/DAC-eGraphMap-draft.pdf;QA guarded_joint_paper_20261016/paper_qa.json. All7relevantcampaign locks idle;no native/build/renderer left active. Work remains incomplete; next version must validate the priority fix on real circuits, preserve complete oldsharedarea recovery, and investigate exact sharedsource compression forbig3 without budget inflation or baseline optimization.

## 2026-10-16: exact signed-source and shared-source transaction probes (no promotion)

Two independent isolated versions, not one mixed mapper: V11signed-source16fresh(4sqrt public named-portCEC/12actualnode failures) and V12joint-source12fresh(0CEC/12node failures); total28fresh/4CEC/24fail. Source/artifact/budget checks pass; all twelve fullscope DA/AD-posthoc/Pareto gates false with missing cases explicit. Prior13403dependency baseline audit referenced bySHA, not freshly repeated; original240groups/520slots372CEC124reportedfail24unmeasured unchanged. Native algorithms/models/parameters untouched; main native/source/defaults/bin8f6unchanged. No promotion or fullEPFL/repeat qualification.

V11 absorbs exact source phases into truth tables, shares true complements via per-source polarity without complement union, counts every realized operator and demanded PO inversion;111debug/release relevant tests pass/5ignored. Allbig3stillfail20000prefix, zero alias savings before failure. Sqrt4A2251.58/D47047.28->2284.18/48381.90; sqrt3A2155.98/D60087.26->2121.12/61924.15. Both delays regress; do not substitute the three-input area tradeoff as DA success. Independent source unit differs from expanded AIG work units; no retirement/reset/input exemption.

V12 uses original frozenV6library-aware source planning, not V11 or main: bothprofiles same binary/planning, only pair feature toggled. Existing source query and bank caps retained, at most16jointtrials/live root. Pair roots have equalrank/loweroperandranks, updates use exact accumulated descendantrefs; strict final count decrease required. Deterministic reference fixture6->5 and bounded3leaf8->7;256layered graphs checked versus independent full reachability;96debug/release relevant tests pass/4ignored. Initial native test called disabled default, corrected test seam; failure archived. These are structural source tests, not full physical CEC.

K4plannedsource single->joint:div20440->20440 (fullimportupper20528),hyp107990->107987(108114),mem_ctrl25061->24999(25564). K3div20613->20613(20701),hyp111494->111494(111621),mem_ctrl28038->28029(28594). Every actual import stillfails after20000admittedprefix BEFORErecovery; these are upper/sourcecounts, not lowerbounds/actual completeimports/PPA or impossibilityproofs. Zero optional-helper change/counterreset. The62source saving is insufficient. No new baseline-referenced delay advantage or area reduction in a legal big-design cover.

Evidence source_probe_evidence_20261016/report.md,validation.json,final_integrity.json,baseline_scope; frozen source/patch/binaries under signed_source_implementation_20261016 and joint_source_transaction_20261016. Next: exact zero-cost/cross-rank source transactions with virtual-parent accounting, old-source-incumbent protection and unchanged bounds; actual complete import/finite physical witness required before full/repeated PPA. Don't blindly rerun these unchanged failures. Baseline ref/deref and required-time ideas used only in our research, no baseline edits. All7relevant locks idle; no campaign remains active.

Source-probe publication QA:6body+1reference, four canonical primary tables byte-identical and preamble/font/margins unchanged. Changedpage6 visually reinspected; other6pages pixel-identical to previously reviewed pages. Negative V11 delay outcomes and V12 insufficient source savings disclosed; rejected8page version retained. Published output/pdf/DAC-eGraphMap-draft.pdf; QA source_probe_paper_20261016/paper_qa.json. All finite sessions finished and runner locks idle; no main promotion.

## 2026-10-16: fixed-delay shared-area recovery and exact source probes (final)

V13 /tmp/emapper-live-shared-area-20261016 implements opt-in EGM_ONLINE_SHARED_BOUND(defaultfalse), refreshed refs/users/active after accepted online commits before further pruning. Conservative bound skips only nonimproving actual-area trials; full-cover repair/prune/unique physical area/dependent-pin delay remain acceptance oracle. RED score2->1, staleactive falsecharge fixture and64completed sharedDAG-pass identity tests;111debug/release egraph testsPASS5ignored. ActualfixedD/pool/globalbudgets unchanged. Old whole-final incumbent across changed pools is not universally guaranteed.

Initial bound-only factor24fresh24CEC:12pairsD0/12/0,A0/9/3; negative sqrt/priority4outcomes retained. Priority4newdynamicbranchinactive and earlier local timings differ; no exclusive causal assignment to bound. Sqrt latearea stage~55ms/0scores motivates newallocation.60ofsamepostseedremainingwindow, notextra2s. Threeprofile screen6fresh6CEC: oldallocation0/bound0,allocation_only.6/bound0,combined.6/bound1. FullfixedEPFL20/two-library matrix80fresh68directoriginalnamedportCEC12actualnodeprefixfailures.34legalpairs ALLDties,A3better31equal0worse:4sqrt2249.05->2246.96D47047.28;3sqrt2153.81->2153.43D60087.26;3arbiter775.02->774.62D900.55. No adoption, no missingcase removal or per-design source choice; oneuniformcompoundprofile.

Two fresh alternating-profile-order sqrt repetitions add12CEC. Balanced screen+repeat view18observations/6triples (notfull20repeat) gives combinedA6better/D6ties; area reduction0.38--2.09. Allocation-onlyA5better/1worse(+0.20in3r0); combinedvsallocation-onlyA6better. Allthreeobs/rootlibkept:nobest-runselection. Fullsqrt4observations separate, notdoublecounted. Samephysicalpoolhashperallocationpair and everyactualnode/cut/pool/SAT/recovery/mappingguard check passes. Fullsource/artifact verification and independentoriginal-input/evaluatorcommand+measurement audit pass. Outputnetlist hashes absentfromcheckpoint are explicitlyfirstcapturedbyreadonlyaudit.

V14 /tmp/emapper-zero-source-20261016 independentlyextendsV12positivepairsourcepass with zero-cost first moves using only leftover original16paired-trial slots/root; no16extra, query/bank/globalbudgetsunchanged, oldpositive source-countincumbentprotected. RED8->8expected7/GREEN8->7;97debug/release testsPASS4ignored,256layeredDAGreachability.12freshbig3/twolib attempts ALL actual20000prefix failures beforephysicalmapping,0CEC. PlanneduppersK4div20523/hyp106472/mem25437;K3div20700/hyp111557/mem28479. Zero-mode source savings5/1642/127 and1/64/115vspositivepass. These are source/planneduppers NOTactualfullimports/PPA/lowerbounds/impossibilityproofs. No newdesigncoverage.

V15 /tmp/emapper-full-source-20261016 independentlytests full literaloriginalsource whencompleteupperfitsglobal20k, defaultfalse. sqrtactual19630countsallAND/PI/constant/POinversions, noinputexemption/retirement/residencyreset; optional18000helperceilingNOTmandatorycap.113debug/release testsPASS5ignored,4freshsqrtCEC/budgetPASS. BADarea4in2248.00->2456.71 and3in2153.55->2249.37atunchangedfastD; reject, nofull20promotion. Morefunctions don'tguaranteebetterphysicalmatchesunder16cut/100kpoolcaps. Initialdiagnostic-insertion assertion archived; finalsource/tests/build frozen before native. This source isNOTmixedwithV13/V14.

Combinedresourceaccountingacross6separatestudies24+6+12+4+80+12=138freshnative,114directCEC,24nodefailures; NOTonecombinedmappercoverage. Frozenpatches/source/Cargo/binaries and RED/green/logs in live_shared_area_implementation_20261016,zero_cost_source_implementation_20261016,full_source_implementation_20261016. Full comparison+targetdata live_area_evidence_v2_20261016/report.md,qualification.json,sqrt_repeats.csv,independent_command_audit.json,final_integrity.json. Initial readonly library-relative-path checker error retained in live_area_evidence_20261016; no rawsource/status/metrics changed to fix it.

Originalbaseline240groups/520slotsremain372CEC124reportedfail24unmeasured, algorithms/models/nativeparams unchanged. Prior13403file zero-drift baselineauditreferencedbySHA, NOTnewreaudit. Sameonce-resyn2EPFL20subjects/originallibrary/publicphysicalmetrics/independentnamedportCEC. Sameglobal20000actualenodes/cuts16/retained100000/SAT16/inclusive2s/setup/finalvalidate/.05returnreserve, mapping-only300/native60(EqMap90historical), no higherbudget. Generatedtransientcandidatesnotretainedmemorycap. Sixfull20DA/ADposthoc/Paretobaselinegatesfalse,MtMunconnected.

Canonicalmaincomparison34commonlegalD14better20equal0worse,A3better20equal11worse,noequalDarea regression;6commonfailedcase slotsstillfail. ControlledlocalareagainsarenotParetofullbaselinewinoroveralljointproductionareaimprovement:keepmain/native/defaults/bin8f6unchanged, no promotion. Next realareastep exactincrementalphysicalref/deref withsafearrival/requiredtimeupdates, full-scorefallbackforunselectedcones, oldsharedarea/fastincumbentpreservation; proof/REDplan incremental_area_design.md.

Additionalreadonlysignaturediagnostic3biginputs:4096randompluswalking/constantpatterns;possibleSIGNEDdirectclasscoalescencesdiv24558/mem12672/hyp36,includingfalsecollisions/biasedfunctions. NOTequivalenceproof or totalsavingbound; alias canreleaseuniquecone, so nodereductioncannotbeconcludedfromdirectcounts. No classes/inputs changed; useonlytoprioritizeexactSATproofbeforeanycanonicalization, nottopromisehypfeasibility ordeclareimpossible. All8relevantlocks idle; allnative/buildsessionsended; onlypaperfinalrender/QA beingfinalized. Projectunfinished,noSOTA/completionguarantee.

Finalpublicationaudit: live_area_paper_20261016/paper_qa.json confirms6body+1reference,standardIEEEpreamble/fonts/marginsunchanged,fourprimarytablesbyteidenticalcanonical8f6. Changedpages4/5/6visuallyinspected(finalwording5/6rechecked);pages1/2/3/7pixelidenticalpriorQA. Rejected8pagePDFretained;proseonlyshortened. OldPDFpluginpathwasautoupdated midturn,firstmarkerfailedBEFOREediting(set-e),new26.1013.11914markerSUCCEEDEDexactlyonce. Publishedoutput/pdf/DAC-eGraphMap-draft.pdfandprojectmain.pdfsameSHA. All6finitecampaigns/138attemptsfinished,8locks idle,no native/build/renderprocessleft. Primarynewreport live_area_evidence_v2_20261016/report.md+evidence_manifest.json. Project/mainperformancepromotionstillNOTcomplete;rootnative/source/defaults/bin8f6unchanged.

Finalsourceproofdiagnosis aftermapping ended:9boundedreadonlySATpairs fromsignaturegroups,7counterexamples/1unknown/1provedhyp pair(136522/136525),4mitercalibrationsPASS. No class/input changes, no new mappedPPA or importedcover,138/114/24mappingaccountingunchanged. Probe/meta/rawmiters/argv/hash/calibration in source_alias_proof_probe_20261016/diagnosis.md. DoNOTtreatlargeSIGcollision countsastruealiases/nodesavings;don'tfeedofflineproofsfreeintoofficialmapping. FutureSATproof/refinementworkmustchargeoriginalnative/mappingbudget. Allprobesfinished,no backgroundprocess.


## 2026-10-16: exact area rejection and bounded source proof (not promoted)

Three independent fixed studies finished: V16 all20EPFL/two-library104runs including24targeted repetitions, and V17 source-only16runs. Total120fresh physical attempts/96direct original named-portCEC/24actual cumulative-node failures. Same once-resyn2 subject report and restricted ASAP7K3/K4, public physical-instance area/unloaded dependent-pin delay. Same actual cumulative egraph20000/cuts16/retained100000/SAT16/inclusive recovery2s (setup/final validate/.05returnreserve), mapper-native60/mapping-only300; no baseline algorithm/model/native parameter changes. Generated transient candidates are not bounded by retainedpool; BDD proof nodes cannot appear as cut/cell functions. Main source/defaults/bin8f6unchanged, no .git edits or promotion.

V16from archivedV13: typed EGM_DELTA_AREA_FILTER defaultfalse. Exact selected-input physical-area rejection protects new inputs BEFORE old producer refs/users drop; unique physical cells/joint pins/aliases/roots accounted. Surviving trials still use full witness/cycle/prune/pin-delay scoring, with old unsupported/unselected fallback. No stale timing cache is treated as an acceptance proof. RED old1unnecessary complete score versus expected0, GREEN;256sharedDAG differential area matches.114related regressionsPASS5ignored. Initial test Binding equality compile error and missing fixture failures retained, fixed test seam/restored unchanged fixtures. Source/Cargo/releasebin/patch/RED/GREEN/argv in delta_area_implementation_20261016.

Full delta_area_full_20261016:80fresh/68CEC/12actual node failures;34legalpairs D0better34equal0worse,A2better31equal1worse. Both profiles same binary/all6V7flags/helperfill1/newINV1/fraction.6/onlinebound1; only new filter0/1.4in sqrt2248.45->2248.34D47047.28;3in sqrt2153.71tieD60087.26;3arbiter774.90->774.78;4priority52.14->52.15regression. Full and repeated budget/source/artifact/independent evaluator-command audits pass. All bad cases preserved.

Alternating repetitions delta_area_repeats_20261016:24fresh24CEC on sqrt/priority/bar, two repeats/two libraries/two profiles, second repeat reverse order.12pairs ALLdelayties, area1better8equal3worse. sqrt4pairs:4in+0.02/-0.34,3in0/+0.09; priority4in0/+0.01,3in0/0;barfourties. No best-run selection or full-suite repetition claim. Single tiny gain notstable; sqrt latestfull area remains~43.9%/30.9%above emap despite rounded timing parity. Relative to canonical main34commonlegal: D14better20equal, A3better20equal11worse, all11costs with lowerD. Main/complete shared-area preservation goal not achieved; keep reference primary paper tables unchanged.

V17from archivedV14 (independent ofV16): typed EGM_PROOF_SOURCE defaultfalse, same joint-source1/zero-cost1/control-and-candidate caps. Exact signed ROBDD identity over all unconstrained originalPIs; unknown retains original source, no simulation-only equivalence/offline cached advice.4095temporary nonterminal decisions/20000calls/.2s search checkpoints, NOT preemptive whole-stage .2s limit. Clone/rewire/scans charged original native60/mapping300; all realized imports globally charged, no resident-as-cumulative or retirement/reset. RED absorption2AND->expected0;GREEN2source tests include64DAGs with exhaustive6PI assignments;97egraph testsPASS4ignored. Broad crate --lib interrupted130 on three unchanged slow synth tests after60s, log retained/not claimedpassed. Initial typedfield patch assertion failed before mutation, corrected to actualV14 seam. Artifacts in proof_source_implementation_20261016.

Source pilot16fresh:4priority public mappingCEC;12big3 actual20000prefix failures BEFOREphysicalrecovery. div112exact aliases release121rawANDs; source-plan4in20435->20373,3in20612->20545, stillnotfullimport. hyp/mem_ctrl0aliases. All caps/audits passed; counts are source estimates, NOT mappedPPA/lowerbounds/impossibility. Independent frozen source-module replay includes existing CLI cleanup and matches BOTH actual library proofcounter sets; div/hyp/mem_ctrl sourceAIGCEC and4positive/negative/phase calibrations all expected. Exact PI/PO name lists checked, including indexednames. Additional readonly post-alias structural cleanup passes div sourceCEC and removes16more rawANDs, NOT integrated/native-feasibility-tested or fed back into officialpreparedinputs. Source diagnostics DO NOT add mappedCEC. Unrun helper build missing original CLI cleanup caught/fixed before first replay; old build/log retained. All data/argv/hashes in proof_source_cec_20261016.

Evidence delta_area_evidence_20261016/report.md,qualification.json,full_pairs.json,repeat_pairs.json,source_validation_reference.json,independent_commands.json,final_integrity.json. Sixfull20DA/ADposthoc/Pareto gates false; MtM unconnected. Earlier13403file baseline audit referencedbySHA NOTnewaudit; original baseline240groups/520slots372CEC124reportedfailure24unmeasured unchanged. No baseline optimization or failed-case relabeling. Next real area method: sparse ref/deref commits with conservative monotone arrival increase/required tightening and full validation/fallback; V16onlyfilters. Source proof selection/propagation remains research;16rawANDpostcleanup gain cannotbe subtracted from e-node estimates to assert feasibility. Allfinite native/diagnostic sessions ended,9relevant runner locks idle; paperfinalQA pending separately at timeofthisrecord.

Final publication QA: delta_area_paper_20261016/paper_qa.json confirms6body+1references, standard preamble/fonts/margins and four reference tables unchanged. Pages4/5/6 visually checked,1/2/3/7 pixel-identical to prior validated pages.8page draft retained; prose shortened only. Published output/pdf/DAC-eGraphMap-draft.pdf, project PDFsameSHA. All finite native/source-audit/build/render sessions finished,9relevant runner locks idle, main/source/defaults/bin8f6unchanged; no promotion.


## 2026-10-16 phase-sensitive delay diagnosis (no promotion)
Read-only six fixed K4 V16 protected_area paths versus original emap reconstructed public area/pin delay and verified input/prepared/library hashes. log2 D4600.49/3874.17, path174/179cells, both1INV; bar188.34/171.90,9/8cells,both1INV; sin1996.31/1788.13,83/80cells. sqrt rounded delay parity47047.28,2077/2077cells. All six supplied generated/retained pool delay certificates pass with identical attainable delay: retained100k truncation does not explain these observed gaps. Candidate coverage before generation/cut retention and phase-composed structures need exact boundary-aligned tests; path histograms do not prove missing candidate or causal replaceability. Positive/negative targets, input phase enumeration and pin permutations already implemented; V16 also has optional unselected INV/input-cone proposals, not unrestricted multi-level phase search. Report results_v2/phase_delay_diagnosis_20261016/report.md, paths.json, manifest.json.
Separate isolated V18 /tmp/emapper-sdc-scan-20261016 default-off EGM_SDC_SCAN_EXISTING: actual generate RED proves optional DC early return skips later existing-key conditional merges. One-branch fix passes RED and differential physical-key-count/old-target preservation fixture. No EPFL mapping/release/full regression qualification yet; NOT a measured delay improvement or main integration. Logs/source/protocol results_v2/sdc_scan_implementation_20261016; native source/main defaults/bin unchanged; baseline algorithms unchanged. All analysis/test sessions ended. Next align emap critical compound cells against original AIG candidate boundaries before choosing phase/cut changes.


## 2026-10-16: V19 phase/pin cut ranking — full and repeated (not promoted)

Isolated /tmp/emapper-pin-cut-20261016 from frozen V16; EGM_CUT_PIN_FLOW defaultfalse. Mandatory-cell per-phase arrival queue ranks cuts using existing real pin variants; label state is transient, not an eligible physical pool or RAM-cap claim. All original preprocessing/source/library/native baseline algorithms unchanged. Same actual cumulative20kenodes/cuts16/retained100k/SAT16/inclusive2s/native60/mapping300. Source/Cargo/bin/patch/RED/GREEN in pin_cut_implementation_20261016;116relatedtestsPASS5ignored. Initial AND8 fixture did not reproduce, retained; actual AO211 map fixture RED3.4->GREEN1.1 plus256assignment truth check. Pin-asymmetric/INV-charge tests pass.

Complete pin_cut_full_20261016:80observations including12identical-FP pilot jobs explicitly adopted (not extra runs),68physical original named-portCEC/12actual node failures.34legalpairs delay13better21equal0worse; area7better14equal13worse. Four sameD area costs remain. Two targeted alternating-order repeats24fresh24CEC,12pairs D6better6equal0worse,A4better4equal4worse; bar/sin/log2 only, not full-suite replication or best-run selection.4bar291.50/188.34->195.42/159.37;4sin486.19/1996.31->580.11/1793.92;3sin462.26/2079.62->482.56/2051.62; log2 bothDunchanged. All these target numbers repeat to0.01. Against main34commonvalid D23better11equal,A7better9equal18worse; K4i2c sameD75.42->77.42 blocks old shared-area incumbent protection. Vs original ABCdelay D13better17equal4worse; emapdelay D2better24equal8worse. Six full20 DA/ADposthoc/Pareto gates false,MtM unconnected. Main/defaults/bin8f6unchanged; no promotion. Evidence pin_cut_evidence_20261016/qualification.json,all_rows.json,validations,baseline_scope. Baseline520slots372CEC124reportedfail24unmeasured unchanged; old13403file audit reused as reference, not a fresh audit.

Causal boundary audit compound_boundary_20261016/bar_restoration.json: identical control/candidate egraph/phase aliases/PI/root interfaces; three missing original-emap compound cuts AO211/AND4/OAI22 restored with exact truth/phase/pin checks; all eight baseline critical cells endpoint correspondences independently ABC proved. Whole supplied pool attainableD188.34->159.37, pre/post100k unchanged and certified. Supports cut-coverage effect, not sole-gate causality. Sin higher-bound proof21/80 fulloriginalcorrespondences/18present3missing independentlybatchproved; unresolved not absent. Log2 only5/179 bounded proof, unmatched INV/tied-input cases cannot be called compound-cut absence. Raw simulation collisions were SAT-refuted and retained; old CNF constant convention corrected, primary older accepted subset independentlyABCverified. All long diagnostic failures/interrupted proofs archived; no offline advice enters mapping. Log2 final19998enodes/18103classes includes1895commutedAND alternatives; source helpers0 because initial planned18103>18000reserve ceiling. This is reserve-allocation evidence, NOT free remaining capacity or a causal improvement.

Independent V20 /tmp/emapper-wide-reduce-20261016 typed EGM_WIDE_CUT_REDUCE defaultfalse. Transient union up to6 evaluated then support-reduced before finalK3/K4 admission; retainedK/cut16/globalnodes/pool/SAT/time unchanged; transient search/work explicitly differs. Baseline stock emap CutSize6/minimizeTruth verified, original baseline untouched. No supergate specification passed: don't infer composed-supergate search. Actual map RED1.1->GREEN0.1 for5->4support, exhaustive256; word-TT/scalar differential and absorption family checks;118relatedtestsPASS5ignored. Frozen source/bin/patch in wide_reduce_implementation_20261016. Log2-only four-case pilot ACTIVE; no result yet. Next algorithmic plans and limits in compound_boundary_20261016/next_experiments.md.


Final V20 log2 probe: four fresh attempts/four independent original named-portCEC, actual final cut support<=K and all20k/16/100k/16/2s/60s/300s guards audited. Two legalpairs delayties, areaK4-0.04/K3tie; tiny unreplicated area change not claimedstable. More transient support minimization has a real toy correctness benefit but no log2 delay benefit in this scope; no blind repeats or fullscope superiority claim. V19 full80+targeted24 and V20pilot4 total108unique physical attempts/96CEC/12originalnode failures.12pilot observations adopted intoV19full are countedonce. Officialbaselinealgorithms/nativeparams unchanged; main8f6unchanged/notpromoted. V19 full vsoriginalemapD2better24equal8worse, ABCdelay13better17equal4worse across34commonvalid; all6fullscope gatesfalse.

Final publication: pin_cut_paper_20261016/paper_qa.json confirms6body+1reference, standardpreamble/fonts/margins and4primarytables byteidentical. Changed1/4/5/6 visuallychecked;2/3/7 pixelidenticalpriorQA. Rejected8pagePDF retained; historical prose shortened, no formattingcompression. Newbody reports full/repeated cutoff gains/area costs/bar causal boundaryproof/log2 capacity census and negative independent support-minimization probe/i2c75.42->77.42 blocker. Publishedoutput/pdf/DAC-eGraphMap-draft.pdf andprojectPDFsameSHA. Allnative/test/build/render sessions finished;136runnerlocksidle. LastsmallcheckABC/Yosys/evaluator/baselineCppbinary+source match recordedbaseline; prior13403file completeaudit onlyreferenced, not rerun. Sources/cmds/failures/diagnostic mistakes allarchived; .git untouched. Next specifics compound_boundary_20261016/next_experiments.md, especially source helper/rewrite tradeoff, tiedpin lineage correspondence, and live cut exchange preserving complete legacy shared cover within existing caps. Project not complete; no SOTA/promotion guarantee.

Final V20 bank audit: identical egraph/aliases under bothsettings, only1(K3)/5(K4) additional retainedcuts and no removedcuts; rootdelay unchanged. Thus feature is active and correct on the toy but these added banks do not improve log2. cut_effects.json retains counts; no causal helper-reallocation result exists yet. Final paper wording polish recompiled/re-rendered, onlypage6changed and visuallyrechecked;publishedSHA is inpaper_qa.json.


## 2026-10-17: guarded cut exchange and exact source propagation (no promotion)

Parent V19 remains isolated. V21 `/tmp/emapper-cut-exchange-20261017` typed default-off EGM_CUT_EXCHANGE_GUARD keeps a raw legacy optional cut prefix unless scalar minimum arrival rank improves. RED/GREEN direct selection fixture,118egraph testsPASS5ignored. Initial missing ASAP7 symlink fixture failure retained; original library restored without algorithm changes. Pilot12/12physicalCEC:K4i2c77.42->75.15,bar195.42->193.24sameD,but sinD+21.66/+13.69(K4/K3). Rejected, no source/config winner mixing.

V23 `/tmp/emapper-phase-cut-exchange-20261017` changes the guard to separate positive/negative output-phase minima; scalar-rank negative-phase fixtureRED/GREEN,119egraph testsPASS5ignored. Full `phase_cut_exchange_full_20261017`80/80 includes12identical-FP pilot observations explicitly adopted/count once:68physicalCEC/12actualnodefailures;34legalpairs D1better26ties7worse,A8better17ties9worse. Six same-D area regressions vs V19 control remain (K4cavlc/ctrl/int2float/router,K3sqrt/arbiter). K4i2c77.42->74.55same165.79;bar195.42->193.24same159.37;sin580.11/1793.92->592.86/1815.58. `phase_cut_exchange_repeat_20261017`16fresh16CEC reversed-order i2c/bar/sin/sqrt:8pairsD6ties2worse,A4better3ties1worse; targeted, not full-suite replication/statistical guarantee. All individual rows including adverse observations retained. Vs canonical main8f6all34commonlegal D22better12ties,A7better10ties17worse; no same-D area regressions vs main, but V19 delay improvements are not all preserved, hence no promotion. All6full20DA/ADposthoc/Pareto gatesfalse;MtMunconnected.

Causal limits: old K4i2c selected physical truth boundaries reconstruct55cut keys present in old but absent in V19 new banks; all55calibrate against old banks. V23 restores53/55 keys, not a full physical-cover replay proof. sqrt old selected cut keys all exist in both original banks. V23sin generated/retained pool certificates bothpass1793.92(control)/1815.58(candidate), with loss before100kcap. Guard placement after diversity can replace that bank with the raw legacy prefix, so this is not an isolated phase-only causal result. A local cut prefix/arrival label is not a full shared-cover certificate. Next concrete obligations in `phase_cut_exchange_implementation_20261017/next_transaction.md`: pin all validated old input cones/phases/joint-output physical bindings before any live eviction, reject conflicts within16cuts/100kpool, one inclusive2s including old search/setup/transfer/final validation, no two full free banks or two2s recoveries. No offline saved best netlists as advice.

Independent V22 `/tmp/emapper-source-propagation-20261017` from V17 (not mixed with V23) adds typed default-off EGM_SOURCE_PROPAGATION: topological signed constant/idempotence/complement/structural-hash propagation after unchanged bounded exact ROBDD aliases. No new semantic queries,4095proof decisions/20000calls/.2s search checkpoints unchanged (not hard whole-stage cap); all transform work inside native60/mapping300 and every actual imported function charged cumulatively. RED3AND->expected1/GREEN;4source tests including64signed DAGs×64assignments and port names;97egraph testsPASS4ignored. `source_propagation_pilot_20261017`16attempts4physicalCEC(priority controls)12node failures. div after BDD40651rawAND->40635,selected source estimate K420373->20366/K320545->20537; actual both stop20000incomplete import. Hyp/mem_ctrl unchanged. These are representation estimates, not minimum bounds/PPA/feasibility. `source_propagation_cec_20261017`3independent original-prepared sourceCEC+4positive/negative calibrations all expected, matching both actual library proofcounter sets; sourceCEC is not physical mapping success. Next bounded existing-choice joint closure hypothesis in `source_propagation_implementation_20261017/next_source_selection.md`.

This round124unique physical attempts/100CEC/24node failures across independent versions; V23pilot12counted once inside80. No combined-version mapper or per-design winners. Same once-resyn2 EPFL20 subjects/two restricted ASAP7 inventories/public unique-instance area/unloaded dependent-pin delay/original named-portCEC. Actual cumulative20kenodes/cut16/retained100k/SAT16/inclusive recovery2s/setup/finalvalidation/.05reserve/native60/mapping300unchanged. Retained pool is not generated transient RAM cap. Baseline original algorithms/models/params unchanged; original240groups520slots372CEC124reportedfailure24unmeasured preserved. Fresh four common binary hashes match original baseline checkpoint; prior13403-file complete audit referenced, not repeated. Main/default/source/bin8f6unchanged; .git not modified (read-only gitstatus failed at existing LFS filter, not rerun). Evidence `cut_exchange_evidence_20261017` includes full/repeat/source audits, all rows/pairs/commands/source hashes/gates and maincomparison. Paper updated only with real full negative study and independent source result; reference four tables unchanged. Publication QA recorded separately after compilation.

Final publication: `cut_exchange_paper_20261017/paper_qa.json` verifies6body+1reference, unchanged standard preamble/fonts/margins/four primary tables. Final4/5/6visuallyinspected;1/2/3/7pixelidentical to previously published PDF. Rejected8page/source retained; prose condensed only. Draft helper-vs-main sqrt attribution corrected before publication. All finite sessions ended,141runnerlocks idle; main8f6notpromoted.


## 2026-10-17: complete-cover transaction seam and bounded source groups (not promoted)

V24 isolated `/tmp/emapper-cover-transaction-20261017` from V19 implements atomic selected-cover physical identity/joint-pin/input-cone/phase/cut/care support checks, exact derivation for new cuts, strict whole-cover(D,A) non-regression, aggregate16cuts/100kpool and caller-deadline checks. New RED catches unproved unused-cut admission;14seam tests plus130egraph-related regressionsPASS6ignored. Engine is NOT invoked by live cut search. Caller still must generate library-legal proposals and one active bank under the original one-inclusive2s recovery window. It cannot protect an archived full2s best cover merely by shortening the initial search.

Read-only replay of34prior V19 selected covers initially29pass/5failclosed due missing unconditional provenance (K4adder/ctrl,K3adder/cavlc/ctrl). Reporting-only capture exports existing Reachability cache care masks,8fresh original named-portCEC on adder/ctrl/cavlc/sin acrossK3/K4. Fixed egraph syntax/aliases/roots/PI/free/cutbanks exactly match for5joined contexts; audit-only reconstructed34/34old covers replay,8/8fresh snapshots replay. These are selected-cover checks, NOT new34mappingCEC/full100kpool qualification or an engine delay gain. Audit context files never feed official mapper inputs. Slowest selected-only transaction audit1.770s leaves original search+live exchange unqualified under2s. Main and baseline algorithms unchanged.

Independent V25 `/tmp/emapper-source-groups-20261017` from V22 adds opt-in EGM_SOURCE_GROUPS defaultoff: up to4same-rank consumers jointly change existing exact choices, reusing original16TOTAL source transactions/root (including pairs), no extra truth/proof queries or globalnodes. RED3consumer fixture7->6reachable;98egraph-related regressionsPASS4ignored withflagoff/on. Fixed16native attempts4priorityCEC/12actual20000incomplete imports. Source estimatesK4div20366unchanged,hyp106345->99143,mem_ctrl24872->24860;K3div20537unchanged,hyp111430->104047,mem_ctrl27914->27903. Not minimum lower bounds/import success/physicalPPA. K4priority area52.29->52.42same2047.24 retained adverse observation, no stable effect claim;K3tie. Native algorithms/models/baselineparams unchanged. Original first runner failed after1successful controlCEC at tuple-key checkpoint serialization; entirepartial/failingharness preserved separately, correctedv2doesnotadopt/overwriteit. Total25uniquephysicalattempts13CEC12actualnodefailures this round across independent versions, not a combined mapper.

Evidence `cover_transaction_evidence_20261017/report.md,qualification.json,next_transaction.md`: live generation provenance tokens and incremental pin/refcount work are next, before full20two-library live-engine PPA/repeats. No new full-engine matrix/gate/win claimed; existing sixfull20 gates false. Original baseline240groups520slots372CEC124reportedfailure24unmeasured remain; fourcommon binaries freshlymatch, prior13403file audit reusednotrerun. Main/source/default/bin8f6unchanged;144runnerlocksidle/allfinitejobsfinished;no.gitwrites. DAC updated factual replay/source limits,6body+1reference,standardpreamble/fonts/margins/fourprimarytablesunchanged;4/5/6visuallyQA,1/2/3/7pixelidenticalprior. PublishedPDF andprojectPDFsameSHA.


## 2026-10-17: single-output tail area recovery, full EPFL and repeated (not promoted)

User primary line is now single-output standard-cell mapping; multi-output remains a separately tested extension, not the source of EPFL gains. Isolated V26 `/tmp/emapper-single-output-area-20261017` from frozen V19 adds typed default-off EGM_REFRESH_AREA_TIMING and EGM_TAIL_SHARED_AREA; original baseline algorithms/models/nativeparams unchanged. A real single-output RED shows stale timing labels after online commit miss a feasible area reduction10->3 atfixedD10; refresh fixes that seam but costs sqrt area in the factor pilot.119related regressionsPASS5ignored,release/source/Cargo/CLI/bin/patch/logs frozen in single_output_area_implementation_20261017.

Factor pilot16/16CEC:control,refresh-only,tail-only,combined over sqrt/sin twoinventories. Choose ONE global tail-only configuration (refresh0/tail1), not perdesign winners; all negative refresh/combined results retained. New tail saves complete visited predecessor, activates better validated deferred cover, allocates existing tail30%flow/70%actualarea, samepool/oneinclusive2s. This does NOT preserve old future area-flow results automatically; no cross-cut exchange is implemented by this factor. NativeV24cut/care transaction seam remains separate/unconnected.

Full single_output_area_full_20261017 complete80/80 includes8explicit identical actual-dependency/factor pilot observations adopted/countonce,72newruns.68independent original named-portCEC/12actual20000incomplete import failures(div/hyp/mem_ctrl bothlibs/bothsettings).34legalpairs ALL delayties; area22better7ties5worse. K4sin580.11->564.42D1793.92;max268.61->232.79D2072.99. K3sin482.56->481.11D2051.62;max249.37->235.91D2111.03. Bothsqrt areas tie in full. Five equal-D area losses:K4ctrl+.03,int2float+.30,priority+.05,router+.01;K3priority+.10, allretained.

Two fresh alternating-order target repeats single_output_area_repeat1/repeat2_20261017 each20/20CEC, sqrt/sin/max/int2float/priority twoinventories, total40CEC/20pairs. D20ties;area10better3ties7worse. Sin/max gains recur bothlibs;oneK4sqrtpair+.04,other3sqrtties,no stable sqrt improvement. K4int2floatandbothpriority regressions repeatbothrounds. Notfull-suite replication/statisticalproof/best-run selection. K4int2floatcontrolrecovery.205s,4slackattempts/3acceptances versusnew.198s,2/1: changingreference/fanouts changesflowtrajectory; NOT totalbudget exhaustion. Fourotherfull losses hadnoimprovedreferenceactivation yet onlinepolicychanged. Nextindependentguard should runoriginal boundedflow onoldworkingcover/refs first, retainthatcomplete result, thenswitchreference onlyifbetter;noaddedflow/newdeadline. No actualarea gainon sqrt claimed.

Round unique128physicalattempts116CEC12nodefails =16pilot+72newfull+40repeats;8adoptionscountonce. Everyvalidactualnode<=20k,cuts<=16,pool<=100k,SAT16,cutrounds32,inclusive recovery2swithsetup/finalvalidation/.05reserve,native60/mappingonly300 unchanged. CEC/preparation/conversion/metricreportexcludedmappingclock;temp generation notRAMcap. Sourcegraph/cut/canonical/rootSHAequal across34fullcontrol/candidatepairs;stored64bit retainedpoolfingerprints equal,notanindependentlyexportedcryptographicpoolproof. Single-output inventoryauditK3=19cells/K4=30,maxoneoutputrecord/name;allvalidmappedmultioutputinstances0.

Evidence single_output_area_evidence_20261017/report.md,qualification.json,paired.csv,all_rows.json,integrity.json,full/repeatvalidations/baseline_scope/frozen_harness. Maincomparison34legal D23better11ties,A14better2ties18worse;K3prioritystill+.10equalDvs main. Full DA/ADposthoc/Pareto sixgatesfalse,MtMunconnected;ADnotthisrunobjective. Originalbaseline240groups520slots372CEC124reportedfail24unmeasured unchanged;4commonbinaryhashes freshlymatch,prior13403file audit onlyreference. Main8f6/source/defaults unchanged,NOTPROMOTED. HistoricalQoRguard REDfull5losses retained, not hidden as a green unit test.

Allfinite mapping/test/build/render/audit sessionsfinished;148runnerlocksidle. DACtitle/abstract/introduction/figure/method/discussion now single-output mapping with delay-constrained area recovery;multioutputonlyextension. New gains AND losses recorded;4old main-reference numericaltables unchanged.6body+1refs standardfont/margins;pages1-6visuallychecked,6also160dpi,7pixel-identicalpreviousQA. Publishedoutput/pdf/DAC-eGraphMap-draft.pdf/projectmain.pdfsameSHA. Paperqa single_output_area_paper_20261017/paper_qa.json. PDFskillmarker ranexactlyoncebeforefirstauthoring;one laterrelative-path figureeditfailedbeforemutation,correctedabsolute/set-e;allfinalPDFre-rendered/verified. Projectnotcomplete;nextplan next_protected_tail.md.


## 2026-10-17: independent single-output component qualification (in progress)

Main/source/defaults/native8f6 and all author baseline algorithms/models/native parameters remain unchanged. See `results_v2/DA_CURRENT_CHECKPOINT.md` before starting any runner; the current finite V28 component campaign is active and must not be duplicated or overlapped with builds/audits.

V27 protected-tail scheduling was tested on the full20/two-library same-binary pair matrix (`single_output_components_guard_full_20261017`,80observations including28 explicit identical-dependency pilot adoptions counted once):68 physical named-port CEC,12 actual prefix node failures.34legalpairs all preserve delay; area7better/25equal/2worse. Adverse K4sqrt+.54 and K3arbiter+.04 are retained. V27 is NOT promoted and does not reproduce V26 sin/max reductions; deferring reference activation can reach the deadline before the switch.

V28 clean ablations live in `/tmp/emapper-single-output-clean-components-20261017`, source/bin/Cargo/RED/GREEN frozen in `single_output_clean_components_implementation_20261017`. A minimized AND3TT plus original(AB,C) sourcecut regression proves that deleting source fallbacks confounds an optional-cross-enode-cut ablation; both settings now retain identical source fallbacks.122related tests pass,5ignored. The fresh320 matrix (`single_output_components_full_20261017`) is RUNNING, not complete:20designs*2inventories*8fixedprofiles. Full-method reference and all factors use uniform tail0; no per-design choice. Profiles separately restrict optional cross-enode cuts, input-negation matching, physical pin permutations, SDC, final actual-area tail, atomic phase recovery, and required-time DAG+slack area module. Current frozen controller description inherited a V27 version label; actual V28 source/native SHA/argv are authoritative and `scope_correction.json` records this metadata error without rewriting the frozen matrix. No V27 mapped result is adopted into V28.

Input-phase-off and canonical-pin controls restrict legal matching coverage, not merely timing ranking. Actual log2/sqrt undriven failures must be retained and cannot be called delay losses or used to claim that ordinary NPN matching is novel. Function-preserving one-legal-assignment/cost-model controls are still a stronger follow-up. Equality in the witness+source cut control does not prove saturation itself has no value. Both controls preserve full saturation, original physical fallbacks and all global counting. Pure primary-input-support SDC remains conservative and external; no conditional union and no ODC.

All original cumulative20kenodes/cuts16/retained100k/SAT16/cutrounds32/one-inclusive2s recovery with mandatory setup/final validation/native60/mapping-only300 remain; generated candidates/RAM are not the retained pool cap. Actual physical area/unloaded dependent-pin delay/original named-port CEC and original once-resyn2 subjects are unchanged. Original baselines remain240groups/520slots/372CEC/124reported failures/24unmeasured; MtM unconnected. Full frozen V28 results/repetitions/source/command/coverage audits and lex/Pareto gates are pending. Paper still the prior V26 draft until verified new evidence; no completion/SOTA/main promotion claim.

Storage interruption during the in-progress V28 run: checkpoint220 survived ENOSPC. Own campaign5091files/4.762GB were SHA-verified and relocated to `/tmp/emapper-artifacts-20261017` with the original project path retained as a symlink. One native-complete but independently unmeasured partial sin/no-SDC attempt is retained intact; one predetermined same-settings retry in a separate directory passed CEC and was explicitly registered. Count the original as an extra unmeasured native attempt, not as a discarded bad case or duplicate success. Resume log/ledger and planned durable verified compressed archives are recorded in the current checkpoint. Full320 is still running, not complete.

## 2026-10-17 单输出组件独立消融：完成，暂不合入

真实证据：[报告](../results_v2/single_output_components_evidence_20261017/report.md)、[qualification](../results_v2/single_output_components_evidence_20261017/qualification.json)、[完整性](../results_v2/single_output_components_evidence_20261017/integrity.json)、[重复汇总](../results_v2/single_output_components_evidence_20261017/repetition_summary.json)。

V27隔离保护版先保留旧flow再切deferred，121相关回归通过/5ignored；全80配置68CEC/12导入失败，34合法对D全相同、A7更低25相同2更高。V28修正消融共享源fallback，RED/GREEN最小回归和122相关测试通过/5ignored，独立全320配置254CEC/66失败。两源码分开冻结，不跨binary采用映射结果。全部phase/pin/SDC/actual-area/required-area等因素固定运行全部20设计两库，tail0统一reference，不按设计选winner。

结构审计：34合法对30个cut bank字节相同，仅ctrl/i2c两库各多1cut；AND交换折叠后没有不同AND child pairs或mixed nonfree forms。关闭SAT新跑40例，40同版同依赖reference含失败显式采用：34合法对D全相同。因此不能把普通NP匹配、候选计数或当前延迟收益归因于跨结构EqSat。phase-mask/pin canonical限制另外删除12/6个必要匹配，属于能力探针，不是保持合法集合的纯时序排序消融。

关闭实际area tail：12/34面积更高，D全相同；关闭required/DAG/slack area模块：30/34面积更高，D全相同。关闭SDC：A3更低24相同7更高，D33相同1更高。候选/中间更新不证明最终采用，开关同时影响后续时间与轨迹。外置SDC是表示与预算策略；固定原PI函数的guarded具体union可以证明安全，不能泛称所有conditional union不安全。ODC仍未实现。

120 fresh目标重复（bar/sin64+sqrt24+保护版32）全CEC，顺序预定反序/正序、不取best run。保护版16重复对D全相同，A4更低7相同5更高，含K4priority+.07、K3sqrt+.04、K3arbiter+.02/+.06与K4sqrt+.23等回退。全套与重复均不支持无回退，且旧sin/max收益丢失，V27不合入；主8f6与默认策略不变。

去重561物理尝试=560qualified记录（476CEC/84记录失败）+1基础设施中断CEC未测。ENOSPC原部分目录完整保留，逐文件校验迁移/tmp并保留逻辑路径，仅一次同配置预定重试，未用partial成功/原net提示或挑PPA。独立审计实际来源/hash/CEC/commands/nodes/cuts/pool/deadline通过，159runner锁空闲（归档前记录）。原始临时结果的8个永久压缩归档逐成员SHA已核验；archives清单随报告保存，未删坏例。

预算不变，但必须准确描述：20000累计导入enodes、cuts16、最终retained100000、SAT上限16/cut32/4轮每轮2000trial、native60/mapping300。2秒是post-admission恢复，包含该阶段setup/finalvalidation/预留；原有限finite/frontier准入排序5秒subguards在这2秒前，仍计native60/mapping300。raw生成（如K4sqrt约152万）不受retained100000的数量或RAM保证。不能把驻留live数、原输入豁免或每region重置当作旧累计预算。

原版baselines240组/520状态（372CEC/124失败/24未测）不变，本轮未重新优化/重跑。小binary新鲜校验匹配，旧13403文件全审计按SHA引用，不称本轮重新全源码审计。34共同合法对D：ABCmap13更好17相同4更差，emap2更好24相同8更差；A仍26/28更差。两库DA/ADposthoc/Pareto六gate全false，MtM未接入。div/hyp/mem_ctrl在SAT前导入超限，减少SAT交换节点不能修复。

下一工作：先证明能产生有用库边界的真实结构规则；补充保留至少一个合法NP assignment的纯排序对照；完整旧恢复结束后只用剩余原trial/round/window做共享面积continuation（sqrt有效窗口已接近耗尽，需先降低评分开销）；安全源压缩另立版本，全计累计节点。新结果通过完整scope/repeats前不合入，不靠放宽预算、同设计混版本、基线改名宣布胜出。论文更新记录在 single_output_components_paper_20261017，主四表保留8f6 reference，新增独立组件表。

## 2026-10-18 完整旧终态保护：局部改进，暂不合入

证据：[报告](../results_v2/terminal_area_evidence_20261018/report.md)、[汇总/全部内部证书](../results_v2/terminal_area_evidence_20261018/qualification.json)、[9个永久归档](../results_v2/terminal_area_evidence_20261018/archives.json)。

V29 `/tmp/emapper-terminal-area-20261018`：旧flow/actual-area/deferred/legacy最终比较全部结束后，才启动固定旧时序界限的area-priority续接；只使用原4个实际area轮次的余额及原每轮2000应用trial额度、原search deadline，不新开窗口。最终原始A和D同时不增加，非法/超期/变差保留完整旧cover。该内部保护不证明另一次独立限时运行会给相同旧结果，不是全局/已访问字典序最优。typed EGM_TERMINAL_AREA默认false；main8f6没有集成。

全EPFL两库2profiles80配置68CEC/12原节点失败，显式采用全部24相同FP试验记录并计一次。34合法对D全相同，独立运行A5更低28相同1更高；K4sqrt2251.12->2251.18(+.06)，该case终态未启动，保留不宣称已消除波动。11case启动、2case实际更新：K3int2float14.15->13.80/D183.34，K4cavlc43.28->40.98/D186.01。两个设计分别两库正反正各3reps共24freshCEC，12pairsD全相同/A6更低6相同；每项实际收益3/3复现，对照库没有收益。不把无实际终态接受的其他跨运行差异称作新机制收益。

V30 `/tmp/emapper-scored-area-20261018`独立缓存full-oracle已评分覆盖，原working-area过滤前按已知快速deferred界限保存，旧最终比较后strictAANDD守护；不新增oracle/trial/round/pool/window。实际路径RED(工作A1/D5忽略已评分A10.1/D1但最终选A20/D1) -> GREEN；64DAG差分旧选择/评分/trial/轮次相同。128相关PASS5ignored。六设计两库24CEC：12Dties、A1更低9相同2更高，但最终archive采用0次，因此不视为改进、不盲跑完整矩阵。记录/复制开销也影响限时轨迹；接管原本要丢弃向量的所有权方案是待测，未实现。

V29相关125PASS5ignored、release完成；初始完整库135PASS2missingfixtureFAIL5ignored保留，隔离补原repo supergates夹具后2失败项PASS。全部源码/Cargo/testfixtures/bin/patch/失败日志冻结。去重128物理尝试116CEC/12记录失败，旧24试验不重复计。所有同输入once-resyn2/同库/publicphysicalarea+unloadedpindelay/originalnamed-portCEC、累计20knodes/cuts16/retained100k/SAT16/cut32/post-admission2s/native60/mapping300保持，线上完整评分不等于应用trial计数。原pre-admission5ssubguards仍在2s外、native/mapping内，retained上限不是RAM/raw生成上限。

check_terminal_area.py/check_scored_area.py确认actualbudget/commands/CEC/SHA及同graph/cuts/aliases/记录poolfp、轮次余额与同运行完整旧终态rawA/D证书。原baseline240/520状态不变，smallbinary重新hash匹配，旧13403文件全审计仅引用；两库DA/ADposthoc/Pareto6gate仍false，MtM未接入。div/hyp/mem_ctrl仍SAT前累计导入超限，本轮后端恢复不能修复。

9个raw目录的永久tar.zst逐成员SHA校验通过，155444702压缩bytes，原失败不删除。归档helper旧restore文案误写20261017，已明确元数据erratum修正20261018；原v1元数据保留，archive/memberhash及实际实验没有变化。PDF6+1已更新局部收益/独立运行限制/缓存负结果，5表源数据与标准字号边距不变，QA见 terminal_area_paper_20261018。下一真正结构工作为已计数信号的全care精确功能分解cuts，方案在 scored_area_implementation_20261018/next_exact_functional_cuts.md，仅计划尚未实现，需candidate-presence RED+实际物理CEC/全套同预算，不得当成已解决大输入。

## 2026-10-19 跨结构功能 cuts：sqrt 可复现收益，保留全套负结果

证据：[完整报告](../results_v2/functional_cuts_evidence_20261019/report.md)、[汇总及六gate](../results_v2/functional_cuts_evidence_20261019/qualification.json)、[选中结构/公共关键路径](../results_v2/functional_cuts_evidence_20261019/selected_cover_analysis.json)、[永久归档](../results_v2/functional_cuts_evidence_20261019/archives.json)。

V31隔离 `/tmp/emapper-functional-cuts-20261019` 默认off EGM_FUNCTIONAL_CUTS。已有计数类g的全care quotient F(S)=Q(S\T,G(T))，最多4伙伴/support、32proof试验/root/原round，在原32round/16cutbank中竞争，无新e-node/条件union。两个profile都terminal_area1，其他固定。130egraph相关回归PASS/5ignored、release和初始失败日志均保留；原mandatorysource witnesses不变但optionalcut和物理pool可变，终态own-run守护不等于保住另一个pool的旧cover。

完整80项68CEC/12导入失败，全部24pilot同FP显式采用且只计一次。34有效pairs：D7better25equal2worse，A15better7equal12worse。sqrt K3 2153.90/60087.26→2021.89/55862.56；K4 2251.18/47047.28→2215.87/47031.68。sqrt/bar/sin两库3轮正反正共36freshCEC，18pairs D6better9equal3worse/A12better6equal。两库sqrt各3/3共同改善；K4sin D+12.63同样3/3重复，adder也D/A回退。固定emap-delay比较，sqrt K3 D约-7.03%而A约+22.9%，K4 D约-.033%而A约+41.8%，不是Pareto全面支配。

只读critical path复核公共f32 pin→f64D及全部instance area；K3sqrt 2951→2052cell，新选目标2332个只匹配新最终bank的精确cut边界（不是新cell数，也不证明旧未导出pool没有SDC匹配）。K4sin路径81→78但D增加，bar该类新边界选中绑定0。新最终cuts、挤出旧cuts和全部paths/局限均存档；不把此分析作为mapper免费候选建议。

V32独立 `/tmp/emapper-source-closure-20261019` 从V25来，不与V31混版本。EGM_SOURCE_CLOSURE_FILTER默认0，bothgroups1；最多256展开的源依赖检查（第257个seen元素作为拒绝标记，不是新增egraph函数），原truth bank及16联合group/pair事务/root不增，原singleton扫描不在计数内且不变。RED首个候选仍依赖共享锥、后三个消费者联合释放→GREEN；19source与101egraph/4ignored两组有重叠，不相加。16attempt4priorityCEC/12实际导入fail，均incomplete/admitted20000/limit20000。源计划 K4 div20366→20366/hyp99143→106344/mem24860→24854；K3 div20537→20537/hyp104047→111430/mem27903→27904。这不是实际完整导入、最小下界或PPA。过保守过滤/触发共享条件可能漏有效组；rank剪枝仅计划，不改冻结版混入实验。

总132unique attempts108CEC24actualnodefail；主native8f6/default/source不变，不合入两新版本。两库DA/ADposthoc/Pareto6gate全false，MtM未接入；原baseline240/520状态、372CEC/124failure/24unmeasured不改。4smallbinary重新hash一致，旧13403文件审计仅引用不冒充本轮全重新核验。所有once-resyn2/同库/publicmetrics/originalnamed-portCEC、累计20k/cut16/retained100k/SAT16/cut32/post-admission2s/native60/mapping300原限额保留；pre-admission5s不在2s内，retained不是生成/RAMcap，线上完整评分非应用trial数，不免计原输入或reset预算。6份tar.zst逐成员SHA通过，所有失败不删除。论文记录新全scope/重复/负结果，标准6正文+1参考、五表源数据不变；完整视觉QA见 functional_cuts_paper_20261019。


## 2026-10-19：完整旧覆盖审计、两相位筛选和保护旧源计划的独立实验

本轮58次真实尝试，28次独立公共CEC通过，30次均实际累计导入20000/20000失败；所有失败和源/命令/参数/SHA保留。详见 `results_v2/phase_source_evidence_20261019/report.md`、qualification.json、failure_stages.json。三个版本独立，不按设计组合优胜配置，未合入主版本。

两相位硬筛选V35：18次全部CEC，针对sqrt/sin/adder两库三固定配置。相对functional-off，D2改善/4持平，A1改善/2持平/3回退；相对functional-on，D2改善/3持平/1回退，A1改善/1持平/4回退。K4 adder恢复90.31/2569.79，K4 sin D恢复1793.92但A580.35仍高于580.11；K4 sqrt保留47031.68的D但A2258.91高于旧2250.29。K3 sqrt由功能cuts的2021.89/55862.56变为2153.85/60084.45，丢失大部分收益。故拒绝这条硬门槛，不以局部标签冒充整网时序/共享面积保证。

旧完整已选覆盖只读审计：K4 sin161个绑定失去所有旧精确full-care cut依据，其中3个旧关键路径绑定；K4 adder1个非关键绑定；sqrt两库均0。计数是target binding不是cell；未审计整个未选pool和SDC替代，不能据此证明候选绝对缺失或单一因果。当前terminal-area仅保护自身运行旧final，尚未实现跨pool完整旧覆盖保护。详细next_complete_cover_exchange.md明确一bank16/一pool100k/剩余旧轮次与deadline，不添第二2秒。

源V33验证严格rank递减后对低/同rank非目标分支剪枝，原256展开未知拒绝兜底。24次6priority CEC/18大输入实际导入拒绝。K4 hyp计划99143→98879（旧无rank硬过滤106344），K3 hyp104047→103973，mem K4 24860→24848、K3 27903不变，div两库不变。这些只是源表示估算，不是下界、完整import或PPA。

独立V34先跑完整旧source plan，只用原每root16次group/pair剩余次数做末端组事务，保留旧引用闭包并仅负delta接受；256随机DAG非回退/独立计数回归。16次4priority CEC/12大输入实际导入拒绝。K4 mem24860→24859，K3 hyp104047→103955、mem27903→27901，div不变。所有before_terminal→after非增；不代表实际enodes或physical cover保证。priority小输入可能literal-auto绕过源选择，不把这些控制CEC当大图压缩已成功。

预算与baseline原算法全部不变：20k实际累计导入、cuts16、retained100k、SAT16、cut32、area4轮×2000应用、post-admission2秒含setup/final/reserve、native60/mapping-only300。pre-admission5秒子guard仍在2秒外/native与mapping内，原singleton不伪装计入group/pair16，raw RAM不等于retained cap。源新版本不与V35混合。12个pilot fullscope gates均false，缺失设计显式记录；AD仅posthoc；MtM未接入。

原baseline240组/520槽372CEC/124报告失败/24未测量不变；四小binary新核SHA一致，前13403文件full audit引用未重跑。本轮负目标pilot尚未达到全EPFL/重复资格，故不盲扩展或声称全覆盖新改进；V31旧80全套和36目标重复完整保留，不改名填入新版本。

独立源码/bin/patch/RED/GREEN/build和初始失败永久保留在ranked_source、terminal_source、phase_screen_implementation_20261019；三raw目录已有逐成员SHA回读的永久tar.zst，运行路径和失败不删除。论文只更新真实负结果/源限制，主五表、字体边距不变；6正文+1参考，改变第5页目视QA，其余六页像素与前QA一致。新PDF证据phase_source_paper_20261019/paper_qa.json。下一步按原预算做actual required-time/全绑定care+phase+pin provenance/共享引用联合交换；源压缩补现有helper的缺失等价组合并独立导出源CEC，不能输入免计或重置累计预算。


## 2026-10-19：按用户授权合入主源码，再全面分析

用户要求“先合并入主版本，然后重新全面分析”，覆盖此前等待所有PPA门槛后才接入的限制。已经实际更新emapper/src、CLI与target/release主binary；没有Git提交或.git写入。主旧8f6与新5390091完整before/after源码、Cargo、binary、patch和所有验证永久存 main_integration_20261019。没有把旧隔离研究的数据改名为新main成绩。

合并选较新physical/recovery分支V35/V31/V29；源V34/V33/V25单独进入egraph_source_import.rs，保留较新相位/输入锥恢复，不整份覆盖较旧source分支regions。TT4解析/权重、source_proof/source_plan及CLI开关已接通。新功能默认off，已淘汰phase screen与source closure gate也off；既有MapParams默认逐项相同。通用默认multioutput仍为true，单输出研究必须显式0。

172完整库测试通过、5ignored；release重编译。新主双库sqrt/sin/adder、functional0/1共12新attempt全部公共CEC，D2改善2回退2同、A4改善1回退1同。sqrt收益保留，同时K4sin/adder已知回退也保留。没有新的full20或重复验证；source处理在这12项off，不能称source+cuts组合全套通过。合成组合路径收紧节点上限6：旧source禁用拒绝，启用source证明/规划+functional+terminal后6节点完成/ABC CEC1/原四PI及索引名保持；5节点拒绝。另普通strict入口6节点探针实际构造16节点后报超限，未接受非法cover：证明全入口还需统一前置账本，不推翻正式bounded-cone实验的逐节点记录。

重分析历史V31对固定emap-delay（不是新main成绩）：每库17/20共同合法；K3 D2好12同3差/A2好1同14差，条件几何比D1.003713/A1.119540；K4 D6好6同5差/A3好1同13差，D1.006580/A1.122529。失败逐项保留，不能将条件均值称完整20胜出。严格(D,A)与A<=old额外约束有区别；旧低A但更慢cover不自动满足新的Dref，不能跨cover拼A/D。

完整报告/流程图/事实与假设/三大输入/优先级在main_integration_20261019/analysis.md，数据analysis_data.json。下一步先打通全部绑定cut/care/phase/pin来源和完整旧共享闭包、统一strict入口前置准入，再在固定强Dref下用增量共享引用/arrival/required-time做area flow/实际面积恢复；源候选单独推进，div最靠近门槛。建议新main固定R/F/S三配置两库全20共120项，预声明目标重复，同库同预算/原baseline不改，不挑设计赢家。此为计划未执行。

主源码合并不是PPA领先或所有功能完成证明。旧论文主表保持旧参考记录，本轮未改PDF或向旧表填新main12项。源/命令/参数/负结果/SHA与永久raw archive均保留，所有finite任务结束，无后台依赖。

## 2026-10-19：绑定来源与受保护恢复的本轮实验（执行中）

本轮新增实验开关，默认关闭：生成时记录 cut/care/原始与规范相位/具名引脚/物理身份，去重保留不同证明；保存本次实际访问的完整覆盖（含非关键共享依赖），交换必须保留其证据与唯一实例面积；固定Dref的required-time area flow和实际面积轨迹独立保存合法已访问(D,A)点。保护范围不是另一流程跑满两秒后的历史最终覆盖，不能据此宣布所有面积回退已消除。

V38目标18次全CEC，但丢失sqrt收益；V39/V40各两次sqrt探针均包含K4全局期限失败（原样保留）；V41两次sqrt全CEC并保住D，但K4面积仍退化。各版独立before/source/bin/patch/RED/GREEN/失败日志在binding_exchange、binding_stage_reserve、binding_transaction、binding_functional_seed_20261019。未按设计择优拼接。另修复到期返回原始候选前缀未规范化、在线提交仍错误收紧D界限、非法昂贵快点排挤合法中间覆盖等反例。

当前V42冻结binary f84b79fb25391e63557c97f636e567df506a2c6468ade8b276a77ed451ebce1a；185库测试通过/5ignored/3既有耗时synth测试未完成并显式排除，初次中断及错误反馈均留存。三配置（legacy、functional、protected）两库全20共120次正在顺序执行，结果/tmp/emapper-binding-noop-full-20261019，证据binding_noop_recovery_20261019。source_hashes.json与work_state.json为本轮检查点；不得在冻结研究执行时改Rust/driver/runner或并发跑新campaign。有限当前任务须同turn等待，不宣称后台跨turn存活。

全局累计20k/cut16/retained100k/SAT16/cut32/recovery2秒/native60/mapping-only300均未增；可选交换另收紧64新cut/2048 offered matches。原baseline240组/520槽及失败保留，算法/模型/native参数不变。新流程仍见K4sin延迟、adder面积、sqrt面积等回退，未获默认采用资格。候选池证书已定位K4sin/adder控制配置的D差距在原始生成候选中，而非100k裁剪或最后恢复；不可扩大为库全局最优证明。完整对照后按finish_binding_round.py有限顺序核验、三次目标交替重复与全scope gates；原发表PDF此时尚未改写。所有新原始目录须在计时任务全部结束后永久逐成员SHA归档。

## 2026-10-19：本轮完整对照结束，重复暴露失败；不得默认采用

V42三配置全套120/120、102独立CEC/18累计导入拒绝；sqrt/sin/adder三轮交替54/54CEC。对legacy的34合法对：D8好25同1差/A25好1同8差；对functional：D1好33同0差/A23好1同10差。全部18 fullscope gate false。重要：K4sqrt第一次重复D85011.84，对照47031.68，另两次才保住47031.68；不能称两库sqrt收益全重复稳定。K4sin仍D+12.63，adder与K4sqrt仍有面积回退。完整证据binding_noop_recovery_20261019/report.md、evidence/qualification.json、八份逐成员SHA永久归档，失败未删除。

已接通本次生成的cut/care/相位/引脚/物理身份来源及完整已访问seed闭包、同Dref required-time与实际共享面积恢复。但完整闭包不是另一旧流程独立跑满预算的历史最优共享面积，不能据此称已保护旧最终PPA。原baseline240组/520槽372CEC124失败24未测不变。

新V43只复用实际候选保留决策已计算的完整area witness与delay frontier，索引重映射/验证仍在2秒内；绝不复用额外只读报告frontier。189测试通过/5ignored/3明确未跑既有耗时测试。诊断证书打开探针2次1CEC1native60失败；统一关闭仅报告证书后四探针2CEC2native60失败，均存generation_frontier_reuse_20261019。V44只改诊断JSON移动/流式序列化，原映射算法/参数/60s/300s与2s等均不变；新source/bin2805993e固定，sqrt探针K4reuse公共CEC保住D47031.68，recompute仍timeout；正在独立全套与重复核验，不混旧版本、不宣称已解决。timing-only探针和完整失败不能改成功。div/hyp/mem_ctrl仍未解决累计节点导入。

## 2026-10-19：输入容量诊断与完整基础覆盖（新资源协议，非旧20k胜出）
原20k是手工固定的累计e-node准入参数，不是EPFL电路不可映射证明。三个拒绝设计双库6次扩大输入相关容量后均完成导入，但后续均触发原native60秒保护；未获有效优化网表，不把导入成功写成延迟收益。baseline原算法/模型/参数和旧20k矩阵未改。
新source-first原型先由原AIG构造AND/NAND/共享INV完整物理覆盖，再允许可选e-graph搜索；全部40case的constructor-only试验已经结束：38独立原始named-port CEC成功，hyp双库2次公共验证60秒保护失败。div/mem_ctrl双库基础覆盖CEC通过，不称e-graph优化成功，不将这种原始结构覆盖与旧full配置拼成赢家。40次搜索均关闭，此轮没有延迟改善结论。
容量公式v1误漏输入负相位，四PI负输出回归实际至少9节点、v1给6。v2已改为2*(PI+AND)+2再加固定20k搜索额度（div101802/hyp443174/mem_ctrl113638）；两项回归通过。旧v1冻结试验保留，不能称它是一般保守上界证明；v2尚未执行全套优化。mapping300秒与baseline算法不变，但这是明确的新节点容量协议，不是旧20k同预算对照。完整来源/命令/失败/审计见results_v2/input_capacity_20261019。原发布PDF与旧主表尚未替换。所有有限mapping任务已经结束。

## 2026-10-19：按输入容量比例分配额外搜索额度（试验入口）

按用户新设定实现统一比例资源协议：B_input=2*(PI+AND)+2，B_search=ceil(rho*B_input)，累计上限G=B_input+B_search；精确有理数计算向上取整，不保留固定20k总额下限。初始rho=1，仅作为验证起点，未按设计调参或称最优。rho=1/2、1、2已生成全20预声明容量表，但尚未完成这些设置的全套映射。额外容量不是独立SAT阶段计数器：压缩输入未用完的预留仍在共同账本，所有helper/RHS/失败重试实际准入都计入G，重建/退役不返还。两库相同AIG使用相同容量公式。

新nested input_capacity_proportional.py、bench_proportional_capacity.py、egraphmap_proportional.py及finite runner/独立checker已接通。native main binary/原baseline算法与models/参数未改；mapping300、native共享60秒、recovery2秒、可选bank100k不变。完整source incumbent加可选bank有单独显式retained上限，不能说与旧total100k/旧20k相同资源协议。当前first=min(20k,G)旧路径优先，只有被证实的import拒绝才用G减已准入量重试，两次共享60秒，不重置额度。

五比例策略回归与五source-cover回归通过。ctrl双库各299累计准入/464上限，2独立CEC，均选择e-graph cover；div双库新上限163604，初次20k拒绝消耗不返还，第二次allowance143604/import完成，累计138301<=G，但后续仍native60秒保护失败。div两次最终基础覆盖都通过独立原始named-port CEC；不把这2次source成功写成e-graph优化或delay提升。四case共4CEC、2优化cover、2基础cover，四个失败native attempts完整保留。证据results_v2/proportional_capacity_20261019，主版本与论文主表仍为旧明确配置，未以这四个pilot替代全20。所有finite mapping已结束。


## 2026-10-20：ρ 全 EPFL 对照与重复验证完成，hyp 搜索仍未解决

证据：[本轮报告](../results_v2/proportional_sweep_20261019/report.txt)、[完整资格核验](../results_v2/proportional_sweep_20261019/qualification/qualification.json)、[阶段诊断](../results_v2/proportional_sweep_20261019/full_stages.json)、[源代码](../results_v2/proportional_sweep_20261019/source_v3_index.json)、[永久归档](../results_v2/proportional_sweep_20261019/archives.json)。

ρ=0.5/1/2 × 两库 × EPFL20，共120/120完整尝试并核验。每个比例40/40有效覆盖，其中38选中e-graph优化覆盖/2为hyp基础覆盖；114次优化输出fresh original named-port CEC与6次同raw/input/lib/evaluator精确SHA的source证明复用，不能把后者称freshCEC或优化成功。三个比例40个逐项D及实际累计准入完全相同、没有用尽G，按预先声明覆盖→共同范围几何D→A→较小ρ规则统一选0.5；不按设计/库挑赢家。几何A为310.585161/310.614683/310.889530，小差别不证明ρ的稳定优越性。

统一B=2*(PI+AND)+2，G=B+ceil(ρB)，无固定20k下限。实际累计准入、重建/退役/失败重试不返还；压缩输入未用完的预留仍在同一账本，而非另开SAT额度。保留mapping300s/native60s/recovery2s（含post-cap可用性重建、重映射、final validation）、cuts16/SAT16/cut32/4×2000local；return reserve0.25s、phase fraction0.25在原2秒内。native retained100k；完整source incumbent另有全局B+100k保守retained界，不能声称与旧total100k/20k同协议或把暂存生成内存等同retained。所有方法共用一次resyn2，baseline算法/模型/native参数未变。

v3隔离版统一exact XOR/TT3压缩（2048mandatory helper、fill0）、按依赖脏cut bank传播、按输出root完整依赖闭包保留和流式物理去重。256随机bank、24逐物理记录差分等回归包括206PASS/5ignored/3明确排除既有synth项。v2新增post-cap availability曾在2秒计时外，撤回其strict2s资格；v3真实RED→GREEN并移入计时窗口，旧源码/bin/探针/失败日志不删。主Rust/default及旧矩阵未改，不将此实验版改名为默认已胜出。

div实际34167/G122703、mem_ctrl34903/G140457（ρ0.5、两库同累计），已选中优化覆盖。hyp实际179014/G634761，容量已足够但native60s仍失败：K4 primary停cut round8，K3已到cone pruning，完成区间含SAT25.7s/generate15.4s/alias9.7s；不把未完成prune记作完成时间，不把input准入成功称优化。ρ0.5主点mem K3 A3448.17/D1228.25、K4 A3802.98/D1087.09，D低于emap但面积明显更高；div K3 A4016.31/D43594.08、K4 A5057.55/D43093.65，仍有面积代价，K4还略慢于emap。

新增54个sqrt/sin/bar三比例两库三轮目标尝试全部freshCEC/优化覆盖；另18个div/hyp/mem三轮统一ρ重复，12fresh优化CEC/6hyp基础source证明复用。总192正式尝试=180fresh优化CEC+12合法source证明复用，不用最优重复替换主点。div/mem两库各3/3重复选中优化；hyp6次重复全60s失败。sqrt K3主点1586.26/60061.59，相对emap A-3.56%/D-0.043%；K4 1915.41/51470.40，A+22.59%/D+9.401%，明显丢失旧D收益。K4bar主点295.98/171.90也不替换成历史195.34/159.37。所有回退、A波动、失败保留。两库DA/AD-posthoc/Pareto六gate全部false，MtM未接入。

只做8个既有hyp baseline物理网表的独立验证补充，7成功、K3emap-area600秒验证仍超时；不重跑mapping/改native算法，不把验证算mapping或静默删除旧fail。完整baseline520槽现在379CEC/117reportedfail/24unmeasured，旧240组及原报告保留；新scope clone记录每个verification-only overlay和旧失败。仅外部CEC/metric给600s，300秒映射与原native guard不变。

论文主表统一ρ0.5配置及水平A/D/T，s明确基础覆盖，旧20k组件消融单独标历史而非新资源配置消融。已发布output/pdf/DAC-eGraphMap-draft.pdf，6正文+1参考，10pt正文/原边距，七页visualQA、0overfull/undefined。数据CSV已改为真实240个EPFL主表记录，完整520槽JSON另留，不再引用旧ISCAS CSV。下一步见next_experiment.json：cut/index增量维护、物理alias生成合并、在已有B+100k总额度内研究共享source/optional bank、保留原AIG有用边界；均仅计划，不能称hyp/面积回退已解决。
