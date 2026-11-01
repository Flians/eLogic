#!/usr/bin/env python3
"""Read-only comparison of a verified supplied-seed cohort; never runs mapping."""
import argparse,collections,csv,hashlib,json,math,statistics
from pathlib import Path
R=Path(__file__).resolve().parents[2]
LABELS={'abc_map':'ABC map','mockturtle_emap_delay':'emap-delay（seed 提供者）','mockturtle_emap_area':'emap-area（目标敏感性）','eqmap_asic_depth':'EqMap-ASIC min-depth（原40项）','eqmap_asic_area':'EqMap-ASIC area（原40项）','esyn_author':'E-Syn 完整作者流程','emorphic_author':'E-morphic 完整作者流程'}
ORDER=list(LABELS)
TABLE_METHODS={'abc_map':'ABC','mockturtle_emap_delay':'emap-delay','mockturtle_emap_area':'emap-area','eqmap_asic_depth':'EqMap-author-min-depth','eqmap_asic_area':'EqMap-author-area','esyn_author':'E-Syn','emorphic_author':'E-morphic'}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def csvwrite(p,rows):
 with Path(p).open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def relation(x,y):return 'win'if x<y else'loss'if x>y else'tie'
def percent(x,y):return 100*(1-x/y)if y else None
def valid(r):
 m=r.get('measured',{});return r.get('status')=='ok'and m.get('cec')==1 and all(isinstance(m.get(k),(int,float))and math.isfinite(m[k])and m[k]>=0 for k in ['area','delay'])
def median(xs):return statistics.median(x for x in xs if x is not None)
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--study',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();P=a.study.resolve();F=P/'final';O=a.output.resolve();assert not O.exists()
 receipt=read(P/'reporting-evidence-manifest.json');inputs=[F/n for n in ['candidate.json','baselines-normalized.json','statistics.json','seed_search_attribution.csv','baseline_snapshot.csv','verification.json','qa.json','completion.json']]
 for p in inputs:assert sha(p)==receipt[str(p.relative_to(P))],p
 protocol=read(P/'protocol.json');candidate=read(F/'candidate.json');baselines=read(F/'baselines-normalized.json');stats=read(F/'statistics.json');assert candidate['library_sha256']==baselines['library_sha256']and candidate['metric_model']==baselines['metric_model'];assert len(candidate['results'])==20
 table={(r['bench'],r['method']):r for r in csv.DictReader((F/'baseline_snapshot.csv').open())};candidate_times={n:float(table[n,'eGraphMap-seeded-refinement']['T'])for n in protocol['order']}
 candidates={r['bench']:r for r in candidate['results']};assert len(candidates)==20 and all(valid(r)for r in candidates.values());rows=[];aggregate=[]
 for method in ORDER:
  original=[b for b in baselines['results']if b['method']==method];assert len(original)==20
  current=[]
  for b in original:
   c=candidates[b['bench']];assert b['input_sha256']==c['input_sha256']==protocol['inputs'][b['bench']]['golden_sha256']
   x=c['measured'];y=b.get('measured',{});ok=valid(b);tr=table[b['bench'],TABLE_METHODS[method]]
   if ok:assert float(tr['A'])==y['area']and float(tr['D'])==y['delay']
   r=dict(design=b['bench'],baseline=method,baseline_status=b['status'],baseline_table_status=tr['status'],baseline_T=float(tr['T'])if tr['T']else None,candidate_refinement_T=candidate_times[b['bench']],T_comparison_scope='candidate refinement excludes historical seed generation; baseline original method mapping time; no end-to-end speedup claimed',comparable=ok,candidate_CEC=x['cec'],baseline_CEC=y.get('cec'),candidate_A=x['area'],candidate_D=x['delay'],baseline_A=y.get('area'),baseline_D=y.get('delay'),DA=relation((x['delay'],x['area']),(y['delay'],y['area']))if ok else'excluded',A=relation(x['area'],y['area'])if ok else'excluded',D=relation(x['delay'],y['delay'])if ok else'excluded',strict_Pareto=ok and x['area']<=y['area']and x['delay']<=y['delay']and(x['area']<y['area']or x['delay']<y['delay']),area_reduction_percent=percent(x['area'],y['area'])if ok else None,delay_reduction_percent=percent(x['delay'],y['delay'])if ok else None,baseline_error=b.get('error'),baseline_report=b.get('source_report'),candidate_report=c['source_report'])
   rows.append(r);current.append(r)
  good=[r for r in current if r['comparable']];da=collections.Counter(r['DA']for r in good);dc=collections.Counter(r['D']for r in good)
  v=dict(baseline=method,label=LABELS[method],valid=len(good),excluded=20-len(good),DA_win=da['win'],DA_tie=da['tie'],DA_loss=da['loss'],D_win=dc['win'],D_tie=dc['tie'],D_loss=dc['loss'],strict_Pareto_wins=sum(r['strict_Pareto']for r in good),median_baseline_T_seconds=median(r['baseline_T']for r in good),median_candidate_refinement_T_seconds=median(r['candidate_refinement_T']for r in good),median_area_reduction_percent=median(r['area_reduction_percent']for r in good),median_delay_reduction_percent=median(r['delay_reduction_percent']for r in good),DA_losses=[r['design']for r in good if r['DA']=='loss'],DA_wins_with_area_cost=[r['design']for r in good if r['DA']=='win'and r['A']=='loss'])
  expected=stats['baseline_comparison'][method];assert(v['DA_win'],v['DA_tie'],v['DA_loss'])==(expected.get('win',0),expected.get('equal',0),expected.get('loss',0));aggregate.append(v)
 # The14 time-only observations are a distinct targeted cohort, never substituted.
 supplemental=F/'paper/eqmap_timeonly_supplement.csv';supp=[]
 for b in csv.DictReader(supplemental.open()):
  assert sha(b['report'])==b['report_sha256'];ok=b['CEC']=='1'and bool(b['area'])and bool(b['delay']);c=candidates[b['bench']]['measured'];v=dict(design=b['bench'],profile=b['profile'],status=b['status'],CEC=b['CEC'],comparable=ok,area=b['area'],delay=b['delay'],seconds=b['mapping_seconds'],candidate_A=c['area'],candidate_D=c['delay'],candidate_refinement_T=candidate_times[b['bench']],DA=relation((c['delay'],c['area']),(float(b['delay']),float(b['area'])))if ok else'excluded',report=b['report'],report_sha256=b['report_sha256']);supp.append(v)
 assert len(supp)==14 and sum(r['comparable']for r in supp)==6
 O.mkdir(parents=True);csvwrite(O/'all_baseline_pairs.csv',rows);csvwrite(O/'baseline_summary.csv',aggregate);csvwrite(O/'eqmap_supplement_pairs.csv',supp)
 dump(O/'report.json',dict(source_commit=stats['source_commit'],scope=stats['scope'],metric_model=candidate['metric_model'],objective='exact(D,A)',comparison='positive A/D reduction means this candidate is smaller/faster; median of per-design relative differences, not geometric ratio',seed_role=stats['seed_role'],comparison_rows=len(rows),aggregate=aggregate,supplement=supp,baseline_runs=0,GitHub_push=False,source_sha256=sha(__file__),inputs={str(p):sha(p)for p in inputs+[supplemental]}))
 lines=['# 当前 eGraphMap 与 baseline：完整对比','', '> 主结果为“统一 emap-delay seed + eGraphMap 精化”，不是独立从 AIG 开始的映射。seed 未作为复杂门覆盖导入原生恢复器。下列质量比较不包含新测 seed 生成成本，不能据此宣称端到端速度优势。','', '## 1. 固定数据口径','', f"- 源码 `{stats['source_commit']}`，同二进制/配置、一次 resyn2 输入、ASAP7 45个物理单输出门+2常量。每设计600秒精化搜索，转换/公共指标/CEC不计时。",'- 全20例构造覆盖、seed、最终输出各20/20 CEC；92,193条案例产物哈希核验。正式 baseline 132个本地配置：101成功、31失败；另8条 E-Syn 文献记录无本地A/D/T或CEC。','- 严格比较原始有限 `(D,A)`，不使用表格四舍五入后的数值决定平局。字典序优胜与双指标 Pareto 优胜分开。','', '## 2. 相对 baseline 的质量','', '|Baseline|共同有效/20|D,A 胜/平/负|仅D 胜/平/负|严格Pareto胜|A中位降幅|D中位降幅|','|---|---:|---|---|---:|---:|---:|']
 for x in aggregate:lines.append(f"|{x['label']}|{x['valid']}|{x['DA_win']}/{x['DA_tie']}/{x['DA_loss']}|{x['D_win']}/{x['D_tie']}/{x['D_loss']}|{x['strict_Pareto_wins']}|{x['median_area_reduction_percent']:.2f}%|{x['median_delay_reduction_percent']:.2f}%|")
 lines+=['','中位降幅 = 各共同有效设计的 `100*(baseline−ours)/baseline` 的中位数；负值表示本方法更大/更慢。各列中位数不一定来自同一设计。失败、缺失和文献项不计胜。','', '- 相对 emap-delay 的9胜11平是对seed的条件精化：6例真正降D，3例同D降A；不是20例独立重新发现了seed质量。','- 相对ABC质量存在局部优势，但仍有4例DA落后；相对完整E-morphic流程仅bar胜，17负，差距主要在延迟。两者保留作者内部结构优化，不能把全部差异归因于e-graph本身。','- emap-area只作目标敏感性比较：20例D更优，但19例为此支付更大面积；这不能说明本方法面积更优。','- 原EqMap两目标均只有8个共同合法设计，不能外推为全20优于EqMap。所有full-scope DA/AD/Pareto gate仍false。','', '### 原版 baseline 的固定协议','', '|方法|流程与配置|成功/失败或文献|','|---|---|---|','|ABC|共享输入上 `strash; map`；不使用map-a|20成功|','|emap|原mockturtle stock emap，库/目标参数按已存命令；delay、area两模式|40成功|','|EqMap-ASIC 原40|两原生目标；内部500秒、48000节点、32轮；已批准最小总搜索截止检查补丁；原greedy/ArcSet提取|16成功、16原生失败、8超时|','|E-Syn|作者默认300秒/25M节点/10M轮、rank0、原map/尺寸后端；转换3600秒/8GiB（旧adder/bar不追溯套用）|12本地：7成功5失败；另8文献|','|E-morphic|README原run_sa.sh完整流程；5次重写、10 samples、0.8；保留代码6轮SA和作者内部预处理|18成功；hyp超时、voter作者CEC失败|','', '所有正式 baseline 每配置一次、同库、外层600秒映射相关搜索、最终CEC不计时；未重新运行或修改。该口径不是同CPU工作量或相同内部搜索算法。','', '### EqMap去节点/轮数限制的补充组','', '仅原先节点拒绝的7设计×2目标=14次，内部仅`-t500`，外层600；6成功、8失败/超时。对6个共同合法输出，本方法均DA优胜。该组与原40项分开，不能逐设计选择更好者拼接成完整新版EqMap。详见 eqmap_supplement_pairs.csv。','', '运行效率尚无优势：18例用满600秒精化预算，而直接ABC/emap通常亚秒至秒级；再考虑未新测计入的seed生成成本，更不能将质量胜出称作加速。all_baseline_pairs.csv同时保留双方T与计时范围，失败运行时间也不删除。','', '## 3. 全20设计的 seed 与搜索贡献','', '|设计|seed A / D|最终 A / D|搜索结果|精化T(s)|','|---|---|---|---|---:|']
 for row in csv.DictReader((F/'seed_search_attribution.csv').open()):
  result='改善'if row['search_outcome']=='win'else'保留seed';result+='；截止'if row['deadline']=='True'else'；完成'
  lines.append(f"|{row['design']}|{float(row['seed_A']):.2f} / {float(row['seed_D']):.2f}|{float(row['output_A']):.2f} / {float(row['output_D']):.2f}|{result}|{float(row['mapping_seconds']):.2f}|")
 lines+=['', '- seed相对原生AND/NAND/INV构造覆盖：20/20同时降A/D，中位33.50%/55.51%；它是seed提供者的贡献，不能归功于本次搜索。','- 搜索相对seed：9改善、11持平、0DA回退；18触及截止。max/cavlc/dec/i2c降D但增A，全部保留。','- 无净收益的11例均截止；记录的原生覆盖没有击败seed，不等于不存在更优解。seed仅在外层保护交付，原生恢复从另一份更弱构造覆盖开始。','', '## 4. 耗时与证据限制','', '18个截止案例AIG饱和约0.000656–3.344360秒，不是全部600秒耗在饱和。div/hyp/mem_ctrl/log2/multiplier/square/voter已闭合面积恢复区间约占在线时段92.5%–97.8%；priority约1.9%。这些区间包括嵌套证明、准备和发布，不能相加或当作独占成本。','', 'bar的末尾2084快照补验为184.49/156.60、CEC通过，正式交付2083为184.51/156.60。没有截止前外层接收收据，不替换正式结果。此处值得分析发布边界，但不能宣称已证明丢失合法截止前赢家。','', '## 5. 结论与后续顺序','', '当前可确认：交付等价与seed保护可靠，9例有净收益，部分对直接mapper的质量有优势。尚不能确认：独立映射全套优势、全局最优、18例搜索完成或新组件的独立因果收益。','', '下一轮优先把经验证的复杂门seed变成可编辑原生覆盖，再分离证明/准备/发布的独占成本；之后验证减少重复恢复和发布是否增加有效搜索。不能仅增加匹配范围或配对数，就期待11例自动获得净收益。','', '算法详见 ALGORITHM.md；正式论文、历史消融、完整收据的位置见 README.md。生成本报告只读取既有数据，不执行mapper/baseline，不改正式数值。']
 (O/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'comparison_rows':len(rows),'methods':len(aggregate),'output':str(O)},ensure_ascii=False))
if __name__=='__main__':main()
