"""Analyze the admitted paired campaign without treating deadlines as completed work."""
import json,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[2]/'results_v28_three_parts_query_budget'
v=json.loads((root/'verified_profiles.json').read_text());runs=v['cache_runs'];native=v['native_build_runs']
rows=[json.loads((root/m/'tab_mfs3/abc_results.json').read_text()) for m in ['cold','warm']];assert rows[0]==rows[1]
abc={r['design']:r for r in rows[0] if r['rc']==0};out={'mfs3_completed':len(abc),'mfs3_timeout':len(rows[0])-len(abc),'controls':{},'primary_completion':{},'paired':{}}
for mode in ['cold','warm']:
 primary=[r for r in runs if r['mode']==mode and (r['stage']!='ablation' or r['arm']=='full')]
 out['primary_completion'][mode]={'runs':len(primary),'completed':sum(r['status']=='complete' for r in primary),'timeouts':sum(r['status']=='timeout' for r in primary),'max_seconds':max(r['optimizer_seconds'] for r in primary)}
 for arm in ['noQUERY','noTRUTH','allPIVOTS']:
  group=[r for r in runs if r['mode']==mode and r['stage']=='ablation' and r['arm']==arm]
  metric=v['cohort_metrics'][mode]['ablation'][arm]
  full={r['design']:r for r in runs if r['mode']==mode and r['stage']=='ablation' and r['arm']=='full'}
  paired=[r for r in group if r['status']=='complete' and full[r['design']]['status']=='complete']
  pairratio=statistics.mean(r['optimizer_seconds']/full[r['design']]['optimizer_seconds'] for r in paired) if paired else None
  out['controls'].setdefault(arm,{})[mode]=dict(metric,timeouts=sum(r['status']=='timeout' for r in group),completed=sum(r['status']=='complete' for r in group),completed_pairs=len(paired),mean_paired_time_ratio=pairratio)
 # Native query statistics exist only at normal completion; never manufacture zeros for TO.
 group=[r for r in native if r['mode']==mode and r['stage']=='ablation' and r['arm']=='full' and r['status']=='complete']
 out.setdefault('full_query_counters',{})[mode]={key:sum(r['live_queries'][key] for r in group) for key in ['inverse','hits','level_skips','walks','tt_hits','local_tt']}
 # Raw ABC ratios and guarded ABC results are kept separate.
 ours={}
 for d in abc:
  f=root/mode/'tab_mfs3'/f'ours.{d}'/'row.tsv';r=f.read_text().split();ours[d]=dict(area=float(r[6]),delay=float(r[7]))
 ratios={d:ours[d]['area']/abc[d]['area'] for d in abc}
 out['paired'][mode]=dict(mean_area_ratio_to_raw_mfs3=statistics.mean(ratios.values()),area_wins=sum(r<1-1e-5 for r in ratios.values()),area_losses=sum(r>1+1e-5 for r in ratios.values()),mean_area_ratio_without_arbiter=statistics.mean(r for d,r in ratios.items() if d!='arbiter'),mfs3_mean_seconds=statistics.mean(r['secs'] for r in abc.values()),ours_mean_seconds=statistics.mean(r['optimizer_seconds'] for r in runs if r['mode']==mode and r['stage']=='tab_mfs3' and r['design'] in abc))
(root/'query_budget_analysis.json').write_text(json.dumps(out,indent=2)+'\n')
lines=['查询、真值表和 pivot 预算消融复核','', '复用消融保留候选变换家族；时间受限 care 计算和贪心轨迹仍可能改变 PPA。allPIVOTS 恢复所有 pivot，扩大候选覆盖。', '时间比的总量口径包含每个 TO 的实际 1000 秒预算；仅完整配对的平均时间比另列，不能把相同预算解释为相同完成工作量。','']
for mode in ['cold','warm']:
 p=out['primary_completion'][mode];lines.append(f"{mode}: 完整方法 {p['completed']}/{p['runs']} 正常完成，{p['timeouts']} TO；最大 {p['max_seconds']:.3f} 秒。")
 for arm in out['controls']:
  r=out['controls'][arm][mode];lines.append(f"  {arm}: 完成 {r['completed']}/16，TO {r['timeouts']}，总时间/full={r['time_ratio']:.4f}，ΣΔA={r['area']:+.2f}；完整配对 {r['completed_pairs']}，平均时间比={r['mean_paired_time_ratio']}。")
 p=out['paired'][mode];lines.append(f"  对原始 mfs3 的平均面积比={p['mean_area_ratio_to_raw_mfs3']:.6f}；面积 {p['area_wins']} 胜/{p['area_losses']} 负；排除 arbiter 后平均比={p['mean_area_ratio_without_arbiter']:.6f}。")
 lines.append(f"  {out['mfs3_completed']}个正常 baseline 配对的平均时间：我们 {p['ours_mean_seconds']:.3f} 秒，ABC {p['mfs3_mean_seconds']:.3f} 秒。")
lines+=['','noQUERY 关闭 INV 查询 memo 与层级快速否定，保留物理相位复用。noTRUTH 关闭预解析仿真和动态表复用，保留共同的标量 cut memo。allPIVOTS 仅将每族 pivot 上限由2恢复为全部。','TO 和外部中断分开：中断尝试单独归档，重跑恢复规定的缓存起点；原生截止仍标记为 TO。']
(root/'QUERY_BUDGET_ANALYSIS.txt').write_text('\n'.join(lines)+'\n')
print(json.dumps(out['primary_completion']))
