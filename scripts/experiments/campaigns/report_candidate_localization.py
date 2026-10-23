#!/usr/bin/env python3
"""Audit diagnostic studies without selecting per-design profile winners."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit_campaign(p):
 c=json.loads(p.read_text());rows=[]
 for j in c['jobs']:
  for key in ['report','audit','pool','candidates']:
   if key in j:assert sha(Path(j[key]))==j[key+'_sha256']
  for r in json.loads(Path(j['report']).read_text())['results']:
   n=r.get('native',{});m=r.get('measured',{});valid=r.get('status')=='ok' and m.get('cec')==1
   a=json.loads(Path(j['audit']).read_text());clock=max((s['mapping_wall_secs'] for s in a['states'].values()),default=0)
   if valid:
    assert n['enodes']<=20000 and n['max_cuts_per_class']<=16 and n['instances']<=100000 and n['heuristic_secs']<=2 and clock<=300
   rows.append({'name':j['name'],'k':j['k'],'bench':j['bench'],'profile':j.get('profile'),'valid':valid,'measured':m,'native':{x:n.get(x) for x in ['enodes','max_cuts_per_class','instances','generated_instances','heuristic_secs','seed_delay','delay']},'mapping_wall_secs':clock,'report':j['report'],'report_sha256':j['report_sha256']})
 return c,rows
out=ROOT/'results_v2/candidate_localization_report_20261014';out.mkdir(exist_ok=True)
c,controls=audit_campaign(ROOT/'results_v2/pool_loss_cases_20261014/campaign.json');_,timed=audit_campaign(ROOT/'results_v2/pool_timing_only_cases_20261014/campaign.json');_,quota=audit_campaign(ROOT/'results_v2/small_cut_quota_cases_20261014/campaign.json');f,frows=audit_campaign(ROOT/'results_v2/candidate_factorial_checkpoint_20261014/campaign.json')
loss=[]
for j in c['jobs']:
 p=json.loads(Path(j['pool']).read_text());d=json.loads(Path(j['candidates']).read_text());crit={r['signal'] for r in d['critical_signals']};lost=[]
 for r in p['signals']:
  b=r['best_before'];a=r['best_after'];bv=b and b['area_witness_arrival_estimate'];av=a and a['area_witness_arrival_estimate']
  if bv is not None and av is not None and av>bv+1e-6:lost.append({'signal':r['signal'],'estimate_delta':av-bv,'on_selected_critical_cone':r['signal'] in crit})
 loss.append({'name':j['name'],'instances_before':p['instances_before'],'instances_after':p['instances_after'],'witness_reserved_instances':p['witness_reserved_instances'],'local_estimate_losses':len(lost),'on_selected_critical_cone':sum(x['on_selected_critical_cone'] for x in lost),'losses':lost})
pairs=[]
for r in timed:
 b=next(x for x in controls if x['k']==r['k'] and x['bench']==r['bench']);pairs.append({'factor':'TIMING_CAP only','bench':r['bench'],'k':r['k'],'control':b['measured'],'trial':r['measured'],'delay_percent':100*(r['measured']['delay']/b['measured']['delay']-1),'area_percent':100*(r['measured']['area']/b['measured']['area']-1)})
b,r=quota;pairs.append({'factor':'SMALL_CUT_QUOTA 0 -> 8 only','bench':'adder','k':4,'control':b['measured'],'trial':r['measured'],'delay_percent':100*(r['measured']['delay']/b['measured']['delay']-1),'area_percent':100*(r['measured']['area']/b['measured']['area']-1)})
old=ROOT/'results_v2/comprehensive_reanalysis_20261013/analysis.json';hashes=json.loads(old.read_text())['sources_sha256'];unchanged={p:sha(ROOT/p)==h for p,h in hashes.items() if p.startswith('emapper/src/')};assert all(unchanged.values())
report={'diagnostic_scope':'4in adder/max/sin +3in adder control; single-factor probes and first fixed factorial checkpoint. No canonical table substitution.', 'main_source_unchanged':unchanged,'pool_loss':loss,'single_factor_pairs':pairs,'factorial_completed':len(frows),'factorial_expected':160,'factorial_profiles':f['fixed_profiles'],'factorial_rows':frows,'notes':['Pool local scores use one common area witness, not certified alternative-cover timing.','Cut correspondence uses deterministic library-independent import/saturation and matching counts/free/roots; no serialized graph identity certificate.','Quota improves adder through different retained cuts; it does not restore the 126 missing reference boundaries.','All instrumented runtime observations are diagnostic; do not claim runtime speedup.','Targeted gains remain slower than ABC delay map; native/scoring models and internal budgets differ.'],'source_campaign_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'results_v2/pool_loss_cases_20261014/campaign.json',ROOT/'results_v2/pool_timing_only_cases_20261014/campaign.json',ROOT/'results_v2/small_cut_quota_cases_20261014/campaign.json',ROOT/'results_v2/candidate_factorial_checkpoint_20261014/campaign.json']}}
(out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
(out/'report.txt').write_text('候选损失定位与固定因素实验（诊断数据，不替换主表）\n'+ '\n'.join(f"{x['name']}: pool {x['instances_before']} -> {x['instances_after']}; 局部估计变差信号 {x['local_estimate_losses']}，其中最终关键锥 {x['on_selected_critical_cone']}" for x in loss)+'\n\n'+'\n'.join(f"{x['k']}in {x['bench']} {x['factor']}: A/D {x['control']['area']:.2f}/{x['control']['delay']:.2f} -> {x['trial']['area']:.2f}/{x['trial']['delay']:.2f}; delay {x['delay_percent']:.2f}%, area {x['area_percent']:.2f}%" for x in pairs)+f"\n\n固定四配置全EPFL矩阵：{len(frows)}/160 已记录。未按设计选赢家；主版本和原版 baseline 算法保持不变。所有成功均独立 CEC 与实际节点/cut/pool/恢复/映射时限核验。\n")
print(out);print([(r['name'],r['measured'].get('area'),r['measured'].get('delay')) for r in frows])
