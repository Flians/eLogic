#!/usr/bin/env python3
"""Publish full-scope component statistics; do not select per-design winners."""
import argparse,collections,copy,csv,hashlib,json,math,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def counts(pairs,metric):return dict(collections.Counter(p[metric+'_outcome']for p in pairs if p['valid']))
def ranges(rows):
 out=[]
 for k,bench,profile in sorted({(r['k'],r['bench'],r['profile'])for r in rows}):
  group=[r for r in rows if (r['k'],r['bench'],r['profile'])==(k,bench,profile)];ok=[r for r in group if r['valid']];z=dict(k=k,bench=bench,profile=profile,attempts=len(group),CEC=len(ok),failures=[r for r in group if not r['valid']])
  for metric in ['area','delay','mapping_wall_upper_secs']:
   values=[r[metric]for r in ok];z[metric]=dict(min=min(values),max=max(values),values=values)if values else None
  out.append(z)
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--full',type=Path,required=True);p.add_argument('--repeats',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False);campaigns=[load(p/'campaign.json')for p in [a.full,*a.repeats]];vs=[load(p/'validation.json')for p in [a.full,*a.repeats]];assert all(v['complete'] and v['source_budget_and_single_factor_audit_pass']for v in vs)
 f=campaigns[0]['fingerprint'];assert all(c['fingerprint']==f for c in campaigns);assert all(sha(p)==h for p,h in f.items());full=vs[0];repeatrows=[dict(r,repetition=i)for i,v in enumerate(vs[1:])for r in v['rows']];repeatpairs=[dict(r,repetition=i)for i,v in enumerate(vs[1:])for r in v['pairs']];profiles=campaigns[0]['profiles'];summary={}
 for profile in profiles:
  fp=[p for p in full['pairs']if p['profile']==profile];rp=[p for p in repeatpairs if p['profile']==profile];rr=[r for r in repeatrows if r['profile']==profile];fr=[r for r in full['rows']if r['profile']==profile];summary[profile]=dict(full_attempts=len(fr),full_CEC=sum(r['valid']for r in fr),full_pair_coverage=len(fp),full_valid_pairs=sum(p['valid']for p in fp),full_area=counts(fp,'area'),full_delay=counts(fp,'delay'),repeat_attempts=len(rr),repeat_CEC=sum(r['valid']for r in rr),repeat_valid_pairs=sum(p['valid']for p in rp),repeat_area=counts(rp,'area'),repeat_delay=counts(rp,'delay'),all_failures=[r for r in [*fr,*rr]if not r['valid']])
 integrity=load(ROOT/'results_v2/single_output_components_evidence_20261017/integrity.json');assert all(sha(p)==h for p,h in integrity['baseline_small_binary_check'].items());G=out/'baseline_scope';G.mkdir();gates=[];required=['abc_map_area_resyn2','abc_map_delay_resyn2','mockturtle_emap_area_resyn2','mockturtle_emap_delay_resyn2','eqmap_lut_amap','esyn_frozen_model_amap_bounded_dag','emorphic_final_sa']
 for k in [3,4]:
  base=ROOT/f'results_v2/guarded_joint_evidence_20261016/baseline_scope/baselines_{k}in.json';b=load(base);d=copy.deepcopy(b);d['results']=[]
  for x in full['rows']:
   if x['k']!=k or x['profile']!='full':continue
   if x['report']:
    report=load(x['report']);r=copy.deepcopy(report['results'][0]);assert report['library_sha256']==b['library_sha256'] and report['metric_model']==b['metric_model'];assert all(r['input_sha256']==z['input_sha256']for z in b['results']if z['bench']==r['bench']);r['method']='eGraphMap';d['results'].append(r)
   else:d['results'].append(dict(bench=x['bench'],method='eGraphMap',status='failed',error=x['error'],input_sha256=next(r['input_sha256']for r in b['results']if r['bench']==x['bench'])))
  assert len(d['results'])==20;candidate=G/f'egraphmap_{k}in.json';candidate.write_text(json.dumps(d,indent=2))
  for goal in ['delay-area','area-delay','pareto']:
   dest=G/f'{k}in_{goal}.json';argv=[sys.executable,str(ROOT/'scripts/experiments/check_mapping_dominance.py'),'--candidate',str(candidate),'--baselines',str(base),'--required-baselines',*required,'--objective',goal,'--output',str(dest)];p=subprocess.run(argv,capture_output=True,text=True);dest.with_suffix('.log').write_text(p.stdout+p.stderr);assert dest.exists();gates.append(dict(k=k,objective=goal,passed=load(dest)['all_passed'],argv=argv,baseline_sha256=sha(base),candidate_sha256=sha(candidate)))
 for fname,rows in [('full.csv',full['rows']),('repeats.csv',repeatrows)]:
  keys=['k','bench','profile','repetition','status','valid','area','delay','mapping_wall_upper_secs','report','report_sha256','audit']
  with (out/fname).open('w')as csv_stream:
   w=csv.DictWriter(csv_stream,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
 (out/'full_rows.json').write_text(json.dumps(full['rows'],indent=2));(out/'repetition_ranges.json').write_text(json.dumps(ranges(repeatrows),indent=2));(out/'full_pairs.json').write_text(json.dumps(full['pairs'],indent=2));(out/'repeat_pairs.json').write_text(json.dumps(repeatpairs,indent=2))
 q=dict(complete=True,frozen_mapper_sha256=f[str(ROOT/'target/release/bench_rs_egraph_map')],full_attempts=full['attempts'],full_CEC=full['cec'],repetition_attempts=len(repeatrows),repetition_CEC=sum(r['valid']for r in repeatrows),summary=summary,gates=gates,uniform_final_configuration='full; V46 algorithm plus default-off same-binary ablation instrumentation, chosen before experiments',source_hashes=f,baseline_small_binary_check=integrity['baseline_small_binary_check'],baseline_algorithms_changed=False,unconnected=['MtM'],AD_checks_are_posthoc=True,paired_sign='ablated minus full; lower favours ablation; higher favours full. Counts restricted to jointly validated pairs, coverage/failures explicit.',metric_guard_scope='seed-referenced metric protection (envelope, seed-area ceiling and bound-feasible DA cache) removed as one module; ordinary working/fast point selection and all cycle/source/binding legality remain. Not a cap-only ablation.',historical_full_budget_incumbent_guaranteed=False)
 q['selected_mechanism_totals']={p:dict(sum((collections.Counter(r.get('selected_mechanisms',{}))for r in full['rows']if r['profile']==p and r['valid']),collections.Counter()))for p in profiles}
 q['final_seed_envelope_violations']={p:sum(bool(r.get('outside_seed_envelope'))for r in full['rows']if r['profile']==p and r['valid'])for p in profiles}
 (out/'qualification.json').write_text(json.dumps(q,indent=2));(out/'report.txt').write_text('\n'.join([f'Full {q["full_attempts"]} attempts / {q["full_CEC"]} CEC; repetitions {q["repetition_attempts"]} attempts / {q["repetition_CEC"]} CEC',*[p+': '+str({k:v for k,v in x.items()if k!='all_failures'})for p,x in summary.items()], 'All failures/regressions retained; no per-design choice; gates '+str([g['passed']for g in gates])])+'\n');print((out/'report.txt').read_text())
if __name__=='__main__':main()
