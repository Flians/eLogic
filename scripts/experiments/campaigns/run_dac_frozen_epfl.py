"""Select ONE preregistered policy, then finish full20 using the frozen mapper.

This adds no mapper branch. Representative r0 cases are explicit same-version
references; repeats are never picked as main-table winners. Baselines are read-only.
"""
import argparse,copy,fcntl,json,pathlib,shutil,statistics
import run_dac_convergence as core
R=core.R;read=core.read;save=core.save;sha=core.sha
def comparable(cfg):
 cfg=copy.deepcopy(cfg)
 for k in ['trace_write','evidence_arrays']:cfg.pop(k)
 for k in ['EGM_TRACE_LINE_WRITE','EGM_EVIDENCE_ARRAY_MOVE']:cfg['environment'].pop(k)
 return cfg
def selection(c):
 expected=[j for p in c['plans'].values()for j in p];assert len(expected)==16 and all(j['key']in c['jobs']for j in expected),'complete representatives/repeats first'
 pairs=[];valid=True
 for n,reps in [('dec',[0,1,2]),('sin',[0,1,2]),('sqrt',[0]),('div',[0])]:
  for rep in reps:
   a,b=[c['jobs'][f'{n}_r{rep}_{v}']for v in ['off','on']]
   ok=all(v['status']=='ok'and v.get('CEC')==1 and v.get('first_independent_CEC')and v.get('qualified_clock_chain')for v in [a,b]);valid &= ok
   p=dict(design=n,rep=rep,valid=bool(ok))
   if ok:
    assert comparable(a['configuration'])==comparable(b['configuration']),'unplanned algorithm/configuration differences'
    p.update(off_T=a['T'],on_T=b['T'],reduction=(a['T']-b['T'])/a['T'],DA_no_worse=(b['D'],b['A'])<=(a['D'],a['A']),both_search_finished=a['search_finished']and b['search_finished'])
   pairs.append(p)
 dec=[p for p in pairs if p['design']=='dec'and p['valid']]
 gain=bool(len(dec)==3 and all(p['both_search_finished']for p in dec)and statistics.median(p['reduction']for p in dec)>=.05 and sum(p['reduction']>0 for p in dec)>=2 and sum(p['on_T']for p in dec)<sum(p['off_T']for p in dec))
 quality=valid and all(p.get('DA_no_worse',False)for p in pairs)
 return dict(variant='on'if gain and quality else'off',gate_passed=bool(gain and quality),runtime_gate=bool(gain),quality_and_evidence_gate=bool(quality),pairs=pairs,selection_policy=c['selection_policy'],per_design_selection=False,baseline_runs=0,default_promoted=False,GitHub_push=False)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--parent',type=pathlib.Path,default=core.DEFAULT);ap.add_argument('--max-jobs',type=int,default=20);a=ap.parse_args();assert a.max_jobs>0
 parent=a.parent.resolve();o=parent.parent/'full-epfl';o.mkdir(parents=True,exist_ok=True);locks=[]
 for p in [R/'results_v2/independent_components_20261019/global.lock',parent/'runner.lock',o/'runner.lock']:
  f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 c=read(parent/'campaign.json');assert c['frozen']==core.fingerprint();selected=selection(c)
 for r in c['jobs'].values():
  assert all(sha(pathlib.Path(r['case'])/n)==h for n,h in r['artifacts'].items())and sha(pathlib.Path(r['sample_file']))==r['sample_sha256']
 for item in c.get('interruptions',[]):
  path=pathlib.Path(item['audit']);assert sha(path)==item['sha256'];audit=read(path);assert not audit['eligible_for_PPA']
  assert all(sha(pathlib.Path(audit['original_case'])/n)==h for n,h in audit['artifacts'].items())
 if(o/'campaign.json').exists():f=read(o/'campaign.json');assert f['selection']==selected and f['wrapper_sha256']==sha(pathlib.Path(__file__))and f['parent_campaign_sha256']==sha(parent/'campaign.json')
 else:
  f=dict(frozen=c['frozen'],selection=selected,wrapper_sha256=sha(pathlib.Path(__file__)),parent_campaign_sha256=sha(parent/'campaign.json'),jobs={},references={},expected=20,baseline_runs=0,GitHub_push=False,default_promoted=False)
  for n in c['subjects']:
   key=f"{n}_r0_{selected['variant']}"
   if key in c['jobs']:
    f['jobs'][n]=copy.deepcopy(c['jobs'][key]);f['references'][n]=dict(parent_key=key,reason='same source, binary, full CLI configuration, input/library and 600s policy; representative r0, not selected best repeat')
  save(o/'campaign.json',f);save(o/'selection.json',selected);shutil.copy2(parent/'campaign.json',o/'parent-campaign-snapshot.json');shutil.copy2(__file__,o/'executed-wrapper.py')
 for row in f['jobs'].values():assert all(sha(pathlib.Path(row['case'])/n)==h for n,h in row['artifacts'].items())
 dispatched=0
 for n in c['subjects']:
  if n in f['jobs']:continue
  if(o/'STOP_AFTER_CURRENT').exists():break
  if dispatched>=a.max_jobs:break
  assert c['frozen']==core.fingerprint()and sha(pathlib.Path(__file__))==f['wrapper_sha256']
  j=dict(key=f"{n}_r0_{selected['variant']}",design=n,rep=0,variant=selected['variant'],phase='full-epfl')
  save(o/'state.json',dict(active=j['key'],completed=len(f['jobs']),expected=20,variant=selected['variant'],baseline_runs=0));print('start',j['key'],flush=True)
  row=core.execute(c,j,o,locks);receipt=o/(j['key']+'-completed.json');assert not receipt.exists();save(receipt,row);assert c['frozen']==core.fingerprint();f['jobs'][n]=row;save(o/'campaign.json',f);dispatched+=1;print('end',j['key'],row['status'],row.get('T'),row.get('A'),row.get('D'),flush=True)
 save(o/'state.json',dict(active=None,completed=len(f['jobs']),expected=20,variant=selected['variant'],baseline_runs=0,default_promoted=False,manually_stopped=(o/'STOP_AFTER_CURRENT').exists(),remaining='independent module ablations and final paper/evidence audit'))
if __name__=='__main__':main()
