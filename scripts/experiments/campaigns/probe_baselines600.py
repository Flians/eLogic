"""Real EqMap and final-SA timing/CEC probes, separate from primary observations."""
import argparse,fcntl,json,os,subprocess,sys
from pathlib import Path
from run_baselines600 import ROOT,HERE,CATALOGUE,command_for,sha
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    locks=[]
    for path in [ROOT/'results_v2/independent_components_20261019/global.lock',ROOT/'results_v2/epfl_original_baselines_resumed_checkpoint_20261013/runner.lock']:
        f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    prior=json.loads(CATALOGUE.read_text());records=[]
    for name in ['r0_6in_eqmap_delay-area_adder','r0_6in_emorphic_adder']:
        old=next(j for j in prior['jobs']if j['name']==name);folder=out/name;audit=out/(name+'.audit.json');log=out/(name+'.log')
        argv=command_for(old,folder,audit)
        env={k:v for k,v in os.environ.items()if not k.startswith(('EGM_','ESYN_','MAPPING_CLOCK_','METER_'))}
        with log.open('w')as f:result=subprocess.run(argv,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        record=dict(name=name,argv=argv,exit_code=result.returncode,primary_data_replacement=False)
        if (folder/'report.json').exists()and audit.exists():
            report=json.loads((folder/'report.json').read_text());clock=json.loads(audit.read_text())
            record['all_method_cec_passed']=all(r['status']=='ok'and r.get('measured',{}).get('cec')==1 for r in report['results'])
            record['clock_keys']=list(clock['clock'])
            if '_eqmap_'in name:
                prep=clock['clock']['adder/__lut_prep__']['used'];assert prep>0
                for method in ['lut_stage_amap','eqmap_lut_amap']:
                    assert clock['clock']['adder/'+method]['lut_prefix_mapping_secs']==prep
                assert 'lut_prefix_mapping_secs'not in clock['clock']['adder/abc_amap']
                record['nested_LUT_mapping_secs']=prep
            else:
                state=clock['clock']['adder/emorphic_final_sa']
                assert len([e for e in state['events']if e['event']=='begin'])>0
                record['feedback_mapping_calls']=sum(e['event']=='begin'for e in state['events'])
                record['feedback_mapping_secs']=state['used']
                record['candidate_cec_passed']=report['results'][0].get('valid_candidates')
                record['candidate_cec_failed']=report['results'][0].get('failed_candidates')
            record['all_mapping_intervals_closed']=all(not s['active']and not s['incomplete']and not s['exhausted']for s in clock['clock'].values())
        else:record['all_method_cec_passed']=False
        records.append(record)
        artifacts=[q for q in out.rglob('*')if q.is_file()and q.name!='validation.json']
        (out/'validation.json').write_text(json.dumps(dict(probes=records,source_sha256={str(q):sha(q)for q in [Path(__file__),HERE/'run_baseline600_budget.py',HERE/'meter_yosys_abc600.py',HERE/'nested_mapping_clock.py']},artifacts={str(q):sha(q)for q in artifacts},primary_data_replacement=False),indent=2)+'\n')
        print(name,result.returncode,record['all_method_cec_passed'],flush=True)
        assert result.returncode==0 and record['all_method_cec_passed']and record['all_mapping_intervals_closed'],'preserve failed probe; audit before changing harness'
if __name__=='__main__':main()
