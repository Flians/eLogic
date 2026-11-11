#!/usr/bin/env python3
"""Re-audit selected probe outputs and compare same-input serial measurements."""
import argparse,hashlib,json,re
from pathlib import Path
from audit_mapped_ppa import audit
from run_buffer_core_validation import buffer_area
from report import read_counters

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(root,design,mode):
    stage='mfs3'if design=='ethernet'else'main';wd=root/stage/'ours'/f'{mode}.{design}';r=json.loads((wd/'result.json').read_text());m=json.loads((root/'manifest.json').read_text())
    assert sha(root/'inputs/engine')==m['engine_sha256']
    selected=root/r['selected_netlist'];assert sha(selected)==r['selected_sha256']and r['cec']==1
    lib=root/'inputs'/('asap7_6in.genlib'if stage=='mfs3'else'asap7.genlib');cost=audit(selected,lib)
    assert abs(cost['area']-r['final']['area'])<1e-4 and abs(cost['delay']-r['final']['delay'])<1e-3
    assert cost['area']<=r['initial']['area']+1e-4 and cost['delay']<=r['initial']['delay']+1e-3
    assert buffer_area(selected,lib)[0]==0 and r['initial_buffer_area']==0
    command=json.loads((wd/'command.json').read_text());env=command['env'];assert env['EPMAPPING_WINDOW_SYN']=='0'and env['EPMAPPING_PRE_BUFFER_CLEANUP']=='1'and env['EPMAPPING_POST_TIMEOUT']=='1000'
    assert r['optimizer_seconds']<=1000.1
    text=(wd/'log').read_text();profile=re.findall(r'egraph_profile (.*)',text)
    counters=read_counters(str(wd/'log'))
    return dict(design=design,mode=mode,status=r['status'],area=cost['area'],delay=cost['delay'],optimizer_seconds=r['optimizer_seconds'],initial_sha256=r['initial_sha256'],selected_sha256=r['selected_sha256'],CEC=1,buffer_area=0,profile={k:float(v)if any(c in v for c in '.eE')else int(v)for k,v in re.findall(r'(\w+)=([0-9.eE+-]+)',profile[-1])}if profile else {},phase_wall={k:v for k,v in counters.items()if k.startswith('secs_')},resource=json.loads((wd/'log.resources.json').read_text()),shared_sum=[line for line in text.splitlines()if line.startswith('[i] shared_sum')])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    bm=json.loads((a.baseline/'manifest.json').read_text());cm=json.loads((a.candidate/'manifest.json').read_text());assert bm['source_inputs_sha256']==cm['source_inputs_sha256']and bm['cpu']==cm['cpu']and bm['jobs']==cm['jobs']==1
    rows=[]
    for design in ['ethernet','mem_ctrl']:
        for mode in ['cold','warm']:
            b=read(a.baseline,design,mode);c=read(a.candidate,design,mode);assert b['initial_sha256']==c['initial_sha256']
            target=(3673.56,546.58)if design=='ethernet'else(1801.85,1461.58)
            c['quality_pass']=c['area']<=target[0]+1e-4 and c['delay']<=target[1]+1e-3
            c['completion_pass']=c['status']=='complete';rows.append(dict(design=design,mode=mode,baseline=b,candidate=c,area_delta=c['area']-b['area'],delay_delta=c['delay']-b['delay'],optimizer_seconds_delta=c['optimizer_seconds']-b['optimizer_seconds']))
    result=dict(complete=True,quality_pass=all(r['candidate']['quality_pass']for r in rows),completion_pass=all(r['candidate']['completion_pass']for r in rows),baseline=str(a.baseline),candidate=str(a.candidate),original_inputs_identical=True,preprocessing_unchanged=True,whole_window_synthesis=False,post_budget_seconds=1000,host_exclusive=False,timing_interpretation='serial CPU-pinned observations, not a statistical speedup estimate; whole-child CPU includes setup/CEC',rows=rows)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k]for k in ['complete','quality_pass','completion_pass']},indent=2))
    for r in rows:print(r['design'],r['mode'],'A/D',r['candidate']['area'],r['candidate']['delay'],'seconds',r['baseline']['optimizer_seconds'],'->',r['candidate']['optimizer_seconds'])
if __name__=='__main__':main()
