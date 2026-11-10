#!/usr/bin/env python3
"""Audit complete/partial buffer campaigns without altering observations."""
import argparse,csv,json,statistics,sys,re
from pathlib import Path
from run_three_parts import DESIGNS
from run_buffer_three_parts import ARMS
from run_epmapping_supplement import sha,save,audit_abc
from audit_mapped_ppa import audit
from run_buffer_core_validation import buffer_area
from exact_core_graph import exact_core_fingerprint as core_fingerprint

MODES=['cold','warm']

def analyze(root,partial=False):
    manifest=json.loads((root/'manifest.json').read_text())
    for name,h in manifest['input_sha256'].items():assert sha(root/'inputs'/name)==h,('input mutated',name)
    expected={(s,'ours',d,m) for s in ['mfs3','main','main_r1','strong_common'] for d in DESIGNS for m in MODES}
    expected|={('ablation',arm,d,m) for arm in ARMS for d in DESIGNS for m in MODES}
    expected|={(s,'abc',d,m) for s in ['mfs3','strong_common'] for d in DESIGNS for m in MODES}
    records=[];index={}
    for stage in ['mfs3','main','main_r1','strong_common','ablation']:
        for path in sorted((root/stage).glob('*/*/result.json')):
            r=json.loads(path.read_text());key=tuple(r[k] for k in ['stage','arm','design','mode']);assert key in expected and key not in index,key
            assert r['status'] in ['complete','timeout'] and r['cec']==1 and r['optimizer_seconds']<=1000.1,path
            command=json.loads((path.parent/'command.json').read_text())
            if r['arm']!='abc':
                flags=command['env'];cap=1 if stage in ['mfs3','main_r1'] else 6
                assert r['round_cap']==cap and flags['EPMAPPING_ROUNDS']==str(cap)
                assert flags['EPMAPPING_POST_TIMEOUT']=='1000' and flags['EPMAPPING_WINDOW_SYN']=='0'
                assert flags['EPMAPPING_AREA_UNDER_DELAY']=='1' and flags['MAP_EGG_PROFILE_TT_CACHE']=='0'
                assert flags['EPMAPPING_BASELINE']==('direct' if stage=='mfs3' else 'high_effort')
                assert command['engine_sha256']==manifest['engine_sha256']
                assert flags['EPMAPPING_BUFFER_CLEANUP']==('0' if r['arm']=='noBUFFER' else '1')
            else:assert command['timeout']==1000
            selected=root/r['selected_netlist'];assert sha(selected)==r['selected_sha256'],path
            lib=root/'inputs'/('asap7_6in.genlib' if stage in ['mfs3','strong_common'] else 'asap7.genlib')
            measured=audit_abc(selected,lib) if r['arm']=='abc' and r['guard_selected']=='optimized' else audit(selected,lib)
            assert abs(measured['area']-r['final']['area'])<.0001 and abs(measured['delay']-r['final']['delay'])<.001,path
            assert measured['area']<=r['initial']['area']+.0001 and measured['delay']<=r['initial']['delay']+.001,path
            start=root/stage/('ours' if r['arm']=='abc' else r['arm'])/f"{r['mode']}.{r['design']}"/'epmapping_baseline.v'
            assert sha(start)==r['initial_sha256'],path
            if r['arm']!='abc':
                log=(path.parent/'log').read_text()
                first=re.search(r'\[i\] emap area=([0-9.eE+-]+) delay=([0-9.eE+-]+)',log)
                assert first,('missing native initial cost',path)
                # Tolerance here matches printed values only; feasibility above
                # retains the shared .0001 area / .001 delay limits.
                assert abs(float(first[1])-r['initial']['area'])<=.005001 and abs(float(first[2])-r['initial']['delay'])<=.005001,('native/independent initial mismatch',path)
                if r['status']=='complete':
                    end=re.search(r'\[i\] epmapping area ([0-9.eE+-]+) -> ([0-9.eE+-]+) delay ([0-9.eE+-]+) -> ([0-9.eE+-]+)',log)
                    assert end and abs(float(end[2])-measured['area'])<=.005001 and abs(float(end[4])-measured['delay'])<=.005001,('native/independent selected mismatch',path)
                    r['native_reported_selected']=dict(area=float(end[2]),delay=float(end[4]))
            b0,_=buffer_area(start,lib);bf,_=buffer_area(selected,lib)
            assert abs(r['buffer_gain']-(b0-bf))<.0001 and abs(r['core_gain']-((r['initial']['area']-b0)-(measured['area']-bf)))<.0001,path
            assert abs(r['buffer_gain']+r['core_gain']-(r['initial']['area']-measured['area']))<.0001,path
            if r['arm']=='abc' and r['status']=='complete':
                raw_netlist=path.parent/'mapped_native_aliases.v'
                raw_buffer,_=buffer_area(raw_netlist,lib)
                r.update(raw_buffer_gain=b0-raw_buffer,raw_core_gain=(r['initial']['area']-b0)-(r['raw']['area']-raw_buffer))
            index[key]=r;records.append(r)
    missing=sorted(expected-index.keys())
    if not partial:
        assert not missing,missing[:5]
        status=json.loads((root/'execution_status.json').read_text());assert status['complete'] and not status['errors']
    stats={};mean=statistics.mean
    for stage in ['mfs3','main','main_r1','strong_common']:
        for arm in (['ours','abc'] if stage in ['mfs3','strong_common'] else ['ours']):
            for mode in MODES:
                group=[r for r in records if (r['stage'],r['arm'],r['mode'])==(stage,arm,mode)]
                if len(group)!=16:continue
                v=dict(completed=sum(r['status']=='complete' for r in group),timeouts=sum(r['status']=='timeout' for r in group),mean_seconds=mean(r['optimizer_seconds'] for r in group),total_seconds=sum(r['optimizer_seconds'] for r in group),buffer_gain_sum=sum(r['buffer_gain'] for r in group),core_gain_sum=sum(r['core_gain'] for r in group),buffer_percent_mean=mean(100*r['buffer_gain']/r['initial']['area'] for r in group),core_percent_mean=mean(100*r['core_gain']/r['initial']['area'] for r in group))
                for q in ['area','delay']:v[q+'_ratio']=mean(r['final'][q]/r['initial'][q] for r in group)
                if arm=='ours' and stage in ['mfs3','strong_common'] and all((stage,'abc',r['design'],mode) in index for r in group):
                    other=lambda r:index[stage,'abc',r['design'],mode]
                    v.update(wins=sum(r['final']['area']<other(r)['final']['area']-.0001 for r in group),losses=[r['design'] for r in group if r['final']['area']>other(r)['final']['area']+.0001],paired_area_percent=mean(100*(r['final']['area']/other(r)['final']['area']-1) for r in group),paired_core_area_percent=mean(100*((r['final']['area']-r['final_buffer_area'])/(other(r)['final']['area']-other(r)['final_buffer_area'])-1) for r in group),paired_without_arbiter_percent=mean(100*(r['final']['area']/other(r)['final']['area']-1) for r in group if r['design']!='arbiter'))
                if arm=='abc':
                    completed=[r for r in group if r['status']=='complete']
                    v.update(delay_violations=sum(r.get('delay_violation',False) for r in group),raw_completed_designs=[r['design'] for r in completed])
                    for q in ['area','delay']:v['raw_'+q+'_ratio']=mean(r['raw'][q]/r['initial'][q] for r in completed)
                    for q in ['buffer','core']:v['raw_'+q+'_gain_sum']=sum(r['raw_'+q+'_gain'] for r in completed)
                stats[stage+'/'+arm+'/'+mode]=v
    ablations={}
    for arm in ARMS:
        for mode in MODES:
            if not all(('ablation',arm,d,mode) in index and ('ablation','full',d,mode) in index for d in DESIGNS):continue
            full=[index['ablation','full',d,mode] for d in DESIGNS];rows=[index['ablation',arm,d,mode] for d in DESIGNS]
            delta=[r['final']['area']-f['final']['area'] for r,f in zip(rows,full)]
            delta_core=sum((r['final']['area']-r['final_buffer_area'])-(f['final']['area']-f['final_buffer_area']) for r,f in zip(rows,full))
            ablations[arm+'/'+mode]=dict(delta_area=sum(delta),delta_core=delta_core,delta_buffer=sum(delta)-delta_core,worse=sum(x>.0001 for x in delta),better=sum(x<-.0001 for x in delta),time_ratio=sum(r['optimizer_seconds'] for r in rows)/sum(r['optimizer_seconds'] for r in full),timeouts=sum(r['status']=='timeout' for r in rows))
    buffer_proofs=[]
    for design in DESIGNS:
        for mode in MODES:
            f=index.get(('ablation','full',design,mode));v=index.get(('ablation','noBUFFER',design,mode))
            if not f or not v:continue
            lib=root/'inputs/asap7.genlib';a=core_fingerprint(root/f['selected_netlist'],lib);b=core_fingerprint(root/v['selected_netlist'],lib)
            same=a==b
            buffer_proofs.append(dict(design=design,mode=mode,complete_pair=f['status']==v['status']=='complete',same_core=same,full_core=a,variant_core=b,delta_area=v['final']['area']-f['final']['area'],full_buffer=f['final_buffer_area'],variant_buffer=v['final_buffer_area']))
    matches=[]
    for stage,arm,design,mode in sorted(index):
        if mode!='cold':continue
        c=index[stage,arm,design,'cold'];w=index.get((stage,arm,design,'warm'))
        if w is None:continue
        assert c['initial_sha256']==w['initial_sha256']
        matches.append(dict(stage=stage,arm=arm,design=design,ppa_equal=all(abs(c['final'][q]-w['final'][q])<(.0001 if q=='area' else .001) for q in ['area','delay']),cold_status=c['status'],warm_status=w['status']))
    out=dict(schema=1,auditor_dependency_sha256={name:sha(Path(__file__).parent/name) for name in ['exact_core_graph.py','run_buffer_core_validation.py','audit_mapped_ppa.py','run_buffer_three_parts.py','run_three_parts.py','run_epmapping_supplement.py']},effective_round_caps=dict(mfs3=1,main_r1=1,main=6,strong_common=6,ablation=6),audit_script_sha256=sha(Path(__file__)),complete=not missing,expected=len(expected),observed=len(records),missing=missing,manifest_sha256=sha(root/'manifest.json'),aggregates=stats,ablations=ablations,buffer_core_proofs=buffer_proofs,cache_mode_matches=matches,records=records)
    save(root/('analysis.partial.json' if missing else 'analysis.json'),out)
    print(json.dumps(dict(complete=out['complete'],observed=len(records),expected=len(expected),aggregates=stats,buffer_proofs=len(buffer_proofs)),indent=2))
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);ap.add_argument('--partial',action='store_true');a=ap.parse_args();analyze(a.root.resolve(),a.partial)
