#!/usr/bin/env python3
"""Same-input three-design experiment with independent buffer/core accounting."""
import argparse, concurrent.futures, json, re, traceback, shutil
from pathlib import Path
import run_epmapping_supplement as run
from analyze_epmapping_supplement import verify_abc_native_outputs

DESIGNS=['ethernet','sasc','usb_phy']
REFERENCE=run.PROJECT/'results_epmapping_supplement_v1_20261110'

def buffer_area(path,library):
    names={}
    for block in re.split(r'(?m)^GATE\s+',library.read_text())[1:]:
        first=block.splitlines()[0].split()
        if first[2]=='Y=A;':names[first[0]]=float(first[1])
    text=path.read_text()
    po={n.strip() for group in re.findall(r'\boutput\s+(.*?);',text,re.S) for n in group.split(',')}
    counts={n:0 for n in names}
    for name,instance,ports in re.findall(r'^\s*(\w+)\s+(\w+)\s*\((.*?)\)\s*;',text,re.M|re.S):
        if name not in names:continue
        pins=dict((p,v.strip()) for p,v in re.findall(r'\.(\w+)\s*\(\s*([^)]*?)\s*\)',ports))
        source=pins['A'];output=pins['Y']
        if source in po and output in po:continue # writer-added PO alias, never a native cell
        counts[name]+=1
    return sum(counts[n]*names[n] for n in names),counts

def analyze(root):
    verify_abc_native_outputs(root)
    manifest=json.loads((root/'manifest.json').read_text())
    for name,h in manifest['input_sha256'].items():assert run.sha(root/'inputs'/name)==h,name
    historical=json.loads((REFERENCE/'analysis.json').read_text())
    old={(r['design'],r['mode'],r['arm']):r for r in historical['records'] if r['part']=='common'}
    records=[];lib=root/'inputs/asap7_6in.genlib'
    for design in DESIGNS:
        for mode in ['cold','warm']:
            ours=json.loads((root/f'common/ours/{mode}.{design}/result.json').read_text())
            abc=json.loads((root/f'common/abc/{mode}.{design}/result.json').read_text())
            assert ours['status']=='complete' and abc['status']=='complete',(design,mode)
            assert ours['initial_sha256']==abc['initial_sha256']==old[design,mode,'ours6']['initial_sha256']
            assert ours['optimizer_seconds']<=1000 and abc['optimizer_seconds']<=1000
            initial=ours['initial'];previous=old[design,mode,'ours6']['final']
            b0,_=buffer_area(root/f'common/ours/{mode}.{design}/epmapping_baseline.v',lib)
            bf,counts=buffer_area(root/ours['selected_netlist'],lib)
            proof=json.loads((root/f'common/abc/{mode}.{design}/native_alias_audit.json').read_text())
            candidate=proof['metrics'];assert proof['cec']==1
            legal=candidate['area']<=initial['area']+.0001 and candidate['delay']<=initial['delay']+.001
            abc_final=candidate if legal else initial
            abpath=root/f'common/abc/{mode}.{design}/mapped_native_aliases.v' if legal else root/f'common/ours/{mode}.{design}/epmapping_baseline.v'
            abuf,_=buffer_area(abpath,lib)
            final=ours['final']
            assert final['area']<=previous['area']+.0001 and final['delay']<=previous['delay']+.001
            assert final['delay']<=initial['delay']+.001 and bf==0, (design,final,counts)
            prior=old[design,mode,'ours6']
            prior_path=Path(prior['source_campaign'])/prior['selected_netlist']
            prior_buffers,_=buffer_area(prior_path,lib)
            core_gain=(initial['area']-b0)-(final['area']-bf)
            assert abs((previous['area']-prior_buffers)-(final['area']-bf))<.0001,('changed core area',design)
            log=(root/f'common/ours/{mode}.{design}/log').read_text()
            cleanup_seconds=float(re.search(r'buffer_cleanup .* seconds=([0-9.e+-]+)',log).group(1))
            abc_core=(initial['area']-b0)-(abc_final['area']-abuf)
            records.append(dict(design=design,mode=mode,initial=initial,previous=previous,ours=final,abc=abc_final,
                ours_seconds=ours['optimizer_seconds'],abc_seconds=abc['optimizer_seconds'],cleanup_seconds=cleanup_seconds,
                ours_buffer_gain=b0-bf,ours_core_gain=core_gain,abc_buffer_gain=b0-abuf,abc_core_gain=abc_core,
                area_vs_abc_percent=100*(final['area']/abc_final['area']-1),cec=1,
                same_initial_sha256=ours['initial_sha256'],selected_sha256=ours['selected_sha256'],
                abc_normalized_sha256=proof['normalized_sha256']))
    for design in DESIGNS:
        c,w=[r for r in records if r['design']==design]
        assert abs(c['ours']['area']-w['ours']['area'])<.0001 and abs(c['ours']['delay']-w['ours']['delay'])<.001
    data=dict(complete=True,records=records,post_timeout=1000,whole_window_synthesis=False,
        accounting='initial buffers minus final buffers; core gain excludes both; output wire aliases have no cells',
        timing='native optimizer including cleanup, checkpoint and final export; preprocessing, mapping and independent CEC excluded',
        cold='empty private persistent pool',warm='paired persistent pool, new process; OS caches not cleared',
        abc_modes='independently timed processes, no e-graph cache',reference_analysis_sha256=run.sha(REFERENCE/'analysis.json'))
    run.save(root/'analysis.json',data)
    print(json.dumps(records,indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);ap.add_argument('--analyze-only',action='store_true');args=ap.parse_args()
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    if args.analyze_only:analyze(root);return
    m=run.initialize(root,run.PROJECT/'build/epmapping/epmapping_check')
    m.update(designs=DESIGNS,common_rounds=6,task='native identity cleanup with buffer/core accounting')
    m['goal']['EPMAPPING_BUFFER_CLEANUP']='1'
    for name in ['run_buffer_core_validation.py','analyze_epmapping_supplement.py']:
        source=run.PROJECT/'scripts/experiments'/name
        frozen=root/'inputs/sources'/name
        if frozen.exists():assert run.sha(source)==run.sha(frozen),('changed harness on resume',name)
        else:shutil.copy2(source,frozen)
    m['input_sha256']={str(p.relative_to(root/'inputs')):run.sha(p) for p in (root/'inputs').rglob('*') if p.is_file()}
    run.save(root/'manifest.json',m)
    def pair(design):
        cache=root/'cache'/design
        for mode in ['cold','warm']:
            out=run.engine(root,m,'common','ours',design,mode,cache,rounds=6)
            old=REFERENCE/f'common/ours/{mode}.{design}/epmapping_baseline.v'
            assert run.sha(old)==out['initial_sha256'],('changed initial',design,mode)
            run.abc(root,m,design,mode,root/f'common/ours/{mode}.{design}/epmapping_baseline.v')
    errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(pair,d):d for d in DESIGNS}
        for future in concurrent.futures.as_completed(futures):
            try:future.result()
            except Exception as e:errors.append((futures[future],repr(e)));traceback.print_exc()
    run.save(root/'execution_status.json',dict(complete=not errors,errors=errors,jobs=3))
    assert not errors,errors
    analyze(root)
if __name__=='__main__':main()
