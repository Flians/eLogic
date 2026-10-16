#!/usr/bin/env python3
"""Validate raw logs, then generate the paper tables/macros from a single verified run root.

python3 scripts/experiments/export_paper.py --root results_v3_audit --out epmapping_paper/DATE27-ePMapping-sub/doc/generated_results.tex
No failed, non-CEC or incorrectly configured noCARE row may enter the paper.
"""
import argparse, csv, hashlib, json, math, re, statistics, subprocess
from pathlib import Path
import report

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def rows(root,stage):
    data=report.read_summary(str(root/stage/'summary.tsv'))
    seen=set()
    for r in data:
        key=(r['stage'],r['design'])
        if key in seen: raise ValueError(f'duplicate {stage}/{key}')
        seen.add(key)
        if stage=='tab_mfs3' and r['stage']=='base': continue
        if not report.verified(r): raise ValueError(f'unverified {stage}/{key}: rc={r.get("rc")} cec={r.get("cec")}')
        log=root/stage/f'{r["stage"]}.{r["design"]}'/'log'
        txt=log.read_text()
        if not re.search(r'window_synth windows=0 applied=0 gain=0 actual=0 gates=0 abandoned=0 delay_rejected=0 secs=0(?:\s|$)',txt):
            raise ValueError(f'whole-window synthesis enabled or not verifiable in {log}')
        env_file=log.parent/'experiment_env.txt'
        if env_file.exists():
            flags=[line for line in env_file.read_text().splitlines() if line.startswith('EPMAPPING_WINDOW_SYN=')]
            if not flags or any(flag!='EPMAPPING_WINDOW_SYN=0' for flag in flags):
                raise ValueError(f'whole-window synthesis configuration is not off in {env_file}')
        if not re.search(r'^\[i\] cec=1$',txt,re.M): raise ValueError(f'no raw CEC in {log}')
        m=re.findall(r'epmapping area ([0-9.eE+-]+) -> ([0-9.eE+-]+) delay ([0-9.eE+-]+) -> ([0-9.eE+-]+)',txt)
        if not m: raise ValueError(f'no final metrics in {log}')
        for k,v in zip(('base_a','out_a','base_d','out_d'),m[-1]):
            if float(r[k])!=float(v): raise ValueError(f'summary/log mismatch {log}: {k}')
        if float(r['out_d'])>float(r['base_d'])+max(.001,abs(float(r['base_d']))*1e-5):
            raise ValueError(f'global delay regression in {log}')
        if float(r['out_a'])>float(r['base_a'])+.001:
            raise ValueError(f'global area regression in area-first experiment: {log}')
        if stage=='ablation' and r['stage']=='noSYN':
            if not re.search(r'epmapping synth roots=0 covers=0 wins=0 secs=0(?:\s|$)',txt):
                raise ValueError(f'noSYN still runs structural synthesis in {log}')
        if stage=='ablation' and r['stage']=='noEGG':
            if ', egg=0,' not in txt:
                raise ValueError(f'noEGG still enables the e-graph in {log}')
        if stage=='ablation' and r['stage']=='noDB':
            if '[i] super library:' in txt:
                raise ValueError(f'noDB still loads the database in {log}')
            if env_file.exists():
                db_flags=[line for line in env_file.read_text().splitlines() if line.startswith('EPMAPPING_SUPER=')]
                if not db_flags or db_flags[-1]!='EPMAPPING_SUPER=':
                    raise ValueError(f'noDB does not override the database in {env_file}')
        if stage=='ablation' and r['stage']=='noCHAIN':
            if ', chain_factor=0,' not in txt or report.read_counters(str(log))['chain_wins']:
                raise ValueError(f'noCHAIN still enables chain factoring in {log}')
        if stage=='ablation' and r['stage']=='noSHARE':
            if ', chain_share=0,' not in txt or ', chain_factor=1,' not in txt:
                raise ValueError(f'noSHARE does not isolate chain roots in {log}')
        if stage=='ablation' and r['stage']=='noCARE':
            if ', dc=0,' not in txt or not re.search(r'\bdc_cuts=0\b',txt):
                raise ValueError(f'noCARE still computes DC in {log}')
    return data

def latex_table(caption,label,headers,body,align):
    env='table*' if label in ('tab:main','tab:mfs3') else 'table'
    return '\n'.join([r'\begin{'+env+'}[t]',r'\centering\footnotesize',r'\setlength{\tabcolsep}{3pt}',
      r'\caption{'+caption+'}'+r'\label{'+label+'}',
      r'\begin{tabular}{'+align+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule']+
      [' & '.join(map(str,r))+r' \\' for r in body]+[r'\bottomrule',r'\end{tabular}',r'\end{'+env+'}'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    root=a.root.resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('whole_window_synthesis') is not False:
        raise ValueError('paper requires a manifest declaring whole-window synthesis disabled')
    engine=root/manifest['archived_engine'] if manifest.get('archived_engine') else Path('build/epmapping/epmapping_check')
    if digest(engine)!=manifest['engine_sha256']:
        raise ValueError('engine changed since the recorded experiment build')
    mainrows=rows(root,'tab_main');abl=rows(root,'ablation');mfs=rows(root,'tab_mfs3');funnel=rows(root,'funnel');lever=rows(root,'lever3')
    index={(r['stage'],r['design']):r for r in mainrows}
    required=[index[('ae',d)] for d in report.PAPER16]
    p=report.read_paper_table('scripts/compare_with_aspdac25.sh')
    ref={d:(p[d][0]*(1+p[d][2]/100),p[d][3]*(1+p[d][5]/100)) for d in report.PAPER16}
    pct=lambda out,base:100*(float(out)/float(base)-1)
    av=lambda xs:statistics.mean(xs)
    all_da=[pct(r['out_a'],ref[r['design']][0]) for r in required];all_dd=[pct(r['out_d'],ref[r['design']][1]) for r in required]
    non=[r for r in required if r['design']!='arbiter']
    metrics=dict(smaller=sum(float(r['out_a'])<ref[r['design']][0] for r in required),
      area_excl=av(pct(r['out_a'],ref[r['design']][0]) for r in non),delay_excl=av(pct(r['out_d'],ref[r['design']][1]) for r in non),
      area_all=av(all_da),delay_all=av(all_dd),own_area=av(pct(r['out_a'],r['base_a']) for r in required),own_delay=av(pct(r['out_d'],r['base_d']) for r in required),total_wall=sum(float(r['secs']) for r in required))
    macros={};fmt=lambda v:f'{v:.2f}'
    def macro(name,value):macros[name]=str(value)
    macro('MainWins',metrics['smaller']);macro('MainAreaExcl',fmt(-metrics['area_excl']));macro('MainDelayExcl',fmt(-metrics['delay_excl']))
    for name,key in [('MainAreaAll','area_all'),('MainDelayAll','delay_all'),('OwnArea','own_area'),('OwnDelay','own_delay')]:macro(name,f'{metrics[key]:+.2f}')
    macro('MainWall',int(metrics['total_wall']))
    body=[];timing=[]
    for r in required:
        d=r['design'];c=report.read_counters(str(root/'tab_main'/f'ae.{d}'/'log'))
        for k in ('preprocess','library','mapping','optimizer','validation','total'):
            if not math.isfinite(c.get('time_'+k,float('nan'))):raise ValueError(f'missing phase timer {d}/{k}')
        timing.append(c)
        body.append([report.esc(d),fmt(ref[d][0]),fmt(ref[d][1]),fmt(float(r['base_a'])),fmt(float(r['base_d'])),fmt(float(r['out_a'])),fmt(float(r['out_d'])),fmt(c['time_optimizer'])])
    macro('MainOptTotal',fmt(sum(c['time_optimizer'] for c in timing)));macro('MainLibraryTotal',fmt(sum(c['time_library'] for c in timing)))
    main_table=latex_table('End-to-end results versus reported ASPDAC\'25 outputs. $A_e,D_e$ are our own initial area and delay; starts differ. $t_{opt}$ times only ePMapping, including preparation and final rebuild. Areas are $\\mu m^2$, delays ps.','tab:main',['Circuit','ASPDAC area','ASPDAC delay','$A_e$','$D_e$','Ours area','Ours delay','$t_{opt}$ (s)'],body,'lrrrrrrr')
    bs={r['design']:r for r in mfs if r['stage']=='base'};ours={r['design']:r for r in mfs if r['stage']=='ours'}
    abc=report.read_summary(str(root/'tab_mfs3'/'abc.tsv'));abc_by={r['design']:r for r in abc}
    if set(abc_by)!=set(report.PAPER16) or len(abc)!=16:raise ValueError('ABC comparison incomplete or duplicated')
    archived=json.loads((root/'abc_inputs.json').read_text()) if (root/'abc_inputs.json').exists() else {}
    native=report.read_summary(str(root/'tab_mfs3/abc_baseline_metrics.tsv'))
    native_by={r['design']:r for r in native}
    if set(native_by)!=set(report.PAPER16) or len(native)!=16:raise ValueError('native ABC baseline measurements incomplete')
    abc_da=[];abc_dd=[];our_da=[];our_dd=[];paired_da=[];paired_dd=[];body=[];abc_complete=0
    for d in report.PAPER16:
        b,o=bs[d],ours[d];ba,bd=float(b['base_a']),float(b['base_d']);r=abc_by[d]
        initial=native_by[d];aba,abd=float(initial['abc_area']),float(initial['abc_delay'])
        if float(initial['emap_area'])!=ba or float(initial['emap_delay'])!=bd:raise ValueError(f'baseline metadata mismatch: {d}')
        initial_log=(root/'tab_mfs3'/f'abc_baseline.{d}.log').read_text()
        for key,value in [('area',aba),('delay',abd)]:
            found=re.findall(r'\b'+key+r'\s*=\s*([0-9.]+)',initial_log)
            if not found or float(found[-1])!=value:raise ValueError(f'native baseline/log mismatch: {d}/{key}')
        # The independent ABC batch can use the earlier stored direct-mapped baseline only
        # after proving that it is byte-identical to the freshly regenerated baseline.
        abc_spec=next(line.split() for line in (root/'abc.specs').read_text().splitlines() if line.split()[1]==d)
        def abc_input(kind,index):
            if d not in archived:return Path(abc_spec[index])
            item=archived[d][kind];path=root/item['archived_path']
            if item['original_path']!=abc_spec[index] or digest(path)!=item['sha256']:
                raise ValueError(f'archived ABC input mismatch: {d}/{kind}')
            original=Path(abc_spec[index])
            if original.exists() and digest(original)!=item['sha256']:
                raise ValueError(f'original ABC input changed: {d}/{kind}')
            return path
        library_content=lambda path:[line.strip() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
        if library_content(abc_input('library',2))!=library_content(root/'tab_mfs3/asap7_6in.genlib'):
            raise ValueError(f'ABC and ePMapping libraries differ: {d}')
        if abc_input('baseline',3).read_bytes()!=(root/'tab_mfs3'/f'base.{d}'/'epmapping_mapped.v').read_bytes():
            raise ValueError(f'ABC and ePMapping did not receive identical mapped starts: {d}')
        od_a,od_d=pct(o['out_a'],ba),pct(o['out_d'],bd);our_da.append(od_a);our_dd.append(od_d)
        abc_cols=['--','--','TO']
        if report.verified(r):
            txt=(root/'tab_mfs3'/f'abc.{d}'/'cec.log').read_text()
            if 'Networks are equivalent' not in txt:raise ValueError(f'ABC CEC log missing {d}')
            raw=(root/'tab_mfs3'/f'abc.{d}'/'log').read_text()
            for key in ('area','delay'):
                found=re.findall(r'\b'+key+r'\s*=\s*([0-9.]+)',raw)
                if not found or float(found[-1])!=float(r[key]):raise ValueError(f'ABC summary/log mismatch: {d}/{key}')
            da,dd=pct(r['area'],aba),pct(r['delay'],abd);abc_da.append(da);abc_dd.append(dd);paired_da.append(od_a);paired_dd.append(od_d)
            abc_cols=[f'{da:+.2f}',f'{dd:+.2f}',r['secs']];abc_complete+=1
        elif r.get('rc')!='142':raise ValueError(f'ABC failed verification: {d}, {r}')
        body.append([report.esc(d),fmt(ba),fmt(bd),fmt(aba),fmt(abd)]+abc_cols+[f'{od_a:+.2f}',f'{od_d:+.2f}',fmt(report.read_counters(str(root/'tab_mfs3'/f'ours.{d}'/'log'))['time_optimizer'])])
    for name,value in [('ABCCompleted',abc_complete),('ABCArea',f'{av(abc_da):+.2f}'),('ABCDelay',f'{av(abc_dd):+.2f}'),('MfsOwnArea',f'{av(our_da):+.2f}'),('MfsOwnDelay',f'{av(our_dd):+.2f}'),('PairedOwnArea',f'{av(paired_da):+.2f}'),('PairedOwnDelay',f'{av(paired_dd):+.2f}')]:macro(name,value)
    mfs_table=latex_table('Common-input comparison on 40 ASAP7 cells. The serialized starts are identical; ABC normalizes some output buffers on read. Each percentage uses its own parsed baseline, shown explicitly. One pass per optimizer; ABC has a 1000 s cap. Completed outputs are CEC checked. Whole-window synthesis is disabled. Times exclude CEC: optimizer phase for ours, process wall for ABC. TO denotes timeout.','tab:mfs3',['Circuit','Ours $A_i$','Ours $D_i$','ABC $A_i$','ABC $D_i$','ABC $\\delta A$','ABC $\\delta D$','ABC s','Ours $\\delta A$','Ours $\\delta D$','Ours s'],body,'lrrrrrrrrrr')
    ai={(r['stage'],r['design']):r for r in abl};designs=sorted({r['design'] for r in abl});arms=['full','noSYN','noEGG','noCARE','noDB']
    for d in designs:
        for arm in arms:ai[(arm,d)]
        if len({(ai[(arm,d)]['base_a'],ai[(arm,d)]['base_d']) for arm in arms})!=1:
            raise ValueError(f'ablation starts differ: {d}')
    expected_abl=set(report.PAPER16)-{'ethernet','RISC','systemcaes','usb_funct','sqrt'}
    if set(designs)!=expected_abl or len(abl)!=55 or {r['stage'] for r in abl}!=set(arms):
        raise ValueError('expected exactly five cut-only arms on the eleven named ablation designs')
    delta={arm:sum(float(ai[(arm,d)]['out_a'])-float(ai[('full',d)]['out_a']) for d in designs) for arm in arms[1:]}
    body=[]
    for arm,label in [('noSYN','cut SOP/DSD synthesis'),('noEGG','e-graph'),('noDB','database'),('noCARE','all cut care masks')]:
        macro({'noSYN':'DeltaSyn','noEGG':'DeltaEgg','noDB':'DeltaDB','noCARE':'DeltaCare'}[arm],f'{delta[arm]:+.2f}')
        body.append([label,f'{delta[arm]:+.2f}',sum(float(ai[(arm,d)]['out_a'])>float(ai[('full',d)]['out_a'])+.005 for d in designs),sum(float(ai[(arm,d)]['out_a'])<float(ai[('full',d)]['out_a'])-.005 for d in designs)])
    ab_table=latex_table('Six-round-cap ablation over eleven designs, relative to the cut-only full configuration; whole-window synthesis is disabled in every arm. Signed area deltas are sums in $\\mu m^2$; columns count worse and better designs. These are path-dependent marginal effects, not additive contributions.','tab:ablation',['Removed component','$\\Delta A$','Worse','Better'],body,'lrrr')
    per,tot=report.aggregate_counters(str(root),'ablation','full',designs,['secs_stage1','secs_stage2','secs_synth','secs_egg','secs_care','secs_window'])
    timekeys=['secs_stage1','secs_stage2','secs_synth','secs_egg','secs_care'];den=sum(tot[k] for k in timekeys)
    body=[[label,fmt(tot[k]),fmt(100*tot[k]/den)] for label,k in zip(['L1','L2','L3','L4','Care'],timekeys)]
    times_table=latex_table('Measured cut-oracle time over the same eleven full runs. The denominator includes L1--L4 and care; cut enumeration, MFFC accounting and whole-window synthesis are excluded.','tab:time',['Stage','Seconds','Share (\\%)'],body,'lrr')
    for name,key in [('TimeSynShare','secs_synth'),('TimeEggShare','secs_egg')]:macro(name,fmt(100*tot[key]/den))
    body=[]
    for d in ['c7552','sin','div','multiplier','log2']:
        c=report.read_counters(str(root/'funnel'/f'r1.{d}'/'log'))
        if d in ('c7552','div'):
            prefix='FunnelIscas' if d=='c7552' else 'FunnelDiv'
            macro(prefix+'Admitted',int(c['egg_admitted']))
            macro(prefix+'Wins',int(c['egg_wins']))
        body.append([d,int(c['single_wins']),int(c['egg_admitted']),int(c['egg_wins'])])
    funnel_table=latex_table('One-pass layer admissions versus committed wins. Counts differ from area saved.','tab:funnel',['Design','L1 wins','L4 admitted','L4 wins'],body,'lrrr')
    # `full` is the six-round iteration control, not a bidirectional-rule-bundle arm.
    bundle=rows(root,'effort_bundle') if not any(r['stage']=='bundle' for r in lever) and (root/'effort_bundle/summary.tsv').exists() else []
    levers={d:[r for r in lever+bundle if r['design']==d and r['stage'] in ('base','iter5','iter10','node2000','bundle')] for d in ['c7552','c3540','c5315','c2670']}
    invariant=0;effort_body=[]
    for d,rr in levers.items():
        if {r['stage'] for r in rr}!={'base','iter5','iter10','node2000','bundle'}:
            raise ValueError(f'effort comparison incomplete: {d}')
        baseline=next(r for r in rr if r['stage']=='base')
        if len({(r['out_a'],r['out_d']) for r in rr})==1:invariant+=1
        effort_body.append([d,baseline['out_a'],baseline['out_d'],
          f"{max(abs(float(r['out_a'])-float(baseline['out_a'])) for r in rr):.2f}",
          f"{max(abs(float(r['out_d'])-float(baseline['out_d'])) for r in rr):.2f}"])
    macro('EffortInvariantDesigns',invariant)
    effort_table=latex_table('One-round search controls: five/ten iterations, 2000 nodes and a genuinely bidirectional rule bundle. Columns give maximum absolute deviation from the default over these four controls. The separate six-round full arm is not a rule-bundle control.','tab:effort',['Design','Base area','Base delay','Max $|\\Delta A|$','Max $|\\Delta D|$'],effort_body,'lrrrr')
    validation=rows(root,'validation')
    vr={(r['stage'],r['design']):r for r in validation}
    body=[];byte_identical=[]
    for d in ('c7552','tv80'):
        on,off=vr[('cache_on',d)],vr[('cache_off',d)]
        paths=[root/'validation'/f'{cfg}.{d}'/'epmapping_mapped.v' for cfg in ('cache_on','cache_off')]
        same_bytes=paths[0].read_bytes()==paths[1].read_bytes()
        if same_bytes:byte_identical.append(d)
        if any(float(on[k])!=float(off[k]) for k in ('base_a','base_d','out_a','out_d')):
            raise ValueError(f'cache changes measured area/delay: {d}')
        c_on,c_off=[report.read_counters(str(path.parent/'log')) for path in paths]
        body.append([d,f"{c_off['secs_synth']:.2f}",f"{c_on['secs_synth']:.2f}",
          f"{100*c_on['synth_cache_hits']/c_on['synth_cache_lookups']:.1f}",'same bytes' if same_bytes else 'CEC, same QoR'])
    cache_table=latex_table('Cache controls with whole-window synthesis disabled. Off/on have identical CEC-checked area and delay; byte equality is reported separately. Timers cover synthesis; c7552 uses one round, tv80 six.','tab:cache',['Design','Off (s)','On (s)','Hit rate (\\%)','Netlist'],body,'lrrrl')
    reference=root/'validation/reference.c7552/epmapping_mapped.v'
    if reference.exists():
        ref=vr[('reference','c7552')];current=vr[('cache_on','c7552')]
        if any(float(ref[k])!=float(current[k]) for k in ('base_a','base_d','out_a','out_d')):
            raise ValueError('database bucketing changes c7552 measured area/delay')
    metrics.update(whole_window_synthesis=False,abc_completed=abc_complete,abc_timeouts=16-abc_complete,
      abc_own_area=av(abc_da),abc_own_delay=av(abc_dd),
      common_subset_ours_area=av(paired_da),common_subset_ours_delay=av(paired_dd),
      mfs_ours_area=av(our_da),mfs_ours_delay=av(our_dd),
      ablation_cec_outputs=len(abl),ablation_deltas=delta,cut_oracle_times=tot,
      effort_invariant_designs=invariant,cache_byte_identical_designs=byte_identical,cache_equal_qor_designs=['c7552','tv80'])
    files={str(path.relative_to(root)):digest(path) for path in root.rglob('*') if path.is_file() and
      (path.name in ('log','cec.log','summary.tsv','abc.tsv','abc_baseline_metrics.tsv','mapped.v','epmapping_mapped.v','abc.specs','abc_inputs.json','manifest.json','experiment_env.txt') or path.name.startswith('abc_baseline.') or 'inputs' in path.relative_to(root).parts)}
    (root/'verified_metrics.json').write_text(json.dumps({'metrics':metrics,'raw_sha256':files},indent=2)+'\n')
    text='% GENERATED by scripts/experiments/export_paper.py; do not edit numeric values.\n'
    text += '\n'.join(f"\\newcommand{{\\{k}}}{{{v}}}" for k,v in macros.items())+'\n'
    for name,t in [('MainResultTable',main_table),('MfsResultTable',mfs_table),('AblationResultTable',ab_table),('TimeResultTable',times_table),('FunnelResultTable',funnel_table),('CacheResultTable',cache_table),('EffortResultTable',effort_table)]:
        text += f"\\newcommand{{\\{name}}}{{%\n"+t+'\n}\n'
    a.out.write_text(text)
    print(json.dumps(metrics,indent=2));print(f'Wrote {a.out}; hashed {len(files)} archived artifacts; superseded evidence is not admitted')
if __name__=='__main__':main()
