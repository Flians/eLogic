#!/usr/bin/env python3
"""Audit paired cold/warm campaigns, then export tables without pooling their times."""
import argparse,csv,hashlib,json,re,statistics,subprocess,sys
from pathlib import Path
import report,export_paper
from audit_mapped_ppa import audit as physical_audit
from run_three_parts import DESIGNS
from timeout_selection import read_rows,selected_netlist
SCRIPT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open() as f:return list(csv.DictReader(f,delimiter='\t'))
def mean(v):return statistics.mean(v)
def pct(a,b):return 100*(float(a)/float(b)-1)
def time_of(root,stage,arm,d):return report.read_counters(str(root/stage/f'{arm}.{d}'/'log'))['time_optimizer']
def display_time(root,stage,arm,d):
    wd=root/stage/f'{arm}.{d}'
    return 'TO' if (wd/'timeout_selection.json').exists() else f'{time_of(root,stage,arm,d):.1f}'
def pair(a,b):return f'{a:.1f}/{b:.1f}'
def fmt(v):return '0.00' if round(v,2)==0 else f'{v:+.2f}'
def table(caption,label,head,body,align,group=None):
    return '\n'.join([r'\begin{table}[t]',r'\centering\scriptsize',r'\setlength{\tabcolsep}{1.5pt}',r'\caption{'+caption+'}'+r'\label{'+label+'}',r'\begin{tabular}{@{}'+align+r'@{}}',r'\toprule',*(group or []),' & '.join(head)+r' \\',r'\midrule',*[line[0] if len(line)==1 else ' & '.join(map(str,line))+r' \\' for line in body],r'\bottomrule',r'\end{tabular}',r'\end{table}'])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path);args=ap.parse_args();root=args.root.resolve();cohorts={m:root/m for m in ['cold','warm']}
    certificates={};metrics={};data={};cache_records=[];native_records=[]
    arms=json.loads((root/'cold/manifest.json').read_text())['arms']
    assert arms==json.loads((root/'warm/manifest.json').read_text())['arms']
    for mode,folder in cohorts.items():
        with (root/f'{mode}_audit.log').open('w') as f:
            subprocess.run([sys.executable,str(SCRIPT/'report_three_parts.py'),'--root',str(folder)],check=True,stdout=f,stderr=subprocess.STDOUT)
        cert=json.loads((folder/'verified_metrics.json').read_text());certificates[mode]=sha(folder/'verified_metrics.json');metrics[mode]=cert['metrics']
        manifest=json.loads((folder/'manifest.json').read_text());assert manifest['profile_cache_mode']==mode and manifest['goal']['MAP_EGG_PROFILE']=='1'
        data[mode]={stage:{(r['stage'],r['design']):r for r in read_rows(folder,stage)} for stage in ['tab_mfs3','tab_main','ablation']}
        for stage,rr in data[mode].items():
            for (arm,d),r in rr.items():
                if arm=='base':continue
                wd=folder/stage/f'{arm}.{d}';flags=(wd/'experiment_env.txt').read_text().splitlines();assert 'MAP_EGG_PROFILE=1' in flags
                trace=json.loads((wd/'cache_provenance.json').read_text());assert trace['mode']==mode
                p=re.search(r'egraph_profile (.*)',(wd/'log').read_text())
                if r['raw_status']=='timeout':
                    assert p is None
                    stats={k:None for k in ['memory_hits','disk_hits','builds','corrupt','write_failures','legacy_fallbacks']}
                else:
                    assert p
                    stats={k:int(v) for k,v in (item.split('=') for item in p[1].split())}
                    assert stats['corrupt']==stats['write_failures']==0,(mode,stage,arm,d,stats)
                expected=Path(manifest['cache_root'])/stage/f'{arm}.{d}'
                assert Path(trace['cache_directory'])==expected
                for name,h in trace['cache_record_sha256'].items():
                    assert sha(expected/name)==h, f'changed cached pool: {mode}/{stage}/{arm}.{d}/{name}'
                # A cold process can reload records it created itself after memory eviction.
                # Empty-at-start is enforced by the archived runner, not by zero disk reads.
                cache_records.append(dict(mode=mode,stage=stage,arm=arm,design=d,status=r['raw_status'],optimizer_seconds=time_of(folder,stage,arm,d),**stats))
                text=(wd/'log').read_text()
                match=re.search(r'epmapping build_reuse (.*)',text)
                native={k:float(v) for k,v in re.findall(r'(\w+)=([\d.]+)',match[1])} if match else None
                cf=re.search(r'epmapping cofactor ([^\n]+)',text)
                cofactor={k:float(v) for k,v in re.findall(r'(\w+)=([\d.]+)',cf[1])} if cf else None
                sc=re.search(r'synth_cache lookups=(\d+) hits=(\d+)',text)
                synthesis=dict(zip(['lookups','hits'],map(int,sc.groups()))) if sc else None
                lq=re.search(r'epmapping live_queries ([^\n]+)',text)
                live_queries={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',lq[1])} if lq else None
                native_records.append(dict(mode=mode,stage=stage,arm=arm,design=d,status=r['raw_status'],native=native,cofactor=cofactor,synthesis=synthesis,live_queries=live_queries))
    matches=[]
    for stage,rr in data['cold'].items():
        assert rr.keys()==data['warm'][stage].keys()
        for (arm,d),r in rr.items():
            if arm=='base':continue
            c=cohorts['cold']/stage/f'{arm}.{d}';w=cohorts['warm']/stage/f'{arm}.{d}';assert (c/'epmapping_baseline.v').read_bytes()==(w/'epmapping_baseline.v').read_bytes()
            other=data['warm'][stage][arm,d]
            lib=cohorts['cold']/'inputs'/('asap7_6in.genlib' if stage=='tab_mfs3' else 'asap7.genlib')
            cp=physical_audit(selected_netlist(c),lib);wp=physical_audit(selected_netlist(w),lib)
            equal=all(abs(float(r[k])-float(other[k]))<=.001 for k in ['out_a','out_d']) and all(abs(cp[k]-wp[k])<=.001 for k in ['area','delay'])
            matches.append(dict(stage=stage,arm=arm,design=d,cold_status=r['raw_status'],warm_status=other['raw_status'],ppa_equal=equal,cold_physical=cp,warm_physical=wp,netlist_equal=selected_netlist(c).read_bytes()==selected_netlist(w).read_bytes()))
    # The common baseline is a freshly executed ABC batch, reused by the warm cohort.
    assert json.loads((cohorts['cold']/'tab_mfs3/abc_results.json').read_text())==json.loads((cohorts['warm']/'tab_mfs3/abc_results.json').read_text())
    raw={str(p.relative_to(root)):sha(p) for m in cohorts for p in (cohorts[m]).rglob('cache_provenance.json')}
    raw['execution_plan.json']=sha(root/'execution_plan.json')
    report_data=dict(raw_sha256=raw,cohort_metrics=metrics,cohort_certificate_sha256=certificates,cache_runs=cache_records,native_build_runs=native_records,matches=matches,all_ppa_equal=all(r['ppa_equal'] for r in matches),all_netlists_equal=all(r['netlist_equal'] for r in matches))
    (root/'verified_profiles.json').write_text(json.dumps(report_data,indent=2)+'\n')
    # Complete per-design costs for both cohorts, including any trajectory difference.
    allrows=[]
    for mode in cohorts:
        for stage,rr in data[mode].items():
            for (arm,d),r in rr.items():
                if arm!='base':allrows.append(dict(mode=mode,part=stage,arm=arm,design=d,status=r['raw_status'],selection=r.get('selection','initial' if r['raw_status']=='timeout' else 'optimized'),initial_area=r['base_a'],initial_delay=r['base_d'],area=r['out_a'],delay=r['out_d'],optimizer_seconds=time_of(cohorts[mode],stage,arm,d),selected_cec=r['cec']))
    with (root/'paired_results.tsv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(allrows[0]),delimiter='\t');writer.writeheader();writer.writerows(allrows)
    # Wall-clock limits can produce different trajectories. Keep both measured outcomes.
    generated=(cohorts['cold']/'generated_results.tex').read_text();generated=generated[:generated.index('\\newcommand{\\MainResultTable}')]
    generated='% GENERATED from paired CEC/PPA/cache audits.\n'+generated
    cold,warm=metrics['cold'],metrics['warm'];times={m:{part:sum(r['optimizer_seconds'] for r in cache_records if r['mode']==m and r['stage']==part and r['arm'] in ('ae','ours','full')) for part in ['tab_mfs3','tab_main','ablation']} for m in cohorts}
    def duo(a,b,fmt):
        aa,bb=fmt(a),fmt(b)
        return aa if aa==bb else r'\begin{tabular}[c]{@{}r@{}}'+aa+r'\\'+bb+r'\end{tabular}'
    def metric_pair(key):
        a,b=f'{cold[key]:+.2f}',f'{warm[key]:+.2f}'
        return a if a==b else a+'/'+b
    def native_cost(part,arm,d,key,fn):return duo(float(data['cold'][part][arm,d][key]),float(data['warm'][part][arm,d][key]),fn)
    def native_time(part,arm,d):return '/'.join(display_time(cohorts[m],part,arm,d) for m in cohorts)
    macros={
        'ColdMainSeconds':f"{times['cold']['tab_main']:.1f}",'WarmMainSeconds':f"{times['warm']['tab_main']:.1f}",
        'ColdMfsSeconds':f"{times['cold']['tab_mfs3']:.1f}",'WarmMfsSeconds':f"{times['warm']['tab_mfs3']:.1f}",
        'WarmTimeEggShare':f"{100*warm['cut_oracle_times']['secs_egg']/sum(warm['cut_oracle_times'].values()):.1f}",
        'WarmTimeSynShare':f"{100*warm['cut_oracle_times']['secs_synth']/sum(warm['cut_oracle_times'].values()):.1f}",
        'ColdNoEggRatio':f"{cold['ablation']['noEGG']['time_ratio']:.2f}",'WarmNoEggRatio':f"{warm['ablation']['noEGG']['time_ratio']:.2f}",
        'SynWorseCount':cold['ablation']['noSYN']['worse'],'EggWorseCount':cold['ablation']['noEGG']['worse'],'DBBetterCount':cold['ablation']['noDB']['better']}
    macros.update(ColdCheckpointSelections=cold.get('method_checkpoint_selections',0),WarmCheckpointSelections=warm.get('method_checkpoint_selections',0))
    macros.update(ProfilePairs=len(matches),ProfileIdentical=sum(r['netlist_equal'] for r in matches))
    macros.update(ProfileEqual=sum(r['ppa_equal'] for r in matches),ColdMethodTimeouts=cold['method_timeouts'],WarmMethodTimeouts=warm['method_timeouts'],ColdCompleted=cold['engine_cec_outputs'],WarmCompleted=warm['engine_cec_outputs'],ColdPhaseRuns=cold['phase_timed_full_runs'],WarmPhaseRuns=warm['phase_timed_full_runs'])
    primary=[r for r in cache_records if r['stage']!='ablation' or r['arm']=='full']
    macros['FullCompletionSentence']=(f'All {len(primary)} runs of the full configuration across the three parts finish within the post-mapping limit. '
        if all(r['status']=='complete' for r in primary) else
        f"The full configuration finishes {sum(r['status']=='complete' for r in primary)} of {len(primary)} runs across the three parts. ")
    for mode,title in [('cold','Cold'),('warm','Warm')]:
        macros[title+'MfsCompleted']=sum(r['raw_status']=='complete' for (arm,d),r in data[mode]['tab_mfs3'].items() if arm=='ours')
        macros[title+'MainCompleted']=sum(r['raw_status']=='complete' for (arm,d),r in data[mode]['tab_main'].items() if arm=='ae')
    # A slash in prose denotes cold/warm costs only when they differ.
    for macro,key in [('OwnArea','main_own_area'),('OwnDelay','main_own_delay'),('OwnOneArea','main_one_area'),('OwnOneDelay','main_one_delay'),('OwnOtherArea','main_other_area'),('OwnOtherDelay','main_other_delay'),('MfsOwnArea','mfs_ours_area'),('MfsOwnDelay','mfs_ours_delay'),('PairedOwnArea','paired_ours_area'),('PairedOwnDelay','paired_ours_delay'),('MainAreaAll','reference_area'),('MainDelayAll','reference_delay')]:
        # Override the single-cohort macro before table definitions.
        generated=re.sub(r'\\newcommand\{\\'+macro+r'\}\{[^}]*\}',lambda m:rf'\newcommand{{\{macro}}}{{{metric_pair(key)}}}',generated)
    arb=data['cold']['tab_main']['ae','arbiter'];warb=data['warm']['tab_main']['ae','arbiter']
    macros.update(ArbiterArea='/'.join(dict.fromkeys(f"{float(r['out_a']):.2f}" for r in [arb,warb])),ArbiterDelay='/'.join(dict.fromkeys(f"{float(r['out_d']):.2f}" for r in [arb,warb])))
    for name,key in [('MainWins','reference_wins'),('OwnAreaWins','main_area_wins'),('DBBetterCount',None)]:
        a,b=(cold[key],warm[key]) if key else (cold['ablation']['noDB']['better'],warm['ablation']['noDB']['better'])
        value=str(a) if a==b else str(a)+'/'+str(b)
        if name in macros:macros[name]=value
        else:generated=re.sub(r'\\newcommand\{\\'+name+r'\}\{[^}]*\}',lambda m:rf'\newcommand{{\{name}}}{{{value}}}',generated)
    a,b=cold['ablation']['noEGG']['worse'],warm['ablation']['noEGG']['worse']
    macros['EggWorseCount']=str(a) if a==b else str(a)+'/'+str(b)
    for arm,name in [('noNARROW','Narrow'),('noPHASE','Phase'),('noSHANNON','Shannon'),('noDAVIO','Davio'),('noREUSE','Reuse'),('noPRICE','Price')]:
        if arm in cold['ablation']:
            v=cold['ablation'][arm];other=warm['ablation'][arm]
            a,b=f"{v['area']:+.2f}",f"{other['area']:+.2f}"
            macros.update({f'Delta{name}':a if a==b else a+'/'+b,f'{name}WorseCount':v['worse'],f'{name}BetterCount':v['better'],f'{name}ColdRatio':f"{v['time_ratio']:.2f}",f'{name}WarmRatio':f"{other['time_ratio']:.2f}"})
    for arm,name in [('cache64K','CacheCapacity'),('noAIGREUSE','AigReuse'),('noLIBSHARE','LibraryShare')]:
        if arm in cold['ablation']:
            c,w=cold['ablation'][arm],warm['ablation'][arm]
            macros.update({name+'ColdRatio':f"{c['time_ratio']:.2f}",name+'WarmRatio':f"{w['time_ratio']:.2f}",name+'AreaCold':f"{c['area']:+.2f}",name+'AreaWarm':f"{w['area']:+.2f}"})
    for arm,name in [('noQUERY','Query'),('noTRUTH','Truth'),('allPIVOTS','Pivot')]:
        if arm in arms:
            c,w=cold['ablation'][arm],warm['ablation'][arm]
            macros.update({name+'ColdRatio':f"{c['time_ratio']:.2f}",name+'WarmRatio':f"{w['time_ratio']:.2f}",name+'AreaCold':f"{c['area']:+.2f}",name+'AreaWarm':f"{w['area']:+.2f}"})
    macros['DeadlineExample']=''
    if 'noDAVIO' in arms and all(data[m]['ablation']['full','aes_core']['raw_status']=='timeout' and data[m]['ablation']['noDAVIO','aes_core']['raw_status']=='complete' and float(data[m]['ablation']['noDAVIO','aes_core']['out_a'])<float(data[m]['ablation']['full','aes_core']['out_a']) for m in cohorts):
        lean='/'.join(dict.fromkeys(data[m]['ablation']['noDAVIO','aes_core']['out_a'] for m in cohorts))
        saved='/'.join(dict.fromkeys(data[m]['ablation']['full','aes_core']['out_a'] for m in cohorts))
        macros['DeadlineExample']=r'For \texttt{aes\_core}, noDAVIO completes at $'+lean+r'\,\mu m^2$, while full reaches the deadline and selects a verified saved output at $'+saved+r'\,\mu m^2$. This difference includes the time budget.'
    macros['CacheDeadlineExample']=''
    if data['cold']['tab_main']['ae.r1','mem_ctrl']['raw_status']=='timeout' and data['warm']['tab_main']['ae.r1','mem_ctrl']['raw_status']=='complete':
        macros['CacheDeadlineExample']=r'One-round \texttt{mem\_ctrl} completes warm but times out cold.'
    if data['cold']['ablation']['full','usb_funct']['raw_status']=='timeout' and data['warm']['ablation']['full','usb_funct']['raw_status']=='complete':
        shift=float(data['cold']['ablation']['full','usb_funct']['out_a'])-float(data['warm']['ablation']['full','usb_funct']['out_a'])
        macros['CacheDeadlineExample']+=r' Full-ablation \texttt{usb\_funct} completes only warm; the selected full areas differ by $'+f'{shift:+.2f}'+r'\,\mu m^2$, which also changes arm-to-full differences.'
    macros['UsbAblationExample']=''
    controls=['noSHANNON','noDAVIO','noREUSE','noPRICE']
    if all(arm in arms for arm in controls) and all(data[m]['ablation'][arm,'usb_phy']['raw_status']=='complete' for m in cohorts for arm in ['full']+controls) and all(float(data[m]['ablation'][arm,'usb_phy']['out_a'])>float(data[m]['ablation']['full','usb_phy']['out_a']) for m in cohorts for arm in controls):
        macros['UsbAblationExample']=r'All four removals increase area on \texttt{usb\_phy}.'
    macros['BuildCaseExample']=''
    byrun={(x['mode'],x['arm'],x['design']):x for x in native_records if x['stage']=='ablation'}
    if 'cache64K' in arms and all(byrun[m,a,'tv80']['status']=='complete' for m in cohorts for a in ['full','cache64K']):
        fullbuild=[byrun[m,'full','tv80']['cofactor']['builds'] for m in cohorts]
        smallbuild=[byrun[m,'cache64K','tv80']['cofactor']['builds'] for m in cohorts]
        unchanged=all(all(data[m]['ablation']['full','tv80'][k]==data[m]['ablation']['cache64K','tv80'][k] for k in ['out_a','out_d']) for m in cohorts)
        def counts(values):return '/'.join(dict.fromkeys(str(int(x)) for x in values))
        if unchanged and all(b>a for a,b in zip(fullbuild,smallbuild)):
            macros['BuildCaseExample']=r'On \texttt{tv80}, cache64K raises cofactor rebuilds from '+counts(fullbuild)+' to '+counts(smallbuild)+r' without changing selected area or delay. Full bypasses '+counts([byrun[m,'full','tv80']['native']['reused'] for m in cohorts])+r' mappings through exact AIG reuse.'
    generated+='\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k,v in macros.items())+'\n'
    reference=report.read_paper_table(str(cohorts['cold']/'inputs/compare_with_aspdac25.sh'))
    ref={d:(reference[d][0]*(1+reference[d][2]/100),reference[d][3]*(1+reference[d][5]/100)) for d in DESIGNS}
    mainbody=[]
    for d in DESIGNS:
        r=data['cold']['tab_main']['ae',d];mainbody.append([d.replace('_',r'\_'),*[f'{x:.2f}' for x in ref[d]],*[f'{float(r[k]):.2f}' for k in ['base_a','base_d']],*[native_cost('tab_main','ae',d,k,lambda x:f'{x:.2f}') for k in ['out_a','out_d']],native_time('tab_main','ae',d)])
    ratio=[mean(float(data['cold']['tab_main']['ae',d][key])/ref[d][i] for d in DESIGNS) for key,i in [('base_a',0),('base_d',1),('out_a',0),('out_d',1)]]
    warmratio=[mean(float(data['warm']['tab_main']['ae',d][key])/ref[d][i] for d in DESIGNS) for key,i in [('base_a',0),('base_d',1),('out_a',0),('out_d',1)]]
    mainbody+=[[r'\midrule'],['Ave. Ratio','1.0000','1.0000',*[duo(a,b,lambda x:f'{x:.4f}') for a,b in zip(ratio,warmratio)],'--']]
    main_completion = r'All runs finish within 1000\,s. ' if all(data[m]['tab_main']['ae',d]['raw_status']=='complete' for m in cohorts for d in DESIGNS) else r'TO: 1000\,s; independently verified saved output selected. '
    main=table(r"Preprocessed mappings, at most six rounds. ASPDAC'25 costs are reconstructed from reported starts and rounded changes; starts differ. Area: $\mu m^2$; delay: ps. $t_c/t_w$: cold/warm optimizer seconds, excluding preprocessing, mapping and CEC. "+main_completion+r"Differing costs are stacked cold above warm. Ave. Ratio averages ratios to the reconstructed reference over all 16 designs.",'tab:main',['','Area','Delay','$A_e$','$D_e$','Area','Delay','$t_c/t_w$'],mainbody,'lrrrrrrr',[r'Circuit & \multicolumn{2}{c}{ASPDAC\textquotesingle25} & \multicolumn{2}{c}{Initial} & \multicolumn{2}{c}{Ours} & Time (s) \\',r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}'])
    ablbody=[]
    for arm in arms:
        v=cold['ablation'][arm];other=warm['ablation'][arm]
        ablbody.append([arm,*[duo(v[k],other[k],lambda x:f'{x:+.2f}') for k in ['own_area','own_delay','area']],duo(v['worse'],other['worse'],str),duo(v['better'],other['better'],str),f"{v['time_ratio']:.2f}/{other['time_ratio']:.2f}"])
    if 'noSHANNON' in arms:
        ablbody=[]
        for arm in arms:
            v,o=cold['ablation'][arm],warm['ablation'][arm]
            tos=[sum(r['raw_status']=='timeout' for (a,d),r in data[m]['ablation'].items() if a==arm) for m in ['cold','warm']]
            ablbody.append([arm,f"{v['area']:+.2f}",f"{o['area']:+.2f}",f"{v['worse']}/{v['better']}",f"{o['worse']}/{o['better']}",f"{v['time_ratio']:.2f}/{o['time_ratio']:.2f}",f'{tos[0]}/{tos[1]}'])
        abl=table(r'Ablations on identical $N_e$ starts; subscripts $c,w$ denote cold/warm. $\Delta A$ ($\mu m^2$) sums area differences from full; W/B counts worse/better areas. Time/full divides total optimizer seconds by the same-mode full total. TO counts cold/warm deadlines; selected netlists are independently validated.','tab:ablation',['Arm','$\Delta A_c$','$\Delta A_w$','W/B$_c$','W/B$_w$','Time/full','TO'],ablbody,'lrrrrrr')
    else:
        abl=table(r'Ablations on identical $N_e$ starts. Differences are cold/warm when distinct; W/B count worse/better areas. Timeout selections are independently validated.','tab:ablation',['Arm','$\delta A$ (\%)','$\delta D$ (\%)','$\Delta A$','W','B','Time/full'],ablbody,'lrrrrrr')
    abc={r['design']:r for r in json.loads((cohorts['cold']/'tab_mfs3/abc_results.json').read_text())};paired=[d for d in DESIGNS if abc[d]['rc']==0];mfsbody=[]
    for d in DESIGNS:
        r=data['cold']['tab_mfs3']['ours',d];b=abc[d]
        abcval=['TO','TO','1000']
        if b['rc']==0:abcval=[fmt(pct(b['area'],b['baseline']['area'])),fmt(pct(b['delay'],b['baseline']['delay']))+(r'$^{\dagger}$' if b['delay']>b['baseline']['delay']+.001 else ''),f"{b['secs']:.1f}"]
        mfsbody.append([d.replace('_',r'\_'),f"{float(r['base_a']):.2f}",f"{float(r['base_d']):.2f}",*abcval,*[duo(pct(data['cold']['tab_mfs3']['ours',d]['out_'+k],r['base_'+k]),pct(data['warm']['tab_mfs3']['ours',d]['out_'+k],r['base_'+k]),fmt) for k in ['a','d']],native_time('tab_mfs3','ours',d)])
    abcrat=[mean(abc[d][k]/abc[d]['baseline'][k] for d in paired) for k in ['area','delay']]
    ownrat=[mean(float(data['cold']['tab_mfs3']['ours',d][f'out_{k}'])/float(data['cold']['tab_mfs3']['ours',d][f'base_{k}']) for d in paired) for k in ['a','d']]
    warmownrat=[mean(float(data['warm']['tab_mfs3']['ours',d][f'out_{k}'])/float(data['warm']['tab_mfs3']['ours',d][f'base_{k}']) for d in paired) for k in ['a','d']]
    mfsbody+=[[r'\midrule'],['Ave. Ratio','1.0000','1.0000',*[f'{v:.4f}' for v in abcrat],f"{mean(abc[d]['secs'] for d in paired):.1f}",*[duo(a,b,lambda x:f'{x:.4f}') for a,b in zip(ownrat,warmownrat)],pair(*[mean(time_of(cohorts[m],'tab_mfs3','ours',d) for d in paired) for m in cohorts])]]
    mfsbody=[mfsbody[-1]]
    mfs=table(r'One-round comparison on identical mappings and the same six-input sublibrary. Ave. Ratio averages native-start-normalized costs over the '+str(len(paired))+r' ABC-completed designs; ABC columns use raw outputs; ours uses independently validated selections. Times are means in seconds, excluding CEC; ours gives cold/warm. Full per-design results are archived.','tab:mfs3',['','$A/A_0$','$D/D_0$','$A/A_0$','$D/D_0$','$t$','$A/A_0$','$D/D_0$','$t_c/t_w$'],mfsbody,'lrrrrrrrr',[r'Statistic & \multicolumn{2}{c}{Initial} & \multicolumn{3}{c}{ABC mfs3} & \multicolumn{3}{c}{Ours} \\',r'\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-9}'])
    mfs=mfs.replace(r'\setlength{\tabcolsep}{1.5pt}',r'\setlength{\tabcolsep}{3pt}')
    for name,t in [('MainResultTable',main),('AblationResultTable',abl)]:generated+=f'\\newcommand{{\\{name}}}{{%\n{t}\n}}\n'
    (root/'generated_profiles.tex').write_text(generated)
    mfs_text='% GENERATED from paired CEC/PPA/cache audits.\n'+f'\\newcommand{{\\MfsComparisonTable}}{{%\n{mfs}\n}}\n'
    (root/'generated_mfs_profiles.tex').write_text(mfs_text)
    if args.out:
        args.out.write_text(generated)
        args.out.with_name('generated_mfs.tex').write_text(mfs_text)
    print(json.dumps(dict(metrics=metrics,totals=times,all_ppa_equal=report_data['all_ppa_equal'],all_netlists_equal=report_data['all_netlists_equal']),indent=2))
if __name__=='__main__':main()
