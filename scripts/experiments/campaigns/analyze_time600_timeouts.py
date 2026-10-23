"""Replay immutable EPFL600 traces; stage attribution, not a CPU/memory profile."""
import argparse,collections,csv,hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
DEFAULT=ROOT/'results_v2/time_only_600_20261022/epfl_checkpoint/campaign.json'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def analyze(campaign):
    cp=json.loads(campaign.read_text());records=[];sources={str(campaign):digest(campaign)}
    for j in cp['jobs']:
        p=Path(j['report']);folder=p.parent
        needed=[p,folder/'attempt0.stages.jsonl',folder/'attempt0.pool.import.json',folder/'attempt0.pool.retention-policy.json',folder/'attempt0.pool.generator-index.jsonl',folder/'attempt0.pool.cut-stage-budget.json']
        for f in needed:
            if f.exists():
                actual=digest(f);assert j['sha256'].get(str(f))==actual,'artifact drift: '+str(f);sources[str(f)]=actual
        report=json.loads(p.read_text());trace=[json.loads(l)for l in(folder/'attempt0.stages.jsonl').read_text().splitlines()]
        times={e['stage']:e['unix_secs']for e in trace};assert all(y['unix_secs']>=x['unix_secs']for x,y in zip(trace,trace[1:]))
        imp=json.loads((folder/'attempt0.pool.import.json').read_text());mapping=report['mapping'];native=report['native']
        rounds=[x for x in trace if re.fullmatch(r'cut_round_\d+_start',x['stage'])]
        spans=[dict(round=int(x['stage'].split('_')[2]),seconds=y['unix_secs']-x['unix_secs'])for x,y in zip(rounds,rounds[1:])]
        span=lambda a,b:times[b]-times[a]if a in times and b in times else None
        state='completed'if native['optimizer_succeeded']else'cuts_unfinished'if 'cuts_start'in times and 'cuts_done_generate_start'not in times else'recovery_unfinished'if 'recover_start'in times else'other_unfinished'
        record=dict(bench=j['bench'],status=report['status'],optimizer_succeeded=native['optimizer_succeeded'],mapping_secs=mapping['wall_secs'],CEC=report.get('measured',{}).get('cec'),source_only=not native['optimizer_succeeded'],import_complete=imp['import_complete'],imported_enodes=imp['actual_imported_enodes'],stage_class=state,last_stage=trace[-1]['stage'],saturation_secs=span('saturation_start','saturation_done'),cuts_secs=span('cuts_start','cuts_done_generate_start'),small_generation_secs=span('cuts_done_generate_start','generate_done'),after_small_generation_to_recovery_secs=span('generate_done','recover_start'),rounds_started=len(rounds),completed_round_spans=spans,
                    recovery_remaining_upper_secs=max(0.,600-(times['recover_start']-trace[0]['unix_secs']))if 'recover_start'in times else None,
                    upper_bound_note='Import_done occurs after mapping start. This bound excludes unknown earlier work; it is NOT actual recovery time or a cross-clock alignment.')
        retention=folder/'attempt0.pool.retention-policy.json'
        if retention.exists():record['retained_instances']=json.loads(retention.read_text()).get('retained_instances')
        index=folder/'attempt0.pool.generator-index.jsonl'
        if index.exists():
            entries=[json.loads(l)for l in index.read_text().splitlines()];record['generator_index_entries']=entries[-1].get('entries')
        if not native['optimizer_succeeded']:
            assert report['status']=='mapping_timeout'and mapping['timeout']and mapping['wall_secs']>=600 and record['CEC']==1
            assert imp['import_complete'],'do not classify partial import as search timeout'
        records.append(record)
    failed=[x for x in records if not x['optimizer_succeeded']]
    return dict(scope='EPFL20 uniform frozen candidate600; no reruns or source changes',records=records,timeout_counts=dict(collections.Counter(x['stage_class']for x in failed)),complete=len(records)-len(failed),timeouts=len(failed),all_final_CEC=sum(x['CEC']==1 for x in records),sources=sources,
        limitations=['Stage wall spans are not CPU attribution.','Last entered stage does not identify an inner hot loop.','No per-round population/RSS/peak-memory/OOM telemetry exists in this run.','Missing recovery final journal is not proof that no valid seed was visited.','Incumbents in memory, if any, cannot be counted as mapped outputs without export andCEC.'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,default=DEFAULT);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--require-complete',action='store_true');a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    data=analyze(a.campaign);(out/'analysis.json').write_text(json.dumps(data,indent=2)+'\n')
    fields=['bench','stage_class','imported_enodes','saturation_secs','cuts_secs','small_generation_secs','after_small_generation_to_recovery_secs','retained_instances','rounds_started','recovery_remaining_upper_secs','mapping_secs','CEC']
    with(out/'timeouts.csv').open('w')as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(x for x in data['records']if not x['optimizer_succeeded'])
    for x in data['records']:
        if not x['optimizer_succeeded']:print(x['bench'],x['stage_class'],x['last_stage'],'nodes',x['imported_enodes'],'retained',x.get('retained_instances'))
    print('optimized',data['complete'],'timeouts',data['timeouts'],'categories',data['timeout_counts'])
    if a.require_complete and data['timeouts']:raise SystemExit(1)
if __name__=='__main__':main()
