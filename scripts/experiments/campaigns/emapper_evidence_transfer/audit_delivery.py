"""Diagnostic only: validate a committed transaction missing from final receipt."""
import argparse,fcntl,json,pathlib,sys
R=pathlib.Path(__file__).resolve().parents[4];sys.path.insert(0,str(R/'scripts/emapper'))
from physical import measure_cover
from verify import verify
from selected_evidence import replay

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--case',default='sin_r1_move');ap.add_argument('--sequence',type=int,default=2014);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args();assert not a.output.exists();locks=[];base=R/'results_v2/emapper_evidence_transfer_20261111/benchmark'
 for path in [R/'results_v2/independent_components_20261019/global.lock',base/'runner.lock']:
  f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 c=json.loads((base/'campaign.json').read_text());assert json.loads((base/'state.json').read_text())['active']is None;r=c['jobs'][a.case];p=pathlib.Path(r['case']);folder=p/'incumbents'/f'{a.sequence:08}';m=json.loads((folder/'manifest.json').read_text());measured=measure_cover(R/'emapper/libraries/asap7/asap7_full.genlib',folder/'mapped.v',a.output,R/'target/release/emapper_measure','yosys');cec=verify(R/'emapper/libraries/asap7/cells.v',pathlib.Path(c['subjects'][r['design']]['golden']),pathlib.Path(measured['netjson']),a.output/'cec');proof=replay(folder/'manifest.json',p/'selected-provenance/objects',pathlib.Path(measured['netjson']));assert cec and proof['valid'];assert[measured['measured']['area'],measured['measured']['delay']]==[m['area'],m['delay']]
 receipt=json.loads((p/'raw.incumbent.receipt.json').read_text());stages=[json.loads(x)for x in(p/'online.stages.jsonl').read_text().splitlines()];last=next(x for x in reversed(stages)if x['stage']=='online_incumbent_published');offset=last['unix_secs']-(p/'raw.incumbent.receipt.json').stat().st_mtime-(receipt['deadline_monotonic']-receipt['observed_monotonic'])
 result=dict(scope='diagnostic only, no new mapping search or replacement of official result',manifest=str(folder/'manifest.json'),measured=measured['measured'],CEC=cec,evidence_valid=proof['valid'],official_AD=[r['A'],r['D']],diagnostic_AD=[m['area'],m['delay']],diagnostic_sequence=a.sequence,receipted_sequence=receipt['sequence'],last_publish_stage=last,estimated_last_publish_minus_outer_deadline_seconds=offset,timing_caveat='nearby filesystem Unix mtime/receipt monotonic calibration; publish-stage event has no sequence or direct monotonic timestamp; cannot certify deadline eligibility or retroactively change official result',original_results_unchanged=True,baseline_runs=0);(a.output/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
