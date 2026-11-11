"""Read-only pin-boundary join. Signature hits are potential, not equivalence proofs."""
import argparse
import json
from pathlib import Path
import dac_function_boundaries as model

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    assert not a.output.exists();a.output.mkdir(parents=True);rows=[]
    for name in ['adder','priority']:
        base=model.S/'fixed-pool-diagnosis'/name
        frozen=model.read(base/'capture/recovery-snapshot.json')
        payload=model.read(model.S/'reference-function-boundaries'/(name+'.json'))
        logic=model.read(Path(payload['logic_path']))
        assert frozen['selected_context']['logic']==logic,'do not join different logical epochs'
        words=[int(v,16)for v in payload['native_class_function_words']]
        def sig(s):return words[s//2]^(model.MASK if s%2 else 0)
        lookup={}
        for t in payload['targets']:
            key=(t['cell'],tuple(sorted((pin,int(word,16))for pin,word in t['input_functions'].items())),int(t['output_function_samples'],16))
            lookup.setdefault(key,[]).append(t['instance'])
        hits={t['instance']:[]for t in payload['targets']}
        for i,cell in enumerate(frozen['pool']):
            inputs=tuple(sorted((pin,sig(s))for pin,s in zip(cell['input_pins'],cell['inputs'])))
            for pin,out in enumerate(cell['outputs']):
                for output_sig in set(sig(s)for s in out['targets']):
                    key=(cell['cell'],inputs,output_sig)
                    for target in lookup.get(key,[]):hits[target].append(dict(instance=i,output_pin=pin,recorded_sources=len(out['sources'])))
        row=dict(design=name,critical_cells=len(hits),potential_exact_cell_pin_boundary_hits=sum(bool(v)for v in hits.values()),
                 no_cell_pin_boundary_match_at_epoch=[k for k,v in hits.items()if not v],hits=hits,
                 scope='same cell name and every named physical pin function plus target function on256 concrete assignments; absence excludes exact pin-boundary reproduction in this captured pool, not alternative fast covers; hits require equivalence/certificate and physical connectivity checks',
                 all_future_candidates_enumerated=False,source_certificates_rechecked=False,baseline_runs=0,CEC_runs=0)
        (a.output/(name+'.json')).write_text(json.dumps(row,indent=2)+'\n');rows.append({k:v for k,v in row.items()if k!='hits'})
        del frozen
    (a.output/'summary.json').write_text(json.dumps(dict(rows=rows,production_algorithm_changed=False,GitHub_push=False),indent=2)+'\n');print(json.dumps(rows))
if __name__=='__main__':main()
