#!/usr/bin/env python3
"""Independently recheck immutable-library pilot receipts and selected PPA/CEC."""
import argparse
import json
from pathlib import Path
import re

from audit_mapped_ppa import audit
from build_readonly_profile_library import snapshot, sha
from run_buffer_core_validation import buffer_area
from run_prebuffer_three_parts import proof


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    m=json.loads((root/'manifest.json').read_text());v=json.loads((root/'validation.json').read_text())
    assert v['complete'] and v['observations']==27
    assert sha(root/'inputs/engine')==m['engine_sha256']
    # pathlib does not recurse through the benchmarks directory symlink. Bind
    # these borrowed raw AIGs explicitly to the previous frozen input hashes.
    control=json.loads((Path(m['control'])/'manifest.json').read_text())
    borrowed={name:digest for name,digest in control['input_sha256'].items()
              if name.startswith('benchmarks/') or name in ['abc','asap7.genlib','asap7_6in.genlib','database.super']}
    for rel,digest in borrowed.items():assert sha(root/'inputs'/rel)==digest,rel
    (root/'original_inputs_receipt.json').write_text(json.dumps(dict(
        control_manifest_sha256=sha(Path(m['control'])/'manifest.json'),input_sha256=borrowed),indent=2)+'\n')
    for rel,digest in m['input_sha256'].items():assert sha(root/'inputs'/rel)==digest,rel
    for rel,digest in m['normalized_sha256'].items():assert sha(root/'normalized'/rel)==digest,rel
    library=root/'shared_library';frozen=json.loads((root/'library_frozen_sha256.json').read_text())
    assert snapshot(library)==frozen
    assert library.stat().st_mode & 0o222==0
    for p in library.iterdir():assert p.stat().st_mode & 0o222==0
    manifest=json.loads((library/'library_manifest.json').read_text())
    for rel,entry in manifest['records'].items():assert sha(library/rel)==entry['sha256'],rel
    previous=json.loads((Path(m['control'])/'analysis.json').read_text())
    previous_index={(r['stage'],r['design']):r for r in previous['records']
                    if r['arm']=='ours' and r['mode']=='cold'}
    reports=[];index={}
    for result in sorted(root.glob('*/*/cold.*/result.json')):
        r=json.loads(result.read_text());wd=result.parent;stage=r['stage'];design=r['design'];arm=r['arm']
        assert r['status']=='complete' and r['cec']==1 and r['optimizer_seconds']<=1000.1
        lib=root/'inputs'/('asap7.genlib' if stage=='main' else 'asap7_6in.genlib')
        net=root/r['selected_netlist'];assert sha(net)==r['selected_sha256']
        cost=audit(net,lib);initial=audit(wd/'epmapping_baseline.v',lib)
        assert abs(cost['area']-r['final']['area'])<.0001 and abs(cost['delay']-r['final']['delay'])<.001
        assert cost['area']<=initial['area']+.0001 and cost['delay']<=initial['delay']+.001
        old=previous_index[stage,design]
        assert r['initial_sha256']==old['initial_sha256']
        assert cost['area']<=old['final']['area']+.0001
        assert cost['delay']<=old['final']['delay']+.001
        assert buffer_area(net,lib)[0]==buffer_area(wd/'epmapping_baseline.v',lib)[0]==0
        cohort='strong47' if stage=='main' else 'strong6' if stage=='strong_common' else 'direct6'
        assert sha(root/'normalized'/cohort/(design+'.v'))==sha(wd/'epmapping_baseline.v')
        command=json.loads((wd/'command.json').read_text());env=command['env']
        assert command['input_sha256']==borrowed['benchmarks/'+design+'.aig']
        assert command['library_sha256']==sha(lib)
        assert command['engine_sha256']==m['engine_sha256']
        for k,val in [('EPMAPPING_WINDOW_SYN','0'),('EPMAPPING_POST_TIMEOUT','1000'),
                      ('EPMAPPING_PRE_BUFFER_CLEANUP','1'),('MAP_EGG_PROFILE_TT_CACHE','0')]:assert env[k]==val
        private=root/'private'/arm/stage/design
        assert Path(env['MAP_EGG_PROFILE_CACHE_DIR'])==private
        log=(wd/'log').read_text()
        line=re.findall(r'egraph_profile (.*)',log)[-1]
        stats={k:float(val) if any(c in val for c in '.eE') else int(val) for k,val in re.findall(r'(\w+)=([0-9.eE+-]+)',line)}
        if arm!='build':
            assert Path(env['MAP_EGG_PROFILE_READONLY_DIR'])==library
            assert stats['builds']==stats['private_disk_hits']==stats['profile_writes']==0
            assert stats['readonly_hits']>0 and not list(private.iterdir())
        cec_hash=proof(root,wd,net,design,lib,'independent_prebuilt_cec')
        reports.append(dict(stage=stage,design=design,arm=arm,area=cost['area'],delay=cost['delay'],
                            selected_sha256=sha(net),CEC=1,cec_log_sha256=cec_hash,profile=stats))
        index[stage,design,arm]=sha(net)
    assert len(reports)==27
    for stage in m['stages']:
        for design in m['design_order']:
            assert len({index[stage,design,a] for a in ['build','prebuilt','prebuilt_reverse']})==1
    assert snapshot(library)==frozen
    output=dict(complete=True,observations=27,independent_PPA=True,independent_CEC=True,
                auditor_sha256=sha(Path(__file__)),protocol_manifest_sha256=sha(root/'manifest.json'),
                library_manifest_sha256=sha(library/'library_manifest.json'),
                immutable_library=True,empty_private_deltas=True,reverse_order_byte_identical=True,
                nonregression_to_previous=True,
                previous_cold_campaign=str(m['control']),records=reports)
    (root/'independent_audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:val for k,val in output.items() if k!='records'},indent=2))


if __name__=='__main__':main()
