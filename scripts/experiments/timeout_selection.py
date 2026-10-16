"""Keep raw timeout evidence and separately validate an initial-netlist fallback.

Only SIGALRM during the explicitly timed optimizer is eligible. Crashes, failed
CEC, outer guards and validation failures are errors, never reclassified.
"""
import csv,hashlib,json,os,shutil,subprocess
from pathlib import Path
from audit_mapped_ppa import audit
from recover_checkpoint import recover

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def selected_netlist(wd):
    if (wd/'timeout_selection.json').exists():
        meta=json.loads((wd/'timeout_selection.json').read_text())
        return wd/meta.get('netlist','timeout_initial.v')
    return wd/'epmapping_mapped.v'
def adapt_row(folder,r):
    m=folder/f"{r['stage']}.{r['design']}"/'timeout_selection.json'
    if not m.exists():return dict(r,raw_status='complete')
    v=json.loads(m.read_text());assert r['rc']=='142' and r['out_a']=='-' and r['out_d']=='-'
    return dict(r,out_a=f"{v['area']:.2f}",out_d=f"{v['delay']:.2f}",cec='1',raw_status='timeout',selection=v['selected'],post_seconds=str(v['post_seconds']))
def read_rows(root,stage):
    with (root/stage/'summary.tsv').open() as f:return [adapt_row(root/stage,r) for r in csv.DictReader(f,delimiter='\t')]

def prepare(root):
    project=Path(__file__).resolve().parents[2];root=root.resolve();campaign=root.parent
    manifest=json.loads((root/'manifest.json').read_text());frozen=root/'inputs';count=0
    budget=manifest.get('post_mapping_timeout',1000)
    for rowfile in sorted(root.glob('*/*/row.tsv')):
        fields=rowfile.read_text().strip().split('\t')
        if fields[-1]=='0':continue
        assert fields[-1]=='142',(rowfile,fields)
        wd=rowfile.parent;text=(wd/'log').read_text();stage=wd.parent.name;design=fields[1]
        assert f'post_mapping_timeout={budget} seconds' in text and '[i] epmapping running (' in text
        assert 'epmapping area ' not in text and fields[6:8]==['-','-'],f'not an optimizer deadline: {rowfile}'
        assert '[i] epmapping synth_cache lookups=' not in text, f'optimizer already returned: {rowfile}'
        assert int(fields[-2])>=budget and int(fields[-2])<3600
        destination=wd/'timeout_selection.json'
        if destination.exists():continue
        if manifest.get('timeout_selection','initial')=='checkpoint' and (wd/'epmapping_checkpoint.json').exists():
            lib=frozen/('asap7_6in.genlib' if stage=='tab_mfs3' else 'asap7.genlib')
            evidence=recover(wd,lib,frozen/'benchmarks'/f'{design}.aig',frozen/'abc',142)
            metadata=dict(raw_returncode=142,status='optimizer_timeout',post_seconds=budget,
                          selected=evidence['selection'],netlist=evidence['checkpoint']['netlist'],
                          area=evidence['area'],delay=evidence['delay'],cec=1,
                          checkpoint=evidence['checkpoint'],selected_netlist_sha256=evidence['netlist_sha256'],
                          cec_source=str(wd/'checkpoint_cec.log'),cec_log_sha256=evidence['proof_sha256'],
                          initial_sha256=evidence['initial_sha256'],raw_row_sha256=sha(rowfile),raw_log_sha256=sha(wd/'log'))
            destination.write_text(json.dumps(metadata,indent=2)+'\n');count+=1;continue
        if stage=='tab_mfs3':
            proof=campaign/'cold/tab_mfs3'/f'base.{design}'
            assert '[i] baseline_cec=1' in (proof/'log').read_text()
        else:
            proof=campaign/'initial_validation'/f'high_effort.{design}'
            proof.mkdir(parents=True,exist_ok=True)
            if not (proof/'verified.json').exists():
                env={k:v for k,v in os.environ.items() if not k.startswith(('EPMAPPING_','MAP_EGG_','EPM_'))}
                env.update(EPM_BIN=str(frozen/'epmapping_check'),EPM_ROOT=str(campaign/'initial_validation'),EPM_TIMEOUT='3600',EPM_RETRIES='1',EPM_CAFFEINATE='0',RAYON_NUM_THREADS='1',PATH=str(frozen)+os.pathsep+os.environ['PATH'])
                flags=dict(manifest['goal'],EPMAPPING_SKIP='1',EPMAPPING_BASELINE_CEC='1',EPMAPPING_SUPER=str(frozen/'database.super'),EPM_GENLIB=str(frozen/'asap7.genlib'))
                cmd=['bash','-c','source "$1"; BENCH="$2"; shift 2; run_engine "$@"','_',str(project/'scripts/experiments/common.sh'),str(frozen/'benchmarks'),str(proof),design,'0','high_effort',*[f'{k}={v}' for k,v in flags.items()]]
                q=subprocess.run(cmd,env=env,cwd=project,capture_output=True,text=True,timeout=3600)
                assert q.returncode==0 and '[i] baseline_cec=1' in (proof/'log').read_text(),q.stderr
                (proof/'validation_row.tsv').write_text(q.stdout)
                (proof/'verified.json').write_text(json.dumps(dict(cec=1,initial_sha256=sha(proof/'epmapping_mapped.v'),log_sha256=sha(proof/'log')),indent=2)+'\n')
        initial=wd/'epmapping_baseline.v';assert initial.read_bytes()==(proof/'epmapping_mapped.v').read_bytes()
        lib=frozen/('asap7_6in.genlib' if stage=='tab_mfs3' else 'asap7.genlib');cost=audit(initial,lib)
        shutil.copy2(initial,wd/'timeout_initial.v')
        metadata=dict(raw_returncode=142,status='optimizer_timeout',post_seconds=budget,selected='initial',area=cost['area'],delay=cost['delay'],cec=1,cec_source=str(proof),cec_log_sha256=sha(proof/'log'),initial_sha256=sha(initial),raw_row_sha256=sha(rowfile),raw_log_sha256=sha(wd/'log'))
        destination.write_text(json.dumps(metadata,indent=2)+'\n');count+=1
    return count

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();print('Prepared initial selections:',prepare(a.root))
