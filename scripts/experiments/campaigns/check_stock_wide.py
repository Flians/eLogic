"""Validate unchanged stock emap wide-capacity wrapper, normalization and named-port CEC."""
import argparse,json,subprocess,sys,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from port_identifiers import escape_indexed_scalars

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root.resolve();out=r/'stock_emap9_cut9_reference/wide_fixture_validation';out.mkdir(exist_ok=False)
 native=r/'stock_emap9_cut9_reference/build/baseline_mapper';lib=r/'asap7_full.genlib';ev=r/'source_kernels/target/release/bench_rs_wide_eval';rows=[];commands=[]
 def run(argv,log,timeout=60):
  commands.append(argv);start=time.monotonic()
  with log.open('w')as stream:p=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
  return p.returncode,time.monotonic()-start
 assert run([str(ev),str(lib),'--models',str(out/'cells.v'),str(out/'library.json')],out/'models.log')[0]==0
 for f in sorted((r/'wide_cell_cec_kernel').glob('*.aag')):
  p=out/f.stem;p.mkdir();rc,_=run(['yosys','-Q','-T','-p',f'read_aiger {f}; write_aiger -symbols {p/"gold.aig"}'],p/'convert.log');row={'cell':f.stem,'conversion_exit':rc}
  if not rc:
   rc,secs=run([str(native),str(lib),str(p/'gold.aig'),str(p/'raw.v'),'area','emap'],p/'map.log');row.update(map_exit=rc,mapping_wall_secs=secs)
   if not rc:
    raw=p/'raw.v';esc=p/'escaped.v';esc.write_text(escape_indexed_scalars(raw.read_text()));row['port_adapter']={'kind':'necessary indexed-scalar escaping','raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'escaped_sha256':hashlib.sha256(esc.read_bytes()).hexdigest()}
    script=f'read_verilog -lib {out/"cells.v"}; read_verilog {esc}; hierarchy -check -auto-top; rename -top mapped; write_verilog -noattr -noexpr {p/"mapped.v"}; write_json {p/"mapped.json"}'
    rc,_=run(['yosys','-Q','-T','-p',script],p/'normalize.log');row['normalization_exit']=rc
    if not rc:
     rc,_=run([str(ev),str(lib),str(p/'gold.aig'),str(p/'mapped.v'),str(p/'mapped.json')],p/'evaluate.log');row['evaluate_exit']=rc
     if not rc:
      records=[json.loads(l)for l in (p/'evaluate.log').read_text().splitlines()if l.startswith('{')];row['public_metrics']=records[-1]
      d=json.loads((p/'mapped.json').read_text());top=max(d['modules'].values(),key=lambda x:len(x.get('cells',{})));row['selected_wide']=[c['type']for c in top['cells'].values()if sum(v=='input'for v in c['port_directions'].values())>6]
  rows.append(row)
 (out/'validation.json').write_text(json.dumps({'scope':'actual-cell raw-structure fixtures, not EPFL PPA or complete structural coverage','library_capacity':9,'structural_cut_capacity':9,'stock_algorithm_modified':False,'commands':commands,'rows':rows},indent=2))
 print([(q['cell'],q.get('evaluate_exit'),q.get('selected_wide'))for q in rows]);assert len(rows)==7 and all(q.get('public_metrics',{}).get('cec')==1 for q in rows);assert any(q.get('selected_wide')for q in rows),'wide matching not activated'
if __name__=='__main__':main()
