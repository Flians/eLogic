#!/usr/bin/env python3
"""Establish ABC's actual buffer-free start, preserving pins and physical sharing."""
import argparse,concurrent.futures,json,os
from pathlib import Path
from run_epmapping_supplement import call,sha,save,audit_abc
from analyze_epmapping_supplement import restore_abc_output_aliases
from exact_core_graph import exact_core_fingerprint
from audit_mapped_ppa import audit

def verify(root):
 def one(p):
  lib=root/'inputs/asap7_6in.genlib';w=root/'normalization_abc_blif'/p.parent.name/p.stem;w.mkdir(parents=True,exist_ok=True)
  cmd=f'read_genlib {lib}; read_blif -c {p}; short_names; print_stats; write_verilog {w/"import.v"}'
  rc,secs=call([str(root/'inputs/abc'),'-c',cmd],dict(os.environ),w/'log',3600);assert rc==0
  text,aliases=restore_abc_output_aliases((w/'import.v').read_text(),lib);out=w/'import_aliases.v';out.write_text(text)
  assert exact_core_fingerprint(p.with_suffix('.v'),lib)==exact_core_fingerprint(out,lib),p
  a=audit(p.with_suffix('.v'),lib);b=audit_abc(out,lib);assert abs(a['area']-b['area'])<.0001 and abs(a['delay']-b['delay'])<.001,(p,a,b)
  row=dict(cohort=p.parent.name,design=p.stem,shared_sha256=sha(p),imported_core_identical=True,area=a['area'],delay=a['delay'],import_log_sha256=sha(w/'log'),imported_sha256=sha(out),command=cmd)
  save(w/'proof.json',row);return row
 files=list((root/'normalized/direct6').glob('*.blif'))+list((root/'normalized/strong6').glob('*.blif'));assert len(files)==32
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(one,files))
 result=dict(complete=True,records=rows,import_protocol='read_blif -c; short_names; gate/core/pin/ordered-CO identity; no logic normalization')
 save(root/'abc_import_proofs.json',result);print('Verified ABC imports of all 32 shared restricted mappings',flush=True);return result
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();verify(a.root.resolve())
