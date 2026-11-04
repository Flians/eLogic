#!/usr/bin/env python3
"""Reproducible experimental native fast mapping, with external named-port CEC.

All native preparation/search/publication is timed, no external mapper seed is
used. The 600s outer guard is only a safety limit. This is not the old profile.
"""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
from provenance import ROOT,source_files,hashes,sha,git_revision
from physical import measure_cover
from verify import verify

EXTRA=[ROOT/'src/bin/emapper_fast.rs',ROOT/'scripts/emapper/build_fast.sh',ROOT/'scripts/emapper/run_fast.py',ROOT/'scripts/emapper/physical.py',ROOT/'scripts/emapper/verify.py',ROOT/'scripts/emapper/provenance.py']
def source():return hashes(source_files()+EXTRA)
def read(p):return json.loads(Path(p).read_text())
def dump(p,j):
 p=Path(p);q=p.with_suffix('.pending');q.write_text(json.dumps(j,indent=2)+'\n');q.replace(p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('input',type=Path);ap.add_argument('--golden',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--cut-frontier',type=int,default=16);ap.add_argument('--critical-quotients',choices=['off','on'],default='on');ap.add_argument('--critical-cuts',choices=['off','on'],default='off');ap.add_argument('--mapped-cut-rank',choices=['off','on'],default='off');ap.add_argument('--shared-area',choices=['off','on'],default='off');ap.add_argument('--shared-prune',choices=['off','on'],default='off');ap.add_argument('--area-passes',type=int,default=2);a=ap.parse_args();folder=a.output.resolve();assert not folder.exists(),'retain old cases, use new output path'
 binary=ROOT/'target/release/emapper_fast';measure=ROOT/'target/release/emapper_measure';receipt=read(ROOT/'target/release/emapper-fast-build.json');assert receipt['source']==source() and receipt['binaries']==hashes([binary,measure]),'rebuild with scripts/emapper/build_fast.sh'
 inp=a.input.resolve();golden=a.golden.resolve();library=ROOT/'emapper/libraries/asap7/asap7_full.genlib';folder.mkdir(parents=True);profile=dict(engine='experimental-native-fast',cut_frontier=a.cut_frontier,critical_quotients=a.critical_quotients,critical_cuts=a.critical_cuts,mapped_cut_rank=a.mapped_cut_rank,shared_area=a.shared_area,shared_prune=a.shared_prune,area_passes=a.area_passes,max_small_inputs=6,mapping_safety_seconds=600,external_seed_used=False,external_seed_mapping_seconds=0)
 out=folder/'mapped.v';native=folder/'native-report.json';argv=[str(binary),str(library),str(inp),str(out),str(native),str(a.cut_frontier),'1'if a.critical_quotients=='on'else'0',str(a.area_passes),'1'if a.critical_cuts=='on'else'0','1'if a.mapped_cut_rank=='on'else'0','1'if a.shared_area=='on'else'0','1'if a.shared_prune=='on'else'0'];tools={k:dict(path=shutil.which(k),sha256=sha(shutil.which(k)))for k in ('yosys','abc')};r=dict(schema=1,status='running',git_head=git_revision(),configuration=profile,input=str(inp),input_sha256=sha(inp),golden=str(golden),golden_sha256=sha(golden),library_sha256=sha(library),code_identity=receipt,argv=argv,tools=tools,CEC_in_mapping_clock=False,baseline_invoked=False);dump(folder/'report.json',r)
 start=time.monotonic();wall=time.time();exitcode=None;timeout=False
 with (folder/'mapping.stdout').open('w')as o,(folder/'mapping.stderr').open('w')as e:
  try:exitcode=subprocess.run(argv,cwd=ROOT,stdout=o,stderr=e,timeout=600).returncode
  except subprocess.TimeoutExpired:timeout=True
 elapsed=time.monotonic()-start;r.update(mapping_wall_seconds=elapsed,native_exit=exitcode,search_timeout=timeout)
 try:
  publication=read(folder/'fast-publication.json');chosen=Path(publication['netlist']);assert publication['library_sha256']==r['library_sha256'] and publication['input_sha256']==r['input_sha256'];assert sha(chosen)==publication['netlist_sha256'];assert publication['elapsed_seconds']<600 and chosen.stat().st_mtime<=wall+600,'late publication'
  # Always deliver the last complete version, never an interrupted final copy.
  shutil.copyfile(chosen,out);r['publication']=publication;r['native']=read(native)if native.exists()else None
  measured=measure_cover(library,out,folder/'physical',measure,tools['yosys']['path']);r['measured']=measured['measured'];r['metric_agreement']=all(measured['measured'][k]==publication[v]for k,v in [('area','A'),('delay','D')]);assert r['metric_agreement'],'native/public metric mismatch'
  os.environ['EGM_YOSYS']=tools['yosys']['path'];os.environ['BENCH_RS_EMAP_ABC']=tools['abc']['path'];r['CEC']=int(verify(library.parent/'cells.v',golden,measured['netjson'],folder/'cec'));assert r['CEC']==1,'final named-port CEC failure'
  r['status']='ok'if exitcode==0 and not timeout else 'mapping_failed_with_verified_incumbent';r['search_finished']=bool(r['native']and r['native']['stats']['search_finished']);r['delivery_sha256']=sha(out)
 except Exception as e:r.update(status='failed',error=type(e).__name__+': '+str(e))
 r['source_unchanged']=receipt['source']==source();assert r['source_unchanged'];r['end_to_end_wall_seconds']=time.monotonic()-start;r['artifacts']={str(p.relative_to(folder)):sha(p)for p in folder.rglob('*')if p.is_file()and p.name not in ('report.json','report.pending')};dump(folder/'report.json',r);print(json.dumps({k:r.get(k)for k in ('status','mapping_wall_seconds','end_to_end_wall_seconds','measured','CEC','error')}));return 0 if r['status']=='ok'else 1
if __name__=='__main__':sys.exit(main())
