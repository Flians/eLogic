#!/usr/bin/env python3
"""Method-only arbiter experiment; every run must reproduce the original mapped input."""
import csv,hashlib,json,os,shutil,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results_v13_dependency_method'
PROFILES={
 'control':{'EPMAPPING_DEPENDENCY_CUTS':'0'},
 'diagnostic':{'EPMAPPING_DEPENDENCY_CUTS':'1'},
 'feasible':{'EPMAPPING_DEPENDENCY_CUTS':'0','EPMAPPING_FEASIBLE_COVERS':'1'},
 'internal':{'EPMAPPING_DEPENDENCY_CUTS':'1','EPMAPPING_DEPENDENCY_INTERNAL':'1'},
 'combined':{'EPMAPPING_DEPENDENCY_CUTS':'1','EPMAPPING_DEPENDENCY_INTERNAL':'1','EPMAPPING_FEASIBLE_COVERS':'1'},
}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 f=OUT/'final_inputs';f.mkdir(parents=True,exist_ok=True)
 manifest=OUT/'manifest.json'
 if not manifest.exists():
  for source in [ROOT/'build/epmapping/epmapping_check',ROOT/'epmapping/epmapping.hpp',ROOT/'epmapping/main.cpp',ROOT/'epmapping/objective.hpp',ROOT/'scripts/experiments/common.sh',Path(__file__),ROOT/'results_v10_three_parts_1000/inputs/abc',ROOT/'results_v10_three_parts_1000/inputs/asap7.genlib',ROOT/'results_v10_three_parts_1000/inputs/database.super',ROOT/'results_v11_arbiter/inputs/arbiter.aig']:
   shutil.copy2(source,f/source.name)
  manifest.write_text(json.dumps(dict(profiles=PROFILES,post_timeout=1000,window_synthesis=False,input_recipe='fraig; resyn2rs; fraig; compress2rs',sha256={p.name:sha(p) for p in f.iterdir()}),indent=2)+'\n')
 m=json.loads(manifest.read_text());assert m['profiles']==PROFILES
 for n,h in m['sha256'].items():assert sha(f/n)==h,n
 def run(name):
  row=OUT/(name+'.tsv')
  if row.exists():return row.read_text()
  env={k:v for k,v in os.environ.items() if not k.startswith(('EPMAPPING_','EPM_','MAP_EGG_'))}
  env.update(ABC=str(f/'abc'),EPM_BIN=str(f/'epmapping_check'),EPM_ROOT=str(OUT),EPM_TIMEOUT='3600',EPM_RETRIES='1',EPM_CAFFEINATE='0',RAYON_NUM_THREADS='1')
  knobs=dict(EPMAPPING_OBJECTIVE='area_delay_bound',EPMAPPING_AREA_UNDER_DELAY='1',EPMAPPING_WINDOW_SYN='0',EPMAPPING_POST_TIMEOUT='1000',EPMAPPING_WRITE_BASELINE='1',EPMAPPING_SAT_DC='0',EPMAPPING_EGG_REQ='1',EPMAPPING_DSE='0',EPMAPPING_SUPER=str(f/'database.super'),EPM_GENLIB=str(f/'asap7.genlib'),**PROFILES[name])
  wd=OUT/'runs'/(name+'.arbiter')
  if knobs['EPMAPPING_DEPENDENCY_CUTS']=='1': knobs['EPMAPPING_DEPENDENCY_TRACE']=str(wd/'dependency.csv')
  command=['bash','-c','source "$1"; BENCH="$2"; shift 2; run_engine "$@"','_',str(f/'common.sh'),str(f),str(wd),'arbiter','6','high_effort',*[k+'='+v for k,v in knobs.items()]]
  proc=subprocess.run(command,capture_output=True,text=True,env=env,cwd=ROOT)
  if proc.returncode:raise RuntimeError(proc.stderr)
  text=proc.stdout.strip().splitlines()[-1]+'\n';row.write_text(text);print(text.strip(),flush=True);return text
 with ThreadPoolExecutor(max_workers=3) as pool: rows=list(pool.map(run,PROFILES))
 header='stage design recipe rounds base_a base_d out_a out_d dA% dD% rnd cec secs rc'.split()
 (OUT/'summary.tsv').write_text('\t'.join(header)+'\n'+''.join(rows))
 records=list(csv.DictReader((OUT/'summary.tsv').open(),delimiter='\t'))
 original=ROOT/'results_v10_three_parts_1000/tab_main/ae.arbiter/epmapping_baseline.v'
 for r in records:
  wd=OUT/'runs'/(r['stage']+'.arbiter');log=(wd/'log').read_text()
  assert (wd/'epmapping_baseline.v').read_bytes()==original.read_bytes(),r['stage']
  assert 'post_mapping_timeout=1000' in log and 'area_under_delay=1, path_objective=0' in log
  if r['rc']=='0':
   assert r['cec']=='1' and '[i] cec=1' in log and 'window_synth windows=0 applied=0' in log
   assert float(r['out_d'])<=float(r['base_d'])+.001 and float(r['out_a'])<=float(r['base_a'])+.0001
  else:assert r['rc']=='142'
  if (wd/'dependency.csv').exists():
   q=subprocess.run(['python',str(ROOT/'scripts/experiments/analyze_dependency_trace.py'),str(wd/'dependency.csv')],check=True,capture_output=True,text=True)
   (wd/'dependency_stats.json').write_text(q.stdout)
 (OUT/'verified.json').write_text(json.dumps(dict(records=records,original_baseline_sha256=sha(original),raw_sha256={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name in ['log','epmapping_baseline.v','epmapping_mapped.v','experiment_env.txt','manifest.json','dependency.csv','dependency_stats.json']}),indent=2)+'\n')
if __name__=='__main__':main()
