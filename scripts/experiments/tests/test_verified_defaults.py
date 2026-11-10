#!/usr/bin/env python3
"""Verify the native preset equals the frozen full arm, with explicit overrides."""
import json, os, re, subprocess, tempfile, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/experiments'))
from audit_mapped_ppa import audit
cfg=json.loads((ROOT/'epmapping/defaults.json').read_text())
header=(ROOT/'epmapping/defaults.hpp').read_text()
pairs=dict(re.findall(r'\{"([A-Z0-9_]+)", "([^"\n]*)"\}',header))
assert pairs==cfg['defaults']
f=ROOT/cfg['campaign']/'inputs'
goal=json.loads((f.parent/'manifest.json').read_text())['goal']
assert all(goal[k]==v for k,v in pairs.items())
base={k:v for k,v in os.environ.items() if not k.startswith(('EPMAPPING_','MAP_EGG_','EPM_'))}
base.update(ABC=str(f/'abc'),RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1')
with tempfile.TemporaryDirectory(prefix='epmapping-defaults-') as td:
 p=Path(td);rows=[]
 # Full-sized original benchmark, default strong preprocessing; compare with
 # explicit frozen settings and the published final output, cold and warm.
 for mode in ['cold','warm']:
  for label in ['defaults','explicit']:
   w=p/label/mode;env=base.copy();env['MAP_EGG_PROFILE_CACHE_DIR']=str(p/label/'pool')
   if label=='explicit':env.update(goal)
   env['MAP_EGG_PROFILE_CACHE_DIR']=str(p/label/'pool')
   q=subprocess.run([str(ROOT/'build/epmapping/epmapping_check'),str(f/'asap7.genlib'),str(f/'benchmarks/usb_phy.aig'),str(w)],env=env,capture_output=True,text=True,timeout=160)
   assert q.returncode==0 and '[i] cec=1' in q.stdout,(q.stdout,q.stderr)
   assert 'configuration='+cfg['profile'] in q.stdout
   assert 'pre_buffer_cleanup' in q.stdout
   assert (w/'epmapping_baseline.v').read_bytes()==(f.parent/'normalized/strong47/usb_phy.v').read_bytes()
   assert 'post_mapping_timeout=1000' in q.stdout
   assert 'super library: 505598 gates, max_vars=4' in q.stdout
   for flag in ['area_under_delay=1', 'max_rounds=6', 'phase_cost=1', 'phase_fanout=1', 'cofactor_modes=6', 'synth_cache=262144', 'cofactor_pivots=2', 'live_query_memo=9', 'shared_mapped_library=1', 'build_reuse=2', 'operator_policy=3']:
    assert flag in q.stdout,flag
   assert 'tt_functions=0' in q.stdout
   if mode=='warm': assert re.search(r'egraph_profile .*disk_hits=[1-9]',q.stdout),q.stdout
   a=audit(w/'epmapping_mapped.v',f/'asap7.genlib');initial=audit(w/'epmapping_baseline.v',f/'asap7.genlib')
   assert abs(a['area']-18.80)<.0001 and abs(a['delay']-158.68)<.001,a
   assert a['area']<=initial['area']+.0001 and a['delay']<=initial['delay']+.001
   if label=='explicit':assert (w/'epmapping_mapped.v').read_bytes()==(p/'defaults'/mode/'epmapping_mapped.v').read_bytes()
   rows.append(dict(mode=mode,profile=label,A=a['area'],D=a['delay'],CEC=1))
 # Explicit zero preserves the earlier complete-flow input and final costs.
 w=p/'legacy';env=base.copy();env.update(EPMAPPING_PRE_BUFFER_CLEANUP='0',MAP_EGG_PROFILE_CACHE_DIR=str(p/'legacy-pool'))
 q=subprocess.run([str(ROOT/'build/epmapping/epmapping_check'),str(f/'asap7.genlib'),str(f/'benchmarks/usb_phy.aig'),str(w)],env=env,capture_output=True,text=True,timeout=160)
 assert q.returncode==0 and '[i] cec=1' in q.stdout and 'pre_buffer_cleanup' not in q.stdout,(q.stdout,q.stderr)
 initial=audit(w/'epmapping_baseline.v',f/'asap7.genlib');final=audit(w/'epmapping_mapped.v',f/'asap7.genlib')
 assert initial['area']>19.95+.0001 and abs(final['area']-18.80)<.0001 and abs(final['delay']-158.68)<.001
 rows.append(dict(mode='cold',profile='explicit legacy zero',A=final['area'],D=final['delay'],CEC=1))
 # Explicit empty values and budget/cache overrides survive preset application.
 h=p/'test.cpp';h.write_text('#include "epmapping/defaults.hpp"\n#include <cassert>\nint main(){setenv("EPMAPPING_POST_TIMEOUT","7",1);setenv("MAP_EGG_PROFILE_TT_CACHE","1",1);setenv("EPMAPPING_SUPER","",1);epmapping::apply_verified_defaults();assert(std::string(getenv("EPMAPPING_POST_TIMEOUT"))=="7");assert(std::string(getenv("MAP_EGG_PROFILE_TT_CACHE"))=="1");epmapping::apply_verified_resources("missing", "'+str(p/'resources')+'");assert(std::string(getenv("EPMAPPING_SUPER")).empty());assert(std::filesystem::is_directory(getenv("MAP_EGG_PROFILE_CACHE_DIR")));}')
 subprocess.run(['g++','-std=c++17','-I',str(ROOT),str(h),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],env=base,check=True)
 print(json.dumps(dict(preset_matches_campaign=True,overrides_preserved=True,rows=rows),indent=2))
