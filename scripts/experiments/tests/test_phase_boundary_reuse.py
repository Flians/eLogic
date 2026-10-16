#!/usr/bin/env python3
"""Regression: reuse a physical inverter's input across the cut boundary on usb_phy."""
import json, os, re, subprocess, sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/experiments'))
from audit_mapped_ppa import audit
frozen=ROOT/'results_v20_three_parts_chain/inputs'
env={k:v for k,v in os.environ.items() if not k.startswith(('EPMAPPING_','MAP_EGG_'))}
env.update(ABC=str(frozen/'abc'),RAYON_NUM_THREADS='1',EPMAPPING_BASELINE='high_effort',EPMAPPING_AREA_MAPPING='1',
 EPMAPPING_WINDOW_SYN='0',EPMAPPING_CUT_LIMIT='12',EPMAPPING_RELAX_REQUIRED='100',EPMAPPING_DSE='0',
 EPMAPPING_NO_DELAY_REGRESSION='1',EPMAPPING_DC_TIME_BUDGET='60',EPMAPPING_ROUNDS='6',
 EPMAPPING_OBJECTIVE='area_delay_bound',EPMAPPING_AREA_UNDER_DELAY='1',EPMAPPING_POST_TIMEOUT='1000',
 EPMAPPING_WRITE_BASELINE='1',EPMAPPING_SAT_DC='0',EPMAPPING_EGG_REQ='1',EPMAPPING_CHAIN_FACTOR='1',
 EPMAPPING_CHAIN_BLOCK='2',EPMAPPING_CHAIN_SHARE='1',EPMAPPING_FEASIBLE_COVERS='1',
 EPMAPPING_SUPER=str(frozen/'database.super'),MAP_EGG_PROFILE='1',EPMAPPING_REPLACEMENT_TRACE='1')
with tempfile.TemporaryDirectory(prefix='epm-phase-boundary-') as folder:
 p=Path(folder);env['MAP_EGG_PROFILE_CACHE_DIR']=str(p/'cache')
 q=subprocess.run([str(ROOT/'build/epmapping/epmapping_check'),str(frozen/'asap7.genlib'),str(frozen/'benchmarks/usb_phy.aig'),str(p/'run')],env=env,capture_output=True,text=True,timeout=1100)
 assert q.returncode==0,q.stdout+q.stderr
 assert 'cec=1' in q.stdout and 'window_synth windows=0 applied=0' in q.stdout
 assert '[phase_reuse] bypass_inverter=' in q.stderr
 wd=p/'run';base=audit(wd/'epmapping_baseline.v',frozen/'asap7.genlib');out=audit(wd/'epmapping_mapped.v',frozen/'asap7.genlib')
 assert (wd/'epmapping_baseline.v').read_bytes()==(ROOT/'results_v20_three_parts_chain/tab_main/ae.usb_phy/epmapping_baseline.v').read_bytes()
 assert out['area']<=22.48001 and out['delay']<=base['delay']+.001,(base,out)
 assert out['serialized_delay']<=base['serialized_delay']+.001
 print(json.dumps(dict(initial_area=base['area'],area=out['area'],delay=out['delay'],cec=1)))
print('Original input/mapping unchanged; physical phase reuse improves usb_phy beyond 22.52 with the initial delay bound intact')
