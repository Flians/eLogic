#!/usr/bin/env python3
"""Persistent pools reload across processes and reject corrupted serialized payloads."""
import json, os, re, subprocess, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
ABC=os.environ.get('ABC',str(ROOT/'results_v20_three_parts_chain/inputs/abc'))
LIB='''GATE ZERO 0 Y=CONST0;
GATE ONE 0 Y=CONST1;
GATE BUF 0.1 Y=A;
PIN A NONINV 1 999 1 0 1 0
GATE INV 0.1 Y=!A;
PIN A INV 1 999 1 0 1 0
GATE AND2 1 Y=A*B;
PIN A NONINV 1 999 10 0 10 0
PIN B NONINV 1 999 20 0 20 0
'''
with tempfile.TemporaryDirectory(prefix='epm-profile-') as name:
    p=Path(name);(p/'lib.genlib').write_text(LIB)
    (p/'input.v').write_text('module top(a,b,c,d,y);\ninput a,b,c,d;\noutput y;\nwire x,z;\nassign x = a & b;\nassign z = c & d;\nassign y = x & z;\nendmodule\n')
    q=subprocess.run([ABC,'-c',f'read_verilog {p}/input.v; strash; write_aiger {p}/input.aig'],check=True,capture_output=True,text=True)
    assert (p/'input.aig').exists(),q.stdout+q.stderr
    env={k:v for k,v in os.environ.items() if not k.startswith(('EPMAPPING_','MAP_EGG_'))}
    env.update(ABC=ABC,EPMAPPING_BASELINE='direct',EPMAPPING_AREA_MAPPING='1',EPMAPPING_OBJECTIVE='area_delay_bound',EPMAPPING_WINDOW_SYN='0',EPMAPPING_POST_TIMEOUT='1000',EPMAPPING_SYN='0',EPMAPPING_SUPER='',EPMAPPING_DC='0',EPMAPPING_CHAIN_FACTOR='0',EPMAPPING_ROUNDS='1',EPMAPPING_WRITE_BASELINE='1',MAP_EGG_PROFILE='1',MAP_EGG_PROFILE_CACHE_DIR=str(p/'cache'))
    def run(arm):
        q=subprocess.run([str(ROOT/'build/epmapping/epmapping_check'),str(p/'lib.genlib'),str(p/'input.aig'),str(p/arm)],env=env,capture_output=True,text=True)
        assert q.returncode==0,q.stdout+q.stderr
        assert '[i] cec=1' in q.stdout and 'window_synth windows=0 applied=0' in q.stdout
        match=re.search(r'egraph_profile (.*)',q.stdout);stats={k:int(v) for k,v in (item.split('=') for item in match[1].split())}
        metrics=tuple(map(float,re.search(r'epmapping area ([\d.]+) -> ([\d.]+) delay ([\d.]+) -> ([\d.]+)',q.stdout).groups()));assert metrics[1]<=metrics[0]+.0001 and metrics[3]<=metrics[2]+.001
        return stats
    cold=run('cold');warm=run('warm');assert cold['builds']>0,cold;assert warm['builds']==0 and warm['disk_hits']>0,warm
    assert (p/'cold/epmapping_mapped.v').read_bytes()==(p/'warm/epmapping_mapped.v').read_bytes()
    files=sorted((p/'cache').glob('*.json'));assert files
    for path in files:
        record=json.loads(path.read_text());record['payload_sha256']='0'*64;path.write_text(json.dumps(record))
    corrupt=run('corrupt');assert corrupt['corrupt']>0 and corrupt['builds']>0,corrupt
    assert (p/'cold/epmapping_mapped.v').read_bytes()==(p/'corrupt/epmapping_mapped.v').read_bytes()
    # Timing-model changes must miss even though the Boolean library functions are unchanged.
    (p/'lib.genlib').write_text(LIB.replace('10 0 10 0','11 0 11 0'))
    changed=run('changed');assert changed['builds']>0 and changed['disk_hits']==0,changed
    # The narrow initialization changes search coverage and must not share keys.
    env['MAP_EGG_PROFILE_NARROW']='0'
    wide_only=run('wide_only');assert wide_only['builds']>0 and wide_only['disk_hits']==0,wide_only
    # Boundary-phase retention changes the context-independent pool, so isolate its key.
    env['MAP_EGG_PROFILE_PHASE_KEEP']='1'
    phase_pool=run('phase_pool');assert phase_pool['builds']>0 and phase_pool['disk_hits']==0,phase_pool
    phase_warm=run('phase_warm');assert phase_warm['builds']==0 and phase_warm['disk_hits']>0,phase_warm
print('Cold/warm/corrupt-rebuild outputs match; timing-library changes invalidate persisted pools; all pass CEC and bounds')
