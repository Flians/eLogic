"""Regression: new cut structures and shared physical phases improve frozen usb_phy."""
import hashlib,json,os,re,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/experiments'))
from audit_mapped_ppa import audit
original=ROOT/'results_v20_three_parts_chain/inputs'
from run_three_parts import GOAL
reference=ROOT/'results_v20_three_parts_chain'
env={k:v for k,v in os.environ.items() if not k.startswith(('EPMAPPING_','MAP_EGG_'))}
env.update(GOAL)
# Test complete generator coverage independently of the default two-pivot budget.
env['EPMAPPING_COFAC_PIVOTS']='0'
env['MAP_EGG_PROFILE']='1'
env.update(ABC=str(original/'abc'),RAYON_NUM_THREADS='1',EPMAPPING_BASELINE='high_effort',EPMAPPING_AREA_MAPPING='1',
 EPMAPPING_CUT_LIMIT='12',EPMAPPING_RELAX_REQUIRED='100',EPMAPPING_NO_DELAY_REGRESSION='1',
 EPMAPPING_DC_TIME_BUDGET='60',EPMAPPING_ROUNDS='6',EPMAPPING_SUPER=str(original/'database.super'),
 EPMAPPING_COFAC='6',EPMAPPING_COFAC_MIN_LEAVES='3',EPMAPPING_PHASE_COST='1',EPMAPPING_PHASE_FANOUT='1',
 MAP_EGG_PROFILE_PHASE_KEEP='0',EPMAPPING_REPLACEMENT_TRACE='1')
with tempfile.TemporaryDirectory(prefix='epm-cofactor-') as folder:
 p=Path(folder);env['MAP_EGG_PROFILE_CACHE_DIR']=str(p/'cache')
 q=subprocess.run([str(ROOT/'build/epmapping/epmapping_check'),str(original/'asap7.genlib'),str(original/'benchmarks/usb_phy.aig'),str(p/'run')],env=env,capture_output=True,text=True,timeout=1100)
 assert q.returncode==0,q.stdout+q.stderr
 assert 'cec=1' in q.stdout and 'window_synth windows=0 applied=0' in q.stdout and 'post_mapping_timeout=1000' in q.stdout
 assert '[phase_reuse] fanout=' in q.stderr
 stat=re.search(r'epmapping cofactor .* replacements=(\d+)',q.stdout);assert stat and int(stat[1])>0
 assert (p/'run/epmapping_baseline.v').read_bytes()==(reference/'tab_main/ae.usb_phy/epmapping_baseline.v').read_bytes()
 a=audit(p/'run/epmapping_mapped.v',original/'asap7.genlib');b=audit(p/'run/epmapping_baseline.v',original/'asap7.genlib')
 assert a['area']<=21.38001 and a['delay']<=158.681 and a['serialized_delay']<=b['serialized_delay']+.001,(a,b)
 print(json.dumps(dict(area=a['area'],delay=a['delay'],initial_area=b['area'],initial_delay=b['delay'],cec=1,cofactor_replacements=int(stat[1]),netlist_sha256=hashlib.sha256((p/'run/epmapping_mapped.v').read_bytes()).hexdigest())))
print('New cut candidates and shared-phase materialization reduce usb_phy area with unchanged original input and delay bound')
