"""Read-only progress while the fixed campaign runs."""
import datetime,json,subprocess,sys,time
from pathlib import Path
here=Path(__file__).resolve().parent
while True:
 q=subprocess.run([sys.executable,str(here/'status.py')],capture_output=True,text=True,check=True);r=json.loads(q.stdout)
 total=sum(r[m][p]['recorded'] for m in ['cold','warm'] for p in ['tab_mfs3','tab_main','ablation'])
 tos=sum(r[m][p]['TO'] for m in ['cold','warm'] for p in ['tab_mfs3','tab_main','ablation'])
 print(datetime.datetime.now().isoformat(timespec='seconds'),total,'/',r['expected_engine_runs'],'TO',tos,'cold',[r['cold'][p]['recorded'] for p in ['tab_mfs3','tab_main','ablation']],'warm',[r['warm'][p]['recorded'] for p in ['tab_mfs3','tab_main','ablation']],flush=True)
 if 'exit_status' in r:print(r['exit_status'],flush=True);break
 time.sleep(45)
