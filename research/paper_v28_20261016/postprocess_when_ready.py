"""Start full validation only after both paired drivers report success."""
import json,subprocess,sys,time
from pathlib import Path
project=Path(__file__).resolve().parents[2]
root=project/'results_v28_three_parts_query_budget'
while not (root/'execution_status.json').exists():time.sleep(15)
status=json.loads((root/'execution_status.json').read_text())
assert status==dict(cold=0,warm=0),status
with (root/'finalization_driver.log').open('w') as log:
 for command in [[sys.executable,'scripts/experiments/finalize_build_campaign.py','--root',str(root)],
                 [sys.executable,'research/paper_v28_20261016/analyze_controls.py']]:
  subprocess.run(command,cwd=project,stdout=log,stderr=subprocess.STDOUT,check=True)
print('Full numerical validation and manuscript build finished; visual QA remains.',flush=True)
