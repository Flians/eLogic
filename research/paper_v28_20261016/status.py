"""Read-only progress for the fixed paired campaign."""
import json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]/'results_v28_three_parts_query_budget'
plan=json.loads((root/'execution_plan.json').read_text())
out={'expected_engine_runs':plan['expected_engine_runs'],'expected_method_runs':plan['expected_method_runs']}
for mode in ['cold','warm']:
 groups={}
 for stage in ['tab_mfs3','tab_main','ablation']:
  records=[p.read_text().strip().split('\t') for p in (root/mode/stage).glob('*/row.tsv')]
  groups[stage]={'recorded':len(records),'normal':sum(r[-1]=='0' for r in records),'TO':sum(r[-1]=='142' for r in records),'errors':[(r[0],r[1],r[-1]) for r in records if r[-1] not in ['0','142']]}
 out[mode]=groups
out['fresh_abc_records']=len(list((root/'cold/tab_mfs3').glob('abc.*/result.json')))
if (root/'execution_status.json').exists():out['exit_status']=json.loads((root/'execution_status.json').read_text())
print(json.dumps(out))
sys.exit(any(out[m][p]['errors'] for m in ['cold','warm'] for p in ['tab_mfs3','tab_main','ablation']))
