"""Commit only the audited ePMapping delivery to the two local repositories; never push."""
import hashlib,json,subprocess
from pathlib import Path
project=Path(__file__).resolve().parents[2]
research=Path(__file__).resolve().parent
root=project/'results_v28_three_parts_query_budget'
manifest=json.loads((root/'delivery_manifest.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(project/manifest['pdf'])==manifest['pdf_sha256']
for rel,h in manifest['result_sha256'].items():assert sha(root/rel)==h,rel
paths=json.loads((research/'commit_paths.json').read_text())
def git(args,cwd=project,output=False):
 command=['git','-c','filter.lfs.process=','-c','filter.lfs.required=false',*args]
 if output:return subprocess.check_output(command,cwd=cwd,text=True).strip()
 subprocess.run(command,cwd=cwd,check=True)
# Preserve every staged entry outside the requested delivery, including pybind11.
before=git(['diff','--cached','--raw','--','.',*[f':(exclude){p}' for p in paths]],output=True)
assert all(line.endswith('\ttools/pybind11') for line in before.splitlines()),before
paper=project/'epmapping_paper'
git(['add','--','DATE27-ePMapping-sub'],cwd=paper)
git(['diff','--cached','--check'],cwd=paper)
if git(['diff','--cached','--name-only','--','DATE27-ePMapping-sub'],cwd=paper,output=True):
 git(['commit','--only','-m','Update DATE paper with paired query-reuse and pivot-budget experiments','--','DATE27-ePMapping-sub'],cwd=paper)
manifest['paper_commit']=git(['rev-parse','HEAD'],cwd=paper,output=True)
(root/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
git(['add','--',*paths])
# Frozen generated headers and the historical lockfile can match global ignores.
references=json.loads((research/'original_inputs_verified.json').read_text())
for mode in ['cold','warm']:
 base=root/mode
 frozen=json.loads((base/'manifest.json').read_text())['frozen_sha256']
 references.update({str((base/rel).relative_to(project)):h for rel,h in frozen.items()})
tracked=set(git(['ls-files','-z'],output=True).split('\0'))
missing=[p for p in references if p not in tracked]
if missing:git(['add','-f','--',*missing])
# Keep the archived genlib fixture byte-identical; its aligned GATE records are data.
git(['diff','--cached','--check','--','epmapping','map_egg','scripts/experiments','scripts/apply_mockturtle_patches.sh','scripts/compare_with_aspdac25.sh',':(exclude)map_egg/tests/data'])
git(['commit','--only','-m','Archive complete paired ePMapping experiments and query-budget ablations','--',*paths])
after=git(['diff','--cached','--raw'],output=True)
assert before==after,dict(before=before,after=after)
print(json.dumps(dict(code_commit=git(['rev-parse','HEAD'],output=True),paper_commit=manifest['paper_commit'],preexisting_index_preserved=True,pushed=False)))
