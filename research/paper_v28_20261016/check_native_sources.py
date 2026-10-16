"""Ensure the submitted native implementation matches the measured frozen sources."""
import hashlib,json
from pathlib import Path
project=Path(__file__).resolve().parents[2];campaign=project/'results_v28_three_parts_query_budget';frozen=campaign/'cold/inputs'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checked={}
for source in (frozen/'build_sources').rglob('*'):
 if not source.is_file():continue
 relative=source.relative_to(frozen/'build_sources')
 if relative.suffix not in ['.rs','.hpp','.h','.cpp','.c','.toml'] and relative.name not in ['build.rs','CMakeLists.txt']:continue
 current=project/relative
 assert current.exists() and sha(current)==sha(source),(relative,'native source drift')
 checked[str(relative)]=sha(source)
for name in ['main.cpp','epmapping.hpp','objective.hpp','bounded_cache.hpp','exact_aig_key.hpp','operator_policy.hpp','live_query_cache.hpp','local_truth_table.hpp']:
 source=frozen/name;current=project/'epmapping'/name
 assert sha(source)==sha(current),(name,'native source drift')
 checked[str(current.relative_to(project))]=sha(current)
assert sha(project/'build/epmapping/epmapping_check')==json.loads((campaign/'execution_plan.json').read_text())['engine_sha256']
out=dict(native_sha256=sha(frozen/'epmapping_check'),sources=checked,packaging_note='The new cut-prefix patch packages an unchanged measured header; the patch installer gained this third patch after source freeze.')
Path(__file__).with_name('native_sources_verified.json').write_text(json.dumps(out,indent=2)+'\n');print('Measured native sources match:',len(checked),'files')
