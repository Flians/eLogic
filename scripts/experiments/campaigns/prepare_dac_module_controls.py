"""Prepare a guarded change set on the ONE main source, after frozen evaluation.

Default: generate an unapplied patch only. --apply requires the complete old
full20 artifact audit to pass; no compiler, mapper or baseline is invoked here.
This adds experimental controls, not a new optimization or default policy.
"""
import argparse
import difflib
import hashlib
import json
import pathlib

R = pathlib.Path(__file__).resolve().parents[3]

PROFILE = '''def dac_module_profile(base, mode):
    """Independent experimental submechanisms; never disable graph validation."""
    if mode not in ('none', 'single-structure', 'rebuild-flow', 'no-fixed-area'):
        raise ValueError('unknown DAC module ablation')
    result = copy.deepcopy(base)
    if mode == 'none':
        return result
    result['dac_module_ablation'] = mode
    env = result['environment']
    if mode == 'single-structure':
        env['EGM_SINGLE_STRUCTURE_CUTS'] = '1'
    elif mode == 'rebuild-flow':
        # Keep the common first complete cover. Ablate steady-state label reuse,
        # not the ability to produce an initial legal mapping before a deadline.
        env['EGM_DAC_REBUILD_FLOW'] = '1'
    else:
        # Joint DelayArea trials stay enabled; this removes only the dedicated
        # fixed-D local and complete-area recovery entries, not all area moves.
        env['EGM_SCHEDULE_RECOVERY'] = '0'
        env['EGM_ONLINE_AREA_RECOVERY'] = '0'
        env['EGM_ONLINE_LEGACY_TAIL'] = '0'
    return result


'''

REBUILD = ''' pub fn rebuild(&mut self,pool:&[Instance],free:&[bool],changed:&[usize]){
  // Reference control: same physical delay relaxation, rebuilt from the entire
  // current pool after each admission. Keep cumulative work counters, complete
  // physical input adjacency and the ORIGINAL deadline/quantum in the caller.
  let(visits,updates)=(self.visits,self.updates);*self=Self::new(free);
  self.visits=visits;self.updates=updates;self.admit(pool,changed);
 }
'''

RUST_TEST = ''' #[test]fn rebuilding_delay_labels_preserves_legal_dag_delay_and_work_counters(){
  for seed in 0..32{
   let mut pool=vec![];for t in 1..10{pool.push(cell(vec![t-1],t,1.+((t+seed)%4)as f32,1.));pool.push(cell(vec![0],t,2.+((t*7+seed)%11)as f32,2.));}
   let free=(0..10).map(|s|s==0).collect::<Vec<_>>();let end=Instant::now()+std::time::Duration::from_secs(5);
   let(mut reuse,mut rebuild)=(Flow::new(&free),Flow::new(&free));
   for len in 1..=pool.len(){reuse.admit(&pool[..len],&[]);let old=rebuild.visits;rebuild.rebuild(&pool[..len],&free,&[]);assert_eq!(old,rebuild.visits);
    while reuse.pending(){assert!(Instant::now()<end);reuse.resume(&pool[..len],&free,1,end);}
    while rebuild.pending(){assert!(Instant::now()<end);rebuild.resume(&pool[..len],&free,1,end);}
    let mut a=reuse.chosen.clone();let mut b=rebuild.chosen.clone();
    if a[9].is_some()&&b[9].is_some(){prune(&pool[..len],&mut a,&free,&[9]).unwrap();prune(&pool[..len],&mut b,&free,&[9]).unwrap();assert_eq!(metrics(&pool[..len],&a,&free,&[9]).1,metrics(&pool[..len],&b,&free,&[9]).1);}
    assert_eq!(reuse.input_users(),rebuild.input_users());
   }
  }
 }
'''

PY_TEST = '''import copy,unittest
from run import dac_module_profile

class DACModules(unittest.TestCase):
    def test_default_has_identical_configuration(self):
        base=dict(environment={'EGM_INCREMENTAL_DELAY':'1'},mapping_seconds=600)
        self.assertEqual(dac_module_profile(base,'none'),base)
    def test_only_target_control_changes(self):
        base=dict(environment={'EGM_INCREMENTAL_DELAY':'1','EGM_JOINT_WINDOWS':'1','EGM_SDC_WINDOW_INPUTS':'0'},mapping_seconds=600)
        old=copy.deepcopy(base)
        expected={'single-structure':{'EGM_SINGLE_STRUCTURE_CUTS':'1'},'rebuild-flow':{'EGM_DAC_REBUILD_FLOW':'1'},'no-fixed-area':{'EGM_SCHEDULE_RECOVERY':'0','EGM_ONLINE_AREA_RECOVERY':'0','EGM_ONLINE_LEGACY_TAIL':'0'}}
        for mode,delta in expected.items():
            actual=dac_module_profile(base,mode)
            self.assertEqual(actual['environment'],dict(base['environment'],**delta))
            self.assertEqual(actual['mapping_seconds'],600)
        self.assertEqual(base,old)
    def test_invalid_rejected(self):
        with self.assertRaises(ValueError):dac_module_profile({'environment':{}},'invalid')

if __name__=='__main__':unittest.main()
'''


def replace_once(text, old, new):
    assert text.count(old) == 1, 'source has drifted; inspect before applying'
    return text.replace(old, new)


def changes():
    paths = ['scripts/emapper/run.py', 'emapper/src/egraph_incremental_delay.rs',
             'emapper/src/egraph_online.rs', 'scripts/emapper/test_dac_modules.py']
    before = {p: (R / p).read_text() if (R / p).exists() else '' for p in paths}
    assert not before[paths[-1]], 'do not overwrite existing tests'
    after = dict(before)
    p = paths[0]
    after[p] = replace_once(after[p], 'def low_overhead_profile(', PROFILE + 'def low_overhead_profile(')
    after[p] = replace_once(after[p], "    ap.add_argument('--observe',", "    ap.add_argument('--dac-module-ablation', choices=['none','single-structure','rebuild-flow','no-fixed-area'], default='none', help='explicit experimental submechanism; main off settings remain unchanged')\n    ap.add_argument('--observe',")
    line = '    profile = recovery_trial_profile(profile,a.trial_safe_area,a.trial_memo,a.trial_timing_dependencies,a.trial_fixed_area,a.trial_release_reuse,a.trial_delta_cost,a.trial_timing_scratch)'
    after[p] = replace_once(after[p], line, line + '\n    profile = dac_module_profile(profile,a.dac_module_ablation)')
    p = paths[1]
    after[p] = replace_once(after[p], ' pub fn input_users(', REBUILD + ' pub fn input_users(')
    after[p] = replace_once(after[p], ' #[test]fn incremental_delay_dag_matches_batch_frontier_and_resumes(){', RUST_TEST + ' #[test]fn incremental_delay_dag_matches_batch_frontier_and_resumes(){')
    p = paths[2]
    after[p] = replace_once(after[p], 'pub(super) struct State {', 'pub(super) struct State {\n dac_rebuild_flow:bool,')
    after[p] = replace_once(after[p], '  Self{match_facts:None,', '  Self{dac_rebuild_flow:std::env::var("EGM_DAC_REBUILD_FLOW").as_deref()==Ok("1"),match_facts:None,')
    after[p] = replace_once(after[p], 'self.flow.admit(&self.pool,&changed);self.joint.inputs_changed', 'if self.dac_rebuild_flow&&self.cover.value().is_some(){self.flow.rebuild(&self.pool,&self.free,&changed);self.flow_witness_changes.invalidate();self.flow_dirty=true;stage_trace("online_ablation_flow_rebuild");}else{self.flow.admit(&self.pool,&changed);}self.joint.inputs_changed')
    after[paths[-1]] = PY_TEST
    return before, after


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=pathlib.Path, required=True)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--audit', type=pathlib.Path,
                    default=R / 'results_v2/emapper_dac_convergence_20261111/full-epfl-final-audit/validation.json')
    a = ap.parse_args()
    assert not a.output.exists(), 'preserve old patch evidence'
    before, after = changes()
    patch = ''.join(''.join(difflib.unified_diff(before[p].splitlines(True), after[p].splitlines(True),
                    fromfile='a/'+p if before[p] else '/dev/null', tofile='b/'+p)) for p in before)
    # Validate Python syntax without changing the actual frozen working source.
    for p in after:
        if p.endswith('.py'):
            compile(after[p], p, 'exec')
    if a.apply:
        audit = json.loads(a.audit.read_text())
        assert audit['completed'] == 20 and not audit['invalid_artifacts']
        assert audit['baseline_runs'] == 0 and not audit['GitHub_push']
        for p, text in after.items():
            assert ((R/p).read_text() if (R/p).exists() else '') == before[p]
            (R/p).write_text(text)
    a.output.mkdir(parents=True)
    (a.output/'controls.patch').write_text(patch)
    (a.output/'preparation.json').write_text(json.dumps(dict(applied=a.apply,
        python_syntax='passed; Rust not compiled and integration CEC not run',
        before_sha256={p:hashlib.sha256(t.encode()).hexdigest() for p,t in before.items()},
        patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
        main_configuration='off, unchanged defaults',baseline_runs=0,GitHub_push=False,
        limitations='three identifiable submechanisms, not disabling essential mapping/validation; '
                    'new paired binary/configuration matrix required; old full20 results stay archived'),indent=2)+'\n')


if __name__ == '__main__':
    main()
