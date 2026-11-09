#!/usr/bin/env python3
"""Fixed single-output (D,A) delivery profile. Input must already be prepared AIG."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from frontend import verify_receipt, OBJECTIVE_POLICY
from provenance import ROOT, sha, hashes, source_files, git_revision
from verify import verify
from physical import measure_cover, contribution


def dump(path, data):
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(data, indent=2)+'\n')
    temp.replace(path)


def safe_path(path):
    # Paths enter Yosys/ABC command strings after mapping, without a shell.
    if any(c in str(path) for c in '\n\r;"\\'):
        raise ValueError('unsupported tool-script path: '+str(path))
    return path


def select_output(raw, native_info, library, subject, native):
    """Describe the output actually selected by the outer deadline supervisor.

    The child may be killed before replacing its initial metadata. Never infer
    the selected cover from that metadata; only a verified receipt authorizes an
    optimized output. Keep observed child metadata separately in the report.
    """
    effective = copy.deepcopy(native_info)
    receipt = raw.with_suffix('.incumbent.receipt.json')
    if receipt.exists():
        chosen = verify_receipt(receipt, library, subject, native)
        shutil.copyfile(chosen['netlist'], raw)
        generated=chosen['kind']=='complete-generated-cover'
        effective.update(optimizer_succeeded=generated, selected='validated_online_incumbent'if generated else'native_source_seed')
        selection = dict(kind='optimized'if generated else'source_only', basis='verified_predeadline_receipt',
                         raw_sha256=sha(raw), incumbent=chosen)
    else:
        if native_info.get('direct_native'):
            raise RuntimeError('direct native mapping produced no verified complete cover')
        if not raw.exists() or sha(raw) != native_info.get('source_sha256'):
            raise RuntimeError('no verified incumbent or unchanged source cover')
        effective.update(optimizer_succeeded=False, selected='source_witness')
        selection = dict(kind='source_only', basis='source_cover_hash', raw_sha256=sha(raw))
    return effective, selection


def ablation_profile(base, ablation):
    """Explicit same-binary variants; preserve all clock/library/cut settings."""
    if ablation not in (None, 'joint-off', 'joint-filtered', 'joint-area-only', 'functional-stream', 'functional-sparse', 'functional-fine', 'functional-lazy', 'source-editable', 'source-coalesced'):
        raise ValueError('unknown explicit experimental configuration')
    profile = copy.deepcopy(base)
    if ablation is not None:
        profile['name'] += '-ablation-' + ablation
        profile['status'] = 'explicit same-binary joint-search ablation'
        if ablation in ('functional-stream','functional-sparse','functional-fine','functional-lazy', 'source-editable', 'source-coalesced'):
            profile['environment']['EGM_FUNCTIONAL_CURSOR'] = '1'
            if ablation in ('functional-sparse','functional-fine','functional-lazy', 'source-editable', 'source-coalesced'):
                profile['environment']['EGM_PRIMITIVE_SPARSE'] = '1'
            if ablation in ('functional-fine','functional-lazy', 'source-editable', 'source-coalesced'):
                profile['environment']['EGM_PROOF_AWARE_JOINT'] = '1'
            if ablation in ('functional-lazy','source-editable', 'source-coalesced'):
                profile['environment']['EGM_SDC_LAZY_LAYOUT'] = '1'
            if ablation in ('source-editable','source-coalesced'):profile['environment']['EGM_EDITABLE_SOURCE'] = '1'
            if ablation == 'source-coalesced':
                profile['environment']['EGM_COALESCE_RECOVERY_PUBLICATION'] = '1'
                profile['recovery_publication_policy'] = 'strict-delay-immediate-same-delay-area-coalesced-v1'
            profile['status'] = 'experimental full-care functional cursor; full EPFL acceptance pending'
            return profile
        if ablation == 'joint-area-only':
            profile['environment']['EGM_JOINT_DELAY_AREA'] = '0'
            return profile
        profile['environment']['EGM_JOINT_WINDOWS'] = '0' if ablation == 'joint-off' else '1'
        profile['environment']['EGM_JOINT_TARGET_GROUPS'] = '0'
    return profile


def search_queues_complete(queues):
    # Internal validated state and externally receipted delivery are distinct.
    return bool(queues) and not any([
        queues.get('pending_small_matchers',0), queues.get('flow_pending',False),
        queues.get('library_pending',False), queues.get('joint_windows',{}).get('pending',False),
        queues.get('publication_state',{}).get('pending',False),
        queues.get('recovery_schedule',{}).get('pending',False)])


def recovery_input_profile(base, mode):
    """Explicit same-binary arrival-screening ablation; the clock is unchanged."""
    if mode not in ('legacy', 'overlay'):
        raise ValueError('unknown recovery input policy')
    profile = copy.deepcopy(base)
    profile['environment']['EGM_RECOVERY_INPUT_OVERLAY'] = '1' if mode == 'overlay' else '0'
    profile['recovery_input_policy'] = mode
    return profile


def recovery_noop_profile(base, mode):
    if mode not in ('off', 'on'):
        raise ValueError('unknown recovery noop mode')
    profile = copy.deepcopy(base)
    profile['environment']['EGM_RECOVERY_NOOP_PREFILTER'] = '1' if mode == 'on' else '0'
    profile['recovery_noop_prefilter'] = mode == 'on'
    return profile


def cover_materialization_profile(base, mode):
    """Change cover compaction/publication only; keep candidate production intact."""
    if mode not in ('eager', 'coalesced', 'deferred', 'event'):
        raise ValueError('unknown cover materialization mode')
    profile = copy.deepcopy(base)
    profile['environment']['EGM_DEFER_COVER_MATERIALIZATION'] = '1' if mode in ('deferred','event') else '0'
    profile['environment']['EGM_COALESCE_MATCH_PUBLICATION'] = '1' if mode == 'event' else '0'
    profile['environment']['EGM_COALESCE_RECOVERY_PUBLICATION'] = '0' if mode == 'eager' else '1'
    profile['cover_materialization'] = mode
    return profile


def frontier_preparation_profile(base, mode):
    """Reuse immutable physical incidences only; rebuild numerical labels."""
    if mode not in ('off', 'on'):
        raise ValueError('unknown frontier preparation mode')
    profile = copy.deepcopy(base)
    profile['environment']['EGM_FRONTIER_PREPARATION'] = '1' if mode == 'on' else '0'
    profile['frontier_preparation'] = mode
    return profile


def batch_frontier_profile(base, mode):
    """Memoize only the unchanged physical-prefix numerical batch witness."""
    if mode not in ('off', 'on'):
        raise ValueError('unknown batch frontier cache mode')
    profile = copy.deepcopy(base)
    profile['environment']['EGM_BATCH_FRONTIER_CACHE'] = '1' if mode == 'on' else '0'
    profile['batch_frontier_cache'] = mode
    return profile


def cut_cost_profile(base, enabled):
    """Diagnostic cost scopes only; never alter candidate or budget policy."""
    if not isinstance(enabled, bool):
        raise ValueError('cut cost detail must be boolean')
    profile = copy.deepcopy(base)
    profile['environment']['EGM_CUT_COST_DETAIL'] = '1' if enabled else '0'
    profile['cut_cost_detail'] = enabled
    return profile


def cut_truth_profile(base, mode):
    """Exact word operations in the event closure, without task rescheduling."""
    if mode not in ('legacy', 'word'):
        raise ValueError('unknown cut truth mode')
    profile = copy.deepcopy(base)
    profile['environment']['EGM_CUT_TRUTH_WORD'] = '1' if mode == 'word' else '0'
    profile['cut_truth_policy'] = mode
    return profile


def cut_join_profile(base, mode):
    """Avoid unowned tuple allocation; preserve legacy task order and visits."""
    if mode not in ('legacy', 'streamed'):
        raise ValueError('unknown cut join mode')
    profile = copy.deepcopy(base)
    profile['environment']['EGM_CUT_JOIN_STREAMED'] = '1' if mode == 'streamed' else '0'
    profile['cut_join_policy'] = mode
    return profile


def initial_cover_profile(base, mode):
    """Native warm start retains the complete main candidate/recovery pipeline."""
    if mode not in ('source', 'native-delay', 'direct'):
        raise ValueError('unknown initial cover mode')
    if mode == 'native-delay' and base['environment'].get('EGM_EDITABLE_SOURCE') != '1':
        raise ValueError('native delay cover requires source-editable ablation')
    profile = copy.deepcopy(base)
    profile['environment']['EGM_NATIVE_INITIAL_DELAY'] = '1' if mode == 'native-delay' else '0'
    profile['environment']['EGM_DIRECT_NATIVE'] = '1' if mode == 'direct' else '0'
    if mode == 'direct':
        profile['environment']['EGM_EDITABLE_SOURCE'] = '0'
    profile['initial_cover_policy'] = mode
    return profile


def flow_witness_profile(base, mode):
    """Same witness semantics; accumulate every Flow delta before score deferral."""
    if mode not in ('legacy', 'delta'):
        raise ValueError('unknown flow witness mode')
    profile = copy.deepcopy(base)
    profile['environment']['EGM_FLOW_WITNESS_DELTA'] = '1' if mode == 'delta' else '0'
    profile['flow_witness_policy'] = mode
    return profile


def recovery_cache_profile(base, mode):
    if mode not in ('legacy', 'split', 'delta', 'lifecycle'):
        raise ValueError('unknown recovery cache mode')
    profile = copy.deepcopy(base)
    profile['environment']['EGM_RECOVERY_CACHE_SPLIT'] = '1' if mode == 'split' else '0'
    profile['environment']['EGM_RECOVERY_CACHE_DELTA'] = '1' if mode == 'delta' else '0'
    profile['environment']['EGM_RECOVERY_CACHE_LIFECYCLE'] = '1' if mode == 'lifecycle' else '0'
    profile['recovery_cache_policy'] = mode
    return profile



def recovery_schedule_profile(base, mode):
    if mode not in ('legacy', 'fair'):
        raise ValueError('unknown recovery schedule mode')
    profile = copy.deepcopy(base)
    profile['environment']['EGM_RECOVERY_FAIR'] = '1' if mode == 'fair' else '0'
    profile['recovery_schedule_policy'] = mode
    return profile


def window_sdc_profile(base, inputs):
    """Explicit exhaustive-window care ablation; does not change other budgets."""
    if not isinstance(inputs, int) or isinstance(inputs, bool) or not 0 <= inputs <= 20:
        raise ValueError('window SDC inputs must be an integer in [0,20]')
    profile = copy.deepcopy(base)
    profile['environment']['EGM_SDC_WINDOW_INPUTS'] = str(inputs)
    profile['sdc_window_inputs'] = inputs
    if inputs:
        profile['name'] += '-window-sdc-'+str(inputs)
    return profile


def monotone_sdc_profile(base, enabled):
    """Explicit supplement policy, independent of window width and all budgets."""
    if not isinstance(enabled, bool):
        raise ValueError('monotone SDC selection must be boolean')
    profile = copy.deepcopy(base)
    profile['environment']['EGM_SDC_MONOTONE'] = str(int(enabled))
    profile['sdc_care_policy'] = 'pi-exact-then-window-supplement' if enabled else 'legacy-selector'
    if enabled:
        profile['name'] += '-monotone-sdc'
    return profile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('aig', type=Path, help='named, combinational binary AIGER; no preprocessing is run')
    ap.add_argument('--output', type=Path, required=True, help='new directory; existing results are never overwritten')
    ap.add_argument('--golden', type=Path, help='optional original named AIG for independent CEC; default: input')
    ap.add_argument('--fresh-emap-seed', type=Path, help='unchanged emap-delay executable; generate seed and charge its process time within the same 600s budget; mutually exclusive with --seed-verilog')
    ap.add_argument('--seed-verilog', type=Path, help='provided complete cover: canonical metrics and independent CEC before mapping; externally seeded refinement, not standalone direct mapping')
    ap.add_argument('--seed-native', choices=['outer','editable'], default='outer', help='explicit external-seed native import ablation; exact internal definitions and fresh prepared-input CEC, charged to mapping budget')
    ap.add_argument('--sdc-window-inputs', type=int, choices=range(0,21), default=0, metavar='0..20', help='explicit exhaustive window SDC ablation; independent window inputs, zero retains legacy PI-support care; fixed 600s')
    ap.add_argument('--sdc-monotone', action='store_true', help='explicit PI-exact care retained, window supplements only full-care fallback; unchanged 600s and candidate domain')
    ap.add_argument('--match-profile', action='store_true', help='explicit diagnostic matching timers; unchanged search, profiling overhead counts in mapping time')
    ap.add_argument('--ablation', choices=['joint-off', 'joint-filtered', 'joint-area-only', 'functional-stream', 'functional-sparse', 'functional-fine', 'functional-lazy', 'source-editable', 'source-coalesced'], help='explicit same-binary joint or functional-search variant; fixed global clock')
    ap.add_argument('--cover-materialization', choices=['eager','coalesced','deferred','event'], help='explicit same-binary cover compaction/publication ablation; event also coalesces one existing small-match event; unchanged candidates and 600s')
    ap.add_argument('--publication-integrity', choices=['fnv64','sha256'], default='fnv64', help='explicit full-byte publication digest protocol; no renewed mapping deadline')
    ap.add_argument('--frontier-numeric',choices=['legacy','local'],default='legacy',help='local: exact dependency-closed numerical reuse on positive full-arc graphs, full-builder fallback otherwise; requires frontier cache/preparation on')
    ap.add_argument('--trial-timing-first',choices=['off','on','goal-aware'],default='off',help='off: area-first; on: timing-first for both goals; goal-aware: timing-first only for DelayArea; acceptance and domain unchanged')
    ap.add_argument('--unchanged-delay-floor', choices=['off','on'], default='off', help='exact untouched-output delay lower bound for conservative DA trial rejection')
    ap.add_argument('--joint-pair-owner', choices=['off','on'], default='off', help='one usable shared helper owns each physical joint pair; retains complete pair set')
    ap.add_argument('--match-facts', choices=['off','on'], default='off', help='invocation-local immutable library layout facts, retains full existing TechLib variants')
    ap.add_argument('--sdc-local-index', choices=['off','on'], default='off', help='cursor-local care-masked target index; fresh boundary targets/proofs, unchanged enumeration')
    ap.add_argument('--sdc-phase-filter', choices=['off','on'], default='off', help='exact boundary-care phase eligibility mask on lazy SDC layouts; skips only proven misses; changes loop cadence, never candidate domain')
    ap.add_argument('--sdc-truth-cache', choices=['off','on'], default='off', help='pure physical phase/lifting truth reuse on lazy SDC layouts; care, target tests and proofs remain fresh')
    ap.add_argument('--recovery-inputs', choices=['legacy', 'overlay'], default='overlay', help='accepted-cover-first arrival screening, or explicit old witness-only control')
    ap.add_argument('--recovery-audit', action='store_true', help='diagnostic physical admission/recovery JSONL; I/O counts in the same mapping clock')
    ap.add_argument('--recovery-audit-cells', default='', help='comma-separated exact cell names for diagnostic output only; empty logs all')
    ap.add_argument('--recovery-noop-prefilter', choices=['off','on'], default='off', help='explicit same-binary identity-only recovery prefilter; fixed global clock')
    ap.add_argument('--frontier-preparation', choices=['off','on'], default='off', help='append-compatible physical incidence preparation; numerical propagation/proofs and600s unchanged')
    ap.add_argument('--batch-frontier-cache', choices=['off','on'], default='off', help='same-prefix numerical frontier cache; source/metric consideration and600s unchanged')
    ap.add_argument('--cut-cost-detail', action='store_true', help='diagnostic exclusive event lanes/admission costs; observation overhead inside600s, not a speed result')
    ap.add_argument('--cut-truth', choices=['legacy','word'], default='legacy', help='exact bit-parallel event cut composition/substitution; unchanged task and candidate order')
    ap.add_argument('--cut-join', choices=['legacy','streamed'], default='legacy', help='same-binary tuple allocation reuse; unchanged task visits/order, domain and 600s clock')
    ap.add_argument('--initial-cover', choices=['source','native-delay','direct'], default='source', help='native source-DAG delay warm start, no external mapper; original full candidate space retained')
    ap.add_argument('--flow-witness', choices=['legacy','delta'], default='legacy', help='accumulated Flow witness deltas; unchanged score/certificate schedule and600s')
    ap.add_argument('--recovery-cache', choices=['legacy','split','delta','lifecycle'], default='legacy', help='explicit same-binary cache dependency ablation; unchanged proposal domain and clock')
    ap.add_argument('--recovery-schedule', choices=['legacy','fair'], default='legacy', help='explicit same-binary fair recovery scheduling; unchanged metric, proof and 600-second clock')
    ap.add_argument('--replay-snapshot', type=Path, help='diagnostic fixed-pool cold recovery replay; not full mapping')
    ap.add_argument('--area-reject-cache', choices=['off','on'], default='off', help='explicit exact-context ordinary area rejection cache; same candidate domain and deadline')
    ap.add_argument('--proposal-observation', action='store_true', help='count complete ordinary recovery proposal identities and global contexts; no pruning, overhead inside budget')
    ap.add_argument('--lifecycle', action='store_true', help='diagnostic cut-to-care lifecycle and exact representative export; overhead inside 600s')
    ap.add_argument('--match-schedule', choices=['legacy','fair'], default='legacy', help='persistent exact/INV/constant/SDC lane rotation; no task pruning or renewed clock')
    ap.add_argument('--selected-provenance', action='store_true', help='selected generator/care witnesses, content-addressed inside mapping budget')
    ap.add_argument('--observe', action='store_true', help='persistent native trajectory and exclusive scope costs; overhead inside 600s')
    ap.add_argument('--snapshot-after', type=float, help='diagnostic snapshot at first recovery boundary after native online seconds; not a search stop or extra budget')
    a = ap.parse_args()
    if a.frontier_numeric == 'local' and (a.batch_frontier_cache != 'on' or a.frontier_preparation != 'on'):
        ap.error('--frontier-numeric local requires --batch-frontier-cache on and --frontier-preparation on')
    if a.snapshot_after is not None and (not __import__('math').isfinite(a.snapshot_after) or not 0 <= a.snapshot_after < 600):
        ap.error('--snapshot-after must be finite and in [0,600)')
    if a.fresh_emap_seed and a.seed_verilog:
        ap.error('--fresh-emap-seed and --seed-verilog are mutually exclusive')
    subject = safe_path(a.aig.resolve())
    golden = safe_path((a.golden or a.aig).resolve())
    folder = safe_path(a.output.resolve())
    profile_path = ROOT/'emapper/profiles/delivery.json'
    profile = recovery_input_profile(ablation_profile(json.loads(profile_path.read_text()), a.ablation), a.recovery_inputs)
    if a.initial_cover in ('native-delay','direct') and (a.seed_verilog or a.fresh_emap_seed):
        raise ValueError('native initial delay cover cannot use an external mapper or seed')
    profile = initial_cover_profile(profile, a.initial_cover)
    profile = recovery_noop_profile(profile, a.recovery_noop_prefilter)
    profile = recovery_cache_profile(profile, a.recovery_cache)
    profile = flow_witness_profile(profile, a.flow_witness)
    profile = cut_join_profile(profile, a.cut_join)
    profile = cut_truth_profile(profile, a.cut_truth)
    profile = cut_cost_profile(profile, a.cut_cost_detail)
    profile = batch_frontier_profile(profile, a.batch_frontier_cache)
    profile = frontier_preparation_profile(profile, a.frontier_preparation)
    if a.cover_materialization is not None:
        profile = cover_materialization_profile(profile, a.cover_materialization)
    profile['environment']['EGM_PUBLICATION_INTEGRITY'] = a.publication_integrity
    profile['publication_integrity'] = a.publication_integrity
    profile['environment']['EGM_SDC_TRUTH_CACHE'] = '1' if a.sdc_truth_cache=='on' else '0'
    profile['sdc_truth_cache'] = a.sdc_truth_cache
    profile['environment']['EGM_SDC_PHASE_FILTER'] = '1' if a.sdc_phase_filter=='on' else '0'
    profile['sdc_phase_filter'] = a.sdc_phase_filter
    profile['environment']['EGM_MATCH_FACTS'] = '1' if a.match_facts=='on' else '0'
    profile['match_facts'] = a.match_facts
    profile['environment']['EGM_JOINT_PAIR_OWNER'] = '1' if a.joint_pair_owner=='on' else '0'
    profile['joint_pair_owner'] = a.joint_pair_owner
    profile['environment']['EGM_UNCHANGED_DELAY_FLOOR'] = '1' if a.unchanged_delay_floor=='on' else '0'
    profile['unchanged_delay_floor'] = a.unchanged_delay_floor
    profile['frontier_numeric'] = a.frontier_numeric
    profile['environment']['EGM_FRONTIER_NUMERIC_LOCAL'] = '1' if a.frontier_numeric == 'local' else '0'
    profile['environment']['EGM_TRIAL_TIMING_FIRST'] = {'off':'0','on':'1','goal-aware':'2'}[a.trial_timing_first]
    profile['trial_timing_first'] = a.trial_timing_first
    profile['environment']['EGM_SDC_LOCAL_INDEX'] = '1' if a.sdc_local_index=='on' else '0'
    profile['sdc_local_index'] = a.sdc_local_index
    profile = recovery_schedule_profile(profile, a.recovery_schedule)
    profile = window_sdc_profile(profile, a.sdc_window_inputs)
    profile = monotone_sdc_profile(profile, a.sdc_monotone)
    if a.match_profile:
        profile['name'] += '-match-profile'
        profile['environment']['EGM_MATCH_PROFILE'] = '1'
    profile['environment']['EGM_MATCH_FAIR'] = '1' if a.match_schedule == 'fair' else '0'
    profile['selected_provenance_requested'] = bool(a.selected_provenance)
    library = ROOT/'emapper/libraries/asap7/asap7_full.genlib'
    native = ROOT/'target/release/emapper_kernel'
    measure = ROOT/'target/release/emapper_measure'
    build_receipt = ROOT/'target/release/emapper-build.json'
    if not build_receipt.exists():
        raise RuntimeError('Run scripts/emapper/build.sh before mapping')
    receipt = json.loads(build_receipt.read_text())
    if receipt != dict(source=hashes(source_files()), binaries=hashes([native, measure])):
        raise RuntimeError('Source/binary drift: rebuild with scripts/emapper/build.sh')
    tools = {name: shutil.which(name) for name in ['yosys','abc']}
    if not all(tools.values()):
        raise RuntimeError('Yosys and ABC must be available in PATH')
    if not subject.is_file() or not golden.is_file():
        raise FileNotFoundError('input or golden AIG missing')
    folder.mkdir(parents=True, exist_ok=False)
    raw = folder/'raw.v'
    # Remove inherited experimental knobs; the profile is the sole algorithm configuration.
    env = {k:v for k,v in os.environ.items() if not k.startswith('EGM_')}
    env.update(profile['environment'])
    env.update(EGM_NATIVE_OPTIMIZER=str(native), EGM_YOSYS=tools['yosys'],
               EGM_RECOVERY_TRACE=str(folder/'recovery.jsonl'))
    if a.replay_snapshot:
        if a.snapshot_after is not None:
            raise ValueError('cannot capture while replaying')
        snapshot = a.replay_snapshot.resolve()
        frozen = json.loads(snapshot.read_text())
        if frozen.get('schema') != 1:
            raise ValueError('unsupported recovery snapshot schema')
        for key, path in [('EGM_DIAG_INPUT_SHA256', subject), ('EGM_DIAG_LIBRARY_SHA256', library), ('EGM_DIAG_NATIVE_SHA256', native)]:
            if frozen['environment'].get(key) != sha(path):
                raise ValueError('snapshot source identity mismatch: '+key)
        for key, value in profile['environment'].items():
            if key != 'EGM_RECOVERY_FAIR' and frozen['environment'].get(key) != value:
                raise ValueError('snapshot profile mismatch: '+key)
        env['EGM_REPLAY_SNAPSHOT'] = str(snapshot)
    if a.snapshot_after is not None or a.replay_snapshot:
        env.update(EGM_DIAG_INPUT_SHA256=sha(subject), EGM_DIAG_LIBRARY_SHA256=sha(library), EGM_DIAG_NATIVE_SHA256=sha(native))
    env['EGM_AREA_REJECT_CACHE'] = '1' if a.area_reject_cache == 'on' else '0'
    if a.proposal_observation:
        env['EGM_PROPOSAL_OBSERVATION'] = '1'
    if a.lifecycle:
        env.update(EGM_LIFECYCLE=str(folder/'lifecycle.jsonl'), EGM_SDC_REPRESENTATIVES=str(folder/'sdc-representatives.json'))
    if a.selected_provenance:
        env['EGM_SELECTED_PROVENANCE'] = str((folder/'selected-provenance').resolve())
    if a.observe:
        env['EGM_OBSERVATION'] = str(folder/'observation.jsonl')
    if a.snapshot_after is not None:
        env.update(EGM_RECOVERY_SNAPSHOT=str(folder/'recovery-snapshot.json'), EGM_RECOVERY_SNAPSHOT_AFTER=str(a.snapshot_after))
    if a.recovery_audit:
        env.update(EGM_RECOVERY_AUDIT=str(folder/'candidate-audit.jsonl'), EGM_RECOVERY_AUDIT_CELLS=a.recovery_audit_cells)
    elif a.recovery_audit_cells:
        raise ValueError('--recovery-audit-cells requires --recovery-audit')
    argv = [sys.executable, str(ROOT/'scripts/emapper/frontend.py'), str(library), str(subject), str(raw)]
    files = source_files()+list((ROOT/'scripts/emapper').glob('*.py'))+[profile_path, build_receipt,
        library, library.parent/'cells.v', library.parent/'library.json', native, measure]
    if a.replay_snapshot:
        files.append(a.replay_snapshot.resolve())
    report = dict(schema=1, git_head=git_revision(), code_identity="dependency hashes are authoritative, including uncommitted changes", configuration=profile, ablation=a.ablation, match_profile_enabled=a.match_profile, base_profile_sha256=sha(profile_path), status='running', CEC_in_mapping_clock=False,
        prepared_sha256=sha(subject), golden_sha256=sha(golden),
        dependencies=hashes(files), tools={k:dict(path=v,sha256=sha(v)) for k,v in tools.items()},
        argv=argv, environment={k:v for k,v in env.items() if k.startswith('EGM_')})
    if a.replay_snapshot:
        report['diagnostic_scope'] = 'fixed candidate/proof snapshot; cold recovery; not full mapping PPA'
        report['snapshot_sha256'] = sha(a.replay_snapshot.resolve())
    dump(folder/'report.json',report)
    seed_generation_seconds = 0.
    if a.fresh_emap_seed:
        from fresh_seed import generate
        try:
            a.seed_verilog, generated = generate(safe_path(a.fresh_emap_seed.resolve()), library,
                                                 subject, folder/'fresh-seed', profile['mapping_seconds'])
            seed_generation_seconds = generated['mapping_seconds']
            report['fresh_seed_generation'] = generated
        except Exception as exc:
            generation_receipt=folder/'fresh-seed/receipt.json'
            generated=json.loads(generation_receipt.read_text()) if generation_receipt.exists() else {}
            report.update(status='seed_generation_failed', error=f'{type(exc).__name__}: {exc}',
                          fresh_seed_generation=generated, mapping_wall_secs=generated.get('mapping_seconds',0),
                          delivery_state='invalid',search_state='not_started')
            dump(folder/'report.json',report)
            return 1
    if a.seed_native=='editable' and not a.seed_verilog:
        raise ValueError('--seed-native editable requires --seed-verilog')
    if a.seed_verilog:
        # Conversion, measurement and independent CEC stay outside the mapping clock.
        # The original seed-generation computation is unknown and never called free mapping work.
        prep_start = time.monotonic()
        try:
            seed_path = safe_path(a.seed_verilog.resolve())
            external = measure_cover(library,seed_path,folder/'external-seed',measure,tools['yosys'])
            os.environ['EGM_YOSYS'],os.environ['BENCH_RS_EMAP_ABC']=tools['yosys'],tools['abc']
            good = verify(library.parent/'cells.v',golden,external['netjson'],folder/'external-seed/cec')
            if a.seed_native=='editable':
                if env.get('EGM_EDITABLE_SOURCE')!='1':
                    raise ValueError('editable external seed requires the source-editable ablation')
                prepared_good=verify(library.parent/'cells.v',subject,external['netjson'],folder/'external-seed/prepared-cec')
                if not prepared_good:raise ValueError('external seed fails fresh prepared-input CEC')
                def file_hash(path):
                    path=Path(path).resolve();return dict(path=str(path),sha256=sha(path))
                proof_folder=folder/'external-seed/prepared-cec'
                proof=dict(schema=1,prepared=file_hash(subject),library=file_hash(library),models=file_hash(library.parent/'cells.v'),netjson=file_hash(external['netjson']),cec_log=file_hash(proof_folder/'cec.log'),verification=file_hash(proof_folder/'verification.json'),artifacts=[file_hash(p)for p in sorted(proof_folder.iterdir())if p.is_file()])
                dump(folder/'external-seed/native-proof.json',proof)
                env.update(EGM_EXTERNAL_EDITABLE='1',EGM_EXTERNAL_PROOF_RECEIPT=str(folder/'external-seed/native-proof.json'))
                external['native_import_requested']=True
            external.update(CEC=int(good),prepared_sha256=sha(subject),golden_sha256=sha(golden),
                experiment_scope='externally seeded refinement; provided seed generation is not timed',
                source_bundle_role='exact external seed becomes editable after native proof and admission'if a.seed_native=='editable'else'native constructive source remains separate from external comparator',
                preparation_wall_secs_excluding_mapping=time.monotonic()-prep_start)
            dump(folder/'external-seed/manifest.json',external)
            report['external_seed']=external
            if not good:
                raise ValueError('external seed fails independent named-port CEC')
            env.update(EGM_SEED_VERILOG=str(seed_path),EGM_SEED_MANIFEST=str(folder/'external-seed/manifest.json'),
                       EGM_MEASURE_BINARY=str(measure))
            report['environment']={k:v for k,v in env.items()if k.startswith('EGM_')}
            report['dependencies'].update(hashes([seed_path]) if seed_path.is_relative_to(ROOT) else
                {str(seed_path):sha(seed_path)})
            if a.fresh_emap_seed:
                external['experiment_scope']='fresh emap-delay seed plus native refinement; seed generation charged within600s; pure normalization and CEC excluded'
                dump(folder/'external-seed/manifest.json',external)
            report['experiment_scope']=external['experiment_scope']
            dump(folder/'report.json',report)
        except Exception as exc:
            report.update(status='seed_failed',error=f'{type(exc).__name__}: {exc}',mapping_wall_secs=0,
                          delivery_state='invalid',search_state='not_started')
            report['artifacts']={str(p.relative_to(folder)):sha(p)for p in folder.rglob('*')if p.is_file()and p.name!='report.json'}
            dump(folder/'report.json',report)
            return 1
    start = time.monotonic()
    from fresh_seed import remaining_mapping_seconds
    deadline = start+remaining_mapping_seconds(profile['mapping_seconds'], seed_generation_seconds)
    env['EGM_MAPPING_DEADLINE_MONOTONIC'] = str(deadline)
    env['EGM_MAPPING_STARTED_MONOTONIC'] = str(start)
    timed_out = False
    try:
        with (folder/'mapping.stdout').open('w') as out, (folder/'mapping.stderr').open('w') as err:
            proc = subprocess.Popen(argv, env=env, stdout=out, stderr=err, start_new_session=True)
            try:
                proc.wait(timeout=max(0,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            except BaseException:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                raise
        refinement_seconds=time.monotonic()-start
        report.update(mapping_wall_secs=seed_generation_seconds+refinement_seconds, refinement_wall_secs=refinement_seconds, seed_generation_in_mapping_clock=bool(a.fresh_emap_seed), mapping_exit_code=proc.returncode,
                      search_timeout=timed_out)
        meta = raw.with_suffix('.complete.json')
        native_info = json.loads(meta.read_text()) if meta.exists() else {}
        report['native_observed'] = copy.deepcopy(native_info)
        effective, selection = select_output(raw, native_info, library, subject, native)
        report['native'] = effective
        report['selection'] = selection
        optimized = selection['kind'] == 'optimized'
        if optimized:
            report['incumbent'] = selection['incumbent']
        elif selection.get('incumbent'):
            report['source_incumbent'] = selection['incumbent']
        first_native=folder/'first-native-cover.json'
        if first_native.exists():report['first_native_cover']=json.loads(first_native.read_text())
        report['direct_native']=a.initial_cover=='direct'
        source_audit=folder/'incumbents/source-seed.json'
        if source_audit.exists():report['editable_source']=json.loads(source_audit.read_text())
        initial_delay_audit=folder/'incumbents/native-initial-delay.json'
        report['native_initial_delay_available']=initial_delay_audit.exists()
        if initial_delay_audit.exists():
            native_initial=json.loads(initial_delay_audit.read_text())
            report['native_initial_delay']=native_initial
        if a.seed_native=='editable':
            report['external_native_import']=dict(requested=True,applied=report.get('editable_source',{}).get('external_seed_editable',False),fallback_remains_legal=True,proof_receipt=str(folder/'external-seed/native-proof.json'))
        report['search_timeout'] |= bool(native_info.get('search_deadline_hit',False))
        report['search_finished'] = not timed_out and bool(native_info.get('search_finished',False))
        publication_audit=folder/'publication.json'
        if publication_audit.exists():report['publication']=json.loads(publication_audit.read_text())
        functional_audit = folder/'online.pool.functional-cursor.json'
        if functional_audit.exists():
            producer = json.loads(functional_audit.read_text())['engine']
            report['candidate_production_complete'] = producer['candidate_production_complete']
            queues_path = folder/'incumbents/seed-recovery.json'
            queues = json.loads(queues_path.read_text()) if queues_path.exists() else {}
            report['search_queues_complete'] = search_queues_complete(queues)
            report['native_publication_state'] = queues.get('publication_state')
            report['search_finished'] &= report['candidate_production_complete'] and report['search_queues_complete']
            report['search_timeout'] |= producer['deadline_hit']

        if not raw.exists():
            raise RuntimeError('mapping produced no complete cover')
        # Pure representation conversion, public metrics and CEC are outside the mapping clock.
        normalized, netjson = folder/'mapped.v', folder/'mapped.json'
        script = f'read_verilog -lib "{library.parent / "cells.v"}"; read_verilog "{raw}"; hierarchy -check -auto-top; rename -top mapped; write_verilog -noattr -noexpr "{normalized}"; write_json "{netjson}"'
        cmd = [tools['yosys'],'-Q','-T','-p',script]
        report['normalization_argv'] = cmd
        result = subprocess.run(cmd, capture_output=True,text=True)
        (folder/'normalize.log').write_text(result.stdout+result.stderr)
        result.check_returncode()
        cmd = [str(measure),str(library),str(netjson)]
        report['measurement_argv'] = cmd
        result = subprocess.run(cmd,capture_output=True,text=True)
        (folder/'measure.log').write_text(result.stdout+result.stderr)
        result.check_returncode()
        report['measured'] = json.loads(result.stdout)
        selected_value=selection.get('incumbent',native_info.get('source',{}))
        same_metrics=all(report['measured'][k]==selected_value.get(k) for k in ['area','delay'])
        report['evaluation']=dict(objective_policy=OBJECTIVE_POLICY,selected_AD={k:selected_value.get(k)for k in ['area','delay']},public_AD={k:report['measured'][k]for k in ['area','delay']},exact_agreement=same_metrics)
        os.environ['EGM_YOSYS'], os.environ['BENCH_RS_EMAP_ABC'] = tools['yosys'], tools['abc']
        good = verify(library.parent/'cells.v', golden, netjson, folder/'cec')
        report['cec'] = int(good)
        if a.selected_provenance:
            from selected_evidence import replay
            if selection.get('incumbent'):
                report['selected_evidence'] = replay(selection['incumbent']['manifest'], folder/'selected-provenance/objects', netjson)
                report['selected_evidence']['validation_outside_mapping_clock'] = True
            else:
                report['selected_evidence'] = dict(valid=None, reason='constructive fallback has no native selected proof set; utilization unknown')
        report['status'] = 'metric_failed' if not same_metrics else 'ok' if good and optimized else 'source_only' if good else 'cec_failed'
        if report.get('first_native_cover'):
            f=report['first_native_cover'];report['search_vs_first_native']=contribution(dict(area=f['area'],delay=f['delay']),report['measured'])
        if report.get('native_initial_delay'):
            area,delay=report['native_initial_delay']['initial_AD']
            report['search_vs_native_initial']=contribution(dict(area=area,delay=delay),report['measured'])
        if report.get('external_seed'):
            report['search_vs_seed']=contribution(report['external_seed']['measured'],report['measured'])
            report['search_vs_seed']['CEC']=int(good)
            report['search_vs_seed']['validated_search_gain']=bool(good and same_metrics and optimized and
                report['search_vs_seed']['outcome']=='improved')
            if report.get('editable_source',{}).get('external_seed_editable'):
                initial=report['editable_source']['native_AD']
                imported=dict(area=initial[0],delay=initial[1])
                report['seed_import_contribution']=contribution(report['external_seed']['measured'],imported)
                report['native_search_net_gain']=contribution(imported,report['measured'])
                report['native_search_net_gain']['CEC']=int(good)
            if effective.get('constructive_source'):
                report['seed_vs_constructive']=contribution(effective['constructive_source'],report['external_seed']['measured'])
            if report['search_vs_seed']['outcome']=='regressed':
                report['status']='seed_regression'
        report['output_sha256'] = sha(normalized)
    except KeyboardInterrupt:
        report.update(status='interrupted',error='Interrupted by user; partial artifacts retained')
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    report['delivery_state']='optimized_verified'if report['status']=='ok'else'source_verified'if report['status']=='source_only'else'invalid'
    report['search_state']='deadline'if report.get('search_timeout')else'finished'if report.get('search_finished')else'stopped'
    report['artifacts'] = {str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='report.json'}
    dump(folder/'report.json',report)
    print(json.dumps({k:report.get(k) for k in ['status','measured','cec','mapping_wall_secs','search_timeout','search_finished']}))
    return 0 if report['status'] in ('ok','source_only') else 1

if __name__ == '__main__':
    sys.exit(main())
