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
        queues.get('publication_state',{}).get('pending',False)])


def recovery_input_profile(base, mode):
    """Explicit same-binary arrival-screening ablation; the clock is unchanged."""
    if mode not in ('legacy', 'overlay'):
        raise ValueError('unknown recovery input policy')
    profile = copy.deepcopy(base)
    profile['environment']['EGM_RECOVERY_INPUT_OVERLAY'] = '1' if mode == 'overlay' else '0'
    profile['recovery_input_policy'] = mode
    return profile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('aig', type=Path, help='named, combinational binary AIGER; no preprocessing is run')
    ap.add_argument('--output', type=Path, required=True, help='new directory; existing results are never overwritten')
    ap.add_argument('--golden', type=Path, help='optional original named AIG for independent CEC; default: input')
    ap.add_argument('--match-profile', action='store_true', help='explicit diagnostic matching timers; unchanged search, profiling overhead counts in mapping time')
    ap.add_argument('--ablation', choices=['joint-off', 'joint-filtered', 'joint-area-only', 'functional-stream', 'functional-sparse', 'functional-fine', 'functional-lazy', 'source-editable', 'source-coalesced'], help='explicit same-binary joint or functional-search variant; fixed global clock')
    ap.add_argument('--recovery-inputs', choices=['legacy', 'overlay'], default='overlay', help='accepted-cover-first arrival screening, or explicit old witness-only control')
    ap.add_argument('--recovery-audit', action='store_true', help='diagnostic physical admission/recovery JSONL; I/O counts in the same mapping clock')
    ap.add_argument('--recovery-audit-cells', default='', help='comma-separated exact cell names for diagnostic output only; empty logs all')
    a = ap.parse_args()
    subject = safe_path(a.aig.resolve())
    golden = safe_path((a.golden or a.aig).resolve())
    folder = safe_path(a.output.resolve())
    profile_path = ROOT/'emapper/profiles/delivery.json'
    profile = recovery_input_profile(ablation_profile(json.loads(profile_path.read_text()), a.ablation), a.recovery_inputs)
    if a.match_profile:
        profile['name'] += '-match-profile'
        profile['environment']['EGM_MATCH_PROFILE'] = '1'
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
    if a.recovery_audit:
        env.update(EGM_RECOVERY_AUDIT=str(folder/'candidate-audit.jsonl'), EGM_RECOVERY_AUDIT_CELLS=a.recovery_audit_cells)
    elif a.recovery_audit_cells:
        raise ValueError('--recovery-audit-cells requires --recovery-audit')
    argv = [sys.executable, str(ROOT/'scripts/emapper/frontend.py'), str(library), str(subject), str(raw)]
    files = source_files()+list((ROOT/'scripts/emapper').glob('*.py'))+[profile_path, build_receipt,
        library, library.parent/'cells.v', library.parent/'library.json', native, measure]
    report = dict(schema=1, git_head=git_revision(), code_identity="dependency hashes are authoritative, including uncommitted changes", configuration=profile, ablation=a.ablation, match_profile_enabled=a.match_profile, base_profile_sha256=sha(profile_path), status='running', CEC_in_mapping_clock=False,
        prepared_sha256=sha(subject), golden_sha256=sha(golden),
        dependencies=hashes(files), tools={k:dict(path=v,sha256=sha(v)) for k,v in tools.items()},
        argv=argv, environment={k:v for k,v in env.items() if k.startswith('EGM_')})
    dump(folder/'report.json',report)
    start = time.monotonic()
    deadline = start+profile['mapping_seconds']
    env['EGM_MAPPING_DEADLINE_MONOTONIC'] = str(deadline)
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
        report.update(mapping_wall_secs=time.monotonic()-start, mapping_exit_code=proc.returncode,
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
        source_audit=folder/'incumbents/source-seed.json'
        if source_audit.exists():report['editable_source']=json.loads(source_audit.read_text())
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
        report['status'] = 'metric_failed' if not same_metrics else 'ok' if good and optimized else 'source_only' if good else 'cec_failed'
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
