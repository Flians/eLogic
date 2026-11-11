"""Evidence-gated, read-only quality diagnosis; no mapper/baseline executions.

The current full artifact audit must finish first. Historical seed preparation
cost is disclosed separately; incomplete candidate logs cannot prove absence.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import pathlib
import shutil
import time

R = pathlib.Path(__file__).resolve().parents[3]
S = R/'results_v2/emapper_dac_convergence_20261111'
H = R/'results_v2/emapper_monotone_sdc_20261104_final'
FOCUS = ['div','hyp','log2','multiplier','sin','sqrt','square','mem_ctrl','voter',
         'adder','arbiter','priority']


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def csv_rows(path):
    with path.open() as stream:return list(csv.DictReader(stream))
def pct(new, old):return 100*(new/old-1) if old else None
def outcome(a, b):return 'better' if a<b else 'equal' if a==b else 'worse'


def analyze(output, audit):
    assert audit['completed']==20 and not audit['invalid_artifacts']
    campaign=S/'full-epfl/campaign.json';c=read(campaign)
    assert sha(campaign)==audit['source_campaign_sha256']
    assert c['selection']['variant']=='off' and len(c['jobs'])==20
    checks=read(H/'derived-checksums.json')
    inputs=['main20_ADT.csv','seed_attribution56.csv','closure.json']
    for name in inputs:assert sha(H/name)==checks[name]
    old_rows=csv_rows(H/'main20_ADT.csv');assert len(old_rows)==20
    old={x['design']:x for x in old_rows};assert len(old)==20
    attrs={x['key']:x for x in csv_rows(H/'seed_attribution56.csv')}
    comparison=read(R/'results_v2/full_baselines_20261025/comparison.json')
    previous=pathlib.Path(comparison['candidate_source']);assert sha(previous)==comparison['candidate_sha256']
    previous_rows={x['bench']:x for x in read(previous)['results']
                   if x['status']=='ok' and x.get('measured',{}).get('cec')==1}
    baselines=read(R/'results_v2/full_baselines_20261025/baselines.json')
    assert sha(R/'results_v2/full_baselines_20261025/baselines.json')==audit['baseline_sha256']
    emap={x['bench']:x for x in baselines['results'] if x['method']=='mockturtle_emap_delay' and x['status']=='ok'}
    assert set(old)==set(emap)==set(c['jobs'])
    rows=[];sources={str(campaign):sha(campaign),str(previous):sha(previous)}
    for name,current in c['jobs'].items():
        old_row=old[name];assert old_row['arm']=='legacy' and old_row['repetition']=='0'
        report=pathlib.Path(old_row['report']);assert sha(report)==old_row['report_sha256']
        historical=read(report);current_report=pathlib.Path(current['case'])/'report.json'
        assert sha(current_report)==current['artifacts']['report.json'];native=read(current_report)
        for key in ['prepared_sha256','golden_sha256']:assert historical[key]==native[key]
        assert historical['native']['library_sha256']==baselines['library_sha256']
        assert historical['evaluation']['exact_agreement'] and native['evaluation']['exact_agreement']
        assert historical['evaluation']['objective_policy']==native['evaluation']['objective_policy']
        assert historical['cec']==native['cec']==1 and current['first_independent_CEC']
        assert [float(old_row['A']),float(old_row['D'])]==[historical['measured']['area'],historical['measured']['delay']]
        a=attrs[old_row['key']];seed_A,seed_D=[float(a[k]) for k in ['S_to_I_A_before','S_to_I_D_before']]
        initial_A,initial_D=current['initial_AD'];old_A,old_D=float(old_row['A']),float(old_row['D'])
        prefix=current.get('observed_prefix',{});details=prefix.get('details',{});cost=prefix.get('exclusive_seconds',{});telemetry=details.get('telemetry',{})
        b=emap[name]['measured'];prev=previous_rows.get(name)
        row=dict(design=name,focus=name in FOCUS,newly_valid_vs_previous_native=name not in previous_rows,
                 current_A=current['A'],current_D=current['D'],current_T=current['T'],
                 initial_A=initial_A,initial_D=initial_D,first_cover_seconds=current['first_cover_time'],
                 emap_A=b['area'],emap_D=b['delay'],
                 current_vs_emap=outcome((current['D'],current['A']),(b['delay'],b['area'])),
                 old_seed_A=seed_A,old_seed_D=seed_D,old_final_A=old_A,old_final_D=old_D,
                 old_status=old_row['status'],old_CEC=1,current_CEC=1,
                 current_vs_old_seeded=outcome((current['D'],current['A']),(old_D,old_A)),
                 current_D_vs_seed_percent=pct(current['D'],seed_D),
                 current_D_vs_old_final_percent=pct(current['D'],old_D),
                 old_refinement_D_percent=pct(old_D,seed_D),old_refinement_A_percent=pct(old_A,seed_A),
                 current_search_D_percent=pct(current['D'],initial_D),
                 current_search_A_percent=pct(current['A'],initial_A),
                 previous_native_A=prev['measured']['area'] if prev else None,
                 previous_native_D=prev['measured']['delay'] if prev else None,
                 current_vs_previous_native=outcome((current['D'],current['A']),(prev['measured']['delay'],prev['measured']['area']))if prev else 'previous-invalid',
                 prefix_elapsed=prefix.get('elapsed'),prefix_cuts=details.get('matched_cuts'),
                 prefix_pool=details.get('pool'),prefix_flow_pending=details.get('flow_pending'),
                 prefix_joint_pending=details.get('joint_windows',{}).get('pending'),
                 exclusive_capture_seconds=cost.get('selected_evidence_capture',0),
                 exclusive_publication_seconds=cost.get('publication',0),
                 exclusive_capture_publication_percent=100*(cost.get('selected_evidence_capture',0)+cost.get('publication',0))/prefix['elapsed'],
                 exclusive_cut_seconds=cost.get('cut_enumeration',0),
                 exclusive_joint_seconds=cost.get('joint_recovery',0),
                 exclusive_delay_resume_seconds=cost.get('delay_flow_resume',0),
                 inclusive_flow_telemetry_seconds=telemetry.get('flow_secs'),
                 prefix_missing_proof=telemetry.get('certificate_missing_proof'),
                 prefix_trial_outcomes=telemetry.get('area_candidate_outcomes'),
                 historical_seed_generation_seconds=float(a['historical_seed_generation_seconds']),
                 historical_seed_generation_in_refinement_clock=a['seed_generation_in_refinement_clock'],
                 report=str(current_report),old_report=str(report),
                 candidate_absence_proven=False,recovery_misselection_proven=False,
                 limits='prefix costs/counters are not full snapshots or same-work comparisons; no complete unselected pool exported')
        rows.append(row);sources[str(report)]=sha(report);sources[str(current_report)]=sha(current_report)
    newly_valid=[x for x in rows if x['newly_valid_vs_previous_native']]
    assert len(newly_valid)==9 and all(x['current_vs_emap']=='worse' for x in newly_valid)
    output.mkdir(parents=True)
    for name in inputs:shutil.copy2(H/name,output/('old-'+name))
    shutil.copy2(campaign,output/'current-campaign-snapshot.json')
    shutil.copy2(pathlib.Path(__file__),output/'analysis-helper.py')
    with(output/'comparison20.csv').open('w')as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'case_evidence.json').write_text(json.dumps(rows,indent=2)+'\n')
    counts={key:{o:sum(x[key]==o for x in rows)for o in ['better','equal','worse']}for key in ['current_vs_emap','current_vs_old_seeded','current_vs_previous_native']}
    validation=dict(current_full_artifact_audit_passed=True,current_audit_sha256=sha(S/'full-epfl-final-audit/validation.json'),
                    current_verified_artifact_references=audit['verified_artifact_references'],
                    current_initial_final_CEC=40,historical_CEC_scope='existing historical closure, not rerun',
                    historical_report_metadata_hashes_checked=20,aligned_prepared_golden_library_objective=20,
                    sources=sources,source_helper_sha256=sha(pathlib.Path(__file__)),counts=counts,
                    pending_candidate_absence_and_fixed_pool_tests=True,mapping_runs=0,baseline_runs=0,
                    GitHub_push=False,interpretation='old seeded timing excludes historical seed generation; no end-to-end runtime equivalence claim')
    (output/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    lines=['# 原生版与旧seed精化版：全EPFL质量差距','',
           '完整当前证据审计已通过；下面不重跑baseline，不把旧seed成本算入已经记录的600秒精化时间。',
           '双方输入/库/公共评价及原报告SHA已对齐；历史CEC沿用原审计，不声称本轮重新执行。','',
           '| 设计 | 原生首次 A/D | 原生最终 A/D | 旧emap seed A/D | 旧精化最终 A/D | 捕获+发布占观测前缀 |',
           '|---|---:|---:|---:|---:|---:|']
    by={x['design']:x for x in rows}
    for name in FOCUS:
        x=by[name];lines.append(f"| {name} | {x['initial_A']:.2f}/{x['initial_D']:.2f} | {x['current_A']:.2f}/{x['current_D']:.2f} | {x['old_seed_A']:.2f}/{x['old_seed_D']:.2f} | {x['old_final_A']:.2f}/{x['old_final_D']:.2f} | {x['exclusive_capture_publication_percent']:.1f}% |")
    lines += ['', '## 三个相对早期原生版的回退','',
              '| 设计 | 旧原生 A/D | 当前 A/D | A变化 | D变化 |',
              '|---|---:|---:|---:|---:|']
    for name in ['adder','arbiter','priority']:
        x=by[name];lines.append(f"| {name} | {x['previous_native_A']:.2f}/{x['previous_native_D']:.2f} | {x['current_A']:.2f}/{x['current_D']:.2f} | {pct(x['current_A'],x['previous_native_A']):+.3f}% | {pct(x['current_D'],x['previous_native_D']):+.3f}% |")
    lines += ['', '这与旧seed版是两个不同的比较，不能把同延迟面积回退和未复现seed精化的延迟收益混为同一原因。',
        '', '## 证据能够支持的判断','',
        '- 九个新增有效案例全部输给emap-delay。div/hyp/multiplier旧结果是source_only：其强质量直接来自emap seed，不能算作旧搜索已经解决了这些电路。',
        '- log2、mem_ctrl旧精化主要保持seed时序并降面积；sqrt/voter只取得较小延迟改善。原生最终尚未达到这些seed时序，首要差距是原生映射未补回原有映射质量，不是证明了某个恢复bug。',
        '- sin、square旧精化允许以较大面积代价交换小幅延迟收益；严格(D,A)获胜并不等于面积也好。应同时报告seed与精化净收益。',
        '- adder已达到emap seed相同延迟且面积更低，但尚未复现旧精化进一步降低延迟的轨迹；仅凭首次覆盖弱不能解释全部剩余差距。',
        '- arbiter的原生最终时序仍远差于seed；priority同seed延迟下面积更高。必须分别检查强实现是否进池、是否可用、是否被提案访问，不能统一归因相位或cut缺失。',
        '- 大设计的观测前缀大量时间用于完整证据捕获/发布，Flow或联合队列仍pending；“触及600秒”不表示候选空间已遍历。scope互斥成本用于占比，不能相加嵌套telemetry。','',
        '注意：inclusive flow_secs包含后续评分、证明与发布调用，不能据它判断延迟标签传播本身很慢。应与exclusive delay_flow_resume分别阅读。priority的主要前缀成本又与大设计不同，不能给全部案例套同一原因。',
        'native_observed可能是截止前未更新的子进程元数据；正式覆盖、CEC及提交状态来自外层核验收据，不用其中no_complete_cover否定已经交付的合法结果。','',
        '## 尚不能证明的原因','',
        '当前收据充分验证所选覆盖，但没有完整未选候选池快照。因此不能证明关键候选不存在，不能把Flow已排空等同于全局最优，也不能仅凭拒绝次数证明恢复误拒绝。',
        '候选缺失与恢复劣选须由同逻辑epoch的完整候选快照、精确输入函数/相位/引脚对应及同池同起点重放区分；旧版本诊断不能直接作为本轮缺失证明。','',
        '## 下一步（先诊断，不立即新增优化）','',
        '1. 先核对adder/priority等同延迟面积差：在同池同起点固定D下检查完整共享面积提案是否存在、被访问、被正确评分和提交；无候选就不是恢复误拒绝。',
        '2. 对div/hyp/mem_ctrl及sqrt逐段对应emap现成网表与原生关键路径，记录缺结构边界、cut未遍历、phase/pin未匹配、候选未选的证据链；不调用新baseline搜索。',
        '3. 在相同候选访问序列上度量证据捕获/发布成本，区分实现开销拖慢搜索与恢复质量。之后再决定是否需要优化交付，而不是继续扩大盲搜索或切换外部seed。',
        '4. 只有定位清楚后才选择单一原生增量：补强原生PI到PO复合门delay映射，或修复已证候选/恢复缺口。保留完整候选空间、600秒和baseline，不恢复外部emap依赖。',
        '5. 单因素验证通过后再全EPFL与重复；主表保留统一r0，不从旧seed/当前/不同重复挑赢家。',
        '', '本报告是质量归因与下一步方案，不是已经修复、收敛、取得runtime优势或完成新消融的证据。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(output),counts=counts,baseline_runs=0)))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=pathlib.Path,required=True)
    ap.add_argument('--wait',action='store_true')
    ap.add_argument('--max-wait-seconds',type=float,default=28800)
    a=ap.parse_args();assert not a.output.exists()
    deadline=time.monotonic()+a.max_wait_seconds;locks=[]
    with(S/'quality-gap-analysis.lock').open('a')as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            audit_path=S/'full-epfl-final-audit/validation.json'
            if audit_path.exists():
                audit=read(audit_path);assert not audit['invalid_artifacts'] and audit['completed']==20
                try:
                    for path in [R/'results_v2/independent_components_20261019/global.lock',S/'full-epfl/runner.lock']:
                        f=path.open('a');locks.append(f);fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    for f in locks:f.close()
                    locks.clear()
            assert a.wait and time.monotonic()<deadline,'complete artifact audit not yet available; no analysis claims emitted'
            time.sleep(15)
        analyze(a.output,audit)


if __name__=='__main__':main()
