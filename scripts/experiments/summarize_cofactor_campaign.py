#!/usr/bin/env python3
"""Write a human-readable summary only after the paired campaign is audited."""
import argparse, csv, hashlib, json, statistics, sys
from pathlib import Path
from timeout_selection import read_rows
from run_three_parts import DESIGNS
import report

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root.resolve()
    verified=json.loads((root/'verified_profiles.json').read_text())
    for m,h in verified['cohort_certificate_sha256'].items():assert sha(root/m/'verified_metrics.json')==h
    metrics=verified['cohort_metrics']
    manifest=json.loads((root/'cold/manifest.json').read_text());arms=manifest['arms']
    method_count=len(DESIGNS)+2*len(DESIGNS)+len(manifest['ablation_designs'])*len(arms)
    title='查询复用、真值表复用与 pivot 预算' if 'noQUERY' in arms else ('构建缓存、精确 AIG 复用和只读库共享' if 'noAIGREUSE' in arms else 'Shannon、Davio、共享相位复用与上下文估价')
    lines=['三部分实验：'+title,'']
    lines+=['原始 AIG、预处理配方、库与初始映射保持不变。目标：固定原始 D0 下最小面积；整窗口综合关闭。',
            '方法与 ABC post-mapping 均限时 1000 秒；外层 3600 秒只覆盖预处理、映射和验证。',
            f'冷/热缓存各 {method_count} 次方法尝试，另各有 16 次初始检查；ABC 新执行 16 次，热组共享经核验的同一结果。',
            'TO 是原始优化器 SIGALRM，未产生完成输出。按归档协议另行选择经独立 CEC/PPA 验证的保存网表或初始映射；该选择不算优化完成。',
            '所有逐电路结果、超时状态、选择结果与时间见 paired_results.tsv；证据链见 verified_profiles.json。','']
    datasets={m:{s:{(r['stage'],r['design']):r for r in read_rows(root/m,s)} for s in ['tab_mfs3','tab_main','ablation']} for m in ['cold','warm']}
    for m in ['cold','warm']:
        v=metrics[m];lines += [f"{m}: 方法完成 {v['engine_cec_outputs']}/{method_count}，TO {v['method_timeouts']}；所有 {method_count} 个选择输出均有 CEC 证据并满足初始面积/延迟界限。"]
    for mode in ['cold','warm']:
        lines += [f"{mode}: TO 中采用改进检查点 {metrics[mode].get('method_checkpoint_selections',0)} 个；其余采用初始检查点或初始网表。"]
    lines += [f"另有 {len(list((root/'initial_validation').glob('*/verified.json')))} 个完整库初始网表的独立 CEC 验证用于超时回退；这些不运行优化器。"]
    lines += [f"冷/热 {len(verified['matches'])} 对中，PPA 相同 {sum(x['ppa_equal'] for x in verified['matches'])}，序列化网表相同 {sum(x['netlist_equal'] for x in verified['matches'])}。",'', '一、同输入 mfs3 比较']
    v=metrics['cold'];lines += [f"同一至多六输入的 40-entry 子库。ABC 完成 {v['abc_completed']}/16，TO {v['abc_timeouts']}，完成结果中延迟违例 {v['abc_delay_violations']}。",f"ABC 完成子集的原始面积/延迟均值变化：{v['abc_raw_area']:+.4f}% / {v['abc_raw_delay']:+.4f}%。"]
    lines += [f"外部选择仅保留满足初始面积和延迟界限的 ABC 完成结果，否则返回初始映射；该子集均值为 {v['abc_guarded_area']:+.4f}% / {v['abc_guarded_delay']:+.4f}%。这是结果选择，不改变 ABC 算法。"]
    # Exported metrics include the externally guarded ABC selection separately.
    for m in ['cold','warm']:
        v=metrics[m];done=sum(r['raw_status']=='complete' for (arm,d),r in datasets[m]['tab_mfs3'].items() if arm=='ours')
        lines += [f"{m}: 我们完成 {done}/16；全部选择输出均值 {v['mfs_ours_area']:+.4f}% / {v['mfs_ours_delay']:+.4f}%；ABC 完成子集 {v['paired_ours_area']:+.4f}% / {v['paired_ours_delay']:+.4f}%。"]
        lines += [f"  方法优化器合计 {sum(x['optimizer_seconds'] for x in verified['cache_runs'] if x['mode']==m and x['stage']=='tab_mfs3'):.4f} 秒（含每项 TO 的 1000 秒预算）。"]
    lines += ['', '二、充分 AIG 预处理后的映射优化（完整 47-entry 库）']
    for m in ['cold','warm']:
        v=metrics[m];d=datasets[m]['tab_main'];done=sum(d['ae',x]['raw_status']=='complete' for x in DESIGNS)
        lines += [f"{m}: 六轮上限完成 {done}/16。单轮选择均值 {v['main_one_area']:+.4f}% / {v['main_one_delay']:+.4f}%；六轮 {v['main_own_area']:+.4f}% / {v['main_own_delay']:+.4f}%；面积减小 {v['main_area_wins']}/16。", f"排除 arbiter 后：{v['main_other_area']:+.4f}% / {v['main_other_delay']:+.4f}%。"]
        for name in ['arbiter','usb_phy']:
            r=d['ae',name];lines += [f"  {name}: {r['base_a']} / {r['base_d']} -> {r['out_a']} / {r['out_d']}，{r['raw_status']}。"]
        lines += [f"相对 ASPDAC'25 重建的最终值：面积更小 {v['reference_wins']}/16，均值差 {v['reference_area']:+.4f}% / {v['reference_delay']:+.4f}%。"]
        lines += [f"  六轮主实验优化器合计 {sum(x['optimizer_seconds'] for x in verified['cache_runs'] if x['mode']==m and x['stage']=='tab_main' and x['arm']=='ae'):.4f} 秒。"]
    lines += ['ASPDAC\'25 无源码；仅以原文起点和舍入百分比重建最终值。初始映射不同，不能据此隔离 post-mapping 算法优势。','', '三、消融；ΔA 为相对 full 的面积差之和（正值表示更差）', 'Arm             冷ΔA       热ΔA       冷W/B   热W/B    冷/热时间比     冷/热TO']
    for arm in metrics['cold']['ablation']:
        c,w=[metrics[m]['ablation'][arm] for m in ['cold','warm']]
        tos=[sum(r['raw_status']=='timeout' for (x,d),r in datasets[m]['ablation'].items() if x==arm) for m in ['cold','warm']]
        lines += [f"{arm:13s} {c['area']:+10.2f} {w['area']:+10.2f} {c['worse']:2d}/{c['better']:<2d} {w['worse']:2d}/{w['better']:<2d} {c['time_ratio']:.3f}/{w['time_ratio']:.3f} {tos[0]}/{tos[1]}"]
    lines += ['', 'usb_phy 的新增算子消融（独立于主实验；面积/延迟、相对 full 的 ΔA、优化器秒数）：']
    for m in ['cold','warm']:
        full=float(datasets[m]['ablation']['full','usb_phy']['out_a'])
        for arm in [x for x in ['full','noSHANNON','noDAVIO','noREUSE','noPRICE','cache64K','noAIGREUSE','noLIBSHARE','noQUERY','noTRUTH','allPIVOTS'] if x in arms]:
            r=datasets[m]['ablation'][arm,'usb_phy'];t=next(x['optimizer_seconds'] for x in verified['cache_runs'] if x['mode']==m and x['stage']=='ablation' and x['arm']==arm and x['design']=='usb_phy')
            lines += [f"  {m} {arm}: {r['out_a']} / {r['out_d']}, ΔA={float(r['out_a'])-full:+.2f}, t={t:.4f}, {r['raw_status']}"]
    lines += ['', 'noSYN 关闭全部 L3（包括新余因子候选）；noSHANNON 只保留 Davio；noDAVIO 只保留 Shannon。',
              'noREUSE 只关闭已有 INV 的扇出检索，保留输入 INV 消除；noPRICE 改用内在面积/时序筛选，保留物理复用；noPHASE 关闭全部物理反相复用。',
              'cache64K 将每个原生结构缓存上限降到 65536；noAIGREUSE 关闭映射前精确 AIG 复用；noLIBSHARE 关闭临时映射网表的只读库共享。',
              'noQUERY 关闭已有 INV 查询的缓存与层级快速判定，保留物理复用；noTRUTH 关闭预解析仿真与动态真值表复用，保留共同的标量 cut memo；allPIVOTS 恢复所有变量的 Shannon/Davio 枚举。',
              '主实验和 full 消融为独立测量。60 秒 care 预算与 1000 秒墙钟截止可影响贪心轨迹；消融差值包含超时影响，不能解释为可相加的纯算子贡献。',
              '', '缓存与计时：冷组各配置/电路独立空持久池；热组只复用对应冷组的池。进程内原生缓存重启；不清操作系统缓存。',
              f"共享主机、每组最多 {manifest['jobs']} 个单工作线程方法进程，另有 ABC；仅各一次观测，时间比不是受控的统计加速比。",
              '超时进程缺少最终候选/阶段计数，不记为零；阶段饼图只覆盖完成的 full 消融运行。']
    for m in ['cold','warm']:
        for d,equal in metrics[m]['full_main_ppa_equal'].items():
            if not equal:
                x=datasets[m]['tab_main']['ae',d];y=datasets[m]['ablation']['full',d]
                lines += [f"独立重复差异：{m} {d} 主实验为 {x['raw_status']}、{x['out_a']}/{x['out_d']}；full 消融为 {y['raw_status']}、{y['out_a']}/{y['out_d']}。保留各自实测结果。"]
    for m in ['cold','warm']:
        measured=[x for x in verified['cache_runs'] if x['mode']==m and x['status']=='complete']
        full=[x for x in verified['cache_runs'] if x['mode']==m and x['stage']=='ablation' and x['arm']=='full']
        lines += [f"{m}: full 消融总优化器时间 {sum(x['optimizer_seconds'] for x in full):.4f} 秒；阶段统计覆盖 {metrics[m]['phase_timed_full_runs']}/16。",f"  仅对 {len(measured)} 个完成运行，可观测持久池 builds={sum(x['builds'] for x in measured)}, disk_hits={sum(x['disk_hits'] for x in measured)}；不能外推为所有运行的缓存统计。"]
    if 'native_build_runs' in verified:
        records=[]
        for x in verified['native_build_runs']:
            if x['stage']!='ablation' or x['arm'] not in ['full','cache64K','noAIGREUSE','noLIBSHARE']:continue
            r=dict(mode=x['mode'],arm=x['arm'],design=x['design'],status=x['status'])
            for group,keys in [('native',['mapping_calls','alias_hits','reused','factor_seconds','mapping_seconds','proof_seconds','key_seconds']),('cofactor',['builds','hits']),('synthesis',['lookups','hits'])]:
                for key in keys:r[group+'_'+key]=(x.get(group) or {}).get(key)
            records.append(r)
        with (root/'construction_counters.tsv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]),delimiter='\t');writer.writeheader();writer.writerows(records)
        lines+=['', '构建计数见 construction_counters.tsv；只包含完成运行的最终统计，TO 单元为空，不能按零计入。']
    construction={}
    for m in ['cold','warm']:
        construction[m]={}
        tt={(x['arm'],x['design']):x for x in verified['cache_runs'] if x['mode']==m and x['stage']=='ablation'}
        for arm in [x for x in ['cache64K','noAIGREUSE','noLIBSHARE','noQUERY','noTRUTH','allPIVOTS'] if x in arms]:
            paired=[d for d in DESIGNS if all(tt[a,d]['status']=='complete' for a in ['full',arm])]
            ratios=[tt[arm,d]['optimizer_seconds']/tt['full',d]['optimizer_seconds'] for d in paired]
            equal=[d for d in paired if all(datasets[m]['ablation'][arm,d][k]==datasets[m]['ablation']['full',d][k] for k in ['out_a','out_d'])]
            item=dict(completed_pairs=len(paired),completed_pair_designs=paired,completed_pairs_equal_ppa=equal,completed_pair_geomean_runtime_ratio=statistics.geometric_mean(ratios) if ratios else None,completed_pair_total_runtime_ratio=sum(tt[arm,d]['optimizer_seconds'] for d in paired)/sum(tt['full',d]['optimizer_seconds'] for d in paired) if paired else None,all_run_time_ratio=metrics[m]['ablation'][arm]['time_ratio'],full_only_timeout=[d for d in DESIGNS if tt['full',d]['status']=='timeout' and tt[arm,d]['status']=='complete'],arm_only_timeout=[d for d in DESIGNS if tt['full',d]['status']=='complete' and tt[arm,d]['status']=='timeout'])
            construction[m][arm]=item
            lines += [f"{m} {arm}: 双方完成 {len(paired)}/16，其中 PPA 相同 {len(equal)}；完成配对的几何平均时间比 {item['completed_pair_geomean_runtime_ratio']}。不包含双方 TO 的相同预算，也不代表统计加速比。"]
    (root/'construction_comparisons.json').write_text(json.dumps(dict(source_sha256=sha(root/'verified_profiles.json'),comparisons=construction,limitation='Single shared-host measurements; completed-pair subsets vary by arm and mode. Not statistically controlled speedups.'),indent=2)+'\n')
    (root/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    # Rounding uncertainty belongs to the published reference, not our native result.
    ref=report.read_paper_table(str(root/'cold/inputs/compare_with_aspdac25.sh'))
    rounding={}
    for m in ['cold','warm']:
        rr=[]
        for d in DESIGNS:
            p=ref[d];bounds={k:[(p[i]-.005)*(1+(p[j]-.005)/100),(p[i]+.005)*(1+(p[j]+.005)/100)] for k,i,j in [('area',0,2),('delay',3,5)]}
            cost=float(datasets[m]['tab_main']['ae',d]['out_a']);point=p[0]*(1+p[2]/100)
            rr.append(dict(design=d,ours_area=cost,reconstructed_area=point,reference_rounding_bounds=bounds,point_area_win=cost<point,robust_area_win=cost<bounds['area'][0]))
        rounding[m]=dict(area_win_count_robust_to_rounding=sum(x['robust_area_win'] for x in rr),rows=rr)
    rounding.update(assumption='Published starts and changes each rounded to two decimals (+/-0.005 in own units)',script_sha256=sha(root/'cold/inputs/compare_with_aspdac25.sh'))
    (root/'reference_rounding_audit.json').write_text(json.dumps(rounding,indent=2)+'\n')
    print(root/'RESULTS.txt')
if __name__=='__main__':main()
