#!/usr/bin/env python3
"""Generate publication figures from the verified three-part experiment archive."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics

os.environ.setdefault('MPLCONFIGDIR', '/tmp/epmapping-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from report import PAPER16
from timeout_selection import adapt_row

def rows(path):
    with path.open() as stream:
        values=list(csv.DictReader(stream, delimiter='\t'))
        return [adapt_row(path.parent,r) if 'stage' in r and 'design' in r else r for r in values]

def pct(out, initial):
    return 100 * (float(out) / float(initial) - 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root, args.out
    verified = json.loads((root / 'verified_metrics.json').read_text())
    assert verified['metrics']['whole_window_synthesis'] is False
    for name, expected in verified['raw_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 8,
                         'axes.labelsize': 8, 'axes.titlesize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'legend.fontsize': 7, 'pdf.fonttype': 42,
                         'mathtext.fontset': 'stix', 'axes.spines.top': False,
                         'axes.spines.right': False})
    blue, orange = '#24628d', '#bd6b2c'
    emitted = {}
    def save(fig, name, data):
        fig.savefig(out / f'{name}.pdf', bbox_inches='tight', pad_inches=.045)
        fig.savefig(out / f'{name}.png', dpi=180, bbox_inches='tight', pad_inches=.045)
        plt.close(fig)
        emitted[name] = data

    main_rows = {(r['stage'], r['design']): r for r in rows(root / 'tab_main/summary.tsv')}
    progression = {}
    fig, ax = plt.subplots(figsize=(3.4, 2.25))
    for key, label, color, marker in [('a', 'Area', blue, 'o'), ('d', 'Delay', orange, 's')]:
        values = [0.] + [-statistics.mean(pct(main_rows[(stage, d)][f'out_{key}'],
                                               main_rows[(stage, d)][f'base_{key}'])
                                          for d in PAPER16) for stage in ['ae.r1', 'ae']]
        progression[key] = values
        ax.plot([0, 1, 6], values, marker=marker, color=color, linewidth=1.6, label=label)
        ax.annotate(f'{values[-1]:.2f}%', (6, values[-1]), xytext=(-4, 7),
                    textcoords='offset points', ha='right', color=color, fontsize=7)
    ax.set(xlabel='Initial mapping / configured round cap',
           ylabel='Mean reduction (%)\nfrom own $A_e$ start', xticks=[0, 1, 6], xticklabels=['Baseline', '1', '6'], xlim=(-.15, 6.25))
    ax.grid(axis='y', color='#dddddd', linewidth=.5)
    ax.legend(loc='lower right', frameon=False)
    fig.tight_layout()
    save(fig, 'round_progress', {'round_caps': [None, 1, 6], 'x_positions': [0, 1, 6], 'baseline': 'initial mapping without post-mapping optimization', 'mean_reduction_percent': progression})

    mfs = rows(root / 'tab_mfs3/summary.tsv')
    ours = {r['design']: r for r in mfs if r['stage'] == 'ours'}
    abc = {r['design']: r for r in rows(root / 'tab_mfs3/abc.tsv')}
    native = {r['design']: r for r in rows(root / 'tab_mfs3/abc_baseline_metrics.tsv')}
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.6), sharey=True)
    data = {}
    for ax, key, title in zip(axes, ['a', 'd'], ['Area change (%)', 'Delay change (%)']):
        points = []
        for y, design in enumerate(PAPER16):
            o = pct(ours[design][f'out_{key}'], ours[design][f'base_{key}'])
            a = None
            if abc[design]['rc'] == '0' and abc[design]['cec'] == '1':
                a = pct(abc[design]['area' if key == 'a' else 'delay'],
                        native[design]['abc_area' if key == 'a' else 'abc_delay'])
                ax.plot([o, a], [y, y], color='#c7c7c7', linewidth=.8, zorder=1)
                ax.scatter(a, y, marker='s', s=18, color=orange, zorder=3)
            else:
                ax.text(.985, y, 'TO', transform=ax.get_yaxis_transform(), ha='right',
                        va='center', color='#666666', fontsize=7)
            interrupted=ours[design]['raw_status']=='timeout'
            ax.scatter(o, y, marker='x' if interrupted else 'o', s=18, color=blue, zorder=3)
            points.append({'design': design, 'ours': o, 'abc': a,'ours_status':ours[design]['raw_status']})
        data[key] = points
        ax.axvline(0, color='#777777', linewidth=.7, linestyle='--')
        ax.set(xlabel=title + '; lower is better', yticks=range(len(PAPER16)), yticklabels=PAPER16)
        ax.grid(axis='x', color='#dddddd', linewidth=.5)
    axes[0].invert_yaxis()
    axes[0].scatter([], [], color=blue, marker='o', s=18, label='ePMapping (x: TO input selection)')
    axes[0].scatter([], [], color=orange, marker='s', s=18, label='ABC mfs3 -a')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .93))
    save(fig, 'abc_comparison', data)

    ablation = rows(root / 'ablation/summary.tsv')
    idx = {(r['stage'], r['design']): r for r in ablation}
    designs = [d for d in PAPER16 if d in {r['design'] for r in ablation}]
    arms = ['noSYN', 'noEGG', 'noDB', 'noCARE']
    labels = ['L3 synthesis', 'L4 e-graph', 'Database', 'Care masks']
    if 'noCHAIN' in verified['metrics']['ablation']:
        arms += ['noCHAIN','noSHARE']; labels += ['Chain blocks','Cross-root sharing']
    if 'noNARROW' in verified['metrics']['ablation']:
        arms += ['noNARROW','noPHASE']; labels += ['Narrow initialization','Physical phase reuse']
    if 'noSHANNON' in verified['metrics']['ablation']:
        arms=['noSHANNON','noDAVIO','noREUSE','noPRICE']
        labels=['Shannon','Davio','Existing-INV reuse','Contextual pricing']
    n=len(arms)
    aggregate = [sum(float(idx[(arm, d)]['out_a']) - float(idx[('full', d)]['out_a'])
                     for d in designs) for arm in arms]
    matrix = np.array([[pct(idx[(arm, d)]['out_a'], idx[('full', d)]['out_a'])
                        for arm in arms] for d in designs])
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0), gridspec_kw={'width_ratios': [1, 1.35]})
    axes[0].barh(range(n), aggregate, color=[blue, orange, '#6e8e67', '#8c6da5', '#508b88', '#b76a82', '#a08a55', '#777777'][:n], height=.55)
    axes[0].set(yticks=range(n), yticklabels=labels, xlabel=r'Summed area change ($\mu m^2$)',
                title='(a) Cumulative marginal effect', xlim=(min(0, min(aggregate)*1.25)-.2, max(0, max(aggregate)*1.25)+.2))
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', color='#dddddd', linewidth=.5)
    for y, value in enumerate(aggregate):
        axes[0].annotate(f'{value:+.2f}',(value,y),xytext=(3 if value>=0 else -3,0),
                         textcoords='offset points',ha='left' if value>=0 else 'right',va='center',fontsize=7)
    bound = max(abs(matrix.min()), abs(matrix.max()), .001)
    im = axes[1].imshow(matrix, cmap='RdBu_r', norm=TwoSlopeNorm(0, -bound, bound), aspect='auto')
    axes[1].set(yticks=range(len(designs)), yticklabels=designs, xticks=range(n),
                xticklabels=arms, title='(b) Per-design area change (%)')
    plt.setp(axes[1].get_xticklabels(), rotation=55, ha='right', fontsize=6.5)
    for y in range(len(designs)):
        for x in range(n):
            value = matrix[y, x]
            annotation = '' if abs(value)<.005 else (f'{value:+.0f}' if abs(value)>=100 else f'{value:+.2f}')
            axes[1].text(x, y, annotation, ha='center', va='center', fontsize=6.5,
                         color='white' if abs(value) > bound*.6 else '#222222')
    fig.colorbar(im, ax=axes[1], fraction=.045, pad=.025).ax.tick_params(labelsize=6)
    fig.tight_layout(w_pad=2.)
    save(fig, 'ablation_effects', {'designs': designs, 'arms': arms,
                                 'summed_area_delta': aggregate, 'per_design_percent': matrix.tolist()})

    times = verified['metrics']['cut_oracle_times']
    keys = ['secs_stage1', 'secs_stage2', 'secs_synth', 'secs_egg', 'secs_care']
    values = [times[k] for k in keys]
    assert all(v >= 0 for v in values) and times['secs_window'] == 0
    total = sum(values)
    names = ['L1', 'L2', 'L3', 'L4', 'Care']
    if times.get('secs_chain',0)>0:
        keys += ['secs_chain']; values += [times['secs_chain']]; names += ['Chain txn']
        total=sum(values)
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    wedges, _ = ax.pie(values, startangle=90, counterclock=False,
                       colors=[blue, '#70a6be', orange, '#8c6da5', '#86917a', '#508b88'],
                       wedgeprops={'width': .38, 'edgecolor': 'white', 'linewidth': .7})
    ax.text(0, .06, f'{total:.1f} s', ha='center', va='center', fontsize=9)
    ax.text(0, -.16, 'operators\n+ care', ha='center', va='center', fontsize=7)
    ax.legend(wedges, [f'{name}: {v:.1f} s (' + ('<0.1%' if 0 < 100*v/total < .1 else f'{100*v/total:.1f}%') + ')' for name, v in zip(names, values)],
              loc='center left', bbox_to_anchor=(.94, .5), frameon=False, fontsize=7.5)
    ax.set(aspect='equal')
    fig.tight_layout()
    save(fig, 'oracle_time_share', {'seconds': dict(zip(keys, values)), 'total_seconds': total})
    emitted['_provenance']={'verified_metrics_sha256':hashlib.sha256((root/'verified_metrics.json').read_bytes()).hexdigest(),
                            'root':str(root), 'whole_window_synthesis':False,
                            'vector_pdf_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.pdf'))}}
    (out / 'plot_data.json').write_text(json.dumps(emitted, indent=2)+'\n')
    print(f'Wrote four vector figures and numeric sidecar to {out}')

if __name__ == '__main__':
    main()
