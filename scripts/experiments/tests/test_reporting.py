#!/usr/bin/env python3
"""Guard against silently reporting failed runs and misnormalizing stage time."""
import importlib.util, subprocess, sys, tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('report',root/'scripts/experiments/report.py')
rep=importlib.util.module_from_spec(spec);spec.loader.exec_module(rep)
assert rep.verified({'rc':'0','cec':'1'})
assert not rep.verified({'rc':'0','cec':'0'})
assert not rep.verified({'rc':'142','cec':'1'})
with tempfile.TemporaryDirectory(prefix='epm-report-') as tmp:
 r=Path(tmp);(r/'ablation/full.sasc').mkdir(parents=True)
 hdr='stage\tdesign\trecipe\trounds\tbase_a\tbase_d\tout_a\tout_d\tdA%\tdD%\trnd\tcec\tsecs\trc\n'
 (r/'ablation/summary.tsv').write_text(hdr+'full\tsasc\thigh_effort\t6\t2\t3\t1\t3\t-50\t0\t1\t1\t9\t0\n'+'noSYN\tsasc\thigh_effort\t6\t2\t3\t2\t3\t0\t0\t1\t0\t9\t0\n')
 (r/'ablation/full.sasc/log').write_text('[i] epmapping staged_secs cut_enum=100 care=1 stage1=2 stage2=3 egg=4 mffc=200\n[i] epmapping synth roots=1 covers=1 wins=1 secs=10\n[i] epmapping window_synth windows=1 applied=1 gain=1 actual=1 gates=1 abandoned=0 delay_rejected=0 secs=300\n')
 assert rep.read_summary_area(str(r),'ablation','noSYN','sasc') is None
 text=[];rep.figure_data(str(r),text);output='\n'.join(text)
 assert 'Total 20.0 s' in output,output
 assert '| L3 synthesis | 10.0 | 50.0 |' in output,output
 (r/'tab_mfs3').mkdir()
 (r/'tab_mfs3/summary.tsv').write_text(hdr+
  'base\tsasc\tdirect\t0\t100\t1000\t-\t-\t-\t-\t-\t-\t0\t0\n'+
  'ours\tsasc\tdirect\t1\t100\t1000\t90\t900\t-10\t-10\t1\t1\t9\t0\n')
 (r/'tab_mfs3/abc.tsv').write_text('design\tarea\tdelay\tsecs\trc\tcec\n'+'sasc\t99\t990\t1\t0\t1\n')
 (r/'tab_mfs3/abc_baseline_metrics.tsv').write_text('design\tabc_area\tabc_delay\n'+'sasc\t110\t1100\n')
 result=subprocess.run([sys.executable,str(root/'scripts/experiments/report.py'),'--root',str(r)],cwd=root,capture_output=True,text=True,check=True)
 assert '**`mfs3 -a`**: mean dA = -10.00%, mean dD = -10.00%' in result.stdout,result.stdout
 print('verified-row and oracle-time-denominator regression checks passed')
