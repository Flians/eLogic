"""Passive /proc observations of the unmodified, frozen native dec pair.

Sampling stays outside case artifacts. Wall minus CPU is NOT attributed to load.
No CPU counters from this experiment are imputed to historical runs.
"""
import fcntl, importlib.util, json, os, pathlib, sys, threading, time
R=pathlib.Path(__file__).resolve().parents[4]
O=R/'results_v2/emapper_stamp_diagnosis_20261111/live-dec'

def stat(text):
    # comm may contain spaces and closing parentheses; fields after final ')' start at 3.
    a=text[text.rfind(')')+2:].split()
    return dict(ppid=int(a[1]),minor_faults=int(a[7]),major_faults=int(a[9]),
                user_ticks=int(a[11]),system_ticks=int(a[12]),threads=int(a[17]),
                start_ticks=int(a[19]),rss_pages=int(a[21]))

def proc(pid):
    p=pathlib.Path('/proc')/str(pid); s=stat((p/'stat').read_text())
    s['io']={k:int(v) for k,v in (line.split(':') for line in (p/'io').read_text().splitlines())}
    s['schedstat']=(p/'schedstat').read_text().strip()
    s['status']={k:v.strip() for k,v in (line.split(':',1) for line in (p/'status').read_text().splitlines()) if k in ['Cpus_allowed_list','voluntary_ctxt_switches','nonvoluntary_ctxt_switches']}
    return s

def sample(case,stop,out):
    identity=None
    with out.open('x',buffering=1) as f:
        while not stop.wait(1):
            row=dict(monotonic=time.monotonic())
            try:
                a=json.loads((case/'native-affinity.json').read_text()); pid=a['pid']; n=proc(pid)
                current=(pid,n['start_ticks'])
                if identity is not None and current!=identity: raise RuntimeError('PID identity changed')
                identity=current; row.update(pid=pid,native=n,parent=proc(n['ppid']))
                row['cpu_ticks']={k:[int(x) for x in v.split()] for k,v in (l.split(' ',1) for l in pathlib.Path('/proc/stat').read_text().splitlines() if l.startswith(('cpu6 ','cpu38 ')))}
                row['pressure']={k:pathlib.Path('/proc/pressure',k).read_text() for k in ['cpu','io','memory']}
                row['cpu6_khz']=pathlib.Path('/sys/devices/system/cpu/cpu6/cpufreq/scaling_cur_freq').read_text().strip()
                row['cgroup']=pathlib.Path('/proc',str(pid),'cgroup').read_text()
                row['cgroup_cpu_stat']=pathlib.Path('/sys/fs/cgroup/cpu.stat').read_text()
            except (OSError,ValueError,KeyError,RuntimeError) as e: row['unavailable']=repr(e)
            f.write(json.dumps(row)+'\n')

def main():
    # Parser regression: process comm is not whitespace-safe.
    assert stat('1 (a b)c) R 2 '+ ' '.join(['0']*21))['ppid']==2
    path=R/'scripts/experiments/campaigns/emapper_physical_stamp/run.py'
    spec=importlib.util.spec_from_file_location('frozen_stamp',path); m=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(path.parent));spec.loader.exec_module(m)
    c=m.read(m.O/'campaign.json'); assert c['frozen']==m.fingerprint(),'frozen source/binary drift'
    O.mkdir(parents=True,exist_ok=True); locks=[]
    for p in [R/'results_v2/independent_components_20261019/global.lock',m.O/'runner.lock',O/'runner.lock']:
        h=p.open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
    assert not (O/'summary.json').exists(),'closed cohort: do not rerun'
    metadata=dict(clock_ticks=os.sysconf('SC_CLK_TCK'),sampler_affinity=sorted(os.sched_getaffinity(0)),
                  sched_schedstats=pathlib.Path('/proc/sys/kernel/sched_schedstats').read_text().strip(),
                  frozen=c['frozen'],scope='new passive diagnostic, original binary; no historical CPU attribution; CEC excluded from native samples')
    m.save(O/'metadata.json',metadata);m.O=O; rows=[]
    for variant,stamp in [('control','construct'),('borrow','borrow')]:
        job=dict(key='dec_resource_'+variant,design='dec',variant=variant,stamp=stamp,rep=0,stage='resource diagnostic')
        done=O/(job['key']+'-completed.json')
        if done.exists(): rows.append(m.read(done));continue
        case=O/'cases'/job['key'];assert not case.exists(),'partial requires independent audit'
        stop=threading.Event();t=threading.Thread(target=sample,args=(case,stop,O/(job['key']+'-samples.jsonl')),daemon=True)
        print('start',job['key'],flush=True); t.start()
        try: row=m.execute(c,job,locks)
        finally: stop.set();t.join()
        m.save(done,row);rows.append(row); print('end',job['key'],row['status'],row.get('T'),flush=True)
        assert c['frozen']==m.fingerprint(),'source drift'
    m.save(O/'summary.json',dict(rows=rows,scope=metadata['scope']))
if __name__=='__main__': main()
