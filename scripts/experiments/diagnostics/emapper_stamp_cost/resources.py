"""Summarize sampled intervals; distinguish observed CPU from wall and shared load."""
import argparse,json,pathlib
def summarize(path,hz=100):
 data=[json.loads(l)for l in pathlib.Path(path).read_text().splitlines()]
 rows=[r for r in data if 'native'in r and 'cpu_ticks'in r];assert len(rows)>1
 a,b=rows[0],rows[-1];span=b['monotonic']-a['monotonic']
 def delta(k):return b['native'][k]-a['native'][k]
 cpu=(delta('user_ticks')+delta('system_ticks'))/hz
 intervals=[]
 for x,y in zip(rows,rows[1:]):
  wall=y['monotonic']-x['monotonic'];c=sum(y['native'][k]-x['native'][k]for k in ['user_ticks','system_ticks'])/hz
  intervals.append(dict(monotonic_start=x['monotonic'],wall=wall,cpu=c,wall_minus_cpu=wall-c,
      write_bytes=y['native']['io']['write_bytes']-x['native']['io']['write_bytes'],major_faults=y['native']['major_faults']-x['native']['major_faults']))
 def busy(v):return sum(v[i]for i in [0,1,2,5,6,7])
 cores={k:dict(nonidle_seconds=(busy(b['cpu_ticks'][k])-busy(a['cpu_ticks'][k]))/hz,
                 fraction=(busy(b['cpu_ticks'][k])-busy(a['cpu_ticks'][k]))/hz/span)for k in a['cpu_ticks']}
 sched=[list(map(int,r['native']['schedstat'].split()))for r in [a,b]]
 return dict(samples=len(rows),sampled_wall=span,native_CPU=cpu,native_user=delta('user_ticks')/hz,native_system=delta('system_ticks')/hz,
     wall_minus_CPU=span-cpu,CPU_fraction=cpu/span,minor_faults=delta('minor_faults'),major_faults=delta('major_faults'),
     io_delta={k:b['native']['io'][k]-a['native']['io'][k]for k in a['native']['io']},
     schedstat_delta=dict(runtime_seconds=(sched[1][0]-sched[0][0])/1e9,runqueue_seconds=(sched[1][1]-sched[0][1])/1e9,slices=sched[1][2]-sched[0][2]),
     core_nonidle=cores,cpu6_khz_range=[min(int(r['cpu6_khz'])for r in rows),max(int(r['cpu6_khz'])for r in rows)],
     largest_nonCPU_intervals=sorted(intervals,key=lambda r:r['wall_minus_cpu'],reverse=True)[:8],
     limitations=['1-second sampling misses process startup/exit and has quantization error',
       'wall minus CPU includes blocking, scheduler waits and sampling boundaries; not load by definition',
       'CPU/core/PSI/cgroup counters are observations, shared counters do not identify an external culprit',
       'old cohort has no such samples; new evidence cannot determine historical CPU time'])
def main():
 p=argparse.ArgumentParser();p.add_argument('samples',type=pathlib.Path,nargs='+');p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();assert not a.output.exists()
 a.output.write_text(json.dumps({str(p):summarize(p)for p in a.samples},indent=2)+'\n')
if __name__=='__main__':main()
