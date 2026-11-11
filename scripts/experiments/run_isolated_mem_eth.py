#!/usr/bin/env python3
"""Serial, CPU-pinned cold/warm probes of unchanged normalized inputs."""
import argparse,json,os,re,shutil,signal,subprocess,time
from pathlib import Path
import run_prebuffer_three_parts as campaign
from report import read_counters

PROJECT=Path(__file__).resolve().parents[2]
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def measured_call(command,env,log,timeout=3600):
    start=time.monotonic();outer=False
    with Path(log).open('w') as out:
        process=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        while True:
            pid,status,usage=os.wait4(process.pid,os.WNOHANG)
            if pid:break
            if time.monotonic()-start>timeout:
                os.killpg(process.pid,signal.SIGKILL);pid,status,usage=os.wait4(process.pid,0);outer=True;break
            time.sleep(.2)
        rc=os.waitstatus_to_exitcode(status);process.returncode=rc
    wall=time.monotonic()-start
    save(Path(str(log)+'.resources.json'),dict(wall_seconds=wall,user_seconds=usage.ru_utime,system_seconds=usage.ru_stime,cpu_seconds=usage.ru_utime+usage.ru_stime,max_rss_kib=usage.ru_maxrss,returncode=rc,outer_guard=outer,affinity=sorted(os.sched_getaffinity(0)),loadavg=os.getloadavg(),scope='whole child including native library setup and CEC; native post timer separate'))
    return (124 if outer else rc),wall

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);ap.add_argument('--engine',type=Path);ap.add_argument('--cpu',type=int,default=15);ap.add_argument('--designs',nargs='+',default=['ethernet','mem_ctrl']);ap.add_argument('--flags-json',type=Path);a=ap.parse_args()
    root=a.root.resolve();source=PROJECT/'results_epmapping_prebuffer_v3_20261111';engine=(a.engine or source/'inputs/engine').resolve()
    os.sched_setaffinity(0,{a.cpu});root.mkdir(parents=True,exist_ok=True);inputs=root/'inputs';inputs.mkdir(exist_ok=True)
    for p in (source/'inputs').iterdir():
        if p.name in ['engine','sources']:continue
        q=inputs/p.name
        if not q.exists():q.symlink_to(p.resolve(),target_is_directory=p.is_dir())
    if not(inputs/'engine').exists():shutil.copy2(engine,inputs/'engine')
    assert campaign.common.sha(inputs/'engine')==campaign.common.sha(engine)
    sources=root/'probe_sources'
    if not sources.exists():
        sources.mkdir()
        for p in (PROJECT/'epmapping').iterdir():
            if p.is_file()and p.suffix in ['.hpp','.cpp','.json']:shutil.copy2(p,sources/p.name)
        shutil.copytree(PROJECT/'map_egg/src',sources/'map_egg_src')
        shutil.copy2(Path(__file__),sources/'run_isolated_mem_eth.py')
    if not(root/'normalized').exists():(root/'normalized').symlink_to(source/'normalized',target_is_directory=True)
    m=json.loads((source/'manifest.json').read_text());m['engine_sha256']=campaign.common.sha(inputs/'engine')
    flags=json.loads(a.flags_json.read_text())if a.flags_json else {}
    manifest=dict(source_snapshot_sha256={str(p.relative_to(sources)):campaign.common.sha(p)for p in sources.rglob('*')if p.is_file()},source=str(source),engine_source=str(engine),engine_sha256=m['engine_sha256'],goal=m['goal'],overrides=flags,cpu=a.cpu,jobs=1,designs=a.designs,cold='empty private pool',warm='new process, completed paired cold pool',host_exclusive=False,source_inputs_sha256={str(p.relative_to(source/'inputs')):campaign.common.sha(p)for p in (source/'inputs').rglob('*')if p.is_file()and p.name.endswith(('.aig','.genlib'))})
    if(root/'manifest.json').exists():assert json.loads((root/'manifest.json').read_text())==manifest
    else:save(root/'manifest.json',manifest)
    campaign.common.call=measured_call;rows=[]
    for design in a.designs:
        stage='mfs3'if design=='ethernet'else'main'
        for mode in ['cold','warm']:
            r=campaign.native(root,m,stage,'ours',design,mode,root/'cache'/design,flags)
            counters=read_counters(str(root/stage/'ours'/f'{mode}.{design}'/'log'))
            target=(3673.56,546.58)if design=='ethernet'else(1801.85,1461.58)
            row=dict(design=design,mode=mode,status=r['status'],area=r['final']['area'],delay=r['final']['delay'],seconds=r['optimizer_seconds'],target_area=target[0],target_delay=target[1],quality_pass=r['final']['area']<=target[0]+.0001 and r['final']['delay']<=target[1]+.001,completion_pass=r['status']=='complete',counters={k:v for k,v in counters.items()if k.startswith(('secs_','time_'))or k=='egg_calls'})
            wd=root/stage/'ours'/f'{mode}.{design}'
            log=(wd/'log').read_text();profile=re.findall(r'egraph_profile (.*)',log)
            if profile:row['profile']={k:float(v) if any(t in v for t in '.eE')else int(v)for k,v in re.findall(r'(\w+)=([0-9.eE+-]+)',profile[-1])}
            phases=[dict(re.findall(r'(\w+)=([^ ]+)',line))for line in log.splitlines()if line.startswith('[phase]')]
            if phases:save(wd/'phase_timing.json',phases);row['phase_timing']=str((wd/'phase_timing.json').relative_to(root))
            row['resource']=json.loads((wd/'log.resources.json').read_text())
            rows.append(row);save(root/'summary.json',rows);print(json.dumps(row),flush=True)
    save(root/'complete.json',dict(complete=True,observations=len(rows),quality_pass=all(r['quality_pass']for r in rows),completion_pass=all(r['completion_pass']for r in rows)))
if __name__=='__main__':main()
