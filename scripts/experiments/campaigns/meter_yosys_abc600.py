#!/usr/bin/env python3
"""Transparent executable shim: unchanged Yosys-selected ABC argv, shared clock."""
import hashlib,json,os,re,shutil,signal,subprocess,sys,uuid
from pathlib import Path
from nested_mapping_clock import Clock
def capture(argv,key,clock_path):
    folder=Path(clock_path).parent/(Path(clock_path).stem+'.yosys-abc')/(re.sub(r'[^A-Za-z0-9_.-]','_',key)+'-'+uuid.uuid4().hex)
    folder.mkdir(parents=True)
    script=Path(argv[argv.index('-f')+1]).resolve()if '-f'in argv else None
    record=dict(original_argv=argv,case=key,cwd=os.getcwd(),ABC_binary_sha256=hashlib.sha256(Path(argv[0]).read_bytes()).hexdigest(),files_before={},files_after={})
    def snapshot(label):
        files=[script]if script and script.is_file()else[]
        if script and ('yosys-abc' in script.parent.name or 'abc-temp' in script.parent.name):
            files=[p for p in script.parent.iterdir()if p.is_file()]
        dest=folder/label;dest.mkdir()
        for p in files:
            shutil.copy2(p,dest/p.name)
            record['files_'+label][str(p)]=dict(archive=str(dest/p.name),sha256=hashlib.sha256((dest/p.name).read_bytes()).hexdigest())
        (folder/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    snapshot('before')
    return record,snapshot,folder
def main():
    clock=Clock(os.environ['MAPPING_CLOCK_PATH'],os.environ['MAPPING_CLOCK_SECONDS'])
    key=os.environ['MAPPING_CLOCK_CASE'];argv=[os.environ['METER_REAL_YOSYS_ABC'],*sys.argv[1:]]
    record,snapshot,folder=capture(argv,key,os.environ['MAPPING_CLOCK_PATH'])
    token,remaining=clock.begin(key,argv)
    if remaining<=0:
        record.update(exit_code=124,not_launched_budget_exhausted=True);snapshot('after');return 124
    code=None
    try:
        p=subprocess.Popen(argv,start_new_session=True)
        try:code=p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL);p.wait();code=124
    finally:
        clock.end(key,token)
        record['exit_code']=code;snapshot('after')
    return code
if __name__=='__main__':raise SystemExit(main())
