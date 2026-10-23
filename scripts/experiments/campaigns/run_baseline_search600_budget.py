"""Search-inclusive600 timing instrumentation; original baseline algorithms unchanged."""
import argparse,fcntl,hashlib,inspect,json,os,re,runpy,subprocess,sys,time
from pathlib import Path
import run_with_mapping_budget as old_budget
from nested_mapping_clock import Clock
HERE=Path(__file__).resolve().parent

def seed_lut_clock(clock,key,prep):
    clock.update(key,'snapshot')
    with clock.path.open('r+')as f:
        fcntl.flock(f,fcntl.LOCK_EX);data=json.load(f);s=data['cases'][key]
        if not s.get('lut_prefix_seeded'):
            prefix=data['cases'].get(prep,{});s['used']+=prefix.get('used',0.)
            s['lut_prefix_seeded']=True;s['lut_prefix_mapping_secs']=prefix.get('used',0.)
            s['incomplete']|=prefix.get('incomplete',False);s['exhausted']|=prefix.get('exhausted',False)
            f.seek(0);f.truncate();json.dump(data,f,indent=2)

def search_command(argv):
    if not isinstance(argv, (list, tuple)) or not argv:
        return False
    # Author native rewriters and extractors perform mapping-driven search.
    # Classification uses executable identity, never path substrings.
    return Path(str(argv[0])).name in {'e-rewriter', 'extraction-gym'}

def execute():
    ap=argparse.ArgumentParser();ap.add_argument('--driver',type=Path,required=True);ap.add_argument('--mapping-seconds',type=float,required=True);ap.add_argument('--audit',type=Path,required=True);ap.add_argument('args',nargs=argparse.REMAINDER)
    a=ap.parse_args();assert a.mapping_seconds==600
    driver=a.driver.resolve();args=a.args[1:]if a.args[:1]==['--']else a.args
    clock_path=a.audit.resolve().with_suffix('.clock.json');assert not clock_path.exists()
    clock=Clock(clock_path,600);real_init=subprocess.Popen.__init__;real_communicate=subprocess.Popen.communicate;real_read=Path.read_text
    commands=[];code=0
    def context():
        frame=inspect.currentframe().f_back;bench=None
        while frame:
            if frame.f_code.co_filename==str(driver):
                row=frame.f_locals.get('row')
                if isinstance(row,dict)and 'bench'in row and 'method'in row:return row['bench']+'/'+row['method']
                bench=frame.f_locals.get('bench',bench)
            frame=frame.f_back
        return str(bench)+'/__lut_prep__'if bench and driver.name=='bench_eqmap_baseline.py'else'shared-setup'
    def init(p,argv,*pos,**kw):
        key=context();argv=list(argv)if isinstance(argv,(list,tuple))else argv
        driver_argv=list(argv)if isinstance(argv,list)else argv
        env=dict(kw.get('env')or os.environ);env.update(MAPPING_CLOCK_PATH=str(clock_path),MAPPING_CLOCK_SECONDS='600',MAPPING_CLOCK_CASE=key)
        if isinstance(argv,list)and argv and Path(str(argv[0])).name.startswith('yosys')and '-p'in argv:
            index=argv.index('-p')+1;script=argv[index]
            if re.search(r'(^|;)\s*abc\s+-lut\b',script):
                abc=Path(argv[0]).resolve().parent/'yosys-abc';assert abc.is_file()
                shim=HERE/'meter_yosys_abc600.py';assert os.access(shim,os.X_OK)
                env['METER_REAL_YOSYS_ABC']=str(abc)
                argv[index]=re.sub(r'(^|;)(\s*)abc(?=\s+-lut\b)',lambda m:m[1]+m[2]+f'abc -exe "{shim}"',script)
        kw['env']=env;search_charged=search_command(argv) and '--version' not in argv
        charged=(old_budget.mapping(argv) or search_charged) and '--version'not in argv
        if driver.name=='bench_eqmap_baseline.py'and key.split('/')[-1]in['eqmap_lut_amap','lut_stage_amap']:
            seed_lut_clock(clock,key,key.split('/')[0]+'/__lut_prep__')
        token=None
        record=dict(argv=None,driver_argv=driver_argv,key=key,mapping_charged=charged,search_charged=search_charged,launched=False)
        commands.append(record)
        if charged:
            token, remaining=clock.begin(key,argv)
            if remaining<=0:
                record['not_launched_budget_exhausted']=True
                raise RuntimeError('600s mapping limit exhausted before launch')
        try:real_init(p,argv,*pos,**kw)
        except BaseException as error:
            record['launch_error']=repr(error)
            if token:clock.end(key,token)
            raise
        # Library/evaluator adapters wrap this constructor. Popen.args records
        # the actual post-adaptation argv, including max-pins9, not the driver's
        # pre-adaptation max-pins6 argument.
        record.update(argv=p.args,launched=True)
        p._meter600=(key,token,charged,time.monotonic())
    def communicate(p,*pos,**kw):
        key,token,charged,start=getattr(p,'_meter600',('shared-setup',None,False,time.monotonic()))
        if charged and token and not getattr(p,"_meter600_cleanup",False):
            remain=clock.update(key,'snapshot');pos=list(pos)
            if len(pos)>1:pos[1]=min(remain,pos[1])if pos[1]is not None else remain
            else:kw['timeout']=min(remain,kw['timeout'])if kw.get('timeout')is not None else remain
        elif not charged:
            # The previous driver-wide watchdog also killed conversion, CEC,
            # and nonmapping work. Search binaries are now charged above.
            pos=list(pos)
            if len(pos)>1:pos[1]=None
            else:kw['timeout']=None
        completed=False
        try:out=real_communicate(p,*pos,**kw);completed=True;return out
        except subprocess.TimeoutExpired:
            p._meter600_cleanup=True;raise
        finally:
            if completed and token:
                clock.end(key,token);p._meter600=(key,None,charged,start)
    capture=driver.parent/'capture_emorphic_abc.py'
    def read(path,*pos,**kw):
        text=real_read(path,*pos,**kw)
        if driver.name=='bench_emorphic_sa.py'and path.resolve()==capture:
            return 'import sys\nsys.path.insert(0,'+repr(str(HERE))+')\nfrom nested_mapping_clock import install_capture_clock\ninstall_capture_clock()\n'+text
        return text
    subprocess.Popen.__init__=init;subprocess.Popen.communicate=communicate;Path.read_text=read
    sys.path.insert(0,str(driver.parent));sys.argv=[str(driver),*args]
    try:runpy.run_path(str(driver),run_name='__main__')
    except SystemExit as e:code=e.code if isinstance(e.code,int)else 1
    except Exception as e:code=1;print(repr(e),file=sys.stderr)
    finally:
        subprocess.Popen.__init__=real_init;subprocess.Popen.communicate=real_communicate;Path.read_text=real_read
        states={}
        if clock_path.exists():
            for key in list(json.loads(clock_path.read_text())['cases']):clock.update(key,'snapshot')
            states=json.loads(clock_path.read_text())['cases']
        audit=dict(policy='600s union of mapping-driven search and mapping intervals; resyn2/pure conversion/independent CEC/metrics excluded',mapping_seconds=600,native_parameters_changed=False,search_included=True,nested_mapping_double_counted=False,driver=str(driver),driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),argv=args,commands=commands,clock=states,external_driver_process_guard_secs=None,exact_eqmap_nested_ABC_meter=True,eqmap_shared_LUT_prefix_charged_once_per_affected_method=True,exit_code=code)
        if '--output'in args:
            path=Path(args[args.index('--output')+1])/'report.json'
            if path.exists():
                report=json.loads(path.read_text());report['wall_budget_policy']={k:audit[k]for k in ['mapping_seconds','policy','external_driver_process_guard_secs']}
                for row in report.get('results',[]):
                    state=states.get(row['bench']+'/'+row['method'],{})
                    if state.get('exhausted')or state.get('incomplete')or state.get('active'):
                        row['partial_status']=row.get('status');row['partial_measured']=row.pop('measured',None);row.update(status='failed',error='600s mapping limit exhausted or incomplete mapping interval');code=1
                path.write_text(json.dumps(report,indent=2)+'\n')
        audit['exit_code']=code;a.audit.write_text(json.dumps(audit,indent=2)+'\n')
    raise SystemExit(code)

def main():
    import run_wide_cells_budget as adapter
    old_budget.main=execute
    adapter.main()
if __name__=='__main__':main()
