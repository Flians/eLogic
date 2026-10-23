import json,os,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from nested_mapping_clock import Clock
from run_baseline_total600_budget import search_command,time_only_native_argv
HERE=Path(__file__).resolve().parent
class TotalClockTests(unittest.TestCase):
    def test_eqmap_resource_flags_only(self):
        args=['/tmp/eqmap_fpga','in.v','out.v','-k','6','-t','2','-s','20000','-n','16','--min-depth']
        changed=time_only_native_argv(args)
        self.assertEqual(changed,['/tmp/eqmap_fpga','in.v','out.v','-k','6','-t','600','--min-depth'])
        self.assertEqual(args[args.index('-s')+1],'20000')

    def test_exact_search_executable_classification(self):
        for name in ['e-rewriter','extraction-gym']:self.assertTrue(search_command(['/tmp/'+name,'input']))
        for name in ['bench_rs_netlist_eval','yosys','e-rewriter-not-real']:self.assertFalse(search_command(['/tmp/'+name]))
    def test_parent_and_nested_mapping_are_union(self):
        with tempfile.TemporaryDirectory()as d:
            c=Clock(Path(d)/'c.json',1);p,_=c.begin('b/m',['extraction-gym']);time.sleep(.04)
            t,_=c.begin('b/m',['abc']);time.sleep(.06);c.end('b/m',t);time.sleep(.04);c.end('b/m',p)
            used=json.loads(c.path.read_text())['cases']['b/m']['used'];self.assertGreater(used,.13);self.assertLess(used,.19)
    def test_search_timeout_and_cec_exclusion(self):
        for seconds,expected in [(.04,0),(.5,1)]:
            with self.subTest(seconds=seconds),tempfile.TemporaryDirectory()as d:
                p=Path(d);binary=p/'e-rewriter';binary.write_text(f'#!{sys.executable}\nimport time\nprint("partial search",flush=True)\ntime.sleep({seconds})\n');binary.chmod(0o700)
                driver=p/'bench_esyn_baseline.py';driver.write_text('import subprocess,os,signal,sys\nbench="fixture"\nrow={"bench":bench,"method":"esyn"}\n'
                  +f'p=subprocess.Popen([{str(binary)!r}],stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)\n'
                  +'try:\n p.communicate(timeout=600)\nexcept subprocess.TimeoutExpired:\n os.killpg(p.pid,signal.SIGKILL);p.communicate();raise SystemExit(1)\n'
                  +'subprocess.run([sys.executable,"-c","import time;time.sleep(.3)"],check=True,timeout=.001)\n')
                audit=p/'a.json'
                code='import sys\nfrom pathlib import Path\nsys.path.insert(0,'+repr(str(HERE))+')\nimport run_baseline_total600_budget as m\nfrom nested_mapping_clock import Clock\nm.Clock=lambda p,n:Clock(p,.15)\nsys.argv='+repr(['meter','--mode','wide-cells','--profile','0','--full-evaluator',str(binary),'--mapping-seconds','600','--driver',str(driver),'--audit',str(audit),'--'])+'\nm.main()\n'
                result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=4)
                self.assertEqual(result.returncode,expected,result.stderr)
                a=json.loads(audit.read_text());state=a['clock']['fixture/esyn'];self.assertFalse(state['active']);self.assertTrue(a['search_included']);self.assertTrue(a['adapter_search_limits_removed'])
                native=next(x for x in a['commands']if x['search_charged'])
                self.assertEqual(native['effective_search_environment'],dict(ESYN_TIME='600',ESYN_NODES=str(2**64-1),ESYN_ITER=str(2**64-1)))
                if expected:self.assertTrue(state['exhausted'])
                else:self.assertLess(state['used'],.15)
if __name__=='__main__':unittest.main()
