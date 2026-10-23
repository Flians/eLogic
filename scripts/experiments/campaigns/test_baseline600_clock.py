"""Real subprocess checks of timing instrumentation; baseline algorithms untouched."""
import json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
from nested_mapping_clock import Clock
from run_baseline600_budget import seed_lut_clock

HERE = Path(__file__).resolve().parent

class BaselineClockTests(unittest.TestCase):
    def test_yosys_shim_preserves_arguments_and_excludes_idle(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            binary = folder/'yosys-abc'
            work=folder/'yosys-abc-fixture';work.mkdir();script=work/'abc.script';script.write_text('read_blif input.blif; if -K 6; write_blif output.blif\n');(work/'input.blif').write_text('input fixture\n')
            binary.write_text('#!/usr/bin/env python3\nimport json,sys,time\nfrom pathlib import Path\nprint(json.dumps(sys.argv[1:]),flush=True)\ntime.sleep(.06)\nPath(sys.argv[sys.argv.index("-f")+1]).with_name("output.blif").write_text("mapped fixture")\n')
            binary.chmod(0o700)
            env = dict(os.environ, MAPPING_CLOCK_PATH=str(folder/'clock.json'),
                       MAPPING_CLOCK_SECONDS='2', MAPPING_CLOCK_CASE='test/lut',
                       METER_REAL_YOSYS_ABC=str(binary))
            args=['-s','argument with spaces','-f',str(script)]
            result = subprocess.run([str(HERE/'meter_yosys_abc600.py'), *args],
                                    env=env, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout), args)
            manifests=list((folder/'clock.yosys-abc').glob('*/manifest.json'));self.assertEqual(len(manifests),1)
            manifest=json.loads(manifests[0].read_text())
            self.assertEqual(manifest['original_argv'],[str(binary),*args])
            self.assertEqual(Path(manifest['files_before'][str(script)]['archive']).read_bytes(),script.read_bytes())
            self.assertEqual(Path(manifest['files_after'][str(work/'output.blif')]['archive']).read_text(),'mapped fixture')
            clock = Clock(folder/'clock.json', 2)
            before = json.loads(clock.path.read_text())['cases']['test/lut']['used']
            import time
            time.sleep(.1)
            clock.update('test/lut', 'snapshot')
            state = json.loads(clock.path.read_text())['cases']['test/lut']
            self.assertEqual(state['used'], before)
            self.assertFalse(state['active'])
            self.assertGreater(before, .05)
            self.assertLess(before, 1)

    def test_yosys_shim_timeout_closes_interval_and_preserves_output(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); binary = folder/'yosys-abc'
            binary.write_text('#!/usr/bin/env python3\nimport time\nprint("partial",flush=True)\ntime.sleep(10)\n')
            binary.chmod(0o700)
            env = dict(os.environ, MAPPING_CLOCK_PATH=str(folder/'clock.json'),
                       MAPPING_CLOCK_SECONDS='.15', MAPPING_CLOCK_CASE='test/lut',
                       METER_REAL_YOSYS_ABC=str(binary))
            result = subprocess.run([str(HERE/'meter_yosys_abc600.py')], env=env,
                                    capture_output=True, text=True, timeout=2)
            self.assertEqual(result.returncode, 124)
            self.assertIn('partial', result.stdout)
            state = json.loads((folder/'clock.json').read_text())['cases']['test/lut']
            self.assertTrue(state['exhausted']); self.assertFalse(state['active'])

    def test_shared_lut_prefix_once_for_each_affected_method(self):
        with tempfile.TemporaryDirectory() as temp:
            clock = Clock(Path(temp)/'clock.json', 600)
            clock.update('test/__lut_prep__', 'snapshot')
            data = json.loads(clock.path.read_text())
            data['cases']['test/__lut_prep__']['used'] = 3.25
            clock.path.write_text(json.dumps(data))
            for method in ['lut_stage_amap', 'eqmap_lut_amap']:
                key = 'test/'+method
                seed_lut_clock(clock, key, 'test/__lut_prep__')
                seed_lut_clock(clock, key, 'test/__lut_prep__')
            clock.update('test/abc_amap', 'snapshot')
            cases = json.loads(clock.path.read_text())['cases']
            self.assertEqual(cases['test/lut_stage_amap']['used'], 3.25)
            self.assertEqual(cases['test/eqmap_lut_amap']['used'], 3.25)
            self.assertEqual(cases['test/abc_amap']['used'], 0)

    def test_driver_attributes_nested_and_direct_mapping_but_not_cec(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp); out=folder/'output';out.mkdir()
            binary=folder/'yosys-abc'
            binary.write_text('#!/usr/bin/env python3\nimport time\ntime.sleep(.06)\n');binary.chmod(0o700)
            abc=folder/'abc';abc.write_bytes(binary.read_bytes());abc.chmod(0o700)
            yosys=folder/'yosys'
            yosys.write_text('#!/usr/bin/env python3\nimport re,subprocess,sys\nshim=re.search(r\'abc -exe "([^\"]+)"\',sys.argv[-1]).group(1)\nraise SystemExit(subprocess.call([shim,"-c","original LUT script"]))\n');yosys.chmod(0o700)
            driver=folder/'bench_eqmap_baseline.py'
            restrict=folder/'restrict_genlib.py';marker=folder/'library-range'
            restrict.write_text(f'import sys\nfrom pathlib import Path\nPath({str(marker)!r}).write_text(sys.argv[-1])\n')
            driver.write_text('import subprocess,sys\nbench="fixture"\n'
                +f'subprocess.run([sys.executable,{str(restrict)!r},"--max-pins","6"],check=True)\n'
                +f'subprocess.run([{str(yosys)!r},"-p","read_aiger input; abc -lut 6; write_verilog output"],check=True)\n'
                +'for method in ["abc_amap","lut_stage_amap","eqmap_lut_amap"]:\n'
                +' row={"bench":bench,"method":method}\n'
                +f' subprocess.run([{str(abc)!r},"-c","amap"],check=True,timeout=600)\n'
                +' subprocess.run([sys.executable,"-c","import time;time.sleep(.1)"],check=True,timeout=.001)\n')
            audit=folder/'audit.json'
            result=subprocess.run([sys.executable,str(HERE/'run_baseline600_budget.py'),
                '--mode','wide-cells','--profile','0','--full-evaluator',str(binary),
                '--mapping-seconds','600','--driver',str(driver),'--audit',str(audit),'--'],
                capture_output=True,text=True,timeout=5)
            self.assertEqual(result.returncode,0,result.stderr)
            data=json.loads(audit.read_text());cases=data['clock']
            adapted=next(x for x in data['commands']if str(restrict)in(x.get('argv')or[]))
            self.assertEqual(adapted['driver_argv'][-1],'6')
            self.assertEqual(adapted['argv'][-1],'9')
            self.assertEqual(marker.read_text(),'9')
            prefix=cases['fixture/__lut_prep__']['used']
            self.assertGreater(prefix,.05)
            self.assertEqual(len(cases['fixture/__lut_prep__']['events']),2)
            for method in ['lut_stage_amap','eqmap_lut_amap']:
                state=cases['fixture/'+method]
                self.assertEqual(state['lut_prefix_mapping_secs'],prefix)
                self.assertGreater(state['used'],prefix+.05)
                self.assertLess(state['used'],prefix+.15)
            self.assertNotIn('lut_prefix_mapping_secs',cases['fixture/abc_amap'])
            self.assertIsNone(data['external_driver_process_guard_secs'])

if __name__ == '__main__': unittest.main()
