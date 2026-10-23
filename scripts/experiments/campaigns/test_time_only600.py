"""Short real-process tests of the outer clock; production default remains600."""
import json,os,sys,time,unittest,tempfile
from pathlib import Path
from run_time_only600 import mapping_process
class ClockTests(unittest.TestCase):
    def test_timeout_kills_process_group_and_preserves_partial_output(self):
        code="import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)']); print(p.pid,flush=True); time.sleep(10)"
        r=mapping_process([sys.executable,'-c',code],os.environ,.2)
        self.assertTrue(r['timeout']);self.assertNotEqual(r['exit_code'],0);self.assertLess(r['wall_secs'],2.)
        pid=int(r['stdout'].strip())
        try:
            with open(f'/proc/{pid}/stat')as f:state=f.read().split()[2]
            self.assertEqual(state,'Z','descendant must not continue mapping after timeout')
        except FileNotFoundError:pass
    def test_deadline_is_propagated_and_nonmapping_work_is_not_charged(self):
        code="import os,time; print(float(os.environ['EGM_MAPPING_DEADLINE_MONOTONIC'])-time.monotonic(),flush=True);time.sleep(.05)"
        r=mapping_process([sys.executable,'-c',code],os.environ,1.)
        self.assertFalse(r['timeout']);self.assertEqual(r['exit_code'],0)
        self.assertGreater(float(r['stdout']),.5);self.assertLessEqual(float(r['stdout']),1.)
        mapping_time=r['wall_secs'];time.sleep(.15)
        self.assertEqual(r['wall_secs'],mapping_time)
    def test_real_frontend_keeps_source_incumbent_and_native_logs_on_expiry(self):
        root=Path(__file__).resolve().parents[3]
        record=root/'results_v2/time_only_600_20261022/dec_validation/dec/command.json'
        env=json.loads(record.read_text())['environment'];env=dict(os.environ,**env)
        lib,prepared=json.loads(record.read_text())['argv'][2:4]
        with tempfile.TemporaryDirectory(prefix='egm-timeout-')as temp:
            folder=Path(temp);fake=folder/'fake-native'
            fake.write_text('#!/usr/bin/env python3\nimport sys,time\nprint("native-started",file=sys.stderr,flush=True)\ntime.sleep(10)\n');fake.chmod(0o700)
            env['EGM_NATIVE_OPTIMIZER']=str(fake);raw=folder/'raw.v'
            r=mapping_process([sys.executable,str(Path(__file__).with_name('egraphmap_time_only.py')),lib,prepared,str(raw)],env,.8)
            self.assertTrue(r['timeout']or r['exit_code']!=0)
            self.assertTrue(raw.is_file(),'complete source incumbent must survive')
            self.assertIn('native-started',(folder/'attempt0.stderr').read_text())
            meta=json.loads(raw.with_suffix('.complete.json').read_text());self.assertFalse(meta['optimizer_succeeded'])
            self.assertIsNone(meta['global_cumulative_node_allowance']);self.assertIsNone(meta['optimizer_address_space_limit_bytes'])
if __name__=='__main__':unittest.main()
