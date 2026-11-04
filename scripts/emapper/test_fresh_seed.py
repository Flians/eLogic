"""Clock arithmetic for the unchanged seed component; CEC is not charged twice."""
import unittest
from fresh_seed import remaining_mapping_seconds

class FreshSeedClockTests(unittest.TestCase):
    def test_seed_charge_reduces_refinement_window_without_renewing_budget(self):
        self.assertEqual(remaining_mapping_seconds(600,0),600)
        self.assertEqual(remaining_mapping_seconds(600,1.25),598.75)
        self.assertEqual(1.25+remaining_mapping_seconds(600,1.25),600)

    def test_invalid_or_exhausted_budget_cannot_launch_search(self):
        for value in [True, -1, float('nan'), float('inf'),600,601]:
            with self.assertRaises(ValueError):remaining_mapping_seconds(600,value)


class FreshSeedReceiptTests(unittest.TestCase):
    def fixture(self, folder, body):
        import sys
        from pathlib import Path
        binary=Path(folder)/'component';binary.write_text('#!'+sys.executable+'\n'+body+'\n');binary.chmod(0o755)
        library=Path(folder)/'lib';library.write_text('library')
        subject=Path(folder)/'input.aig';subject.write_bytes(b'aig')
        return binary,library,subject

    def test_receipt_binds_unchanged_argv_and_escaped_output(self):
        import tempfile,json
        from pathlib import Path
        from fresh_seed import generate
        with tempfile.TemporaryDirectory() as temp:
            paths=self.fixture(temp,"import pathlib,sys; pathlib.Path(sys.argv[3]).write_text('module top(input a[0], output y); endmodule')")
            output,receipt=generate(*paths,Path(temp)/'seed',10)
            self.assertEqual(receipt['argv'][-2:],['delay','emap'])
            self.assertIn('\\a[0] ',output.read_text())
            self.assertEqual(receipt['status'],'ok')
            self.assertFalse(receipt['baseline_algorithm_changed'])
            self.assertEqual(json.loads((output.parent/'receipt.json').read_text()),receipt)
            self.assertGreaterEqual(receipt['mapping_seconds'],0)

    def test_failed_seed_keeps_raw_command_logs_and_receipt(self):
        import tempfile,json,subprocess
        from pathlib import Path
        from fresh_seed import generate
        with tempfile.TemporaryDirectory() as temp:
            paths=self.fixture(temp,"import sys; print('native failure',file=sys.stderr); sys.exit(3)")
            folder=Path(temp)/'seed'
            with self.assertRaises(subprocess.CalledProcessError):generate(*paths,folder,10)
            receipt=json.loads((folder/'receipt.json').read_text())
            self.assertEqual(receipt['exit_code'],3)
            self.assertEqual(receipt['status'],'failed')
            self.assertIn('native failure',(folder/'stderr.log').read_text())

if __name__=='__main__':unittest.main()
