"""Verify whole-set conflict protection and timestamp-stable installation."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('overlay', ROOT / 'tools/apply_mockturtle_overlay.py')
overlay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlay)
MANIFEST = json.loads((overlay.OVERLAY / 'manifest.json').read_text())


class OverlayTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.destination = Path(self.temporary.name)
        self.before = {}
        for entry in MANIFEST['files']:
            relative = entry['path']
            data = subprocess.check_output(['git', '-C', str(ROOT / 'tools/mockturtle'),
                                            'show', MANIFEST['recorded_upstream_commit'] + ':' + relative])
            path = self.destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            self.before[relative] = data

    def tearDown(self):
        self.temporary.cleanup()

    def test_repeat_install_does_not_trigger_recompile(self):
        self.assertEqual(overlay.apply(self.destination), 4)
        stamps = {e['path']: (self.destination / e['path']).stat().st_mtime_ns for e in MANIFEST['files']}
        self.assertEqual(overlay.apply(self.destination), 0)
        self.assertEqual(overlay.apply(self.destination, check=True), 0)
        for relative, stamp in stamps.items():
            self.assertEqual((self.destination / relative).stat().st_mtime_ns, stamp)

    def test_one_conflict_does_not_replace_other_headers(self):
        last = MANIFEST['files'][-1]['path']
        (self.destination / last).write_bytes(self.before[last] + b'\n// unrelated local change\n')
        with self.assertRaises(ValueError):
            overlay.apply(self.destination)
        for relative, data in self.before.items():
            self.assertEqual((self.destination / relative).read_bytes(),
                             data + b'\n// unrelated local change\n' if relative == last else data)

    def test_read_only_check_does_not_install(self):
        with self.assertRaises(ValueError):
            overlay.apply(self.destination, check=True)
        for relative, data in self.before.items():
            self.assertEqual((self.destination / relative).read_bytes(), data)


if __name__ == '__main__':
    unittest.main()
