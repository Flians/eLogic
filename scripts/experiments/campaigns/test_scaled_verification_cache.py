import json
import tempfile
import unittest
from pathlib import Path

from scaled_verification_cache import key, lookup, sha


class VerificationReuseTest(unittest.TestCase):
    def test_changed_input_netlist_and_proof_cannot_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = [root / name for name in ["original", "library", "raw", "evaluator"]]
            for i, path in enumerate(paths):
                path.write_text(str(i))
            proof = root / "cec.log"
            proof.write_text("checked fixture")
            cache = root / "cache.json"
            entry = dict(measured=dict(cec=1), raw_verilog=str(paths[2]),
                         proof_files={str(proof): sha(proof)})
            cache.write_text(json.dumps(dict(entries={key(*paths): entry})))
            self.assertEqual(lookup(cache, *paths), entry)
            paths[0].write_text("changed original")
            self.assertIsNone(lookup(cache, *paths))
            paths[0].write_text("0")
            paths[2].write_text("changed netlist")
            self.assertIsNone(lookup(cache, *paths))
            paths[2].write_text("2")
            proof.write_text("corrupt")
            with self.assertRaises(ValueError):
                lookup(cache, *paths)


if __name__ == "__main__":
    unittest.main()
