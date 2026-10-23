import tempfile
import unittest
from pathlib import Path

from input_capacity_policy_v2 import input_capacity


class InputCapacityPolicyTest(unittest.TestCase):
    def test_all_negative_pi_outputs_fit_the_bound(self):
        # Four PIs, four NOTs and at least one constant: v1's bound was six.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "negative_pi.aag"
            path.write_text("aag 4 4 0 4 0\n2\n4\n6\n8\n3\n5\n7\n9\n")
            capacity = input_capacity(path)
            self.assertGreaterEqual(capacity["base_enode_upper"], 9)
            self.assertFalse(capacity["refund_on_rebuild"])

    def test_latches_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sequential.aag"
            path.write_text("aag 1 0 1 1 0\n")
            with self.assertRaises(ValueError):
                input_capacity(path)


if __name__ == "__main__":
    unittest.main()
