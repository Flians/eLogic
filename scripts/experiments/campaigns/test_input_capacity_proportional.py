import tempfile
import unittest
from pathlib import Path

from input_capacity_proportional import input_capacity


class ProportionalCapacityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "four_negative_pis.aag"
        self.path.write_text("aag 4 4 0 4 0\n2\n4\n6\n8\n3\n5\n7\n9\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_one_to_one_without_fixed_floor(self):
        result = input_capacity(self.path)
        self.assertEqual(result["base_enode_upper"], 10)
        self.assertEqual(result["search_allowance"], 10)
        self.assertEqual(result["node_limit"], 20)
        self.assertFalse(result["refund_on_rebuild"])
        self.assertEqual(result["pool_limit"], 100000)

    def test_fraction_rounds_up_exactly(self):
        result = input_capacity(self.path, "1/3")
        self.assertEqual(result["search_allowance"], 4)
        self.assertEqual(result["node_limit"], 14)
        self.assertEqual(input_capacity(self.path, "0.25")["search_allowance"], 3)

    def test_zero_search_still_reserves_complete_input_bound(self):
        result = input_capacity(self.path, "0")
        self.assertEqual(result["node_limit"], 10)

    def test_invalid_ratios_rejected(self):
        for ratio in ["-1", "nan", "inf", "1/0", "bad"]:
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                input_capacity(self.path, ratio)

    def test_pool_budget_is_separate(self):
        with self.assertRaises(ValueError):
            input_capacity(self.path, "1", pool_extra=1)


if __name__ == "__main__":
    unittest.main()
