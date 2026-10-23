import unittest
from scaled_ratio_selection import select


class WholeSuiteSelectionTest(unittest.TestCase):
    def row(self, rho, bench, delay, status="ok"):
        return dict(ratio=rho, bench=bench, k=3, delay=delay, area=1.0, mapping_secs=1.0, status=status)

    def test_one_global_ratio_even_when_local_winners_differ(self):
        rows = [self.row("1/2", "a", 1), self.row("1/2", "b", 100),
                self.row("1", "a", 4), self.row("1", "b", 20)]
        chosen, summary, common = select(rows, ["1/2", "1"])
        self.assertEqual(chosen, "1")
        self.assertEqual(len(common), 2)
        self.assertAlmostEqual(summary["1"]["delay_geomean"], 80 ** 0.5)

    def test_missing_case_is_not_silently_dropped(self):
        rows = [self.row("1/2", "a", 1), self.row("1/2", "b", 1, "failed"),
                self.row("1", "a", 10), self.row("1", "b", 100)]
        chosen, _, common = select(rows, ["1/2", "1"])
        self.assertEqual(chosen, "1")
        self.assertEqual(len(common), 1)

    def test_ties_prefer_lower_ratio(self):
        rows = [self.row(rho, "a", 2) for rho in ["2", "1", "1/2"]]
        self.assertEqual(select(rows, ["2", "1", "1/2"])[0], "1/2")


if __name__ == "__main__":
    unittest.main()
