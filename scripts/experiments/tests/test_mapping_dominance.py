"""Do not hide conflicting or failed observations sharing a baseline name."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


class ObservationCoverage(unittest.TestCase):
    def check_gate(self, observations):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def save(name, results):
                path = root / name
                path.write_text(json.dumps({"library_sha256": "same-lib", "metric_model": "same-model",
                                            "results": results}))
                return str(path)
            candidate = save("candidate.json", [self.row("egraph", 4, 4)])
            paths = [save(f"baseline{i}.json", [row]) for i, row in enumerate(observations)]
            output = root / "gate.json"
            tool = Path(__file__).resolve().parents[1] / "check_mapping_dominance.py"
            run = subprocess.run(["python3", str(tool), "--candidate", candidate,
                                  "--baselines", *paths, "--output", str(output)], capture_output=True)
            return run.returncode, json.loads(output.read_text())

    @staticmethod
    def row(method, area, delay):
        return {"bench": "test", "method": method, "status": "ok", "input_sha256": "same-input",
                "measured": {"area": area, "delay": delay, "cec": 1}}

    def test_different_covers_with_same_name_both_checked(self):
        code, result = self.check_gate([self.row("emorphic_random_amap", 2, 6),
                                       self.row("emorphic_random_amap", 6, 6)])
        self.assertEqual(code, 1)
        self.assertFalse(result["all_passed"])
        row = result["results"][0]
        self.assertEqual(row["baseline_count"], 2)
        self.assertEqual(len(row["dominated_baselines"]), 1)

    def test_identical_covers_coalesce_with_all_sources(self):
        row = self.row("abc_amap", 6, 6)
        code, result = self.check_gate([row, row])
        self.assertEqual(code, 0)
        row = result["results"][0]
        self.assertEqual(row["baseline_count"], 1)
        self.assertEqual(len(row["baseline_observations"][0]["sources"]), 2)

    def test_failed_observation_cannot_be_overwritten_by_success(self):
        failed = dict(self.row("abc_amap", 6, 6), status="failed", error="timeout")
        code, result = self.check_gate([failed, self.row("abc_amap", 6, 6)])
        self.assertEqual(code, 1)
        self.assertEqual(result["results"][0]["invalid_baselines"], ["abc_amap"])


if __name__ == "__main__": unittest.main()
