import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import runtime_estimate as estimator  # noqa: E402


class RuntimeEstimateTests(unittest.TestCase):
    def test_fit_separates_startup_from_per_example_cost(self):
        records = [dict(rows=rows, seconds=20 + 0.5 * rows) for rows in (30, 100, 300)]
        startup, per_example = estimator.fit(records)
        self.assertAlmostEqual(startup, 20)
        self.assertAlmostEqual(per_example, 0.5)

    def test_fit_is_conservative_without_size_spread(self):
        self.assertEqual(estimator.fit([dict(rows=10, seconds=50), dict(rows=10, seconds=70)]), (0.0, 6.0))

    def test_makespan_follows_fifo_slot_reuse_and_long_tail(self):
        self.assertEqual(estimator.makespan([10, 10, 10, 10], 2), 20)
        self.assertEqual(estimator.makespan([100, 1, 1, 1], 2), 100)
        self.assertEqual(estimator.makespan([1, 1, 1, 100], 2), 101)

    def test_calibration_packages_are_spread_and_leave_training_data(self):
        rows = {f"P{n}": n for n in (0, 25, 40, 90, 280, 5000)}
        chosen = estimator.calibration_packages(list(rows), rows, 3)
        self.assertEqual(chosen, ["P25", "P90", "P280"])
        with self.assertRaises(ValueError):
            estimator.calibration_packages(["P25", "P40"], rows, 2)

    def test_estimate_counts_only_pending_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "calibration").mkdir()
            write = lambda path, value: (run / path).write_text(json.dumps(value))
            write("workers.json", dict(packages=["A", "B", "C"]))
            write("corpus.json", dict(rowsByPackage=dict(A=100, B=200, C=1000)))
            for phase in ("normal", "reranker"):
                write(f"calibration/{phase}.json", dict(jobs=2, packages=[
                    dict(rows=10, seconds=11), dict(rows=110, seconds=111)]))
            write("calibration/training.json", dict(trainingSeconds=500, evaluationSeconds=50,
                                                    trainingDetail="t", evaluationDetail="e"))
            (run / "packages/0001").mkdir(parents=True)
            write("packages/0001/normal.json", {})
            result = estimator.estimate(run, 2, elapsed=60)
            normal, training, evaluation, reranker = result["stages"]
            self.assertAlmostEqual(normal["seconds"], 1001)  # B and C in parallel; C dominates
            self.assertAlmostEqual(reranker["seconds"], 101 + 1001)  # all pending; C waits for A's slot
            self.assertEqual(normal["largestPackage"]["name"], "C")
            self.assertAlmostEqual(result["totalSeconds"], 60 + 1001 + 500 + 50 + 1102)
            self.assertIn("RENT AT LEAST", estimator.report(result))
            (run / "training-ready").touch()
            with self.assertRaises(ValueError):
                estimator.estimate(run, 4, elapsed=0)
            stages = estimator.estimate(run, 2, elapsed=0, normal_only=True)["stages"]
            self.assertEqual([s["stage"] for s in stages], ["normal benchmarks"])


if __name__ == "__main__":
    unittest.main()
