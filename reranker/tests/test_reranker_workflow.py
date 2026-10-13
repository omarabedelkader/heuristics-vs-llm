"""Leakage guards and an opt-in real PyTorch/ONNX training smoke test."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import reranker_workflow as workflow
from ranking_corpus import write_json


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="four partition training ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.split = dict(schema="coo-package-split-v2", seed=42,
                          eligible=["Train", "Validation", "Test", "Benchmark"],
                          train=["Train"], validation=["Validation"], test=["Test"], benchmark=["Benchmark"])
        write_json(self.root / "split.json", self.split)

    def test_each_model_file_rejects_other_partitions_including_benchmarks(self):
        path = self.root / "rows.jsonl"
        for partition in ("train", "validation", "test"):
            for name in self.split["eligible"]:
                with self.subTest(partition=partition, group=name):
                    path.write_text(json.dumps(dict(schema="coo-ranking-v1", group=name)) + "\n")
                    if name not in self.split[partition]:
                        with self.assertRaisesRegex(ValueError, "Forbidden"):
                            workflow.read_partition(path, self.split, partition)
                    else:
                        self.assertEqual(len(workflow.read_partition(path, self.split, partition)), 1)

    def test_training_rejects_leakage_before_loading_ml_or_creating_model(self):
        # Even an unavailable repository/model module cannot hide the leakage error.
        for partition, name in (("training", "Benchmark"), ("validation", "Benchmark")):
            for file, group in (("training", "Train"), ("validation", "Validation")):
                (self.root / f"{file}.jsonl").write_text(json.dumps(dict(schema="coo-ranking-v1", group=group)) + "\n")
            (self.root / f"{partition}.jsonl").write_text(json.dumps(dict(schema="coo-ranking-v1", group=name)) + "\n")
            with self.assertRaisesRegex(ValueError, "Forbidden"):
                workflow.train(argparse.Namespace(split=self.root / "split.json", training=self.root / "training.jsonl",
                                                 validation=self.root / "validation.jsonl"))

    def test_ranking_metrics_include_misses_and_cap_mrr_at_ten(self):
        metrics = workflow.ranking_metrics([1, 2, 10, 11, 0])
        self.assertAlmostEqual(metrics["mrr"], 0.32)
        self.assertEqual(metrics["accuracyAt1"], 0.2)
        self.assertEqual(metrics["accuracyAt3"], 0.4)
        self.assertEqual(metrics["accuracyAt10"], 0.6)
        self.assertEqual([workflow.rank_bucket(rank) for rank in (1, 3, 10, 11, 0)], list(range(5)))

    @unittest.skipUnless(os.environ.get("RERANKER_TEST_REPOSITORY"), "Set RERANKER_TEST_REPOSITORY for real ML smoke test")
    def test_real_training_selects_validation_checkpoint_and_exports_test_figures(self):
        repo = Path(os.environ["RERANKER_TEST_REPOSITORY"])
        for partition, group in (("training", "Train"), ("validation", "Validation"), ("test", "Test")):
            rows = [dict(schema="coo-ranking-v1", group=group, kind="messages", prefix="si",
                         sourcePrefix="example ^ self si", receiverKind="self", target="size", candidateLimit=50,
                         candidates=[dict(name="silly", rank=1), dict(name="size", rank=2)]) for _ in range(5)]
            rows.append(dict(rows[0], target="six", candidates=[]))
            (self.root / f"{partition}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        script = str(ROOT / "scripts/reranker_workflow.py")
        common = ["--repository", str(repo), "--split", str(self.root / "split.json")]
        model = self.root / "model"
        commands = [
            ["train", *common, "--training", str(self.root / "training.jsonl"), "--validation",
             str(self.root / "validation.jsonl"), "--output", str(model), "--epochs", "2", "--width", "16"],
            ["evaluate", *common, "--data", str(self.root / "test.jsonl"), "--model", str(model),
             "--output", str(self.root / "evaluation.json")],
            ["report", "--model", str(model), "--evaluation", str(self.root / "evaluation.json"),
             "--output", str(self.root / "learning")],
        ]
        for command in commands:
            process = subprocess.run([sys.executable, script, *command], capture_output=True, text=True, timeout=90)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        history = json.loads((model / "learning-history.json").read_text())
        metadata = json.loads((model / "metadata.json").read_text())
        expected = max(history["epochs"][1:], key=lambda r: r["validationMRR"])
        self.assertEqual(metadata["bestEpoch"], expected["epoch"])
        self.assertEqual(metadata["packageSplit"], self.split)
        self.assertEqual(metadata["trainingRows"], 6)
        self.assertEqual(metadata["trainingMisses"], 1)
        evaluation = json.loads((self.root / "evaluation.json").read_text())
        self.assertEqual(evaluation["testGroups"], ["Test"])
        self.assertEqual(len(evaluation["summaries"]), 4)
        for summary in evaluation["summaries"]:
            self.assertEqual(sum(map(sum, summary["rankTransitions"])), 6)
            self.assertEqual(summary["candidateRecall"], 5 / 6)
        for name in ("learning-curves", "test-performance", "test-rank-transitions"):
            self.assertTrue((self.root / f"learning/{name}.png").read_bytes().startswith(b"\x89PNG"))
            self.assertTrue((self.root / f"learning/{name}.pdf").read_bytes().startswith(b"%PDF"))
        # The same evaluator must reject the benchmark group, even with a valid model.
        forbidden = self.root / "benchmark.jsonl"
        forbidden.write_text(json.dumps(dict(rows[0], group="Benchmark")) + "\n")
        process = subprocess.run([sys.executable, script, "evaluate", *common, "--data", str(forbidden),
                                  "--model", str(model), "--output", str(self.root / "forbidden.json")],
                                 capture_output=True, text=True, timeout=20)
        self.assertEqual(process.returncode, 1)
        self.assertIn("Forbidden", process.stderr)
        self.assertFalse((self.root / "forbidden.json").exists())


if __name__ == "__main__":
    unittest.main()
