import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ranking_corpus", ROOT / "scripts/ranking_corpus.py")
corpus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corpus)


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "corpus"
        self.data.mkdir()
        self.image = self.root / "Pharo.image"
        self.image.write_bytes(b"frozen image")
        self.packages = ["Train-A", "Train-B", "Validation", "Test", "Benchmark", "No-Rows"]
        corpus.write_json(self.data / "packages.json", self.packages)
        self.rows = [dict(schema="coo-ranking-v1", group=name, target="size")
                     for name in ["Train-A", "Benchmark", "Train-B", "Benchmark", "Validation", "Test"]]
        self.write_rows()
        self.split = dict(schema="coo-package-split-v2", seed=42, eligible=self.packages,
                          train=["Train-A", "Train-B", "No-Rows"], validation=["Validation"],
                          test=["Test"], benchmark=["Benchmark"], ratios=[80, 10, 10])
        self.split_path = self.root / "split.json"
        self.write_split()
        self.output = self.root / "run"

    def write_rows(self):
        (self.data / "all.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")

    def write_split(self):
        corpus.write_json(self.split_path, self.split)

    def finalize(self):
        return corpus.finalize(self.data, self.image)

    def partition(self):
        return corpus.partition(self.data, self.image, self.split_path, self.output)

    def test_four_partitions_exclude_benchmarks_from_every_model_file(self):
        manifest = self.finalize()
        self.assertEqual(manifest["rowsByPackage"]["No-Rows"], 0)
        audit = self.partition()
        train = [json.loads(line) for line in (self.output / "training.jsonl").read_text().splitlines()]
        test = [json.loads(line) for line in (self.output / "test.jsonl").read_text().splitlines()]
        self.assertEqual(train, [self.rows[0], self.rows[2]])
        self.assertEqual(test, [self.rows[5]])
        validation = [json.loads(line) for line in (self.output / "validation.jsonl").read_text().splitlines()]
        self.assertEqual(validation, [self.rows[4]])
        for rows in (train, validation, test):
            self.assertNotIn("Benchmark", {row["group"] for row in rows})
        self.assertEqual(audit["trainingRows"], 2)
        self.assertEqual(audit["testRows"], 1)
        self.assertEqual(audit["validationRows"], 1)
        self.assertEqual(audit["excludedBenchmarkRows"], 2)
        self.assertEqual(audit["packageSplit"], self.split)
        self.assertEqual(corpus.verify(self.data, self.image, self.split_path), manifest)

    def test_new_split_reuses_same_corpus(self):
        self.finalize()
        self.split.update(train=["Benchmark", "Train-B", "No-Rows"], benchmark=["Train-A"])
        self.write_split()
        self.assertEqual(self.partition()["trainingRows"], 3)

    def test_rejects_overlap_missing_duplicate_and_changed_pool(self):
        self.finalize()
        for key, value in [("train", self.packages), ("train", ["Train-A"]),
                           ("benchmark", ["Benchmark", "Benchmark"]),
                           ("eligible", self.packages + ["New"]), ("seed", True)]:
            with self.subTest(key=key, value=value):
                changed = dict(self.split, **{key: value})
                corpus.write_json(self.split_path, changed)
                with self.assertRaises(ValueError):
                    self.partition()
                self.assertFalse((self.output / "training.jsonl").exists())

    def test_every_pair_of_partitions_must_be_disjoint(self):
        self.finalize()
        from itertools import combinations
        for left, right in combinations(corpus.PARTITIONS, 2):
            with self.subTest(left=left, right=right):
                changed = dict(self.split, **{right: self.split[right] + self.split[left][:1]})
                with self.assertRaisesRegex(ValueError, "disjoint"):
                    corpus.validate_split(changed)

    def test_seeded_split_reserves_benchmarks_before_80_10_10(self):
        self.finalize()
        selection = self.root / "selection.json"
        corpus.write_json(selection, dict(eligible=self.packages, benchmark=["Benchmark"], seed=42))
        first = corpus.create_split(self.data, selection, self.root / "new-split.json")
        second = corpus.create_split(self.data, selection, self.root / "second-split.json")
        self.assertEqual(first, second)
        self.assertEqual(first["benchmark"], ["Benchmark"])
        self.assertEqual([len(first[key]) for key in ("train", "validation", "test")], [3, 1, 1])
        for key in ("train", "validation", "test"):
            self.assertNotIn("Benchmark", first[key])
        with self.assertRaisesRegex(ValueError, "ratios"):
            corpus.create_split(self.data, selection, self.root / "new-split.json", (70, 15, 15))

    def test_legacy_two_way_split_is_rejected(self):
        self.finalize()
        self.split["schema"] = "coo-package-split-v1"
        self.write_split()
        with self.assertRaisesRegex(ValueError, "four-part"):
            self.partition()

    def test_wrong_image_rejected(self):
        self.finalize()
        self.image.write_bytes(b"other image")
        with self.assertRaisesRegex(ValueError, "Image does not match the prepared corpus"):
            self.partition()

    def test_truncated_corpus_never_publishes_training_data(self):
        self.finalize()
        self.rows.pop()
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.partition()
        self.assertEqual(list(self.output.iterdir()), [])
        with self.assertRaisesRegex(ValueError, "checksum"):
            corpus.verify(self.data, self.image)

    def test_malformed_or_mislabeled_rows_fail_before_publication(self):
        self.finalize()
        for row in [dict(self.rows[0], package="Benchmark"), dict(self.rows[0], group="Unknown"),
                    dict(self.rows[0], schema="other"), {"group": []}, []]:
            with self.subTest(row=row):
                self.rows[0] = row
                self.write_rows()
                with self.assertRaises(ValueError):
                    self.partition()
                self.assertEqual(list(self.output.iterdir()), [])

    def test_incomplete_export_has_no_usable_manifest(self):
        (self.data / "all.jsonl").write_text('{"group":')
        with self.assertRaises(ValueError):
            self.finalize()
        self.assertFalse((self.data / "manifest.json").exists())

    def test_empty_export_rejected(self):
        self.rows.clear()
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "Empty"):
            self.finalize()

    def test_zero_row_validation_fails_without_redrawing(self):
        self.finalize()
        self.split.update(validation=["No-Rows"], train=["Train-A", "Train-B", "Validation"])
        self.write_split()
        original = self.split_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.partition()
        self.assertEqual(self.split_path.read_bytes(), original)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_existing_artifacts_not_overwritten(self):
        self.finalize()
        with self.assertRaisesRegex(ValueError, "already finalized"):
            self.finalize()
        self.partition()
        original = (self.output / "training.jsonl").read_bytes()
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.partition()
        self.assertEqual((self.output / "training.jsonl").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
