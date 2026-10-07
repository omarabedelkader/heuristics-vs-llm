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
        self.packages = ["Train-A", "Train-B", "Held-Out", "No-Rows"]
        corpus.write_json(self.data / "packages.json", self.packages)
        self.rows = [dict(schema="coo-ranking-v1", group=name, target="size")
                     for name in ["Train-A", "Held-Out", "Train-B", "Held-Out"]]
        self.write_rows()
        self.split = dict(schema="coo-package-split-v1", seed=42, eligible=self.packages,
                          train=["Train-A", "Train-B", "No-Rows"], benchmark=["Held-Out"])
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

    def test_every_remaining_package_and_only_held_out_test(self):
        manifest = self.finalize()
        self.assertEqual(manifest["rowsByPackage"]["No-Rows"], 0)
        audit = self.partition()
        train = [json.loads(line) for line in (self.output / "training.jsonl").read_text().splitlines()]
        test = [json.loads(line) for line in (self.output / "test.jsonl").read_text().splitlines()]
        self.assertEqual(train, [self.rows[0], self.rows[2]])
        self.assertEqual(test, [self.rows[1], self.rows[3]])
        self.assertEqual(audit["trainingRows"], 2)
        self.assertEqual(audit["testRows"], 2)
        self.assertEqual(audit["packageSplit"], self.split)
        self.assertEqual(corpus.verify(self.data, self.image, self.split_path), manifest)

    def test_new_split_reuses_same_corpus(self):
        self.finalize()
        self.split.update(train=["Held-Out", "Train-B", "No-Rows"], benchmark=["Train-A"])
        self.write_split()
        self.assertEqual(self.partition()["trainingRows"], 3)

    def test_rejects_overlap_missing_duplicate_and_changed_pool(self):
        self.finalize()
        for key, value in [("train", self.packages), ("train", ["Train-A"]),
                           ("benchmark", ["Held-Out", "Held-Out"]),
                           ("eligible", self.packages + ["New"]), ("seed", True)]:
            with self.subTest(key=key, value=value):
                changed = dict(self.split, **{key: value})
                corpus.write_json(self.split_path, changed)
                with self.assertRaises(ValueError):
                    self.partition()
                self.assertFalse((self.output / "training.jsonl").exists())

    def test_wrong_image_rejected(self):
        self.finalize()
        self.image.write_bytes(b"other image")
        with self.assertRaisesRegex(ValueError, "different image"):
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
        for row in [dict(self.rows[0], package="Held-Out"), dict(self.rows[0], group="Unknown"),
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

    def test_zero_row_holdout_fails_without_redrawing(self):
        self.finalize()
        self.split.update(benchmark=["No-Rows"], train=self.packages[:3])
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
