"""Check scheduling, process cleanup and exact aggregation without model inference."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import package_benchmarks as workers


def result(package, run_id, count=1, rank_sum=1, memory=-10):
    return dict(schema="coo-package-result-v2", package=package, runId=run_id,
                corpus=dict(packages=1, classes=2, methods=3),
                rows=[dict(kind=kind, strategy=strategy, prefix=prefix, count=count,
                           reciprocalRankSum=rank_sum, timeMs=2 * count, memoryBytes=memory,
                           **dict.fromkeys(workers.HITS, min(rank_sum, count)))
                      for kind, strategy, prefix in sorted(workers.row_keys())])


class PackageWorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="package workers ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.run = self.root / "run"
        self.image = self.root / "image"
        self.run.mkdir()
        self.image.mkdir()
        (self.image / "Pharo.image").write_bytes(b"template")
        (self.image / "Pharo.changes").write_bytes(b"changes")
        (self.image / "Pharo.sources").write_bytes(b"sources")
        workers.atomic_json(self.run / "selection.json", dict(
            schema="coo-package-split-v1", seed=42, eligible=["A", "B", "C", "Other"],
            benchmark=["A", "B", "C"], train=["Other"]))
        self.manifest = workers.prepare(self.run, self.image)

    def checkpoints(self):
        for index, package in enumerate(self.manifest["packages"]):
            directory = workers.package_directory(self.run, index)
            directory.mkdir(parents=True, exist_ok=True)
            workers.atomic_json(directory / "result.json", result(
                package, self.manifest["runId"], count=(1, 9, 0)[index],
                rank_sum=(1, 0, 0)[index], memory=(-10, -90, 0)[index]))

    def test_aggregate_weights_observations_and_preserves_negative_memory(self):
        self.checkpoints()
        data = workers.read_json(workers.aggregate(self.run, self.image))
        self.assertEqual(data["corpus"], dict(packages=3, classes=6, methods=9))
        self.assertEqual(len(data["rows"]), len(workers.row_keys()))
        for row in data["rows"]:
            self.assertEqual(row["count"], 10)
            self.assertEqual(row["reciprocalRankSum"] / row["count"], 0.1)
            self.assertEqual(row["timeMs"], 20)
            self.assertEqual(row["memoryBytes"], -100)
            self.assertEqual([row[field] for field in workers.HITS], [1, 1, 1, 1])

    def test_top_k_hits_must_be_ordered_and_bounded_by_count(self):
        self.checkpoints()
        path = workers.package_directory(self.run, 1) / "result.json"
        for hits in ([1, 0, 0, 0], [0, 0, 0, 10], [0.5, 1, 1, 1]):
            with self.subTest(hits=hits):
                data = workers.read_json(path)
                for row in data["rows"]:
                    row.update(zip(workers.HITS, hits))
                workers.atomic_json(path, data)
                with self.assertRaisesRegex(ValueError, "top-k hit counts"):
                    workers.validate_result(path, self.manifest["packages"][1], self.manifest["runId"])

    def test_missing_or_mismatched_results_block_aggregation(self):
        self.checkpoints()
        path = workers.package_directory(self.run, 1) / "result.json"
        original = workers.read_json(path)
        for key, value in (("package", "A"), ("runId", "another-run"),
                           ("corpus", dict(packages=2, classes=4, methods=3)),
                           ("rows", original["rows"][:-1]),
                           ("rows", original["rows"] + original["rows"][:1])):
            with self.subTest(key=key):
                workers.atomic_json(path, dict(original, **{key: value}))
                with self.assertRaises(ValueError):
                    workers.aggregate(self.run, self.image)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            workers.aggregate(self.run, self.image)

    def test_changed_image_or_packages_rejected_on_resume(self):
        (self.image / "Pharo.image").write_bytes(b"another image")
        with self.assertRaisesRegex(ValueError, "Resume inputs changed"):
            workers.prepare(self.run, self.image)
        (self.image / "Pharo.image").write_bytes(b"template")
        workers.prepare(self.run, self.image)
        selection = workers.read_json(self.run / "selection.json")
        workers.atomic_json(self.run / "selection.json", dict(selection, benchmark=["A", "B"], train=["C", "Other"]))
        with self.assertRaisesRegex(ValueError, "Resume inputs changed"):
            workers.prepare(self.run, self.image)

    def test_user_package_list_is_checked_against_the_image(self):
        eligible = self.root / "eligible.json"
        eligible.write_text(json.dumps(["A", "B", "C", "Other"]))
        packages = self.root / "packages.txt"
        output = self.root / "benchmark-selection.json"
        packages.write_text("# mine\nB  # this one\n\nA\n")
        selection = workers.select(packages, eligible, output)
        self.assertEqual(selection["benchmark"], ["A", "B"])
        self.assertEqual(selection["train"], ["C", "Other"])
        self.assertEqual(workers.load_selection(output), selection)
        # The same list is reused; a different one needs a new experiment.
        packages.write_text(json.dumps(["B", "A"]))
        self.assertEqual(workers.select(packages, eligible, output), selection)
        packages.write_text("C\n")
        with self.assertRaisesRegex(ValueError, "different benchmark package list"):
            workers.select(packages, eligible, output)
        packages.write_text("Not-A-Package\n")
        with self.assertRaisesRegex(ValueError, "not benchmark-eligible"):
            workers.select(packages, eligible, self.root / "other.json")
        # A saved selection keeps its seed.
        packages.write_text(json.dumps(dict(selection, seed=7)))
        self.assertEqual(workers.select(packages, eligible, self.root / "other.json")["seed"], 7)

    def stub(self, fail=False, slow=False):
        script = self.image / "pharo"
        script.write_text(f'''#!{sys.executable}
import json, os, sys, time, subprocess
from pathlib import Path
sys.path.insert(0, {str(ROOT / 'tests')!r})
from test_package_benchmarks import result
root = Path({str(self.root)!r})
package = os.environ['BENCHMARK_PACKAGE']
def event(kind):
    with (root / 'events.jsonl').open('a') as out:
        out.write(json.dumps(dict(event=kind, package=package, time=time.monotonic(), pid=os.getpid())) + '\\n')
event('start')
assert Path('Pharo.image').read_bytes() == b'template'
assert Path('Pharo.sources').read_bytes() == b'sources'
if {slow!r}:
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    (root / ('child-' + package)).write_text(str(child.pid))
time.sleep(60 if {slow!r} else (0.2 if package == 'B' else 0.7))
if {fail!r} and package == 'C':
    sys.exit(24)
assert Path(os.environ['BENCHMARK_RUN_SELECTION']).name == 'selection.json'
Path(os.environ['BENCHMARK_OUTPUT']).write_text(json.dumps(result(
    package, os.environ['BENCHMARK_RUN_ID'])))
event('finish')
''')
        script.chmod(0o755)

    def command(self):
        return [sys.executable, str(ROOT / "scripts/package_benchmarks.py"), "run",
                "--run", str(self.run), "--image", str(self.image), "--jobs", "2"]

    def events(self):
        return [json.loads(line) for line in (self.root / "events.jsonl").read_text().splitlines()]

    def test_two_slots_refill_and_resume_keeps_completed_packages(self):
        self.stub(fail=True)
        failed = subprocess.run(self.command(), capture_output=True, text=True, timeout=10)
        self.assertEqual(failed.returncode, 1, failed.stdout + failed.stderr)
        events = self.events()
        active = peak = 0
        for event in events:
            active += 1 if event["event"] == "start" else -1
            peak = max(peak, active)
        self.assertEqual(peak, 2)
        # B finishes first: C must start while A is still running.
        self.assertLess(next(e["time"] for e in events if e["package"] == "C"),
                        next(e["time"] for e in events if e["package"] == "A" and e["event"] == "finish"))
        self.assertTrue((workers.package_directory(self.run, 0) / "result.json").exists())
        self.assertTrue((workers.package_directory(self.run, 1) / "result.json").exists())
        self.assertFalse((workers.package_directory(self.run, 2) / "result.json").exists())
        self.stub()
        resumed = subprocess.run(self.command(), capture_output=True, text=True, timeout=10)
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        starts = [e["package"] for e in self.events() if e["event"] == "start"]
        self.assertEqual(starts.count("A"), 1)
        self.assertEqual(starts.count("B"), 1)
        self.assertEqual(starts.count("C"), 2)
        self.assertEqual((self.image / "Pharo.image").read_bytes(), b"template")
        self.assertNotEqual((workers.package_directory(self.run, 0) / "image/Pharo.image").stat().st_ino,
                            (workers.package_directory(self.run, 1) / "image/Pharo.image").stat().st_ino)

    def test_termination_stops_worker_processes(self):
        self.stub(slow=True)
        process = subprocess.Popen(self.command(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 5
            while not (self.root / "child-B").exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue((self.root / "child-B").exists())
            process.terminate()
            _, errors = process.communicate(timeout=8)
            self.assertEqual(process.returncode, 130, errors)
            for event in self.events():
                with self.assertRaises(ProcessLookupError):
                    os.kill(event["pid"], 0)
            for path in self.root.glob("child-*"):
                # A killed orphan may briefly remain a zombie on some systems.
                status = subprocess.run(["ps", "-o", "stat=", "-p", path.read_text()],
                                        capture_output=True, text=True).stdout.strip()
                self.assertTrue(not status or status.startswith("Z"), status)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=8)


if __name__ == "__main__":
    unittest.main()
