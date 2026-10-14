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


def candidates(strategy, count):
    """LLM completion rows also count the candidates answered and the real names among them."""
    if strategy in workers.LLM:
        return dict(generated=3 * count, valid=3 * count)
    if strategy in workers.BEFORE_VALIDATION.values():
        return dict(generated=4 * count, valid=count)
    return {}


def result(package, run_id, count=1, rank_sum=1, memory=-10):
    return dict(schema="coo-package-result-v3", package=package, runId=run_id,
                corpus=dict(packages=1, classes=2, methods=3),
                rows=[dict(kind=kind, strategy=strategy, prefix=prefix, count=count,
                           reciprocalRankSum=rank_sum, timeMs=2 * count, memoryBytes=memory,
                           **dict.fromkeys(workers.HITS, min(rank_sum, count)),
                           **candidates(strategy, count))
                      for kind, strategy, prefix in sorted(workers.row_keys())])


def part(package, run_id, kind, strategy):
    """The output of one benchmark worker: the seven prefix rows of one kind and strategy."""
    data = result(package, run_id)
    return dict(data, schema="coo-package-part-v1", kind=kind, strategy=strategy,
                rows=[row for row in data["rows"] if (row["kind"], row["strategy"]) == (kind, strategy)])


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

    def test_aggregate_separates_results_before_and_after_pharo_validation(self):
        self.checkpoints()
        workers.aggregate(self.run, self.image)
        before = workers.read_json(self.run / "aggregate-before.json")
        after = workers.read_json(self.run / "aggregate-after.json")
        for data in (before, after):
            self.assertEqual({row["strategy"] for row in data["rows"]}, set(workers.STRATEGIES))
            self.assertEqual(len(data["rows"]), 2 * len(workers.STRATEGIES) * 7)
            self.assertEqual(data["corpus"], dict(packages=3, classes=6, methods=9))
        llm = lambda data: {(row["generated"], row["valid"]) for row in data["rows"] if row["strategy"] in workers.LLM}
        self.assertEqual(llm(before), {(40, 10)})
        self.assertEqual(llm(after), {(30, 30)})
        other = lambda data: sorted((row["kind"], row["strategy"], row["prefix"], row["count"])
                                    for row in data["rows"] if row["strategy"] not in workers.LLM)
        self.assertEqual(other(before), other(after))
        table = (self.run / "pharo-validation.tex").read_text()
        # Seven prefix sizes: 70 completions, 280 candidates, 70 real names, per model and kind.
        self.assertEqual(table.count("& 70 & 280 & 70 & 25.0 \\\\"), 8)
        for label in ("Methods", "Classes", "& 0.5B &", "& 1.5B &", "& 3B &", "& 7B &"):
            self.assertIn(label, table)

    def test_candidate_counts_are_checked(self):
        self.checkpoints()
        path = workers.package_directory(self.run, 1) / "result.json"
        for strategy, counts in (("llmUnvalidated05B", dict(generated=1, valid=2)),
                                 ("llmUnvalidated05B", dict(generated=91, valid=0)),
                                 ("llm7B", dict(generated=5, valid=4)),
                                 ("llm7B", dict(generated=None, valid=None))):
            with self.subTest(strategy=strategy, counts=counts):
                data = result(self.manifest["packages"][1], self.manifest["runId"], count=9, rank_sum=0)
                for row in data["rows"]:
                    if row["strategy"] == strategy:
                        row.update(counts)
                workers.atomic_json(path, data)
                with self.assertRaisesRegex(ValueError, "candidate counts"):
                    workers.validate_result(path, self.manifest["packages"][1], self.manifest["runId"])

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

    def stub(self, fail=False, slow=False, pause=0):
        script = self.image / "pharo"
        script.write_text(f'''#!{sys.executable}
import json, os, sys, time, subprocess
from pathlib import Path
sys.path.insert(0, {str(ROOT / 'tests')!r})
from test_package_benchmarks import part
root = Path({str(self.root)!r})
package = os.environ['BENCHMARK_PACKAGE']
kind, strategy = os.environ['BENCHMARK_KIND'], os.environ['BENCHMARK_STRATEGY']
first = (kind, strategy) == ('messages', 'llmUnvalidated7B')
def event(name):
    with (root / 'events.jsonl').open('a') as out:
        out.write(json.dumps(dict(event=name, package=package, kind=kind, strategy=strategy,
                                  time=time.monotonic(), pid=os.getpid())) + '\\n')
event('start')
assert Path('Pharo.image').read_bytes() == b'template'
assert Path('Pharo.sources').read_bytes() == b'sources'
if {slow!r}:
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    (root / ('child-' + package)).write_text(str(child.pid))
time.sleep(60 if {slow!r} else ({pause!r} if not first else 0.2 if package == 'B' else 0.7))
if {fail!r} and package == 'C':
    sys.exit(24)
assert Path(os.environ['BENCHMARK_RUN_SELECTION']).name == 'selection.json'
Path(os.environ['BENCHMARK_OUTPUT']).write_text(json.dumps(part(
    package, os.environ['BENCHMARK_RUN_ID'], kind, strategy)))
event('finish')
''')
        script.chmod(0o755)

    def command(self):
        return [sys.executable, str(ROOT / "scripts/package_benchmarks.py"), "run",
                "--run", str(self.run), "--image", str(self.image), "--jobs", "2"]

    def events(self):
        return [json.loads(line) for line in (self.root / "events.jsonl").read_text().splitlines()]

    def parts(self):
        return {(path.parent.parent.parent.name, path.parent.name)
                for path in self.run.glob("packages/*/benchmarks/*/part.json")}

    def test_two_slots_refill_and_resume_keeps_completed_benchmarks(self):
        self.stub(fail=True)
        failed = subprocess.run(self.command(), capture_output=True, text=True, timeout=10)
        self.assertEqual(failed.returncode, 1, failed.stdout + failed.stderr)
        events = self.events()
        active = peak = 0
        for event in events:
            active += 1 if event["event"] == "start" else -1
            peak = max(peak, active)
        self.assertEqual(peak, 2)
        # The longest benchmark (7B before validation) runs first for every package.
        starts = [(e["package"], e["kind"], e["strategy"]) for e in events if e["event"] == "start"]
        self.assertEqual(set(starts[:2]), {("A", "messages", "llmUnvalidated7B"),
                                           ("B", "messages", "llmUnvalidated7B")})
        # B finishes first: C must start while A is still running.
        self.assertLess(next(e["time"] for e in events if e["package"] == "C"),
                        next(e["time"] for e in events if e["package"] == "A" and e["event"] == "finish"))
        self.assertIn(("0002", "messages-llmUnvalidated7B"), self.parts())
        self.assertFalse(any(name == "0003" for name, _ in self.parts()))
        self.assertFalse(list(self.run.glob("packages/*/result.json")))
        done = self.parts()
        self.stub()
        (self.root / "events.jsonl").unlink()
        resumed = subprocess.run(self.command(), capture_output=True, text=True, timeout=30)
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        # Every benchmark runs once more only if it had not finished before.
        starts = [(f"{'ABC'.index(e['package']) + 1:04d}", f"{e['kind']}-{e['strategy']}")
                  for e in self.events() if e["event"] == "start"]
        self.assertEqual(len(starts), len(set(starts)))
        self.assertFalse(done & set(starts))
        self.assertEqual(len(starts) + len(done), 3 * len(workers.KINDS) * len(workers.MEASURED))
        for index, package in enumerate("ABC"):
            workers.validate_result(workers.package_directory(self.run, index) / "result.json",
                                    package, self.manifest["runId"])
        self.assertEqual(len((self.run / "timings.jsonl").read_text().splitlines()), 3)
        self.assertEqual((self.image / "Pharo.image").read_bytes(), b"template")
        # A finished benchmark keeps its part, not its private image copy.
        self.assertFalse(list(self.run.glob("packages/*/benchmarks/*/image")))

    def test_one_package_uses_every_slot(self):
        selection = workers.read_json(self.run / "selection.json")
        workers.atomic_json(self.run / "selection.json", dict(selection, benchmark=["A"], train=["B", "C", "Other"]))
        (self.run / "workers.json").unlink()
        self.manifest = workers.prepare(self.run, self.image)
        self.stub(pause=0.5)
        command = self.command()
        command[-1] = "4"
        finished = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(finished.returncode, 0, finished.stdout + finished.stderr)
        active = peak = 0
        for event in self.events():
            active += 1 if event["event"] == "start" else -1
            peak = max(peak, active)
        self.assertEqual(peak, 4)
        workers.validate_result(workers.package_directory(self.run, 0) / "result.json",
                                "A", self.manifest["runId"])

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
