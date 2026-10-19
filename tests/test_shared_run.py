"""Several hosts sharing one run directory: no benchmark runs twice or is lost."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import package_benchmarks as workers
import test_package_benchmarks as single
from test_package_benchmarks import part


class SharedRunTests(unittest.TestCase):
    setUp, stub, events = single.PackageWorkerTests.setUp, single.PackageWorkerTests.stub, single.PackageWorkerTests.events

    def shared(self, job, jobs=2, lease=900):
        env = dict(os.environ, OAR_JOB_ID=job, BENCHMARK_CLAIM_LEASE=str(lease),
                   BENCHMARK_CLAIM_RESCAN="0.5")
        return subprocess.Popen(
            [sys.executable, str(ROOT / "scripts/shared_run.py"), "run", "--run", str(self.run),
             "--image", str(self.image), "--jobs", str(jobs)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def finish(self, *processes):
        outputs = []
        for process in processes:
            output, _ = process.communicate(timeout=60)
            self.assertEqual(process.returncode, 0, output)
            outputs.append(output)
        return outputs

    def starts(self):
        return [(e["package"], e["kind"], e["strategy"]) for e in self.events() if e["event"] == "start"]

    def test_two_hosts_share_the_rest_of_a_resumed_run(self):
        # The interrupted run: package A has its 7B benchmarks done.
        done = {("A", kind, strategy) for kind in workers.KINDS
                for strategy in ("llmUnvalidated7B", "llm7B")}
        for package, kind, strategy in done:
            directory = workers.task_directory(self.run, 0, kind, strategy)
            directory.mkdir(parents=True)
            workers.atomic_json(directory / "part.json", part(package, self.manifest["runId"], kind, strategy))
        # A benchmark the killed job was running: its private image and pending output remain.
        stale = workers.task_directory(self.run, 0, "messages", "hybrid7B")
        (stale / "image").mkdir(parents=True)
        (stale / "part.pending.json").write_text("{")
        self.stub(pause=0.05)
        for output in self.finish(self.shared("1"), self.shared("2")):
            self.assertIn("Started", output)
        starts = self.starts()
        self.assertEqual(len(starts), len(set(starts)))
        self.assertFalse(done & set(starts))
        self.assertEqual(len(starts) + len(done), 3 * len(workers.KINDS) * len(workers.MEASURED))
        for index, package in enumerate("ABC"):
            workers.validate_result(workers.package_directory(self.run, index) / "result.json",
                                    package, self.manifest["runId"])
        self.assertFalse(list(self.run.glob("packages/*/benchmarks/*/claim*.json")))
        self.assertFalse(list(self.run.glob("packages/*/result.pending.*")))
        self.assertEqual((self.image / "Pharo.image").read_bytes(), b"template")

    def test_expired_claims_are_taken_over_and_live_ones_waited_for(self):
        dead = workers.task_directory(self.run, 0, "messages", "llmUnvalidated7B") / "claim.json"
        live = workers.task_directory(self.run, 1, "messages", "llmUnvalidated7B") / "claim.json"
        for claim in (dead, live):
            claim.parent.mkdir(parents=True)
            claim.write_text(json.dumps(dict(host="other", id="other")))
        os.utime(dead, (time.time() - 3600, time.time() - 3600))
        self.stub(pause=0.05)
        host = self.shared("1", lease=60)
        # The live host finishes its benchmark a little later.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and len(list(self.run.glob("packages/*/benchmarks/*/part.json"))) < 10:
            time.sleep(0.1)
        workers.atomic_json(live.parent / "part.json",
                            part("B", self.manifest["runId"], "messages", "llmUnvalidated7B"))
        live.unlink()
        self.finish(host)
        starts = self.starts()
        self.assertIn(("A", "messages", "llmUnvalidated7B"), starts)
        self.assertNotIn(("B", "messages", "llmUnvalidated7B"), starts)
        self.assertEqual(len(starts), 3 * len(workers.KINDS) * len(workers.MEASURED) - 1)
        for index, package in enumerate("ABC"):
            workers.validate_result(workers.package_directory(self.run, index) / "result.json",
                                    package, self.manifest["runId"])


if __name__ == "__main__":
    unittest.main()
