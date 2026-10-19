"""Run the benchmarks of one run directory from several hosts at the same time.

Each host (one OAR job) runs this on the shared run directory. Before starting a
benchmark, a host claims it by creating claim.json in its directory (atomic on NFS);
while the benchmark runs, the host refreshes the claim every HEARTBEAT seconds. A claim
not refreshed for LEASE seconds belongs to a host that died (walltime reached, node
lost) and is taken over. Completed benchmarks (part.json) are never run again, so the
run continues exactly where it stopped, whatever the number of hosts.

The checkpoints, their validation and the measured code are those of
package_benchmarks.py, which this module leaves untouched: its checksum is pinned in
workers.json, and any change there would make the run impossible to resume."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import sys
import time
import uuid

import package_benchmarks as workers


HEARTBEAT = 60
LEASE = int(os.environ.get("BENCHMARK_CLAIM_LEASE", "900"))
# Seconds between two scans for claimable benchmarks while every free one is taken.
RESCAN = float(os.environ.get("BENCHMARK_CLAIM_RESCAN", "30"))
OWNER = dict(host=socket.gethostname(), job=os.environ.get("OAR_JOB_ID", ""),
             pid=os.getpid(), id=uuid.uuid4().hex)


def claim_path(run, index, kind, strategy):
    return workers.task_directory(run, index, kind, strategy) / "claim.json"


def try_claim(path):
    """Claim a benchmark for this host; False while another live host holds it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age < LEASE:
                return False
            # Expired: only the host whose rename succeeds takes it over.
            try:
                path.rename(path.with_name(f"claim.expired.{OWNER['id']}.json"))
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(descriptor, "w") as out:
            json.dump(dict(OWNER, time=time.time()), out)
        return True
    return False


def owns(path):
    try:
        return workers.read_json(path).get("id") == OWNER["id"]
    except (OSError, ValueError):
        return False


def release(path):
    if owns(path):
        path.unlink(missing_ok=True)
    for expired in path.parent.glob("claim.expired.*.json"):
        expired.unlink(missing_ok=True)


def complete_packages(run, manifest):
    """Merge every package whose 28 benchmarks are done; returns the packages left."""
    left = 0
    for index, package in enumerate(manifest["packages"]):
        result = workers.package_directory(run, index) / "result.json"
        if result.exists():
            continue
        if all((workers.task_directory(run, index, kind, strategy) / "part.json").exists()
               for kind in workers.KINDS for strategy in workers.MEASURED):
            merge_parts(run, index, package, manifest["runId"])
            print(f"Finished {package}: {result}", flush=True)
        else:
            left += 1
    return left


def merge_parts(run, index, package, run_id):
    """package_benchmarks.merge_parts, with a temporary file of this host only:
    two hosts may complete the same package at the same time."""
    parts = [workers.read_json(workers.task_directory(run, index, kind, strategy) / "part.json")
             for kind in workers.KINDS for strategy in workers.MEASURED]
    corpus = parts[0]["corpus"]
    if any(part["corpus"] != corpus for part in parts):
        raise ValueError(f"Benchmarks of {package} counted different corpora")
    result = dict(schema="coo-package-result-v3", package=package, runId=run_id,
                  corpus=corpus, rows=[row for part in parts for row in part["rows"]])
    path = workers.package_directory(run, index) / "result.json"
    temporary = path.with_name(f"result.pending.{OWNER['id']}.json")
    workers.atomic_json(temporary, result)
    workers.validate_result(temporary, package, run_id)
    temporary.replace(path)


def run_shared(run, image, jobs):
    manifest = workers.prepare(run, image)
    run_id = manifest["runId"]
    packages = manifest["packages"]
    # Validate every checkpoint once, as a single-host resume does.
    for index, package in enumerate(packages):
        result = workers.package_directory(run, index) / "result.json"
        if result.exists():
            workers.validate_result(result, package, run_id)
            continue
        for kind in workers.KINDS:
            for strategy in workers.MEASURED:
                part = workers.task_directory(run, index, kind, strategy) / "part.json"
                if part.exists():
                    workers.validate_part(part, package, run_id, kind, strategy)
    complete_packages(run, manifest)
    pending = [(index, packages[index], kind, strategy)
               for strategy in workers.ORDER for kind in workers.KINDS
               for index in range(len(packages))
               if not (workers.package_directory(run, index) / "result.json").exists()
               and not (workers.task_directory(run, index, kind, strategy) / "part.json").exists()]
    host = OWNER["host"]
    with (run / f"worker-invocations.{host}.jsonl").open("a") as out:
        out.write(json.dumps(dict(jobs=jobs, pending=len(pending), time=time.time(), **OWNER)) + "\n")
    print(f"{len(pending)} benchmarks left in this run; this host ({host}) runs up to {jobs} "
          f"at a time, sharing them with the other hosts of the run", flush=True)
    code = (workers.SCRIPTS / "benchmark-package.st").read_text()

    def label(task):
        _, package, kind, strategy = task
        return f"{package} [{kind} {strategy}]"

    def done(task):
        index, _, kind, strategy = task
        return (workers.task_directory(run, index, kind, strategy) / "part.json").exists()

    active = {}
    progress = {}
    last_beat = last_scan = 0.0
    waiting = None
    try:
        while True:
            for task, (process, log, temporary, started) in list(active.items()):
                status = process.poll()
                progress[task] = workers.report_progress(log, label(task), progress.get(task, 0))
                if status is None:
                    continue
                log.close()
                del active[task]
                index, package, kind, strategy = task
                claim = claim_path(run, index, kind, strategy)
                if status:
                    release(claim)
                    raise RuntimeError(f"Benchmark failed for {label(task)} (exit {status}); see {log.name}")
                seconds = time.monotonic() - started
                workers.validate_part(temporary, package, run_id, kind, strategy)
                temporary.replace(temporary.parent / "part.json")
                shutil.rmtree(temporary.parent / "image", ignore_errors=True)
                release(claim)
                with (run / f"task-timings.{host}.jsonl").open("a") as out:
                    out.write(json.dumps(dict(package=package, kind=kind, strategy=strategy,
                                              seconds=seconds, jobs=jobs, host=host)) + "\n")
                print(f"  {label(task)}: done in {seconds / 60:.1f} min", flush=True)
                left = complete_packages(run, manifest)
                print(f"Packages left in this run: {left}/{len(packages)}", flush=True)
                last_scan = 0.0
            now = time.monotonic()
            if active and now - last_beat >= HEARTBEAT:
                for index, _, kind, strategy in active:
                    try:
                        os.utime(claim_path(run, index, kind, strategy))
                    except FileNotFoundError:
                        pass
                last_beat = now
            if len(active) < jobs and now - last_scan >= RESCAN:
                last_scan = now
                pending = [task for task in pending if task not in active and not done(task)]
                for task in pending:
                    if len(active) >= jobs:
                        break
                    if task in active:
                        continue
                    index, package, kind, strategy = task
                    claim = claim_path(run, index, kind, strategy)
                    if not try_claim(claim):
                        continue
                    if done(task):
                        release(claim)
                        continue
                    process, log, temporary = workers.launch(
                        image, workers.task_directory(run, index, kind, strategy), package, kind,
                        strategy, run_id, run / "selection.json", code)
                    active[task] = (process, log, temporary, time.monotonic())
                    print(f"Started {label(task)} ({len(active)}/{jobs} slots)", flush=True)
                if not active:
                    if not pending:
                        break
                    if len(pending) != waiting:
                        waiting = len(pending)
                        print(f"Waiting: the {waiting} benchmarks left run on other hosts", flush=True)
            time.sleep(0.1 if active else RESCAN)
    finally:
        workers.stop_processes(active)
        for index, _, kind, strategy in active:
            release(claim_path(run, index, kind, strategy))
    left = complete_packages(run, manifest)
    if left:
        raise RuntimeError(f"{left} packages are still incomplete")
    print(f"All {len(packages)} packages are complete", flush=True)


def interrupted(signum, _frame):
    raise KeyboardInterrupt(f"Signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run",))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--jobs", type=workers.positive_jobs, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupted)
    try:
        run_shared(args.run.resolve(), args.image.resolve(), args.jobs)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"Package benchmark error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
