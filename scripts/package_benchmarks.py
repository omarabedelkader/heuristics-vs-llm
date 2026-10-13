"""Saved package list, bounded package workers, resumable checkpoints and aggregation."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

from benchmark_source import read_json, sha256, write_json


SCRIPTS = Path(__file__).resolve().parent
STRATEGIES = ("heuristicsBaseline", "heuristicsDependency", "llm05B", "llm15B",
              "llm3B", "llm7B", "hybrid05B", "hybrid15B", "hybrid3B", "hybrid7B")
SELECTION = "coo-package-split-v1"
TOTALS = ("count", "reciprocalRankSum", "timeMs", "memoryBytes", "top1", "top2", "top3", "top10")
# Observations whose target is ranked within the first k (Top-k accuracy, Recall@k).
HITS = ("top1", "top2", "top3", "top10")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def positive_jobs(value):
    if not value or not value.isascii() or not value.isdecimal() or int(value) < 1:
        raise argparse.ArgumentTypeError("BENCHMARK_JOBS must be an explicit positive integer")
    return int(value)


def package_names(value, label):
    if (not isinstance(value, list) or not value
            or any(not isinstance(name, str) or not name for name in value)
            or len(value) != len(set(value))):
        raise ValueError(f"Invalid or duplicate package names in {label}")
    return set(value)


def load_selection(path):
    """The saved benchmark packages and the image's eligible packages they were chosen from."""
    selection = read_json(path)
    if selection.get("schema") != SELECTION or type(selection.get("seed")) is not int:
        raise ValueError(f"{path}: not a saved benchmark selection")
    if not package_names(selection.get("benchmark"), "benchmark") <= package_names(selection.get("eligible"), "eligible"):
        raise ValueError(f"{path}: benchmark packages outside the eligible packages")
    return selection


def select(packages_path, eligible_path, output, seed=42):
    """Save a user-supplied list of benchmark packages as the experiment's selection.

    The file is a saved selection, a JSON array of package names, or plain text with
    one package name per line (# starts a comment)."""
    eligible = package_names(read_json(eligible_path), str(eligible_path))
    text = packages_path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [line.split("#", 1)[0].strip() for line in text.splitlines()]
        data = [name for name in data if name]
    if isinstance(data, dict):
        if data.get("schema") != SELECTION or type(data.get("seed")) is not int:
            raise ValueError(f"{packages_path}: not a saved benchmark selection")
        benchmark, seed = data.get("benchmark"), data["seed"]
    else:
        benchmark = data
    names = package_names(benchmark, str(packages_path))
    unknown = sorted(names - eligible)
    if unknown:
        raise ValueError(f"{len(unknown)} listed packages are not benchmark-eligible in this image: "
                         + ", ".join(unknown[:5]) + (" ..." if len(unknown) > 5 else ""))
    if names == eligible:
        raise ValueError("The saved selection format needs at least one eligible package outside the list")
    if output.exists():
        if set(read_json(output).get("benchmark", [])) != names:
            raise ValueError(f"{output} already saves a different benchmark package list; "
                             "use a new EXPERIMENT_DIR or remove BENCHMARK_PACKAGES_FILE")
        return read_json(output)
    selection = dict(schema=SELECTION, seed=seed, eligible=sorted(eligible),
                     benchmark=sorted(names), train=sorted(eligible - names))
    write_json(output, selection)
    return selection


def prepare(run, image):
    """Pin every input needed to safely reuse a completed package checkpoint."""
    packages = load_selection(run / "selection.json")["benchmark"]
    inputs = {
        "selection": sha256(run / "selection.json"),
        "image": sha256(image / "Pharo.image"),
        "scripts": {name: sha256(SCRIPTS / name) for name in (
            "package_benchmarks.py", "benchmark-package.st", "benchmark-export.st")},
    }
    manifest = run / "workers.json"
    if manifest.exists():
        saved = read_json(manifest)
        if saved["inputs"] != inputs or saved["packages"] != packages:
            raise ValueError("Resume inputs changed (packages, image or worker scripts); use a new run")
        return saved
    saved = dict(schema="coo-package-workers-v3", runId=uuid.uuid4().hex,
                 inputs=inputs, packages=packages)
    atomic_json(manifest, saved)
    return saved


def row_keys():
    return {(kind, strategy, prefix) for kind in ("messages", "variables")
            for strategy in STRATEGIES for prefix in range(2, 9)}


def validate_result(path, package, run_id):
    result = read_json(path)
    if (result.get("schema") != "coo-package-result-v2" or result.get("package") != package
            or result.get("runId") != run_id):
        raise ValueError(f"Wrong package or run identity: {path}")
    corpus = result["corpus"]
    if (set(corpus) != {"packages", "classes", "methods"} or corpus["packages"] != 1
            or any(type(n) is not int or n < 0 for n in corpus.values())):
        raise ValueError(f"Invalid corpus counts: {path}")
    seen = set()
    for row in result["rows"]:
        key = (row["kind"], row["strategy"], row["prefix"])
        if key in seen or key not in row_keys() or type(row["prefix"]) is not int:
            raise ValueError(f"Duplicate or unexpected measurement: {path}: {key}")
        seen.add(key)
        if type(row["count"]) is not int or row["count"] < 0:
            raise ValueError(f"Invalid observation count: {path}")
        for field in TOTALS[1:]:
            value = row[field]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Invalid {field}: {path}")
        if not 0 <= row["reciprocalRankSum"] <= row["count"] + 1e-8 or row["timeMs"] < 0:
            raise ValueError(f"Invalid reciprocal rank or time totals: {path}")
        if row["count"] == 0 and any(row[field] != 0 for field in TOTALS[1:]):
            raise ValueError(f"Totals without observations: {path}")
        hits = [row[field] for field in HITS]
        if any(type(n) is not int for n in hits) or not 0 <= hits[0] <= hits[1] <= hits[2] <= hits[3] <= row["count"]:
            raise ValueError(f"Invalid top-k hit counts: {path}")
    if seen != row_keys():
        raise ValueError(f"Incomplete measurements: {path}")
    return result


def package_directory(run, index):
    return run / "packages" / f"{index + 1:04d}"


def copy_image(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    # VM and sources are shared read-only; each worker owns its image, changes,
    # working directory and pharo-local. No worker saves the template image.
    for name in ("Pharo.image", "Pharo.changes"):
        if (source / name).exists():
            shutil.copy2(source / name, destination / name)
    for path in source.glob("*.sources"):
        target = destination / path.name
        if not target.exists():
            target.symlink_to(path.resolve())


def stop_processes(active):
    for process, *_ in active.values():
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            # macOS reports EPERM instead of ESRCH for a group that already exited.
            pass
    deadline = time.monotonic() + 5
    for process, log, *_ in active.values():
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            process.wait()
        log.close()


def launch(image, directory, package, run_id, selection_path, code):
    """Start one isolated Pharo worker; returns (process, log, pending result path)."""
    working = directory / "image"
    copy_image(image, working)
    atomic_json(directory / "package.json", {"package": package})
    temporary = directory / "result.pending.json"
    temporary.unlink(missing_ok=True)
    env = dict(os.environ, BENCHMARK_PACKAGE=package, BENCHMARK_RUN_ID=run_id,
               BENCHMARK_OUTPUT=str(temporary), BENCHMARK_RUN_SELECTION=str(selection_path))
    log = (directory / "worker.log").open("w")
    try:
        process = subprocess.Popen(
            [str(image / "pharo"), "--headless", "Pharo.image",
             "--no-default-preferences", "eval", code],
            cwd=working, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    except BaseException:
        log.close()
        raise
    return process, log, temporary


PROGRESS = "Progress: "


def report_progress(log, package, offset):
    """Show the benchmarks a worker finished since offset; returns the new offset."""
    with open(log.name, errors="replace") as source:
        source.seek(offset)
        while True:
            line = source.readline()
            if not line.endswith("\n"):
                return offset
            offset = source.tell()
            if line.startswith(PROGRESS):
                print(f"  {package}: {line[len(PROGRESS):].strip()}", flush=True)


def execute(pending, jobs, start, finish):
    """Keep at most `jobs` workers running, starting the next package as a slot frees.

    start(index, package) -> (process, log, temporary);
    finish(index, package, temporary, seconds) validates and keeps the result."""
    active = {}
    progress = {}
    try:
        while pending or active:
            for index, (process, log, package, temporary, started) in list(active.items()):
                status = process.poll()
                progress[index] = report_progress(log, package, progress.get(index, 0))
                if status is None:
                    continue
                log.close()
                if status:
                    raise RuntimeError(f"Benchmark failed for {package} (exit {status}); see {log.name}")
                finish(index, package, temporary, time.monotonic() - started)
                del active[index]
            while pending and len(active) < jobs:
                index, package = pending.pop(0)
                process, log, temporary = start(index, package)
                active[index] = (process, log, package, temporary, time.monotonic())
                print(f"Started {package} ({len(active)}/{jobs} slots)", flush=True)
            if active:
                time.sleep(0.1)
    finally:
        stop_processes(active)


def run_workers(run, image, jobs):
    manifest = prepare(run, image)
    total = len(manifest["packages"])
    pending = []
    for index, package in enumerate(manifest["packages"]):
        result = package_directory(run, index) / "result.json"
        if result.exists():
            validate_result(result, package, manifest["runId"])
            print(f"Reusing {package}", flush=True)
        else:
            pending.append((index, package))
    done = [total - len(pending)]
    code = (SCRIPTS / "benchmark-package.st").read_text()
    # Record the user-selected cap for each invocation (including resumed runs).
    with (run / "worker-invocations.jsonl").open("a") as out:
        out.write(json.dumps(dict(jobs=jobs, pending=len(pending), time=time.time())) + "\n")

    def start(index, package):
        return launch(image, package_directory(run, index), package,
                      manifest["runId"], run / "selection.json", code)

    def finish(index, package, temporary, seconds):
        validate_result(temporary, package, manifest["runId"])
        result = package_directory(run, index) / "result.json"
        temporary.replace(result)
        with (run / "timings.jsonl").open("a") as out:
            out.write(json.dumps(dict(package=package, seconds=seconds, jobs=jobs)) + "\n")
        done[0] += 1
        print(f"Finished {package} in {seconds / 60:.1f} min ({done[0]}/{total} packages): {result}", flush=True)

    execute(pending, jobs, start, finish)


def aggregate(run, image):
    manifest = prepare(run, image)
    combined = dict(packages=manifest["packages"], corpus=dict(packages=0, classes=0, methods=0))
    totals = {}
    for index, package in enumerate(manifest["packages"]):
        result = validate_result(package_directory(run, index) / "result.json", package, manifest["runId"])
        for key, value in result["corpus"].items():
            combined["corpus"][key] += value
        for row in result["rows"]:
            key = (row["kind"], row["strategy"], row["prefix"])
            target = totals.setdefault(key, dict(zip(("kind", "strategy", "prefix"), key),
                                                 **dict.fromkeys(TOTALS, 0)))
            for field in TOTALS:
                target[field] += row[field]
    combined["rows"] = list(totals.values())
    path = run / "aggregate.json"
    atomic_json(path, combined)
    return path


def interrupted(signum, _frame):
    raise KeyboardInterrupt(f"Signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("select", "prepare", "run", "aggregate"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--jobs", type=positive_jobs)
    parser.add_argument("--packages", type=Path, help="select: the benchmark package list")
    parser.add_argument("--eligible", type=Path, help="select: the image's eligible package names")
    parser.add_argument("--output", type=Path, help="select: the experiment's saved selection")
    parser.add_argument("--seed", type=int, default=42, help="select: seed recorded for a plain list")
    args = parser.parse_args()
    if args.action == "select":
        if args.packages is None or args.eligible is None or args.output is None:
            parser.error("select requires --packages, --eligible and --output")
    elif args.run is None or args.image is None:
        parser.error(f"{args.action} requires --run and --image")
    if args.action == "run" and args.jobs is None:
        parser.error("run requires --jobs (no automatic concurrency)")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.action == "select":
            result = select(args.packages, args.eligible, args.output, args.seed)
            print(f"Benchmark packages: {len(result['benchmark'])} saved in {args.output}")
        elif args.action == "prepare":
            prepare(args.run.resolve(), args.image.resolve())
        elif args.action == "run":
            run_workers(args.run.resolve(), args.image.resolve(), args.jobs)
        else:
            print(aggregate(args.run.resolve(), args.image.resolve()))
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"Package benchmark error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
