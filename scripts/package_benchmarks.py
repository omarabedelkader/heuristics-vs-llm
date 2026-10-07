"""Bounded package workers, resumable checkpoints and publication aggregation."""
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

from ranking_corpus import read_json, sha256


SCRIPTS = Path(__file__).resolve().parent
STRATEGIES = {
    "normal": ("heuristicsBaseline", "heuristicsDependency", "llm05B", "llm15B",
               "llm3B", "llm7B", "hybrid05B", "hybrid15B", "hybrid3B", "hybrid7B"),
    "reranker": ("neuralRank10", "neuralRank20", "neuralRank30", "neuralRank50"),
}
TOTALS = ("count", "reciprocalRankSum", "timeMs", "memoryBytes")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def positive_jobs(value):
    if not value or not value.isascii() or not value.isdecimal() or int(value) < 1:
        raise argparse.ArgumentTypeError("BENCHMARK_JOBS must be an explicit positive integer")
    return int(value)


def prepare(run, image):
    """Pin every input needed to safely reuse a completed package checkpoint."""
    split = read_json(run / "split.json")
    packages = split["benchmark"]
    if not packages or len(set(packages)) != len(packages):
        raise ValueError("Empty or duplicate benchmark packages")
    inputs = {
        "split": sha256(run / "split.json"),
        "corpus": sha256(run / "corpus.json"),
        "image": sha256(image / "Pharo.image"),
        "model": {p.name: sha256(p) for p in sorted((run / "model").iterdir()) if p.is_file()},
        "scripts": {name: sha256(SCRIPTS / name) for name in (
            "package_benchmarks.py", "benchmark-package.st", "benchmark-export.st")},
    }
    if "ranker.onnx" not in inputs["model"] or "metadata.json" not in inputs["model"]:
        raise ValueError("Missing trained model or metadata; cannot checkpoint this run")
    manifest = run / "workers.json"
    if manifest.exists():
        saved = read_json(manifest)
        if saved["inputs"] != inputs or saved["packages"] != packages:
            raise ValueError("Resume inputs changed (split, corpus, image, model or worker scripts); use a new run")
        return saved
    saved = dict(schema="coo-package-workers-v1", runId=uuid.uuid4().hex,
                 inputs=inputs, packages=packages)
    atomic_json(manifest, saved)
    return saved


def row_keys(phase):
    return {(kind, strategy, prefix) for kind in ("messages", "variables")
            for strategy in STRATEGIES[phase] for prefix in range(2, 9)}


def validate_result(path, package, phase, run_id):
    result = read_json(path)
    if (result.get("schema") != "coo-package-result-v1" or result.get("package") != package
            or result.get("phase") != phase or result.get("runId") != run_id):
        raise ValueError(f"Wrong package, phase or run identity: {path}")
    corpus = result["corpus"]
    if (set(corpus) != {"packages", "classes", "methods"} or corpus["packages"] != 1
            or any(type(n) is not int or n < 0 for n in corpus.values())):
        raise ValueError(f"Invalid corpus counts: {path}")
    seen = set()
    for row in result["rows"]:
        key = (row["kind"], row["strategy"], row["prefix"])
        if key in seen or key not in row_keys(phase) or type(row["prefix"]) is not int:
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
    if seen != row_keys(phase):
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
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    for process, log, *_ in active.values():
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        log.close()


def run_workers(run, image, phase, jobs):
    manifest = prepare(run, image)
    pending = []
    for index, package in enumerate(manifest["packages"]):
        directory = package_directory(run, index)
        result = directory / f"{phase}.json"
        if result.exists():
            validate_result(result, package, phase, manifest["runId"])
            print(f"[{phase}] Reusing {package}", flush=True)
        else:
            pending.append((index, package))
    code = (SCRIPTS / "benchmark-package.st").read_text()
    active = {}
    # Record the user-selected cap for each invocation (including resumed runs).
    with (run / "worker-invocations.jsonl").open("a") as out:
        out.write(json.dumps(dict(phase=phase, jobs=jobs, pending=len(pending), time=time.time())) + "\n")
    try:
        while pending or active:
            for index, (process, log, package, temporary, result) in list(active.items()):
                status = process.poll()
                if status is None:
                    continue
                log.close()
                if status:
                    raise RuntimeError(f"{phase} failed for {package} (exit {status}); see {log.name}")
                validate_result(temporary, package, phase, manifest["runId"])
                temporary.replace(result)
                del active[index]
                print(f"[{phase}] Finished {package}: {result}", flush=True)
            while pending and len(active) < jobs:
                index, package = pending.pop(0)
                directory = package_directory(run, index)
                working = directory / "image"
                copy_image(image, working)
                atomic_json(directory / "package.json", {"package": package})
                temporary = directory / f"{phase}.pending.json"
                temporary.unlink(missing_ok=True)
                result = directory / f"{phase}.json"
                env = dict(os.environ, BENCHMARK_PACKAGE=package, BENCHMARK_PHASE=phase,
                           BENCHMARK_RUN_ID=manifest["runId"], BENCHMARK_OUTPUT=str(temporary),
                           BENCHMARK_SPLIT_FILE=str(run / "split.json"))
                log = (directory / f"{phase}.log").open("w")
                try:
                    process = subprocess.Popen(
                        [str(image / "pharo"), "--headless", "Pharo.image",
                         "--no-default-preferences", "eval", code],
                        cwd=working, env=env, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                active[index] = (process, log, package, temporary, result)
                print(f"[{phase}] Started {package} ({len(active)}/{jobs} slots)", flush=True)
            if active:
                time.sleep(0.1)
    finally:
        stop_processes(active)


def aggregate(run, image):
    manifest = prepare(run, image)
    combined = dict(packages=manifest["packages"], corpus=dict(packages=0, classes=0, methods=0))
    totals = {phase: {} for phase in STRATEGIES}
    for index, package in enumerate(manifest["packages"]):
        results = {phase: validate_result(package_directory(run, index) / f"{phase}.json",
                                         package, phase, manifest["runId"])
                   for phase in STRATEGIES}
        corpus = results["normal"]["corpus"]
        if corpus != results["reranker"]["corpus"]:
            raise ValueError(f"Normal and re-ranker corpus counts differ for {package}")
        for key, value in corpus.items():
            combined["corpus"][key] += value
        for phase, result in results.items():
            for row in result["rows"]:
                key = (row["kind"], row["strategy"], row["prefix"])
                target = totals[phase].setdefault(key, dict(zip(("kind", "strategy", "prefix"), key),
                                                           **dict.fromkeys(TOTALS, 0)))
                for field in TOTALS:
                    target[field] += row[field]
    for phase in STRATEGIES:
        combined[phase] = list(totals[phase].values())
    path = run / "aggregate.json"
    atomic_json(path, combined)
    return path


def interrupted(signum, _frame):
    raise KeyboardInterrupt(f"Signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "aggregate"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--phase", choices=tuple(STRATEGIES))
    parser.add_argument("--jobs", type=positive_jobs)
    args = parser.parse_args()
    if args.action == "run" and (args.phase is None or args.jobs is None):
        parser.error("run requires --phase and --jobs (no automatic concurrency)")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.action == "prepare":
            prepare(args.run.resolve(), args.image.resolve())
        elif args.action == "run":
            run_workers(args.run.resolve(), args.image.resolve(), args.phase, args.jobs)
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
