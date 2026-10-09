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

from ranking_corpus import load_split, read_json, sha256, validate_split
from runtime_estimate import calibration_packages


SCRIPTS = Path(__file__).resolve().parent
STRATEGIES = {
    "normal": ("heuristicsBaseline", "heuristicsDependency", "llm05B", "llm15B",
               "llm3B", "llm7B", "hybrid05B", "hybrid15B", "hybrid3B", "hybrid7B"),
    "reranker": ("neuralRank10", "neuralRank20", "neuralRank30", "neuralRank50"),
}
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


def prepare(run, image):
    """Pin every input needed to safely reuse a completed package checkpoint."""
    split = load_split(run / "split.json")
    packages = split["benchmark"]
    if not packages or len(set(packages)) != len(packages):
        raise ValueError("Empty or duplicate benchmark packages")
    inputs = {
        "split": sha256(run / "split.json"),
        "corpus": sha256(run / "corpus.json"),
        "image": sha256(image / "Pharo.image"),
        # Only re-ranker runs train, so only they have a training configuration.
        "trainingConfig": sha256(run / "training-config.json") if (run / "training-config.json").exists() else None,
        # Runs without the re-ranker (training-config "reranker": false) have no partitions.
        "partitions": {name: sha256(run / f"{name}.jsonl") if (run / f"{name}.jsonl").exists() else None
                       for name in ("training", "validation", "test")},
        "scripts": {name: sha256(SCRIPTS / name) for name in (
            "package_benchmarks.py", "benchmark-package.st", "benchmark-export.st",
            "ranking_corpus.py", "reranker_workflow.py")},
    }
    if (run / "model-inputs.json").exists():
        pin_model(run)
    manifest = run / "workers.json"
    if manifest.exists():
        saved = read_json(manifest)
        if saved["inputs"] != inputs or saved["packages"] != packages:
            raise ValueError("Resume inputs changed (split, corpus, image, model or worker scripts); use a new run")
        return saved
    saved = dict(schema="coo-package-workers-v2", runId=uuid.uuid4().hex,
                 inputs=inputs, packages=packages)
    atomic_json(manifest, saved)
    return saved


def pin_model(run):
    """Bind re-ranker checkpoints to the model produced after the normal phase."""
    split = load_split(run / "split.json")
    directory = run / "model"
    metadata = read_json(directory / "metadata.json")
    if metadata.get("packageSplit") != split:
        raise ValueError("Trained model must record the same four-part package split")
    inputs = {p.name: sha256(p) for p in sorted(directory.iterdir()) if p.is_file()}
    if "ranker.onnx" not in inputs:
        raise ValueError("Missing trained model")
    path = run / "model-inputs.json"
    if path.exists() and read_json(path) != inputs:
        raise ValueError("Resume inputs changed: trained model differs from saved re-ranker checkpoints")
    if not path.exists():
        atomic_json(path, inputs)
    return inputs


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
        hits = [row[field] for field in HITS]
        if any(type(n) is not int for n in hits) or not 0 <= hits[0] <= hits[1] <= hits[2] <= hits[3] <= row["count"]:
            raise ValueError(f"Invalid top-k hit counts: {path}")
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


def launch(image, directory, package, phase, run_id, split_path, code):
    """Start one isolated Pharo worker; returns (process, log, pending result path)."""
    working = directory / "image"
    copy_image(image, working)
    atomic_json(directory / "package.json", {"package": package})
    temporary = directory / f"{phase}.pending.json"
    temporary.unlink(missing_ok=True)
    env = dict(os.environ, BENCHMARK_PACKAGE=package, BENCHMARK_PHASE=phase,
               BENCHMARK_RUN_ID=run_id, BENCHMARK_OUTPUT=str(temporary),
               BENCHMARK_SPLIT_FILE=str(split_path))
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
    return process, log, temporary


def execute(pending, jobs, phase, start, finish):
    """Keep at most `jobs` workers running, starting the next package as a slot frees.

    start(index, package) -> (process, log, temporary);
    finish(index, package, temporary, seconds) validates and keeps the result."""
    active = {}
    try:
        while pending or active:
            for index, (process, log, package, temporary, started) in list(active.items()):
                status = process.poll()
                if status is None:
                    continue
                log.close()
                if status:
                    raise RuntimeError(f"{phase} failed for {package} (exit {status}); see {log.name}")
                finish(index, package, temporary, time.monotonic() - started)
                del active[index]
            while pending and len(active) < jobs:
                index, package = pending.pop(0)
                process, log, temporary = start(index, package)
                active[index] = (process, log, package, temporary, time.monotonic())
                print(f"[{phase}] Started {package} ({len(active)}/{jobs} slots)", flush=True)
            if active:
                time.sleep(0.1)
    finally:
        stop_processes(active)


def run_workers(run, image, phase, jobs):
    manifest = prepare(run, image)
    if phase == "reranker":
        pin_model(run)
    rows = read_json(run / "corpus.json")["rowsByPackage"]
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
    # Record the user-selected cap for each invocation (including resumed runs).
    with (run / "worker-invocations.jsonl").open("a") as out:
        out.write(json.dumps(dict(phase=phase, jobs=jobs, pending=len(pending), time=time.time())) + "\n")

    def start(index, package):
        return launch(image, package_directory(run, index), package, phase,
                      manifest["runId"], run / "split.json", code)

    def finish(index, package, temporary, seconds):
        validate_result(temporary, package, phase, manifest["runId"])
        result = package_directory(run, index) / f"{phase}.json"
        temporary.replace(result)
        # Actual durations, to compare against estimate.json.
        with (run / "timings.jsonl").open("a") as out:
            out.write(json.dumps(dict(phase=phase, package=package, rows=rows[package],
                                      seconds=seconds, jobs=jobs)) + "\n")
        print(f"[{phase}] Finished {package} in {seconds / 60:.1f} min: {result}", flush=True)

    execute(pending, jobs, phase, start, finish)


def prepare_calibration(run, jobs):
    """Choose the small TRAINING packages that calibration times, once per concurrency."""
    directory = run / "calibration"
    info_path = directory / "info.json"
    if not (info_path.exists() and read_json(info_path)["jobs"] == jobs):
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir()
        split = load_split(run / "split.json")
        rows = read_json(run / "corpus.json")["rowsByPackage"]
        unused = split.get("unused", [])
        chosen = calibration_packages(split["train"] + unused, rows, max(jobs, 2))
        # Only the chosen packages are worker targets; real benchmark packages are relabelled
        # so the split stays a complete partition, and are never run here.
        calibration_split = dict(split, benchmark=sorted(chosen),
                                 train=sorted((set(split["train"]) - set(chosen)) | set(split["benchmark"])))
        if "unused" in split:
            calibration_split["unused"] = sorted(set(unused) - set(chosen))
        validate_split(calibration_split)
        atomic_json(directory / "split.json", calibration_split)
        atomic_json(info_path, dict(jobs=jobs, runId=uuid.uuid4().hex, packages=chosen,
                                    rows={name: rows[name] for name in chosen}))
    return read_json(info_path)


def calibrate(run, image, phase, jobs):
    """Time real workers on small TRAINING packages at the chosen concurrency.

    Benchmark packages are never touched; results are discarded after timing."""
    directory = run / "calibration"
    info = prepare_calibration(run, jobs)
    if info["jobs"] != jobs:
        raise ValueError("Calibration concurrency differs; recalibrate the normal phase first")
    output = directory / f"{phase}.json"
    if output.exists():
        print(f"[calibration {phase}] Reusing saved timings", flush=True)
        return read_json(output)
    code = (SCRIPTS / "benchmark-package.st").read_text()
    records = []

    def start(index, package):
        return launch(image, directory / "packages" / f"{index + 1:04d}", package, phase,
                      info["runId"], directory / "split.json", code)

    def finish(index, package, temporary, seconds):
        validate_result(temporary, package, phase, info["runId"])
        records.append(dict(package=package, rows=info["rows"][package], seconds=seconds))
        print(f"[calibration {phase}] {package}: {info['rows'][package]} examples in {seconds:.0f} s", flush=True)

    started = time.monotonic()
    execute(list(enumerate(info["packages"])), jobs, f"calibration {phase}", start, finish)
    result = dict(jobs=jobs, wallSeconds=time.monotonic() - started, packages=records)
    atomic_json(output, result)
    for path in (directory / "packages").glob("*/image"):
        shutil.rmtree(path)
    return result


def aggregate(run, image, normal_only=False, phases=None):
    manifest = prepare(run, image)
    if phases is None:
        phases = ("normal",) if normal_only else tuple(STRATEGIES)
    if "reranker" in phases:
        pin_model(run)
    combined = dict(packages=manifest["packages"], benchmarkPhases=list(phases),
                    corpus=dict(packages=0, classes=0, methods=0))
    totals = {phase: {} for phase in phases}
    for index, package in enumerate(manifest["packages"]):
        results = {phase: validate_result(package_directory(run, index) / f"{phase}.json",
                                         package, phase, manifest["runId"])
                   for phase in phases}
        corpus = results[phases[0]]["corpus"]
        if "reranker" in results and corpus != results["reranker"]["corpus"]:
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
    for phase in phases:
        combined[phase] = list(totals[phase].values())
    path = run / "aggregate.json"
    atomic_json(path, combined)
    return path


def interrupted(signum, _frame):
    raise KeyboardInterrupt(f"Signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "pin-model", "run", "prepare-calibration",
                                           "calibrate", "aggregate"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--phase", choices=tuple(STRATEGIES))
    parser.add_argument("--jobs", type=positive_jobs)
    parser.add_argument("--normal-only", action="store_true", help="Aggregate only normal benchmarks")
    args = parser.parse_args()
    if args.action in ("run", "calibrate") and (args.phase is None or args.jobs is None):
        parser.error(f"{args.action} requires --phase and --jobs (no automatic concurrency)")
    if args.normal_only and args.action != "aggregate":
        parser.error("--normal-only applies only to aggregate")
    if args.action == "prepare-calibration" and args.jobs is None:
        parser.error("prepare-calibration requires --jobs")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.action == "prepare":
            prepare(args.run.resolve(), args.image.resolve())
        elif args.action == "pin-model":
            prepare(args.run.resolve(), args.image.resolve())
            pin_model(args.run.resolve())
        elif args.action == "run":
            run_workers(args.run.resolve(), args.image.resolve(), args.phase, args.jobs)
        elif args.action == "prepare-calibration":
            prepare_calibration(args.run.resolve(), args.jobs)
        elif args.action == "calibrate":
            calibrate(args.run.resolve(), args.image.resolve(), args.phase, args.jobs)
        else:
            # --phase aggregates that phase alone (separate normal and re-ranker pipelines).
            phases = (args.phase,) if args.phase else None
            print(aggregate(args.run.resolve(), args.image.resolve(), args.normal_only, phases))
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"Package benchmark error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
