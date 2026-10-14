"""Saved package list, bounded benchmark workers, resumable checkpoints and aggregation.

Each package is split into its 28 independent benchmarks (2 kinds x 14 strategies).
Every benchmark runs in its own Pharo process, so all --jobs slots stay busy even with
one package, and the Ollama server batches the concurrent requests of the workers."""
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
# LLM completion is measured twice: answering whatever the model generates (before
# Pharo validation), and answering only real names of the image (after).
LLM = ("llm05B", "llm15B", "llm3B", "llm7B")
BEFORE_VALIDATION = {name: name.replace("llm", "llmUnvalidated") for name in LLM}
MEASURED = STRATEGIES + tuple(BEFORE_VALIDATION.values())
RESULT_SETS = ("before", "after")
# Candidates an LLM completion answered, and how many are real names of the image.
CANDIDATES = ("generated", "valid")
SELECTION = "coo-package-split-v1"
TOTALS = ("count", "reciprocalRankSum", "timeMs", "memoryBytes", "top1", "top2", "top3", "top10")
# Observations whose target is ranked within the first k (Top-k accuracy, Recall@k).
HITS = ("top1", "top2", "top3", "top10")
KINDS = ("messages", "variables")
PREFIXES = range(2, 9)
# Longest benchmarks first (largest model, then the unvalidated, validated and hybrid
# runs): the slowest work never starts last, and concurrent workers query the same
# model at the same time, which Ollama batches.
ORDER = tuple(strategy for size in ("7B", "3B", "15B", "05B")
              for strategy in (f"llmUnvalidated{size}", f"llm{size}", f"hybrid{size}")) + (
    "heuristicsBaseline", "heuristicsDependency")
assert sorted(ORDER) == sorted(MEASURED)


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
    saved = dict(schema="coo-package-workers-v4", runId=uuid.uuid4().hex,
                 inputs=inputs, packages=packages)
    atomic_json(manifest, saved)
    return saved


def row_keys():
    return {(kind, strategy, prefix) for kind in KINDS
            for strategy in MEASURED for prefix in PREFIXES}


def check_corpus(corpus, path):
    if (set(corpus) != {"packages", "classes", "methods"} or corpus["packages"] != 1
            or any(type(n) is not int or n < 0 for n in corpus.values())):
        raise ValueError(f"Invalid corpus counts: {path}")


def check_rows(rows, expected, path):
    seen = set()
    for row in rows:
        key = (row["kind"], row["strategy"], row["prefix"])
        if key in seen or key not in expected or type(row["prefix"]) is not int:
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
        if row["strategy"] in LLM or row["strategy"] in BEFORE_VALIDATION.values():
            generated, valid = (row.get(field) for field in CANDIDATES)
            if (type(generated) is not int or type(valid) is not int
                    or not 0 <= valid <= generated <= 10 * row["count"]
                    or (row["strategy"] in LLM and valid != generated)):
                raise ValueError(f"Invalid candidate counts: {path}: {key}")
    if seen != expected:
        raise ValueError(f"Incomplete measurements: {path}")


def validate_result(path, package, run_id):
    result = read_json(path)
    if (result.get("schema") != "coo-package-result-v3" or result.get("package") != package
            or result.get("runId") != run_id):
        raise ValueError(f"Wrong package or run identity: {path}")
    check_corpus(result["corpus"], path)
    check_rows(result["rows"], row_keys(), path)
    return result


def validate_part(path, package, run_id, kind, strategy):
    """One benchmark of a package: its seven prefix rows and the package corpus."""
    part = read_json(path)
    if (part.get("schema") != "coo-package-part-v1" or part.get("package") != package
            or part.get("runId") != run_id or part.get("kind") != kind
            or part.get("strategy") != strategy):
        raise ValueError(f"Wrong package, benchmark or run identity: {path}")
    check_corpus(part["corpus"], path)
    check_rows(part["rows"], {(kind, strategy, prefix) for prefix in PREFIXES}, path)
    return part


def package_directory(run, index):
    return run / "packages" / f"{index + 1:04d}"


def task_directory(run, index, kind, strategy):
    return package_directory(run, index) / "benchmarks" / f"{kind}-{strategy}"


def merge_parts(run, index, package, run_id):
    """Combine a package's 28 benchmark parts into its checked result.json."""
    parts = [read_json(task_directory(run, index, kind, strategy) / "part.json")
             for kind in KINDS for strategy in MEASURED]
    corpus = parts[0]["corpus"]
    if any(part["corpus"] != corpus for part in parts):
        raise ValueError(f"Benchmarks of {package} counted different corpora")
    result = dict(schema="coo-package-result-v3", package=package, runId=run_id,
                  corpus=corpus, rows=[row for part in parts for row in part["rows"]])
    path = package_directory(run, index) / "result.json"
    temporary = path.with_suffix(".pending.json")
    atomic_json(temporary, result)
    validate_result(temporary, package, run_id)
    temporary.replace(path)
    return path


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


def launch(image, directory, package, kind, strategy, run_id, selection_path, code):
    """Start one isolated Pharo worker for one benchmark of a package;
    returns (process, log, pending part path)."""
    working = directory / "image"
    copy_image(image, working)
    atomic_json(directory.parent.parent / "package.json", {"package": package})
    temporary = directory / "part.pending.json"
    temporary.unlink(missing_ok=True)
    env = dict(os.environ, BENCHMARK_PACKAGE=package, BENCHMARK_KIND=kind,
               BENCHMARK_STRATEGY=strategy, BENCHMARK_RUN_ID=run_id,
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


def report_progress(log, label, offset):
    """Show the benchmarks a worker finished since offset; returns the new offset."""
    with open(log.name, errors="replace") as source:
        source.seek(offset)
        while True:
            line = source.readline()
            if not line.endswith("\n"):
                return offset
            offset = source.tell()
            if line.startswith(PROGRESS):
                print(f"  {label}: {line[len(PROGRESS):].strip()}", flush=True)


def execute(pending, jobs, start, finish, label=str):
    """Keep at most `jobs` workers running, starting the next task as a slot frees.

    start(task) -> (process, log, temporary);
    finish(task, temporary, seconds) validates and keeps the result."""
    active = {}
    progress = {}
    try:
        while pending or active:
            for task, (process, log, temporary, started) in list(active.items()):
                status = process.poll()
                progress[task] = report_progress(log, label(task), progress.get(task, 0))
                if status is None:
                    continue
                log.close()
                if status:
                    raise RuntimeError(f"Benchmark failed for {label(task)} (exit {status}); see {log.name}")
                finish(task, temporary, time.monotonic() - started)
                del active[task]
            while pending and len(active) < jobs:
                task = pending.pop(0)
                process, log, temporary = start(task)
                active[task] = (process, log, temporary, time.monotonic())
                print(f"Started {label(task)} ({len(active)}/{jobs} slots)", flush=True)
            if active:
                time.sleep(0.1)
    finally:
        stop_processes(active)


def run_workers(run, image, jobs):
    manifest = prepare(run, image)
    run_id = manifest["runId"]
    total = len(manifest["packages"])
    # Benchmarks still to run, per package; a package is complete when none remain.
    remaining = {}
    for index, package in enumerate(manifest["packages"]):
        result = package_directory(run, index) / "result.json"
        if result.exists():
            validate_result(result, package, run_id)
            print(f"Reusing {package}", flush=True)
            continue
        remaining[index] = set()
        for kind in KINDS:
            for strategy in MEASURED:
                part = task_directory(run, index, kind, strategy) / "part.json"
                if part.exists():
                    validate_part(part, package, run_id, kind, strategy)
                else:
                    remaining[index].add((kind, strategy))
        if not remaining[index]:
            merge_parts(run, index, package, run_id)
            del remaining[index]
            print(f"Reusing {package} (all benchmarks were done)", flush=True)
    pending = [(index, manifest["packages"][index], kind, strategy)
               for strategy in ORDER for kind in KINDS
               for index in sorted(remaining) if (kind, strategy) in remaining[index]]
    done = [total - len(remaining)]
    started = {}
    code = (SCRIPTS / "benchmark-package.st").read_text()
    # Record the user-selected cap for each invocation (including resumed runs).
    with (run / "worker-invocations.jsonl").open("a") as out:
        out.write(json.dumps(dict(jobs=jobs, pending=len(pending), time=time.time())) + "\n")
    print(f"{len(pending)} benchmarks of {len(remaining)} packages to run, {jobs} at a time", flush=True)

    def label(task):
        _, package, kind, strategy = task
        return f"{package} [{kind} {strategy}]"

    def start(task):
        index, package, kind, strategy = task
        started.setdefault(index, time.monotonic())
        return launch(image, task_directory(run, index, kind, strategy), package, kind, strategy,
                      run_id, run / "selection.json", code)

    def finish(task, temporary, seconds):
        index, package, kind, strategy = task
        validate_part(temporary, package, run_id, kind, strategy)
        directory = temporary.parent
        temporary.replace(directory / "part.json")
        # The part is kept; its private image copy is not needed any more.
        shutil.rmtree(directory / "image", ignore_errors=True)
        with (run / "task-timings.jsonl").open("a") as out:
            out.write(json.dumps(dict(package=package, kind=kind, strategy=strategy,
                                      seconds=seconds, jobs=jobs)) + "\n")
        remaining[index].discard((kind, strategy))
        if remaining[index]:
            return
        result = merge_parts(run, index, package, run_id)
        elapsed = time.monotonic() - started.get(index, time.monotonic() - seconds)
        with (run / "timings.jsonl").open("a") as out:
            out.write(json.dumps(dict(package=package, seconds=elapsed, jobs=jobs)) + "\n")
        done[0] += 1
        print(f"Finished {package} in {elapsed / 60:.1f} min ({done[0]}/{total} packages): {result}", flush=True)

    execute(pending, jobs, start, finish, label)


def result_set(rows, name):
    """The ten published strategies; before validation, LLM completion is the unvalidated run."""
    if name == "after":
        return [row for row in rows if row["strategy"] in STRATEGIES]
    renamed = {before: after for after, before in BEFORE_VALIDATION.items()}
    return [dict(row, strategy=renamed.get(row["strategy"], row["strategy"]))
            for row in rows if row["strategy"] not in LLM]


def validation_table(rows):
    """LaTeX: how many of the candidates the models generate are real Pharo names."""
    lines = [
        r"\begin{table}[ht]", r"\centering",
        r"\caption{Candidates generated by LLM completion before Pharo validation, and the share that "
        r"are real Pharo names (class names for Classes, selectors for Methods), over all packages and "
        r"prefixes 2--8.}",
        r"\renewcommand{\arraystretch}{1.3}", r"\begin{tabular}{|c|l|r|r|r|r|}", r"\hline",
        r"\textbf{} & \textbf{Model} & \textbf{Completions} & \textbf{Candidates} & "
        r"\textbf{Valid in Pharo} & \textbf{Valid (\%)} \\", r"\hline"]
    for kind, label in (("messages", "Methods"), ("variables", "Classes")):
        lines.append(rf"\multirow{{{len(LLM)}}}{{*}}{{\textbf{{{label}}}}}")
        for index, (after, before) in enumerate(BEFORE_VALIDATION.items()):
            chosen = [row for row in rows if row["kind"] == kind and row["strategy"] == before]
            count, generated, valid = (sum(row[field] for row in chosen)
                                       for field in ("count", *CANDIDATES))
            share = f"{100 * valid / generated:.1f}" if generated else "--"
            lines.append(f"& {after[3:].replace('05', '0.5').replace('15', '1.5')} & {count} & {generated} "
                         f"& {valid} & {share} \\\\" + (r" \cline{2-6}" if index < len(LLM) - 1 else ""))
        lines.append(r"\hline")
    return "\n".join(lines + [r"\end{tabular}", r"\end{table}"]) + "\n"


def aggregate(run, image):
    """Sum the packages; write one aggregate per result set and the validation table."""
    manifest = prepare(run, image)
    combined = dict(packages=manifest["packages"], corpus=dict(packages=0, classes=0, methods=0))
    totals = {}
    for index, package in enumerate(manifest["packages"]):
        result = validate_result(package_directory(run, index) / "result.json", package, manifest["runId"])
        for key, value in result["corpus"].items():
            combined["corpus"][key] += value
        for row in result["rows"]:
            key = (row["kind"], row["strategy"], row["prefix"])
            fields = TOTALS + (CANDIDATES if "generated" in row else ())
            target = totals.setdefault(key, dict(zip(("kind", "strategy", "prefix"), key),
                                                 **dict.fromkeys(fields, 0)))
            for field in fields:
                target[field] += row[field]
    rows = list(totals.values())
    for name in RESULT_SETS:
        atomic_json(run / f"aggregate-{name}.json", dict(combined, rows=result_set(rows, name)))
    (run / "pharo-validation.tex").write_text(validation_table(rows), encoding="utf-8")
    path = run / "aggregate.json"
    atomic_json(path, dict(combined, rows=rows))
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
