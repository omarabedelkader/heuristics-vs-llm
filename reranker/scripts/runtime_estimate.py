"""Predict the remaining wall time of a run from timings measured on this machine.

Calibration benchmarks a few small TRAINING packages (never benchmark packages) at
the chosen concurrency, so the measured cost includes Pharo startup and contention
on the shared Ollama and re-ranker servers. Package time is modelled as
startup + perExample * examples, then the real FIFO worker queue is simulated.
"""
import argparse
import heapq
import json
import math
from pathlib import Path
import sys

from ranking_corpus import read_json, write_json


MARGIN = 1.25


def calibration_packages(candidates, rows, count, smallest=30, largest=300):
    """Spread sizes geometrically so a linear fit separates startup from per-example cost."""
    pool = sorted((rows[name], name) for name in candidates if rows[name] > 0)
    if len(pool) < count or len(candidates) <= count:
        raise ValueError(f"Calibration needs more than {count} nonempty training packages")
    chosen = []
    for i in range(count):
        target = smallest * (largest / smallest) ** (i / max(count - 1, 1))
        _, name = min((p for p in pool if p[1] not in chosen),
                      key=lambda p: (abs(math.log(p[0] / target)), p[1]))
        chosen.append(name)
    return chosen


def fit(records):
    """Least squares seconds = startup + perExample * examples, both non-negative."""
    xs = [r["rows"] for r in records]
    ys = [r["seconds"] for r in records]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    if slope <= 0:
        # Sizes too similar or noisy: charge the average cost per example (conservative).
        return 0.0, sum(ys) / max(sum(xs), 1)
    startup = my - slope * mx
    if startup < 0:
        return 0.0, sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    return startup, slope


def makespan(durations, jobs):
    """Same order and slot reuse as package_benchmarks.execute."""
    slots = [0.0] * jobs
    for duration in durations:
        heapq.heappush(slots, heapq.heappop(slots) + duration)
    return max(slots)


def duration(seconds):
    minutes = math.ceil(seconds / 60)
    return f"{minutes // 60}h{minutes % 60:02d}m"


def estimate(run, jobs, elapsed, normal_only=False, training=True, reranker_only=False):
    workers = read_json(run / "workers.json")
    rows = read_json(run / "corpus.json")["rowsByPackage"]
    calibration = run / "calibration"
    stages = []
    largest = max(workers["packages"], key=lambda name: rows[name])
    phases = ("normal",) if normal_only else ("reranker",) if reranker_only else ("normal", "reranker")
    for position, phase in enumerate(phases):
        if phase == "reranker" and reranker_only:
            add_training(run, calibration, stages, training)
        measured = read_json(calibration / f"{phase}.json")
        if measured["jobs"] != jobs:
            raise ValueError(f"Calibration used {measured['jobs']} jobs, not {jobs}; recalibrate")
        startup, per_example = fit(measured["packages"])
        pending = [name for index, name in enumerate(workers["packages"])
                   if not (run / "packages" / f"{index + 1:04d}" / f"{phase}.json").exists()]
        examples = sum(rows[name] for name in pending)
        seconds = makespan([startup + per_example * rows[name] for name in pending], jobs)
        stages.append(dict(stage=f"{phase} benchmarks", seconds=seconds,
                           detail=f"{len(pending)} packages, {examples:,} examples, "
                                  f"{per_example:.2f} s/example + {startup:.0f} s startup per package"))
        if position == 0:
            stages[-1]["largestPackage"] = dict(name=largest, rows=rows[largest],
                                                seconds=startup + per_example * rows[largest])
        if phase == "normal":
            add_training(run, calibration, stages, training)
    remaining = sum(stage["seconds"] for stage in stages)
    total = elapsed + remaining
    result = dict(schema="coo-runtime-estimate-v1", jobs=jobs, elapsedSeconds=elapsed,
                  stages=stages, remainingSeconds=remaining, totalSeconds=total,
                  recommendedSeconds=total * MARGIN)
    write_json(run / "estimate.json", result)
    return result


def add_training(run, calibration, stages, training):
    if training and not (run / "training-ready").exists():
        measured = read_json(calibration / "training.json")
        stages.append(dict(stage="re-ranker training", seconds=measured["trainingSeconds"],
                           detail=measured["trainingDetail"]))
        stages.append(dict(stage="test evaluation", seconds=measured["evaluationSeconds"],
                           detail=measured["evaluationDetail"]))


def report(result):
    lines = ["", "=" * 72,
             f"RUNTIME ESTIMATE for this machine with BENCHMARK_JOBS={result['jobs']}",
             "=" * 72,
             f"  {'already spent (setup + calibration)':<36}{duration(result['elapsedSeconds']):>9}"]
    for stage in result["stages"]:
        lines.append(f"  {stage['stage']:<36}{duration(stage['seconds']):>9}   {stage['detail']}")
        if "largestPackage" in stage:
            big = stage["largestPackage"]
            lines.append(f"  {'':<36}{'':>9}   longest single package: {big['name']} "
                         f"({big['rows']:,} examples) ~{duration(big['seconds'])}")
    lines += ["-" * 72,
              f"  {'PREDICTED TOTAL':<36}{duration(result['totalSeconds']):>9}",
              f"  {'RENT AT LEAST (+25% margin)':<36}{duration(result['recommendedSeconds']):>9}",
              "=" * 72,
              "Assumes time grows linearly with examples per package; aggregation/export"
              " take minutes.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--jobs", type=int, required=True)
    parser.add_argument("--elapsed", type=float, default=0.0, help="Seconds already spent in this invocation")
    parser.add_argument("--normal-only", action="store_true")
    parser.add_argument("--no-training", action="store_true", help="The run skips re-ranker training")
    parser.add_argument("--reranker-only", action="store_true",
                        help="Training, test evaluation and re-ranker benchmarks only")
    args = parser.parse_args()
    try:
        print(report(estimate(args.run.resolve(), args.jobs, args.elapsed, args.normal_only,
                                     not args.no_training, args.reranker_only)), flush=True)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Runtime estimate error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
