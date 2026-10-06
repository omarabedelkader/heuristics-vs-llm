#!/usr/bin/env bash
set -euo pipefail

# Adjust these defaults here or through environment variables.
export TRAIN_PACKAGE_COUNT="${TRAIN_PACKAGE_COUNT:-16}"
export VALIDATION_PACKAGE_COUNT="${VALIDATION_PACKAGE_COUNT:-4}"
export BENCHMARK_PACKAGE_COUNT="${BENCHMARK_PACKAGE_COUNT:-40}"
export RANKING_SEED="${RANKING_SEED:-42}"
export RANKING_EPOCHS="${RANKING_EPOCHS:-10}"
export RESULTS_DIR="${RESULTS_DIR:-/Users/omar/Desktop/HeuristicCompletion-Benchmarks-Multiples/benchmark-results}"
case "$RESULTS_DIR" in
    /*) ;;
    *) RESULTS_DIR="$PWD/$RESULTS_DIR" ;;
esac

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "Creating Image"
# 1. Download Pharo + VM
mkdir -p image
cd image
curl -fsSL https://get.pharo.org/140+vm | bash

echo "1-DOWNLOAD IMAGE OK"

echo "Installation Repository"
# 2. Install the requested GitHub branch into the headless image.
./pharo --headless Pharo.image eval --save "
Metacello new
  githubUser: 'omarabedelkader' project: 'HeuristicCompletion-Benchmarks-Multiples' commitish: 'neural-ranking' path: 'src';
  baseline: 'ExtendedHeuristicCompletionBenchmarks';
  load
"

echo "2-INSTALL REPO OK"

# 3. Fetch the Python trainer/server from the same branch and install dependencies.
# A fresh run directory prevents stale exports or an old model entering this run.
RERANKER_RUN_DIR="$(mktemp -d "$PWD/reranker-run.XXXXXX")"
export RERANKER_RUN_DIR
RERANKER_MODEL_DIR="$RERANKER_RUN_DIR/ranking-models/rank32"
echo "Preparing Re-ranker: $RERANKER_RUN_DIR"
curl -fsSL https://codeload.github.com/omarabedelkader/HeuristicCompletion-Benchmarks-Multiples/tar.gz/refs/heads/neural-ranking \
    -o "$RERANKER_RUN_DIR/repository.tar.gz"
tar -xzf "$RERANKER_RUN_DIR/repository.tar.gz" -C "$RERANKER_RUN_DIR"
REPO_DIR="$RERANKER_RUN_DIR/HeuristicCompletion-Benchmarks-Multiples-neural-ranking"
python3 -m venv reranker-venv
reranker-venv/bin/python -m pip install -r "$REPO_DIR/reranker/requirements.txt"

# 4. Select train/validation/test packages before collecting any examples.
./pharo --headless Pharo.image eval "
| directory |
directory := (OSEnvironment current at: 'RERANKER_RUN_DIR') asFileReference.
(directory / 'eligible-packages.json') writeStreamDo: [ :out |
    out truncate; nextPutAll: (STONJSON toString: CooBenchRunner eligiblePackageNames) ].
"

reranker-venv/bin/python - "$RERANKER_RUN_DIR" <<'PY'
import json
import os
from pathlib import Path
import random
import sys

folder = Path(sys.argv[1])
counts = {key: int(os.environ[env]) for key, env in (
    ("train", "TRAIN_PACKAGE_COUNT"), ("validation", "VALIDATION_PACKAGE_COUNT"),
    ("test", "BENCHMARK_PACKAGE_COUNT"))}
if any(n < 1 for n in counts.values()):
    raise SystemExit("All package counts must be positive")
packages = sorted(set(json.loads((folder / "eligible-packages.json").read_text())))
# Keep naming families (e.g. Collections-* and Collections-Tests-*) together.
# This is an automatic naming rule, not a claim of project-level independence.
families = {}
for package in packages:
    families.setdefault(package.split("-", 1)[0], []).append(package)
rng = random.Random(int(os.environ["RANKING_SEED"]))
remaining = sorted(families)
rng.shuffle(remaining)
split = {"seed": int(os.environ["RANKING_SEED"]), "grouping": "package name before first hyphen"}
for partition in ("test", "validation", "train"):
    pool = []
    while len(pool) < counts[partition] and remaining:
        pool.extend(families[remaining.pop()])
    if len(pool) < counts[partition]:
        raise SystemExit(f"Not enough separate package families for {counts}; lower the package counts")
    split[partition] = sorted(rng.sample(pool, counts[partition]))
split["groups"] = {p: p.split("-", 1)[0]
                   for part in ("train", "validation", "test") for p in split[part]}
(folder / "split.json").write_text(json.dumps(split, indent=2) + "\n")
for part in ("train", "validation", "test"):
    print(f"{part}: {', '.join(split[part])}", flush=True)
PY

# 5. Collect real candidateRecall examples for both completion kinds. No model needed.
echo "Exporting Training Corpus"
./pharo --headless Pharo.image eval "
| directory split corpus |
directory := (OSEnvironment current at: 'RERANKER_RUN_DIR') asFileReference.
split := STONJSON fromString: (directory / 'split.json') contents.
corpus := directory / 'corpus'.
corpus ensureCreateDirectory.
#('train' 'validation' 'test') do: [ :partition |
    (split at: partition) do: [ :packageName |
        { CooBenchRunnerMessage. CooBenchRunnerVariables } do: [ :runnerClass |
            | runner recall kind |
            kind := runnerClass = CooBenchRunnerMessage
                ifTrue: [ 'messages' ] ifFalse: [ 'variables' ].
            Stdio stdout nextPutAll: partition, ': ', packageName, ' / ', kind; lf; flush.
            runner := runnerClass new
                package: (PackageOrganizer default packageNamed: packageName);
                baseline: #candidateRecall; others: #(); run; yourself.
            recall := runner results at: #candidateRecall.
            recall exportTrainingTo: corpus / (packageName, '-', kind, '.jsonl') ] ] ].
"

# 6. Concatenate the exports, preserve family holdouts, and train the ONNX model.
echo "Training Re-ranker"
reranker-venv/bin/python - "$RERANKER_RUN_DIR" "$REPO_DIR" "$RERANKER_MODEL_DIR" <<'PY'
import json
import os
from pathlib import Path
import subprocess
import sys

folder, repo, model = map(Path, sys.argv[1:])
split = json.loads((folder / "split.json").read_text())
groups = {part: set() for part in ("train", "validation", "test")}
counts = {part: 0 for part in groups}
positives = 0
with (folder / "corpus.jsonl").open("w") as out:
    for part in groups:
        for package in split[part]:
            for kind in ("messages", "variables"):
                with (folder / "corpus" / f"{package}-{kind}.jsonl").open() as source:
                    for line in source:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if row["group"] != package or row["kind"] != kind:
                            raise SystemExit(f"Unexpected package/kind in {package}-{kind}.jsonl")
                        row["package"] = package
                        row["group"] = split["groups"][package]
                        groups[part].add(row["group"])
                        counts[part] += 1
                        if part == "train" and row["target"] in [c["name"] for c in row["candidates"]]:
                            positives += 1
                        out.write(json.dumps(row) + "\n")
if any(n == 0 for n in counts.values()) or not positives:
    raise SystemExit(f"Corpus needs nonempty partitions and training hits: {counts}, hits={positives}")
if any(groups[a] & groups[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
    raise SystemExit("Training, validation and test groups overlap")
print(f"Corpus rows: {counts}; training hits: {positives}", flush=True)
subprocess.run([
    sys.executable, str(repo / "reranker/train.py"), str(folder / "corpus.jsonl"), str(model),
    "--validation-groups", *sorted(groups["validation"]),
    "--test-groups", *sorted(groups["test"]),
    "--epochs", os.environ["RANKING_EPOCHS"], "--width", "32", "--seed", os.environ["RANKING_SEED"],
], check=True)
PY

echo "Evaluating Held-out Corpus"
reranker-venv/bin/python "$REPO_DIR/reranker/evaluate.py" \
    "$RERANKER_MODEL_DIR" "$RERANKER_MODEL_DIR/test.jsonl" > "$RERANKER_RUN_DIR/evaluation.json"

echo "Starting Re-ranker"
reranker-venv/bin/python "$REPO_DIR/reranker/serve.py" "$RERANKER_MODEL_DIR" > "$RERANKER_RUN_DIR/reranker.log" 2>&1 &
RERANKER_PID=$!
cleanup() {
    kill "$RERANKER_PID" 2>/dev/null || true
    wait "$RERANKER_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Exercise actual ONNX inference and check that the requested model is serving.
if ! reranker-venv/bin/python - "$RERANKER_MODEL_DIR" "$RERANKER_PID" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

model_id = hashlib.sha256((Path(sys.argv[1]) / "ranker.onnx").read_bytes()).hexdigest()
payload = json.dumps({
    "schema": "coo-ranking-v1", "kind": "messages", "prefix": "si",
    "sourcePrefix": "example ^ self si", "receiverKind": "self",
    "candidates": [{"name": "size", "rank": 1}],
}).encode()
deadline = time.monotonic() + 60
while time.monotonic() < deadline:
    os.kill(int(sys.argv[2]), 0)
    try:
        request = Request("http://127.0.0.1:8765/rank", data=payload,
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=2) as response:
            result = json.load(response)
        if (result.get("schema") != "coo-ranking-v1"
                or result.get("modelId") != model_id
                or result.get("scores") != [["size", 1.0]]):
            raise SystemExit("Unexpected reranker response or another model is using port 8765")
        print("RE-RANKER READY")
        break
    except (URLError, TimeoutError):
        time.sleep(1)
else:
    raise SystemExit("Re-ranker did not become ready within 60 seconds")
PY
then
    cat "$RERANKER_RUN_DIR/reranker.log" >&2
    exit 1
fi

if ! kill -0 "$RERANKER_PID" 2>/dev/null; then
    cat "$RERANKER_RUN_DIR/reranker.log" >&2
    exit 1
fi

echo "Run Benchmark"
# 7. Benchmark exactly the saved test packages; never redraw from the training pool.
./pharo --headless Pharo.image eval --save "
| directory split comparison file |
directory := (OSEnvironment current at: 'RERANKER_RUN_DIR') asFileReference.
split := STONJSON fromString: (directory / 'split.json') contents.
comparison := CooBenchRunner
    compareMessagesAndVariablesFor: (split at: 'test')
    strategies: CooBenchmarkChart rerankerStrategies.
file := CooBenchRunner
    exportReranker: comparison
    to: (OSEnvironment current at: 'RESULTS_DIR').
Stdio stdout nextPutAll: file fullName; lf.
"

echo "7-RUN BENCHMARKS OK"
echo "Model, corpus, split, evaluation and service log: $RERANKER_RUN_DIR"
