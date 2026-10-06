#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/pipeline-common.sh"

# Each training run gets fresh artifacts, but always reuses the experiment split.
export RERANKER_RUN_DIR="$(mktemp -d "$EXPERIMENT_DIR/reranker-run.XXXXXX")"
RERANKER_MODEL_DIR="$RERANKER_RUN_DIR/model"
cp "$BENCHMARK_SPLIT_FILE" "$RERANKER_RUN_DIR/split.json"
if [[ -z "${RERANKER_PYTHON:-}" ]]; then
    python3 -m venv "$EXPERIMENT_DIR/reranker-venv"
    RERANKER_PYTHON="$EXPERIMENT_DIR/reranker-venv/bin/python"
fi
"$RERANKER_PYTHON" -m pip install -r "$REPO_DIR/reranker/requirements.txt"

echo "Exporting training examples from every non-benchmark package"
./pharo --headless Pharo.image eval "
| split directory |
directory := (OSEnvironment current at: 'RERANKER_RUN_DIR') asFileReference.
split := CooBenchmarkSplit readFrom: directory / 'split.json'.
CooBenchRunner exportTrainingForSplit: split to: directory / 'training.jsonl'.
"

echo "Training re-ranker (fixed epochs; no benchmark data)"
TRAINING_SEED="$("$RERANKER_PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["seed"])' "$BENCHMARK_SPLIT_FILE")"
"$RERANKER_PYTHON" "$REPO_DIR/reranker/train.py" \
    "$RERANKER_RUN_DIR/training.jsonl" "$RERANKER_MODEL_DIR" \
    --package-split "$RERANKER_RUN_DIR/split.json" \
    --epochs "$RANKING_EPOCHS" --width 32 --seed "$TRAINING_SEED"

# Test examples are collected only AFTER the model is trained, into a separate file.
echo "Evaluating the saved benchmark packages"
./pharo --headless Pharo.image eval "
| split directory |
directory := (OSEnvironment current at: 'RERANKER_RUN_DIR') asFileReference.
split := CooBenchmarkSplit readFrom: directory / 'split.json'.
CooBenchRunner exportTestForSplit: split to: directory / 'test.jsonl'.
"
"$RERANKER_PYTHON" "$REPO_DIR/reranker/evaluate.py" \
    "$RERANKER_MODEL_DIR" "$RERANKER_RUN_DIR/test.jsonl" > "$RERANKER_RUN_DIR/evaluation.json"

echo "Starting re-ranker"
"$RERANKER_PYTHON" "$REPO_DIR/reranker/serve.py" "$RERANKER_MODEL_DIR" > "$RERANKER_RUN_DIR/reranker.log" 2>&1 &
RERANKER_PID=$!

# Exercise actual ONNX inference and check that the requested model is serving.
if ! "$RERANKER_PYTHON" - "$RERANKER_MODEL_DIR" "$RERANKER_PID" <<'PY'
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

echo "Running re-ranking benchmarks on the saved benchmark packages"
./pharo --headless Pharo.image eval "
| directory split comparison file |
directory := (OSEnvironment current at: 'RERANKER_RUN_DIR') asFileReference.
split := CooBenchmarkSplit readFrom: directory / 'split.json'.
comparison := CooBenchRunner rerankerBenchmarksForSplit: split.
file := CooBenchRunner exportReranker: comparison to: (OSEnvironment current at: 'RESULTS_DIR').
Stdio stdout nextPutAll: file fullName; lf.
"
echo "Re-ranking benchmarks complete: $RESULTS_DIR"
echo "Model, separate training/test corpora, split, evaluation and log: $RERANKER_RUN_DIR"
