#!/usr/bin/env bash
# The neural re-ranker, end to end and on its own:
#   dataset -> saved benchmark packages -> train/validation/test split of the OTHER
#   packages -> training (best epoch by validation MRR) -> test evaluation -> learning
#   figures -> serve the model -> NeuralRank-10/20/30/50 benchmarks on the saved
#   benchmark packages -> publish.
# The benchmark packages are the same saved list pipeline-benchmarks.sh uses, and they
# never enter train, validation or test. No Ollama/LLM is needed here.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ESTIMATE=run
for option in "$@"; do
    case "$option" in
        --estimate-only) ESTIMATE=only ;;
        --skip-estimate) ESTIMATE=skip ;;
        --help|-h)
            echo "Usage: BENCHMARK_JOBS=N $0 [--estimate-only | --skip-estimate]"
            echo "Trains, validates and tests the neural re-ranker on packages outside the saved"
            echo "benchmark list, then benchmarks NeuralRank-10/20/30/50 on the saved benchmark packages."
            echo "--estimate-only: prepare, time training and a few small training packages, print the predicted runtime, then stop."
            echo "--skip-estimate: start without the runtime estimate (no calibration)."
            echo "BENCHMARK_PACKAGES_FILE=path: benchmark exactly these packages (one name per line, a JSON array, or a saved selection)."
            echo "RANKING_EPOCHS (10), RANKING_SPLIT_RATIOS (80,10,10), TRAINING_PACKAGE_COUNT (all): training settings."
            exit 0 ;;
        *) echo "Unknown option: $option (use --help)" >&2; exit 2 ;;
    esac
done
export RANKING_SPLIT_RATIOS="${RANKING_SPLIT_RATIOS:-80,10,10}"
export RANKING_EPOCHS="${RANKING_EPOCHS:-10}"
# Optional: train on only this many (seeded random) training packages; empty = all.
export TRAINING_PACKAGE_COUNT="${TRAINING_PACKAGE_COUNT:-}"
if [[ -n "$TRAINING_PACKAGE_COUNT" && ! "$TRAINING_PACKAGE_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "TRAINING_PACKAGE_COUNT must be a positive integer (or unset to train on all training packages)." >&2
    exit 1
fi
if [[ ! "$RANKING_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
    echo "RANKING_EPOCHS must be a positive integer." >&2
    exit 1
fi
RUN_PREFIX=reranker-run
source "$SCRIPT_DIR/scripts/pipeline-common.sh"

prepare_run_directory --ratios "$RANKING_SPLIT_RATIOS" \
    ${TRAINING_PACKAGE_COUNT:+--training-packages "$TRAINING_PACKAGE_COUNT"}
MODEL_DIR="$RUN_DIR/model"
if [[ -f "$RUN_DIR/training-config.json" ]]; then
    if [[ "$("$PIPELINE_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["epochs"])' "$RUN_DIR/training-config.json")" != "$RANKING_EPOCHS" ]]; then
        echo "Resume training epochs differ; restore RANKING_EPOCHS or start a new run." >&2
        exit 1
    fi
else
    printf '{"epochs": %s, "width": 32, "selection": "validation-MRR"}\n' "$RANKING_EPOCHS" > "$RUN_DIR/training-config.json"
fi
if [[ ! -f "$RUN_DIR/corpus.json" ]]; then
    echo "Preparing train/validation/test files (benchmark packages excluded from ALL three)"
    "$PIPELINE_PYTHON" "$SCRIPT_DIR/scripts/ranking_corpus.py" partition \
        "$RANKING_CORPUS_DIR" --repository "$REPO_DIR" \
        --split "$RUN_DIR/split.json" --output "$RUN_DIR"
fi

if [[ -z "${RERANKER_PYTHON:-}" ]]; then
    python3 -m venv "$EXPERIMENT_DIR/reranker-venv"
    PIPELINE_PYTHON="$EXPERIMENT_DIR/reranker-venv/bin/python"
fi
"$PIPELINE_PYTHON" -m pip install -r "$REPO_DIR/reranker/requirements.txt" \
    -r "$SCRIPT_DIR/scripts/reranker-requirements.txt"

"$PIPELINE_PYTHON" "$SCRIPT_DIR/scripts/package_benchmarks.py" prepare \
    --run "$RUN_DIR" --image "$EXPERIMENT_DIR/image"

RERANKER_PID=""
# Serve a model on port 8765 and exercise actual ONNX inference before use.
start_reranker() {
    "$PIPELINE_PYTHON" "$REPO_DIR/reranker/serve.py" "$1" > "$2" 2>&1 &
    RERANKER_PID=$!
    CLEANUP_PIDS+=("$RERANKER_PID")
    if ! "$PIPELINE_PYTHON" - "$1" "$RERANKER_PID" <<'PY'
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
        cat "$2" >&2
        exit 1
    fi
    if ! kill -0 "$RERANKER_PID" 2>/dev/null; then
        cat "$2" >&2
        exit 1
    fi
}

stop_reranker() {
    kill "$RERANKER_PID" 2>/dev/null || true
    wait "$RERANKER_PID" 2>/dev/null || true
    RERANKER_PID=""
}

# Predict the runtime before the long phases: sampled training steps, test evaluation,
# and real workers on a few small TRAINING packages (never benchmark packages) served
# by an untrained model at BENCHMARK_JOBS concurrency.
if [[ "$ESTIMATE" != skip ]]; then
    CALIBRATION_DIR="$RUN_DIR/calibration"
    echo "Estimating runtime: timing training steps and small training packages at $BENCHMARK_JOBS jobs (takes minutes)"
    "$PIPELINE_PYTHON" "$SCRIPT_DIR/scripts/package_benchmarks.py" prepare-calibration \
        --run "$RUN_DIR" --image "$EXPERIMENT_DIR/image" --jobs "$BENCHMARK_JOBS"
    if [[ ! -f "$CALIBRATION_DIR/training.json" ]]; then
        "$PIPELINE_PYTHON" "$SCRIPT_DIR/scripts/reranker_workflow.py" calibrate \
            --repository "$REPO_DIR" --split "$RUN_DIR/split.json" \
            --calibration-split "$CALIBRATION_DIR/split.json" \
            --training "$RUN_DIR/training.jsonl" --validation "$RUN_DIR/validation.jsonl" \
            --test "$RUN_DIR/test.jsonl" --epochs "$RANKING_EPOCHS" --width 32 \
            --output "$CALIBRATION_DIR/model" --report "$CALIBRATION_DIR/training.json"
    fi
    if [[ ! -f "$CALIBRATION_DIR/reranker.json" ]]; then
        start_reranker "$CALIBRATION_DIR/model" "$CALIBRATION_DIR/reranker-server.log"
        "$PIPELINE_PYTHON" "$SCRIPT_DIR/scripts/package_benchmarks.py" calibrate \
            --run "$RUN_DIR" --image "$EXPERIMENT_DIR/image" --phase reranker --jobs "$BENCHMARK_JOBS"
        stop_reranker
    fi
    "$PIPELINE_PYTHON" "$SCRIPT_DIR/scripts/runtime_estimate.py" \
        --run "$RUN_DIR" --jobs "$BENCHMARK_JOBS" --reranker-only --elapsed "$SECONDS"
    if [[ "$ESTIMATE" == only ]]; then
        echo "Estimate only: nothing was trained or benchmarked. To run this exact prepared experiment:"
        echo "  BENCHMARK_RESUME_DIR=\"$RUN_DIR\" BENCHMARK_JOBS=$BENCHMARK_JOBS $0"
        exit 0
    fi
fi

# Only train/validation data enter model selection; test is evaluated once afterwards.
if [[ ! -f "$RUN_DIR/training-ready" ]]; then
    echo "Training re-ranker; selecting the best epoch using validation MRR"
    if ! "$PIPELINE_PYTHON" "$SCRIPT_DIR/scripts/reranker_workflow.py" train \
        --repository "$REPO_DIR" --split "$RUN_DIR/split.json" \
        --training "$RUN_DIR/training.jsonl" --validation "$RUN_DIR/validation.jsonl" \
        --output "$MODEL_DIR" --epochs "$RANKING_EPOCHS" --width 32 \
        > "$RUN_DIR/training.log" 2>&1; then
        echo "Training failed; see $RUN_DIR/training.log" >&2
        exit 1
    fi
    echo "Evaluating the selected checkpoint on the separate test packages"
    "$PIPELINE_PYTHON" "$SCRIPT_DIR/scripts/reranker_workflow.py" evaluate \
        --repository "$REPO_DIR" --split "$RUN_DIR/split.json" \
        --model "$MODEL_DIR" --data "$RUN_DIR/test.jsonl" \
        --output "$RUN_DIR/evaluation.json"
    printf '%s\n' "$RANKING_EPOCHS" > "$RUN_DIR/training-ready"
else
    echo "Reusing validation-selected model and independent test evaluation"
fi
"$PIPELINE_PYTHON" "$SCRIPT_DIR/scripts/package_benchmarks.py" pin-model \
    --run "$RUN_DIR" --image "$EXPERIMENT_DIR/image"

# Rebuild paper figures from retained metrics; this never retrains or re-evaluates.
"$PIPELINE_PYTHON" "$SCRIPT_DIR/scripts/reranker_workflow.py" report \
    --model "$MODEL_DIR" --evaluation "$RUN_DIR/evaluation.json" \
    --output "$RUN_DIR/learning"
for figure in learning-curves test-performance test-rank-transitions; do
    for extension in png pdf; do
        if [[ ! -s "$RUN_DIR/learning/$figure.$extension" ]]; then
            echo "Missing learning figure: $figure.$extension" >&2
            exit 1
        fi
    done
done

echo "Starting re-ranker"
start_reranker "$MODEL_DIR" "$RUN_DIR/reranker.log"
echo "Running re-ranking benchmarks on the saved benchmark packages"
run_package_phase reranker
stop_reranker

echo "Aggregating all packages and exporting the re-ranker files"
publish_phase reranker results-table-re-ranker.tex performance-reranker.png recall-top-re-ranker.tex
# Retire the former filename only after the complete new set was published.
rm -f -- "$RESULTS_DIR/performance-re-ranker.png"
echo "Model, train/validation/test corpora, split, learning figures, evaluation and logs: $RUN_DIR"
print_runtime_summary
