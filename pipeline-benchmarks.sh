#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_RERANKER_BENCHMARKS=0
ESTIMATE=run
for option in "$@"; do
    case "$option" in
        --skip-reranker-benchmarks) SKIP_RERANKER_BENCHMARKS=1 ;;
        --estimate-only) ESTIMATE=only ;;
        --skip-estimate) ESTIMATE=skip ;;
        --help|-h)
            echo "Usage: BENCHMARK_JOBS=N $0 [--skip-reranker-benchmarks] [--estimate-only | --skip-estimate]"
            echo "--skip-reranker-benchmarks: skip only live neural re-ranker benchmarks; training, validation, test and learning figures still run."
            echo "--estimate-only: prepare, time a few training packages on this machine, print the predicted runtime, then stop."
            echo "--skip-estimate: start benchmarks without the runtime estimate (no calibration)."
            exit 0 ;;
        *) echo "Unknown option: $option (use --help)" >&2; exit 2 ;;
    esac
done
if [[ ! "${BENCHMARK_JOBS:-}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Set BENCHMARK_JOBS to the number of simultaneous package jobs you want (positive integer)." >&2
    exit 1
fi
export BENCHMARK_JOBS
if [[ -n "${BENCHMARK_RESUME_DIR:-}" ]]; then
    BENCHMARK_RESUME_DIR="$(cd "$BENCHMARK_RESUME_DIR" && pwd -P)"
fi
export MINING_DIR="${MINING_DIR:-$SCRIPT_DIR/mining}"
mkdir -p "$MINING_DIR"
MINING_DIR="$(cd "$MINING_DIR" && pwd -P)"
export RANKING_CORPUS_DIR="$MINING_DIR/corpus"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$SCRIPT_DIR/experiment}"
mkdir -p "$EXPERIMENT_DIR"
EXPERIMENT_DIR="$(cd "$EXPERIMENT_DIR" && pwd -P)"
if [[ "$EXPERIMENT_DIR" == "$MINING_DIR" ]]; then
    echo "EXPERIMENT_DIR must differ from MINING_DIR: mining and benchmarks need separate images." >&2
    exit 1
fi
export RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/resutls}"
mkdir -p "$RESULTS_DIR"
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"
export REPO_DIR="$EXPERIMENT_DIR/repository"
export BENCHMARK_SPLIT_FILE="$EXPERIMENT_DIR/split.json"
export BENCHMARK_SELECTION_FILE="$EXPERIMENT_DIR/benchmark-selection.json"
export RANKING_SPLIT_RATIOS="${RANKING_SPLIT_RATIOS:-80,10,10}"
export BENCHMARK_PACKAGE_COUNT="${BENCHMARK_PACKAGE_COUNT:-}"
export RANKING_SEED="${RANKING_SEED:-}"
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
RERANKER_PID=""
OLLAMA_PID=""
RESULTS_LOCKED=""
WORKER_PID=""

if ! mkdir "$EXPERIMENT_DIR/.running" 2>/dev/null; then
    echo "Experiment is already running: $EXPERIMENT_DIR (.running lock)" >&2
    exit 1
fi
pipeline_cleanup() {
    if [[ -n "$WORKER_PID" ]]; then
        kill "$WORKER_PID" 2>/dev/null || true
        wait "$WORKER_PID" 2>/dev/null || true
    fi
    if [[ -n "$RERANKER_PID" ]]; then
        kill "$RERANKER_PID" 2>/dev/null || true
        wait "$RERANKER_PID" 2>/dev/null || true
    fi
    if [[ -n "$OLLAMA_PID" ]]; then
        kill "$OLLAMA_PID" 2>/dev/null || true
        wait "$OLLAMA_PID" 2>/dev/null || true
    fi
    if [[ -n "$RESULTS_LOCKED" ]]; then
        rmdir "$RESULTS_DIR/.running"
    fi
    rmdir "$EXPERIMENT_DIR/.running"
}
trap pipeline_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if ! mkdir "$RESULTS_DIR/.running" 2>/dev/null; then
    echo "Results directory is already in use: $RESULTS_DIR" >&2
    exit 1
fi
RESULTS_LOCKED=1

# Dataset source, in order: local corpus, then the pinned Hugging Face snapshot,
# then mining. Mining runs only when the snapshot is absent (exit 3) or unusable
# (exit 4); local problems that mining cannot fix (exit 1) stop the pipeline.
if [[ ! -f "$RANKING_CORPUS_DIR/manifest.json" ]]; then
    echo "No completed local dataset at $RANKING_CORPUS_DIR. Checking the Hugging Face snapshot"
    snapshot_status=0
    "${RERANKER_PYTHON:-python3}" "$SCRIPT_DIR/scripts/download_snapshot.py" "$MINING_DIR" || snapshot_status=$?
    case "$snapshot_status" in
        0) echo "Using downloaded dataset: $RANKING_CORPUS_DIR" ;;
        3|4)
            if [[ "$snapshot_status" == 3 ]]; then
                echo "WARNING: dataset not found locally and no snapshot is published remotely." >&2
            else
                echo "WARNING: dataset not found locally and the remote snapshot is NOT usable (reason above)." >&2
            fi
            echo "WARNING: mining a new dataset with pipeline-mine-training-data.sh (this takes a long time)." >&2
            bash "$SCRIPT_DIR/pipeline-mine-training-data.sh"
            echo "Using mined dataset: $RANKING_CORPUS_DIR" ;;
        *) exit "$snapshot_status" ;;
    esac
else
    echo "Using saved local dataset: $RANKING_CORPUS_DIR"
fi
# Verify before copying the benchmark image. Invalid saved data is never silently replaced.
"${RERANKER_PYTHON:-python3}" "$SCRIPT_DIR/scripts/ranking_corpus.py" verify \
    "$RANKING_CORPUS_DIR" --repository "$MINING_DIR/repository"
if [[ ! -d "$REPO_DIR" ]]; then
    if [[ -f "$EXPERIMENT_DIR/image-ready" || -f "$BENCHMARK_SPLIT_FILE" ]]; then
        echo "Frozen benchmark source is missing; restore it or use a new EXPERIMENT_DIR." >&2
        exit 1
    fi
    source_stage="$(mktemp -d "$EXPERIMENT_DIR/source.XXXXXX")"
    cp -R "$MINING_DIR/repository/." "$source_stage/"
    mv "$source_stage" "$REPO_DIR"
fi
# This small source check also rejects an incompatible repository on reruns.
"${RERANKER_PYTHON:-python3}" - "$SCRIPT_DIR/scripts" "$RANKING_CORPUS_DIR" "$REPO_DIR" <<'PYCODE'
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from ranking_corpus import load_manifest
load_manifest(Path(sys.argv[2]), repository=Path(sys.argv[3]))
PYCODE

if [[ ! -e "$EXPERIMENT_DIR/image" ]]; then
    if [[ -f "$BENCHMARK_SPLIT_FILE" || -f "$EXPERIMENT_DIR/image-ready" ]]; then
        echo "The experiment image is missing; restore it or use a new EXPERIMENT_DIR." >&2
        exit 1
    fi
    if [[ ! -f "$MINING_DIR/image/Pharo.image" || ! -x "$MINING_DIR/image/pharo" ]]; then
        echo "The original mining image is required; restore MINING_DIR/image from the dataset's snapshot." >&2
        exit 1
    fi
    echo "Copying the original mining image for benchmarks"
    image_stage="$(mktemp -d "$EXPERIMENT_DIR/image.XXXXXX")"
    cp -R "$MINING_DIR/image/." "$image_stage/"
    # Verify the copied bytes before publishing the template; never reinstall code.
    "${RERANKER_PYTHON:-python3}" - "$SCRIPT_DIR/scripts" "$RANKING_CORPUS_DIR" "$image_stage/Pharo.image" <<'PYCODE'
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from ranking_corpus import load_manifest
load_manifest(Path(sys.argv[2]), image=Path(sys.argv[3]))
PYCODE
    mv "$image_stage" "$EXPERIMENT_DIR/image"
fi
if [[ ! -f "$EXPERIMENT_DIR/image/Pharo.image" || ! -x "$EXPERIMENT_DIR/image/pharo" ]]; then
    echo "The experiment image is incomplete; restore it or use a new EXPERIMENT_DIR." >&2
    exit 1
fi
# Check on reruns too, rejecting older experiments that downloaded a different image.
"${RERANKER_PYTHON:-python3}" - "$SCRIPT_DIR/scripts" "$RANKING_CORPUS_DIR" "$EXPERIMENT_DIR/image/Pharo.image" <<'PYCODE'
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from ranking_corpus import load_manifest
load_manifest(Path(sys.argv[2]), image=Path(sys.argv[3]))
PYCODE
"${RERANKER_PYTHON:-python3}" "$SCRIPT_DIR/scripts/pharo_runtime.py" "$EXPERIMENT_DIR/image"
touch "$EXPERIMENT_DIR/image-ready"
cd "$EXPERIMENT_DIR/image"

# A fresh default experiment reuses the published selection. Explicit package/seed
# settings request a new selection through the existing Pharo selection workflow.
if [[ ! -e "$BENCHMARK_SELECTION_FILE" && ! -e "$BENCHMARK_SPLIT_FILE" &&
      -f "$MINING_DIR/snapshot-selection/split.json" &&
      -z "$BENCHMARK_PACKAGE_COUNT" && -z "$RANKING_SEED" && -z "$TRAINING_PACKAGE_COUNT" &&
      "$RANKING_SPLIT_RATIOS" == "80,10,10" ]]; then
    cp "$MINING_DIR/snapshot-selection/benchmark-selection.json" "$BENCHMARK_SELECTION_FILE"
    cp "$MINING_DIR/snapshot-selection/split.json" "$BENCHMARK_SPLIT_FILE"
    echo "Using the snapshot's saved benchmark selection and train/validation/test split"
fi

# Select and save benchmark names in the independent benchmark image.
./pharo --headless Pharo.image eval "
| file split countText seedText |
file := (OSEnvironment current at: 'BENCHMARK_SELECTION_FILE') asFileReference.
countText := OSEnvironment current at: 'BENCHMARK_PACKAGE_COUNT'.
seedText := OSEnvironment current at: 'RANKING_SEED'.
file exists
    ifTrue: [
        split := CooBenchmarkSplit readFrom: file.
        countText ifNotEmpty: [
            countText asNumber = split benchmarkPackageNames size ifFalse: [
                Error signal: 'Package count differs from saved split; use a new EXPERIMENT_DIR' ] ].
        seedText ifNotEmpty: [
            seedText asNumber = split seed ifFalse: [
                Error signal: 'Seed differs from saved split; use a new EXPERIMENT_DIR' ] ] ]
    ifFalse: [
        split := CooBenchmarkSplit
            benchmarkPackages: (countText ifEmpty: [ '50' ]) asNumber
            seed: (seedText ifEmpty: [ '42' ]) asNumber.
        split writeTo: file ].
split validateAgainst: CooBenchRunner eligiblePackageNames.
((OSEnvironment current at: 'EXPERIMENT_DIR') asFileReference / 'llm-models.json')
    writeStreamDo: [ :out |
        out truncate; nextPutAll: (STONJSON toString: CooBenchRunner llmModels values asArray); lf ].
Stdio stdout
    nextPutAll: 'Saved split: ', split benchmarkPackageNames size asString, ' benchmark packages; ',
        split trainingPackageNames size asString, ' remaining packages before train/validation/test partitioning'; lf;
    nextPutAll: file fullName; lf.
"

# The selection file adapts the existing Pharo API; split.json is the authoritative four-way split.
"${RERANKER_PYTHON:-python3}" "$SCRIPT_DIR/scripts/ranking_corpus.py" split \
    "$RANKING_CORPUS_DIR" --selection "$BENCHMARK_SELECTION_FILE" \
    --output "$BENCHMARK_SPLIT_FILE" --ratios "$RANKING_SPLIT_RATIOS" \
    ${TRAINING_PACKAGE_COUNT:+--training-packages "$TRAINING_PACKAGE_COUNT"}

# New invocations train once. Explicit resumes retain the same model and results.
if [[ -n "${BENCHMARK_RESUME_DIR:-}" ]]; then
    RERANKER_RUN_DIR="$(cd "$BENCHMARK_RESUME_DIR" && pwd -P)"
    if [[ "$(dirname "$RERANKER_RUN_DIR")" != "$EXPERIMENT_DIR" ||
          ! -f "$RERANKER_RUN_DIR/training-config.json" || ! -f "$RERANKER_RUN_DIR/workers.json" ]]; then
        echo "BENCHMARK_RESUME_DIR must be a prepared package run inside EXPERIMENT_DIR." >&2
        exit 1
    fi
    if ! cmp -s "$BENCHMARK_SPLIT_FILE" "$RERANKER_RUN_DIR/split.json" ||
            [[ "$("${RERANKER_PYTHON:-python3}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["epochs"])' "$RERANKER_RUN_DIR/training-config.json")" != "$RANKING_EPOCHS" ]]; then
        echo "Resume split or training epochs differ; restore the original settings or start a new run." >&2
        exit 1
    fi
else
    RERANKER_RUN_DIR="$(mktemp -d "$EXPERIMENT_DIR/reranker-run.XXXXXX")"
    cp "$BENCHMARK_SPLIT_FILE" "$RERANKER_RUN_DIR/split.json"
    printf '{"epochs": %s, "width": 32, "selection": "validation-MRR"}\n' "$RANKING_EPOCHS" > "$RERANKER_RUN_DIR/training-config.json"
fi
export RERANKER_RUN_DIR
RERANKER_MODEL_DIR="$RERANKER_RUN_DIR/model"
export PUBLICATION_DIR="$(mktemp -d "$RERANKER_RUN_DIR/publication.XXXXXX")"
echo "Run directory: $RERANKER_RUN_DIR"
echo "Package concurrency selected by you: $BENCHMARK_JOBS"

if [[ -z "${BENCHMARK_RESUME_DIR:-}" ]]; then
    echo "Preparing train/validation/test files (benchmark packages excluded from ALL three)"
    "${RERANKER_PYTHON:-python3}" "$SCRIPT_DIR/scripts/ranking_corpus.py" partition \
        "$RANKING_CORPUS_DIR" --repository "$REPO_DIR" \
        --split "$RERANKER_RUN_DIR/split.json" --output "$RERANKER_RUN_DIR"
fi

if [[ -z "${RERANKER_PYTHON:-}" ]]; then
    python3 -m venv "$EXPERIMENT_DIR/reranker-venv"
    RERANKER_PYTHON="$EXPERIMENT_DIR/reranker-venv/bin/python"
fi
"$RERANKER_PYTHON" -m pip install -r "$REPO_DIR/reranker/requirements.txt" \
    -r "$SCRIPT_DIR/scripts/reranker-requirements.txt"

"$RERANKER_PYTHON" "$SCRIPT_DIR/scripts/package_benchmarks.py" prepare \
    --run "$RERANKER_RUN_DIR" --image "$EXPERIMENT_DIR/image"

# Serve a model on port 8765 and exercise actual ONNX inference before use.
start_reranker() {
    "$RERANKER_PYTHON" "$REPO_DIR/reranker/serve.py" "$1" > "$2" 2>&1 &
    RERANKER_PID=$!
    if ! "$RERANKER_PYTHON" - "$1" "$RERANKER_PID" <<'PY'
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

run_package_phase() {
    "$RERANKER_PYTHON" "$SCRIPT_DIR/scripts/package_benchmarks.py" run \
        --run "$RERANKER_RUN_DIR" --image "$EXPERIMENT_DIR/image" \
        --phase "$1" --jobs "$BENCHMARK_JOBS" &
    WORKER_PID=$!
    local status=0
    wait "$WORKER_PID" || status=$?
    WORKER_PID=""
    if [[ "$status" != 0 ]]; then
        echo "Package phase failed. Resume with BENCHMARK_RESUME_DIR=\"$RERANKER_RUN_DIR\" and your BENCHMARK_JOBS setting." >&2
        return "$status"
    fi
}

# The Pharo client uses localhost:11434. Reuse an existing healthy server, or
# start a local one that this pipeline owns and cleans up on success or failure.
export OLLAMA_HOST="127.0.0.1:11434"
OLLAMA_LOG="$RERANKER_RUN_DIR/ollama.log"
if ! curl -fsS --connect-timeout 2 --max-time 5 \
        http://127.0.0.1:11434/api/tags > "$RERANKER_RUN_DIR/ollama-tags.json" 2>/dev/null; then
    if [[ -z "${OLLAMA_BIN:-}" ]]; then
        OLLAMA_BIN="$(command -v ollama || true)"
    fi
    if [[ -z "$OLLAMA_BIN" ]]; then
        # Local installation, without sudo or system services (also works on OAR).
        if [[ "$(uname -s)" != Linux ]]; then
            echo "Ollama is not installed. Install it or set OLLAMA_BIN to its executable." >&2
            exit 1
        fi
        case "$(uname -m)" in
            x86_64|amd64) ollama_arch=amd64 ;;
            aarch64|arm64) ollama_arch=arm64 ;;
            *) echo "Unsupported Ollama architecture: $(uname -m)" >&2; exit 1 ;;
        esac
        ollama_install_dir="$EXPERIMENT_DIR/ollama-bin"
        OLLAMA_BIN="$ollama_install_dir/bin/ollama"
        if [[ ! -x "$OLLAMA_BIN" ]]; then
            mkdir -p "$ollama_install_dir"
            curl -fsSL "https://ollama.com/download/ollama-linux-$ollama_arch.tar.zst" \
                -o "$ollama_install_dir/ollama.tar.zst"
            if tar --help 2>/dev/null | grep -q zstd; then
                tar --zstd -xf "$ollama_install_dir/ollama.tar.zst" -C "$ollama_install_dir"
            elif command -v zstd >/dev/null 2>&1; then
                zstd -dc "$ollama_install_dir/ollama.tar.zst" | tar -xf - -C "$ollama_install_dir"
            else
                echo "Extracting Ollama requires tar with zstd support or the zstd command." >&2
                exit 1
            fi
        fi
    fi
    export OLLAMA_MODELS="${OLLAMA_MODELS_DIR:-$SCRIPT_DIR/.ollama-models}"
    mkdir -p "$OLLAMA_MODELS"
    OLLAMA_MODELS="$(cd "$OLLAMA_MODELS" && pwd)"
    "$OLLAMA_BIN" serve > "$OLLAMA_LOG" 2>&1 &
    OLLAMA_PID=$!
    ollama_ready=""
    for ((attempt=0; attempt<120; attempt++)); do
        if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
            cat "$OLLAMA_LOG" >&2
            exit 1
        fi
        if curl -fsS --connect-timeout 2 --max-time 5 \
                http://127.0.0.1:11434/api/tags > "$RERANKER_RUN_DIR/ollama-tags.json" 2>/dev/null; then
            ollama_ready=1
            break
        fi
        sleep 1
    done
    if [[ -z "$ollama_ready" ]]; then
        echo "Ollama did not become ready; see $OLLAMA_LOG" >&2
        exit 1
    fi
else
    echo "Using the existing Ollama server at $OLLAMA_HOST"
fi

# Pull only missing models, using the exact names specified by the frozen source.
"$RERANKER_PYTHON" "$SCRIPT_DIR/scripts/ollama_models.py" \
    "$EXPERIMENT_DIR/llm-models.json" "$RERANKER_RUN_DIR/ollama-tags.json"

# Predict the runtime before the long phases. Calibration times real workers on a few
# small TRAINING packages (never benchmark packages) at BENCHMARK_JOBS concurrency,
# times sampled training steps, and serves an untrained model only to time NeuralRank.
if [[ "$ESTIMATE" != skip ]]; then
    CALIBRATION_DIR="$RERANKER_RUN_DIR/calibration"
    echo "Estimating runtime: timing small training packages at $BENCHMARK_JOBS jobs (takes minutes)"
    "$RERANKER_PYTHON" "$SCRIPT_DIR/scripts/package_benchmarks.py" calibrate \
        --run "$RERANKER_RUN_DIR" --image "$EXPERIMENT_DIR/image" --phase normal --jobs "$BENCHMARK_JOBS"
    if [[ ! -f "$CALIBRATION_DIR/training.json" ]]; then
        "$RERANKER_PYTHON" "$SCRIPT_DIR/scripts/reranker_workflow.py" calibrate \
            --repository "$REPO_DIR" --split "$RERANKER_RUN_DIR/split.json" \
            --calibration-split "$CALIBRATION_DIR/split.json" \
            --training "$RERANKER_RUN_DIR/training.jsonl" --validation "$RERANKER_RUN_DIR/validation.jsonl" \
            --test "$RERANKER_RUN_DIR/test.jsonl" --epochs "$RANKING_EPOCHS" --width 32 \
            --output "$CALIBRATION_DIR/model" --report "$CALIBRATION_DIR/training.json"
    fi
    estimate_options=(--run "$RERANKER_RUN_DIR" --jobs "$BENCHMARK_JOBS")
    if [[ "$SKIP_RERANKER_BENCHMARKS" == 0 ]]; then
        if [[ ! -f "$CALIBRATION_DIR/reranker.json" ]]; then
            start_reranker "$CALIBRATION_DIR/model" "$CALIBRATION_DIR/reranker-server.log"
            "$RERANKER_PYTHON" "$SCRIPT_DIR/scripts/package_benchmarks.py" calibrate \
                --run "$RERANKER_RUN_DIR" --image "$EXPERIMENT_DIR/image" --phase reranker --jobs "$BENCHMARK_JOBS"
            stop_reranker
        fi
    else
        estimate_options+=(--normal-only)
    fi
    "$RERANKER_PYTHON" "$SCRIPT_DIR/scripts/runtime_estimate.py" "${estimate_options[@]}" --elapsed "$SECONDS"
    if [[ "$ESTIMATE" == only ]]; then
        echo "Estimate only: no benchmark package was run. To run this exact prepared experiment:"
        echo "  BENCHMARK_RESUME_DIR=\"$RERANKER_RUN_DIR\" BENCHMARK_JOBS=$BENCHMARK_JOBS $0"
        exit 0
    fi
fi

echo "Running baseline, dependency, LLM completion and hybrid benchmarks on the saved packages"
run_package_phase normal

# Release an owned Ollama server before measuring the re-rankers.
if [[ -n "$OLLAMA_PID" ]]; then
    kill "$OLLAMA_PID" 2>/dev/null || true
    wait "$OLLAMA_PID" 2>/dev/null || true
    OLLAMA_PID=""
fi

# Normal benchmarks are complete. Only train/validation data enter model selection.
if [[ ! -f "$RERANKER_RUN_DIR/training-ready" ]]; then
    echo "Training re-ranker; selecting the best epoch using validation MRR"
    if ! "$RERANKER_PYTHON" "$SCRIPT_DIR/scripts/reranker_workflow.py" train \
        --repository "$REPO_DIR" --split "$RERANKER_RUN_DIR/split.json" \
        --training "$RERANKER_RUN_DIR/training.jsonl" --validation "$RERANKER_RUN_DIR/validation.jsonl" \
        --output "$RERANKER_MODEL_DIR" --epochs "$RANKING_EPOCHS" --width 32 \
        > "$RERANKER_RUN_DIR/training.log" 2>&1; then
        echo "Training failed; see $RERANKER_RUN_DIR/training.log" >&2
        exit 1
    fi
    echo "Evaluating the selected checkpoint on the separate test packages"
    "$RERANKER_PYTHON" "$SCRIPT_DIR/scripts/reranker_workflow.py" evaluate \
        --repository "$REPO_DIR" --split "$RERANKER_RUN_DIR/split.json" \
        --model "$RERANKER_MODEL_DIR" --data "$RERANKER_RUN_DIR/test.jsonl" \
        --output "$RERANKER_RUN_DIR/evaluation.json"
    printf '%s\n' "$RANKING_EPOCHS" > "$RERANKER_RUN_DIR/training-ready"
else
    echo "Reusing validation-selected model and independent test evaluation"
fi
"$RERANKER_PYTHON" "$SCRIPT_DIR/scripts/package_benchmarks.py" pin-model \
    --run "$RERANKER_RUN_DIR" --image "$EXPERIMENT_DIR/image"

# Rebuild paper figures from retained metrics; this never retrains or re-evaluates.
"$RERANKER_PYTHON" "$SCRIPT_DIR/scripts/reranker_workflow.py" report \
    --model "$RERANKER_MODEL_DIR" --evaluation "$RERANKER_RUN_DIR/evaluation.json" \
    --output "$RERANKER_RUN_DIR/learning"
for figure in learning-curves test-performance test-rank-transitions; do
    for extension in png pdf; do
        if [[ ! -s "$RERANKER_RUN_DIR/learning/$figure.$extension" ]]; then
            echo "Missing learning figure: $figure.$extension" >&2
            exit 1
        fi
    done
done

if [[ "$SKIP_RERANKER_BENCHMARKS" == 0 ]]; then
echo "Starting re-ranker"
start_reranker "$RERANKER_MODEL_DIR" "$RERANKER_RUN_DIR/reranker.log"

echo "Running re-ranking benchmarks on the saved benchmark packages"
run_package_phase reranker
else
    echo "Skipping live neural re-ranker benchmarks (--skip-reranker-benchmarks)"
fi

echo "Aggregating all packages and exporting the requested benchmark files"
export BENCHMARK_AGGREGATE="$RERANKER_RUN_DIR/aggregate.json"
aggregate_options=(--run "$RERANKER_RUN_DIR" --image "$EXPERIMENT_DIR/image")
publication_files=(results-table.tex performance.png dataset-summary.tex)
if [[ "$SKIP_RERANKER_BENCHMARKS" == 1 ]]; then
    aggregate_options+=(--normal-only)
else
    publication_files+=(results-table-re-ranker.tex performance-reranker.png)
fi
"$RERANKER_PYTHON" "$SCRIPT_DIR/scripts/package_benchmarks.py" aggregate \
    "${aggregate_options[@]}"
./pharo --headless Pharo.image --no-default-preferences eval "$(cat "$SCRIPT_DIR/scripts/benchmark-export.st")"
# The upstream exporter uses an extra hyphen; publish the requested filename.
if [[ "$SKIP_RERANKER_BENCHMARKS" == 0 ]]; then
    mv "$PUBLICATION_DIR/performance-re-ranker.png" "$PUBLICATION_DIR/performance-reranker.png"
fi

# Publish only a complete set from this run, so a failure cannot combine old and
# new benchmark artifacts. Training files, logs and metadata stay in the run dir.
for name in "${publication_files[@]}"; do
    if [[ ! -s "$PUBLICATION_DIR/$name" ]]; then
        echo "Missing or empty benchmark artifact: $PUBLICATION_DIR/$name" >&2
        exit 1
    fi
done
for name in "${publication_files[@]}"; do
    cp "$PUBLICATION_DIR/$name" "$RESULTS_DIR/$name"
done
# Retire the former filename only after the complete new set was published.
rm -f -- "$RESULTS_DIR/performance-re-ranker.png"
if [[ "$SKIP_RERANKER_BENCHMARKS" == 1 ]]; then
    # A skipped phase must not leave old neural results beside this run's outputs.
    rm -f -- "$RESULTS_DIR/results-table-re-ranker.tex" "$RESULTS_DIR/performance-reranker.png"
fi
echo "Requested benchmarks complete. Published ${#publication_files[@]} files: $RESULTS_DIR"
echo "Model, train/validation/test corpora, split, learning figures, evaluation and logs: $RERANKER_RUN_DIR"
if [[ -f "$RERANKER_RUN_DIR/estimate.json" ]]; then
    "$RERANKER_PYTHON" -c 'import json, sys; e = json.load(open(sys.argv[1])); print("Predicted total: %.1f h; this invocation took: %.1f h (per-package times: timings.jsonl)" % (e["totalSeconds"] / 3600, int(sys.argv[2]) / 3600))' \
        "$RERANKER_RUN_DIR/estimate.json" "$SECONDS"
fi
