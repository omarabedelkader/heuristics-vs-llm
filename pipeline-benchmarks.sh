#!/usr/bin/env bash
#OAR -q default
#OAR -p chirop
#OAR -l host=1,walltime=14:00:00
#OAR -n heuristic-reranker-benchmark
#OAR -O heuristic-reranker.%jobid%.out
#OAR -E heuristic-reranker.%jobid%.err
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! "${BENCHMARK_JOBS:-}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Set BENCHMARK_JOBS to the number of simultaneous package jobs you want (positive integer)." >&2
    exit 1
fi
export BENCHMARK_JOBS
export MINING_DIR="${MINING_DIR:-$SCRIPT_DIR/mining}"
if [[ ! -f "$MINING_DIR/corpus/manifest.json" ]]; then
    echo "Saved dataset missing. Run MINING_DIR=\"$MINING_DIR\" $SCRIPT_DIR/pipeline-mine-training-data.sh first." >&2
    exit 1
fi
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
export BENCHMARK_PACKAGE_COUNT="${BENCHMARK_PACKAGE_COUNT:-}"
export RANKING_SEED="${RANKING_SEED:-}"
export RANKING_EPOCHS="${RANKING_EPOCHS:-10}"
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

# Verify saved data before any downloads. Never access or run the mining image.
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

mkdir -p "$EXPERIMENT_DIR/image"
cd "$EXPERIMENT_DIR/image"
if [[ ! -f Pharo.image || ! -x pharo ]]; then
    if [[ -f "$BENCHMARK_SPLIT_FILE" || -f "$EXPERIMENT_DIR/image-ready" ]]; then
        echo "The experiment image is missing; restore it or use a new EXPERIMENT_DIR." >&2
        exit 1
    fi
    echo "Downloading separate benchmark Pharo image"
    curl -fsSL "${PHARO_DOWNLOAD_URL:-https://get.pharo.org/140+vm}" -o get-pharo.sh
    bash get-pharo.sh
fi

if [[ ! -f "$EXPERIMENT_DIR/image-ready" ]]; then
    echo "Installing frozen benchmark source"
    ./pharo --headless Pharo.image eval --save "
Metacello new
    repository: 'tonel://', ((OSEnvironment current at: 'REPO_DIR') asFileReference / 'src') fullName;
    baseline: 'ExtendedHeuristicCompletionBenchmarks';
    load
"
    touch "$EXPERIMENT_DIR/image-ready"
fi

# Select and save benchmark names in the independent benchmark image.
./pharo --headless Pharo.image eval "
| file split countText seedText |
file := (OSEnvironment current at: 'BENCHMARK_SPLIT_FILE') asFileReference.
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
        split trainingPackageNames size asString, ' training packages'; lf;
    nextPutAll: file fullName; lf.
"

# New invocations train once. Explicit resumes retain the same model and results.
if [[ -n "${BENCHMARK_RESUME_DIR:-}" ]]; then
    RERANKER_RUN_DIR="$(cd "$BENCHMARK_RESUME_DIR" && pwd -P)"
    if [[ "$(dirname "$RERANKER_RUN_DIR")" != "$EXPERIMENT_DIR" ||
          ! -f "$RERANKER_RUN_DIR/training-ready" || ! -f "$RERANKER_RUN_DIR/workers.json" ]]; then
        echo "BENCHMARK_RESUME_DIR must be a prepared package run inside EXPERIMENT_DIR." >&2
        exit 1
    fi
    if ! cmp -s "$BENCHMARK_SPLIT_FILE" "$RERANKER_RUN_DIR/split.json" ||
            [[ "$(cat "$RERANKER_RUN_DIR/training-ready")" != "$RANKING_EPOCHS" ]]; then
        echo "Resume split or training epochs differ; restore the original settings or start a new run." >&2
        exit 1
    fi
else
    RERANKER_RUN_DIR="$(mktemp -d "$EXPERIMENT_DIR/reranker-run.XXXXXX")"
    cp "$BENCHMARK_SPLIT_FILE" "$RERANKER_RUN_DIR/split.json"
fi
export RERANKER_RUN_DIR
RERANKER_MODEL_DIR="$RERANKER_RUN_DIR/model"
export PUBLICATION_DIR="$(mktemp -d "$RERANKER_RUN_DIR/publication.XXXXXX")"
echo "Run directory: $RERANKER_RUN_DIR"
echo "Package concurrency selected by you: $BENCHMARK_JOBS"

if [[ -z "${BENCHMARK_RESUME_DIR:-}" ]]; then
    echo "Preparing training/test files from the saved corpus (benchmark packages excluded from training)"
    "${RERANKER_PYTHON:-python3}" "$SCRIPT_DIR/scripts/ranking_corpus.py" partition \
        "$RANKING_CORPUS_DIR" --repository "$REPO_DIR" \
        --split "$RERANKER_RUN_DIR/split.json" --output "$RERANKER_RUN_DIR"
fi

if [[ -z "${RERANKER_PYTHON:-}" ]]; then
    python3 -m venv "$EXPERIMENT_DIR/reranker-venv"
    RERANKER_PYTHON="$EXPERIMENT_DIR/reranker-venv/bin/python"
fi
"$RERANKER_PYTHON" -m pip install -r "$REPO_DIR/reranker/requirements.txt"

if [[ -z "${BENCHMARK_RESUME_DIR:-}" ]]; then
    echo "Training re-ranker (fixed epochs; no benchmark data)"
    TRAINING_SEED="$("$RERANKER_PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["seed"])' "$BENCHMARK_SPLIT_FILE")"
    "$RERANKER_PYTHON" "$REPO_DIR/reranker/train.py" \
        "$RERANKER_RUN_DIR/training.jsonl" "$RERANKER_MODEL_DIR" \
        --package-split "$RERANKER_RUN_DIR/split.json" \
        --epochs "$RANKING_EPOCHS" --width 32 --seed "$TRAINING_SEED"

    # The trainer only receives training.jsonl. Evaluate the held-out file after training.
    echo "Evaluating the saved benchmark packages"
    "$RERANKER_PYTHON" "$REPO_DIR/reranker/evaluate.py" \
        "$RERANKER_MODEL_DIR" "$RERANKER_RUN_DIR/test.jsonl" > "$RERANKER_RUN_DIR/evaluation.json"
    printf '%s\n' "$RANKING_EPOCHS" > "$RERANKER_RUN_DIR/training-ready"
else
    echo "Reusing trained model and completed package checkpoints: $RERANKER_RUN_DIR"
fi

"$RERANKER_PYTHON" "$SCRIPT_DIR/scripts/package_benchmarks.py" prepare \
    --run "$RERANKER_RUN_DIR" --image "$EXPERIMENT_DIR/image"

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

echo "Running baseline, dependency, LLM completion and hybrid benchmarks on the saved packages"
run_package_phase normal

# Release an owned Ollama server before measuring the re-rankers.
if [[ -n "$OLLAMA_PID" ]]; then
    kill "$OLLAMA_PID" 2>/dev/null || true
    wait "$OLLAMA_PID" 2>/dev/null || true
    OLLAMA_PID=""
fi

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
run_package_phase reranker

echo "Aggregating all packages and exporting the five publication files"
export BENCHMARK_AGGREGATE="$RERANKER_RUN_DIR/aggregate.json"
"$RERANKER_PYTHON" "$SCRIPT_DIR/scripts/package_benchmarks.py" aggregate \
    --run "$RERANKER_RUN_DIR" --image "$EXPERIMENT_DIR/image"
./pharo --headless Pharo.image --no-default-preferences eval "$(cat "$SCRIPT_DIR/scripts/benchmark-export.st")"
# The upstream exporter uses an extra hyphen; publish the requested filename.
mv "$PUBLICATION_DIR/performance-re-ranker.png" "$PUBLICATION_DIR/performance-reranker.png"

# Publish only a complete set from this run, so a failure cannot combine old and
# new benchmark artifacts. Training files, logs and metadata stay in the run dir.
publication_files=(results-table.tex results-table-re-ranker.tex performance.png performance-reranker.png dataset-summary.tex)
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
echo "All benchmarks complete. Tables, figures and shared dataset summary: $RESULTS_DIR"
echo "Model, separate training/test corpora, split, evaluation and log: $RERANKER_RUN_DIR"
