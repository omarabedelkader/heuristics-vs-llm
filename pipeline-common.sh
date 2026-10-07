#!/usr/bin/env bash
# Shared experiment setup, sourced by both entry points.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$SCRIPT_DIR/experiment}"
mkdir -p "$EXPERIMENT_DIR"
EXPERIMENT_DIR="$(cd "$EXPERIMENT_DIR" && pwd)"
export RESULTS_DIR="${RESULTS_DIR:-$EXPERIMENT_DIR/results}"
mkdir -p "$RESULTS_DIR"
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"
export BENCHMARK_PACKAGE_COUNT="${BENCHMARK_PACKAGE_COUNT:-}"
export RANKING_SEED="${RANKING_SEED:-}"
export RANKING_EPOCHS="${RANKING_EPOCHS:-10}"
export BENCHMARK_REF="${BENCHMARK_REF:-main}"
export BENCHMARK_SPLIT_FILE="$EXPERIMENT_DIR/split.json"
export REPO_DIR="$EXPERIMENT_DIR/repository"
RERANKER_PID=""

if ! mkdir "$EXPERIMENT_DIR/.running" 2>/dev/null; then
    echo "Experiment is already running: $EXPERIMENT_DIR (.running lock)" >&2
    exit 1
fi
pipeline_cleanup() {
    if [[ -n "$RERANKER_PID" ]]; then
        kill "$RERANKER_PID" 2>/dev/null || true
        wait "$RERANKER_PID" 2>/dev/null || true
    fi
    rmdir "$EXPERIMENT_DIR/.running"
}
trap pipeline_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Freeze the code once as well as the image, so reruns cannot mix revisions.
if [[ ! -d "$REPO_DIR" ]]; then
    source_stage="$(mktemp -d "$EXPERIMENT_DIR/source.XXXXXX")"
    local_repo="${BENCHMARK_REPO_DIR:-$SCRIPT_DIR/../HeuristicCompletion-Benchmarks-Multiples}"
    if [[ -d "$local_repo/src" ]]; then
        cp -R "$local_repo/src" "$local_repo/scripts" "$local_repo/reranker" "$source_stage/"
        printf 'Local snapshot: %s\n' "$local_repo" > "$source_stage/origin.txt"
    else
        if [[ -n "${BENCHMARK_REPO_DIR:-}" ]]; then
            echo "BENCHMARK_REPO_DIR has no src directory: $local_repo" >&2
            exit 1
        fi
        curl -fsSL "https://codeload.github.com/omarabedelkader/HeuristicCompletion-Benchmarks-Multiples/tar.gz/$BENCHMARK_REF" \
            -o "$source_stage/repository.tar.gz"
        tar -xzf "$source_stage/repository.tar.gz" --strip-components=1 -C "$source_stage"
        printf 'GitHub ref: %s\n' "$BENCHMARK_REF" > "$source_stage/origin.txt"
    fi
    if [[ ! -f "$source_stage/src/ExtendedHeuristicCompletion-Benchmarks/CooBenchmarkSplit.class.st" ]]; then
        echo "Benchmark source is missing shared-package-split support; update neural-ranking or BENCHMARK_REPO_DIR." >&2
        exit 1
    fi
    mv "$source_stage" "$REPO_DIR"
fi

mkdir -p "$EXPERIMENT_DIR/image"
cd "$EXPERIMENT_DIR/image"
if [[ ! -f Pharo.image || ! -x pharo ]]; then
    if [[ -f "$BENCHMARK_SPLIT_FILE" || -f "$EXPERIMENT_DIR/image-ready" ]]; then
        echo "The experiment image is missing; restore it or use a new EXPERIMENT_DIR." >&2
        exit 1
    fi
    echo "Creating Pharo image"
    curl -fsSL https://get.pharo.org/140+vm -o get-pharo.sh
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

# Select before any benchmark or training collection. Existing splits are never redrawn.
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
Stdio stdout
    nextPutAll: 'Saved split: ', split benchmarkPackageNames size asString, ' benchmark packages; ',
        split trainingPackageNames size asString, ' training packages'; lf;
    nextPutAll: file fullName; lf.
"
