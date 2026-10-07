#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MINING_DIR="${MINING_DIR:-$SCRIPT_DIR/mining}"
mkdir -p "$MINING_DIR"
MINING_DIR="$(cd "$MINING_DIR" && pwd -P)"
export REPO_DIR="$MINING_DIR/repository"
export RANKING_CORPUS_DIR="$MINING_DIR/corpus"
export BENCHMARK_REF="${BENCHMARK_REF:-main}"
CORPUS_PYTHON="${RERANKER_PYTHON:-python3}"

if ! mkdir "$MINING_DIR/.running" 2>/dev/null; then
    echo "Mining is already running: $MINING_DIR (.running lock)" >&2
    exit 1
fi
trap 'rmdir "$MINING_DIR/.running"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# A completed dataset is usable even after its mining image has been removed.
if [[ -d "$RANKING_CORPUS_DIR" ]]; then
    "$CORPUS_PYTHON" "$SCRIPT_DIR/scripts/ranking_corpus.py" verify \
        "$RANKING_CORPUS_DIR" --repository "$REPO_DIR"
    echo "Reusing saved dataset: $RANKING_CORPUS_DIR"
    exit 0
fi

# Freeze the code once as well as the image, so reruns cannot mix revisions.
if [[ ! -d "$REPO_DIR" ]]; then
    source_stage="$(mktemp -d "$MINING_DIR/source.XXXXXX")"
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
        echo "Benchmark source is missing shared-package-split support; update BENCHMARK_REF or BENCHMARK_REPO_DIR." >&2
        exit 1
    fi
    mv "$source_stage" "$REPO_DIR"
fi

mkdir -p "$MINING_DIR/image"
cd "$MINING_DIR/image"
if [[ ! -f Pharo.image || ! -x pharo ]]; then
    if [[ -f "$MINING_DIR/image-ready" ]]; then
        echo "The experiment image is missing; restore it or use a new MINING_DIR." >&2
        exit 1
    fi
    echo "Creating Pharo image"
    curl -fsSL "${PHARO_DOWNLOAD_URL:-https://get.pharo.org/140+vm}" -o get-pharo.sh
    bash get-pharo.sh
fi

if [[ ! -f "$MINING_DIR/image-ready" ]]; then
    echo "Installing frozen benchmark source"
    ./pharo --headless Pharo.image eval --save "
Metacello new
    repository: 'tonel://', ((OSEnvironment current at: 'REPO_DIR') asFileReference / 'src') fullName;
    baseline: 'ExtendedHeuristicCompletionBenchmarks';
    load
"
    touch "$MINING_DIR/image-ready"
fi

# Publish only after the complete export and validation succeed. An interrupted
# attempt stays in corpus-pending.* and cannot be mistaken for a usable corpus.
export CORPUS_STAGE_DIR="$(mktemp -d "$MINING_DIR/corpus-pending.XXXXXX")"
echo "Mining all eligible image packages once (no benchmark selection yet)"
./pharo --headless Pharo.image eval "
| directory packages |
directory := (OSEnvironment current at: 'CORPUS_STAGE_DIR') asFileReference.
packages := CooBenchRunner eligiblePackageNames.
(directory / 'packages.json') writeStreamDo: [ :out |
    out nextPutAll: (STONJSON toString: packages); lf ].
CooBenchRunner exportRankingCorpusForPackages: packages to: directory / 'all.jsonl'.
"
"$CORPUS_PYTHON" "$SCRIPT_DIR/scripts/ranking_corpus.py" finalize \
    "$CORPUS_STAGE_DIR" --image "$MINING_DIR/image/Pharo.image" --repository "$REPO_DIR"
mv "$CORPUS_STAGE_DIR" "$RANKING_CORPUS_DIR"
echo "Prepared corpus: $RANKING_CORPUS_DIR"
echo "Run pipeline-benchmarks.sh with the same MINING_DIR; it downloads a separate benchmark image."
