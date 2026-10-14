#!/usr/bin/env bash
# Baseline, dependency, LLM completion (0.5B/1.5B/3B/7B) and hybrid benchmarks on the
# saved benchmark packages of one experiment. It builds a Pharo image from your
# benchmark checkout and benchmarks the image's own packages: no dataset is mined or
# downloaded. The neural re-ranker and its mined training data live in reranker/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for option in "$@"; do
    case "$option" in
        --help|-h)
            echo "Usage: BENCHMARK_JOBS=N [BENCHMARK_PACKAGE_COUNT=N] $0"
            echo "Runs baseline, dependency, LLM completion and hybrid benchmarks on the saved benchmark packages."
            echo "BENCHMARK_PACKAGE_COUNT=N: benchmark N seeded-random packages of the image (default 50; RANKING_SEED, default 42)."
            echo "BENCHMARK_PACKAGES_FILE=path: benchmark exactly these packages (one name per line, a JSON array, or a saved selection)."
            echo "BENCHMARK_REPO_DIR=path: the benchmark checkout (default ../HeuristicCompletion-Benchmarks-Multiples)."
            echo "BENCHMARK_RESUME_DIR=path: continue an interrupted benchmark-run.* directory."
            echo "The benchmarks keep running when you press Ctrl-C, close the terminal or leave the SSH session;"
            echo "their output goes to a log file (BENCHMARK_LOG=path, default benchmark-<date>.log in the current directory)."
            echo "--foreground: stay attached to this terminal instead (for job schedulers)."
            echo "Results: RESULTS_DIR/results-before-pharo-validation and RESULTS_DIR/results-after-pharo-validation."
            exit 0 ;;
        --foreground) ;;
        *) echo "Unknown option: $option (use --help)" >&2; exit 2 ;;
    esac
done
if [[ " $* " != *" --foreground "* ]]; then
    # Start again in a session of its own, without a terminal: closing this one, or
    # Ctrl-C, no longer reaches the benchmarks. This shell only shows their log.
    BENCHMARK_LOG="${BENCHMARK_LOG:-$PWD/benchmark-$(date '+%Y%m%d-%H%M%S').log}"
    : > "$BENCHMARK_LOG"
    "${PIPELINE_PYTHON:-python3}" -c 'import os, signal, sys
os.setsid()
signal.signal(signal.SIGHUP, signal.SIG_IGN)
os.execvp("bash", ["bash", *sys.argv[1:]])' "$SCRIPT_DIR/pipeline-benchmarks.sh" --foreground \
        > "$BENCHMARK_LOG" 2>&1 < /dev/null &
    pipeline=$!
    echo "Benchmarks started (PID $pipeline). They keep running if you press Ctrl-C or close this session."
    echo "See them again:  tail -f $BENCHMARK_LOG"
    echo "Stop them:       kill $pipeline"
    tail -n +1 -f "$BENCHMARK_LOG" &
    viewer=$!
    trap 'kill "$viewer" 2>/dev/null; echo; echo "Still running (PID $pipeline). See them again: tail -f $BENCHMARK_LOG"; exit 0' INT
    status=0
    wait "$pipeline" || status=$?
    sleep 1
    kill "$viewer" 2>/dev/null || true
    exit "$status"
fi
source "$SCRIPT_DIR/scripts/pipeline-common.sh"

prepare_run_directory
"$PIPELINE_PYTHON" "$SCRIPT_DIR/scripts/package_benchmarks.py" prepare \
    --run "$RUN_DIR" --image "$EXPERIMENT_DIR/image"

# The Pharo client reads OLLAMA_HOST. Reuse an existing healthy server recent
# enough to return log-probabilities, or start one that this pipeline owns and
# cleans up on success or failure.
OLLAMA_MIN_VERSION="${OLLAMA_MIN_VERSION:-0.12.11}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
OLLAMA_LOG="$RUN_DIR/ollama.log"
OLLAMA_PID=""
version_at_least() {
    [[ -n "$1" && "$(printf '%s\n%s\n' "$OLLAMA_MIN_VERSION" "$1" | sort -V | head -n1)" == "$OLLAMA_MIN_VERSION" ]]
}
server_version() {
    curl -fsS --connect-timeout 2 --max-time 5 "http://127.0.0.1:$1/api/version" 2>/dev/null \
        | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1 || true
}
binary_version() {
    # An unreachable host makes the binary report only its own (client) version.
    OLLAMA_HOST=127.0.0.1:1 "$1" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | tail -n1 || true
}
server_usable() {
    curl -fsS --connect-timeout 2 --max-time 5 \
        "http://127.0.0.1:$1/api/tags" > "$RUN_DIR/ollama-tags.json" 2>/dev/null \
        && version_at_least "$(server_version "$1")"
}
if ! server_usable "$OLLAMA_PORT"; then
    old_version="$(server_version "$OLLAMA_PORT")"
    if [[ -n "$old_version" ]]; then
        echo "Ollama $old_version at port $OLLAMA_PORT is older than $OLLAMA_MIN_VERSION" \
            "(no log-probabilities); using port ${OLLAMA_FALLBACK_PORT:-11436} instead" >&2
        OLLAMA_PORT="${OLLAMA_FALLBACK_PORT:-11436}"
    fi
fi
export OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"
if ! server_usable "$OLLAMA_PORT"; then
    if [[ -n "$(server_version "$OLLAMA_PORT")" ]]; then
        echo "Ollama at $OLLAMA_HOST is older than $OLLAMA_MIN_VERSION; stop it or set OLLAMA_PORT" >&2
        exit 1
    fi
    if [[ -z "${OLLAMA_BIN:-}" ]]; then
        # The first recent enough installation: PATH, a user-local one, or a previous download.
        for candidate in "$(command -v ollama || true)" "$HOME/.local/ollama/bin/ollama" \
                "$EXPERIMENT_DIR/ollama-bin/bin/ollama"; do
            [[ -n "$candidate" && -x "$candidate" ]] || continue
            if version_at_least "$(binary_version "$candidate")"; then
                OLLAMA_BIN="$candidate"
                echo "Starting Ollama $(binary_version "$OLLAMA_BIN") from $OLLAMA_BIN at $OLLAMA_HOST"
                break
            fi
            echo "Ignoring $candidate $(binary_version "$candidate"), older than $OLLAMA_MIN_VERSION" >&2
        done
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
        if [[ ! -x "$OLLAMA_BIN" ]] || ! version_at_least "$(binary_version "$OLLAMA_BIN")"; then
            rm -rf "$ollama_install_dir"
            mkdir -p "$ollama_install_dir"
            curl -fsSL "https://ollama.com/download/ollama-linux-$ollama_arch.tar.zst" \
                -o "$ollama_install_dir/ollama.tar.zst"
            if command -v zstd >/dev/null 2>&1; then
                zstd -dc "$ollama_install_dir/ollama.tar.zst" | tar -xf - -C "$ollama_install_dir"
            elif "$PIPELINE_PYTHON" -m pip install --quiet --target "$ollama_install_dir/python" zstandard; then
                # tar --zstd also needs the zstd program; without it, decompress in Python.
                PYTHONPATH="$ollama_install_dir/python" "$PIPELINE_PYTHON" -c \
                    'import sys, zstandard; zstandard.ZstdDecompressor().copy_stream(sys.stdin.buffer, sys.stdout.buffer)' \
                    < "$ollama_install_dir/ollama.tar.zst" | tar -xf - -C "$ollama_install_dir"
            else
                echo "Extracting Ollama requires tar with zstd support or the zstd command." >&2
                exit 1
            fi
        fi
    fi
    export OLLAMA_MODELS="${OLLAMA_MODELS_DIR:-$SCRIPT_DIR/.ollama-models}"
    mkdir -p "$OLLAMA_MODELS"
    OLLAMA_MODELS="$(cd "$OLLAMA_MODELS" && pwd)"
    # Reuse models already pulled by a system Ollama (often read-only) through links,
    # instead of downloading them again.
    shared_models="${OLLAMA_SHARED_MODELS:-/usr/share/ollama/.ollama/models}"
    if [[ -r "$shared_models/blobs" && -r "$shared_models/manifests" ]]; then
        mkdir -p "$OLLAMA_MODELS/blobs"
        for blob in "$shared_models"/blobs/*; do
            [[ -r "$blob" && ! -e "$OLLAMA_MODELS/blobs/${blob##*/}" ]] && ln -s "$blob" "$OLLAMA_MODELS/blobs/"
        done
        cp -rn "$shared_models/manifests" "$OLLAMA_MODELS/" 2>/dev/null || true
    fi
    "$OLLAMA_BIN" serve > "$OLLAMA_LOG" 2>&1 &
    OLLAMA_PID=$!
    CLEANUP_PIDS+=("$OLLAMA_PID")
    ollama_ready=""
    for ((attempt=0; attempt<120; attempt++)); do
        if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
            cat "$OLLAMA_LOG" >&2
            exit 1
        fi
        if curl -fsS --connect-timeout 2 --max-time 5 \
                "http://127.0.0.1:$OLLAMA_PORT/api/tags" > "$RUN_DIR/ollama-tags.json" 2>/dev/null; then
            ollama_ready=1
            break
        fi
        sleep 1
    done
    if [[ -z "$ollama_ready" ]]; then
        echo "Ollama did not become ready; see $OLLAMA_LOG" >&2
        exit 1
    fi
    if ! version_at_least "$(server_version "$OLLAMA_PORT")"; then
        echo "$OLLAMA_BIN is older than Ollama $OLLAMA_MIN_VERSION (no log-probabilities)" >&2
        exit 1
    fi
else
    echo "Using the existing Ollama server at $OLLAMA_HOST"
fi

# Pull only missing models, using the exact names specified by the frozen source.
"$PIPELINE_PYTHON" "$SCRIPT_DIR/scripts/ollama_models.py" \
    "$EXPERIMENT_DIR/llm-models.json" "$RUN_DIR/ollama-tags.json"

echo "Running baseline, dependency, LLM completion and hybrid benchmarks on the saved packages"
run_packages

echo "Aggregating all packages and exporting the benchmark files"
publish_results results-table.tex performance.png dataset-summary.tex recall-top.tex
echo "Packages, package results and logs: $RUN_DIR"
echo "This invocation took $((SECONDS / 60)) min (per-package times: $RUN_DIR/timings.jsonl)"
