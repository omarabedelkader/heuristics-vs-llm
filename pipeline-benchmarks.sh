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
            echo "BENCHMARK_JOBS=N: benchmarks run at the same time. Every package is split into its 28 benchmarks"
            echo "  (2 kinds x 14 strategies), each in its own Pharo process, so N slots stay busy even for one package."
            echo "Ollama: the pipeline starts its own server on a free port with OLLAMA_NUM_PARALLEL=N"
            echo "  (default BENCHMARK_JOBS) and OLLAMA_MAX_LOADED_MODELS (default 4)."
            echo "  BENCHMARK_OLLAMA_HOST=host:port uses an existing server instead (start it with those settings)."
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

# The benchmarks use their own Ollama server, on a free port next to any desktop
# Ollama, started with one request slot per worker and every model kept loaded, so
# the concurrent workers are served in parallel instead of queuing. This pipeline owns
# it and stops it. BENCHMARK_OLLAMA_HOST=host:port uses that server instead (job.oar.sh
# starts one with the same settings), or starts one there if nothing answers.
if [[ -n "${BENCHMARK_OLLAMA_HOST:-}" ]]; then
    export OLLAMA_HOST="$BENCHMARK_OLLAMA_HOST"
else
    export OLLAMA_HOST="127.0.0.1:$("$PIPELINE_PYTHON" -c 'import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])')"
fi
export BENCHMARK_OLLAMA_URL="http://$OLLAMA_HOST"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-$BENCHMARK_JOBS}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-4}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}"
OLLAMA_LOG="$RUN_DIR/ollama.log"
OLLAMA_PID=""
if ! curl -fsS --connect-timeout 2 --max-time 5 \
        "$BENCHMARK_OLLAMA_URL/api/tags" > "$RUN_DIR/ollama-tags.json" 2>/dev/null; then
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
    # Reuse the models a desktop Ollama already downloaded, instead of fetching them again.
    if [[ -z "${OLLAMA_MODELS_DIR:-}" && -d "$HOME/.ollama/models" ]]; then
        OLLAMA_MODELS_DIR="$HOME/.ollama/models"
    fi
    export OLLAMA_MODELS="${OLLAMA_MODELS_DIR:-$SCRIPT_DIR/.ollama-models}"
    echo "Starting Ollama at $OLLAMA_HOST: $OLLAMA_NUM_PARALLEL parallel requests, up to $OLLAMA_MAX_LOADED_MODELS models loaded"
    mkdir -p "$OLLAMA_MODELS"
    OLLAMA_MODELS="$(cd "$OLLAMA_MODELS" && pwd)"
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
                "$BENCHMARK_OLLAMA_URL/api/tags" > "$RUN_DIR/ollama-tags.json" 2>/dev/null; then
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
    echo "Using the existing Ollama server at $OLLAMA_HOST (start it with OLLAMA_NUM_PARALLEL and OLLAMA_MAX_LOADED_MODELS for parallel requests)"
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
