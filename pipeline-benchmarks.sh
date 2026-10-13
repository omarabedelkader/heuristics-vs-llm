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
            exit 0 ;;
        *) echo "Unknown option: $option (use --help)" >&2; exit 2 ;;
    esac
done
source "$SCRIPT_DIR/scripts/pipeline-common.sh"

prepare_run_directory
"$PIPELINE_PYTHON" "$SCRIPT_DIR/scripts/package_benchmarks.py" prepare \
    --run "$RUN_DIR" --image "$EXPERIMENT_DIR/image"

# The Pharo client uses localhost:11434. Reuse an existing healthy server, or
# start a local one that this pipeline owns and cleans up on success or failure.
export OLLAMA_HOST="127.0.0.1:11434"
OLLAMA_LOG="$RUN_DIR/ollama.log"
OLLAMA_PID=""
if ! curl -fsS --connect-timeout 2 --max-time 5 \
        http://127.0.0.1:11434/api/tags > "$RUN_DIR/ollama-tags.json" 2>/dev/null; then
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
    CLEANUP_PIDS+=("$OLLAMA_PID")
    ollama_ready=""
    for ((attempt=0; attempt<120; attempt++)); do
        if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
            cat "$OLLAMA_LOG" >&2
            exit 1
        fi
        if curl -fsS --connect-timeout 2 --max-time 5 \
                http://127.0.0.1:11434/api/tags > "$RUN_DIR/ollama-tags.json" 2>/dev/null; then
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
"$PIPELINE_PYTHON" "$SCRIPT_DIR/scripts/ollama_models.py" \
    "$EXPERIMENT_DIR/llm-models.json" "$RUN_DIR/ollama-tags.json"

echo "Running baseline, dependency, LLM completion and hybrid benchmarks on the saved packages"
run_packages

echo "Aggregating all packages and exporting the benchmark files"
publish_results results-table.tex performance.png dataset-summary.tex recall-top.tex
echo "Packages, package results and logs: $RUN_DIR"
echo "This invocation took $((SECONDS / 60)) min (per-package times: $RUN_DIR/timings.jsonl)"
