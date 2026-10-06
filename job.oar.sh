#!/usr/bin/env bash

#OAR -q default
#OAR -p chirop
#OAR -l host=1,walltime=14:00:00
#OAR -n heuristic-completion-benchmark
#OAR -O heuristic-completion.%jobid%.out
#OAR -E heuristic-completion.%jobid%.err

set -euo pipefail


###############################################################################
# Configuration
###############################################################################

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_ID="${OAR_JOB_ID:-manual-$(date '+%Y%m%d-%H%M%S')}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/oar-runs/$RUN_ID}"

# Keep paths stable when Ollama installation changes the working directory.
case "$RUN_DIR" in
    /*) ;;
    *) RUN_DIR="$PWD/$RUN_DIR" ;;
esac

# Both pipelines share this image, package split and results directory.
export EXPERIMENT_DIR="$RUN_DIR"
export RESULTS_DIR="$RUN_DIR/results"
export BENCHMARK_PACKAGE_COUNT="${BENCHMARK_PACKAGE_COUNT:-50}"
export RANKING_SEED="${RANKING_SEED:-42}"
export RANKING_EPOCHS="${RANKING_EPOCHS:-10}"
export BENCHMARK_REF="${BENCHMARK_REF:-main}"
# BENCHMARK_REPO_DIR and RERANKER_PYTHON, if set, are inherited by the pipelines.

OLLAMA_INSTALL_DIR="$RUN_DIR/ollama-bin"

# Persistent model directory.
# Models downloaded by one job can be reused by future jobs.
OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-$ROOT_DIR/.ollama-models}"

OLLAMA_LOG="$RUN_DIR/ollama.log"
OLLAMA_PID=""

OLLAMA_MODELS_TO_PULL=(
    "pharo-llm/Qwen2.5-Coder-SFT:0.5B"
    "pharo-llm/Qwen2.5-Coder-SFT:1.5B"
    "pharo-llm/Qwen2.5-Coder-SFT:3b"
    "pharo-llm/Qwen2.5-Coder-SFT:7b"
)


###############################################################################
# Logging
###############################################################################

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}


###############################################################################
# Cleanup
###############################################################################

cleanup() {

    if [ -n "${OLLAMA_PID:-}" ]; then

        if kill -0 "$OLLAMA_PID" 2>/dev/null; then

            log "Stopping Ollama (PID $OLLAMA_PID)"

            kill "$OLLAMA_PID" || true

            wait "$OLLAMA_PID" 2>/dev/null || true

        fi

    fi
}

trap cleanup EXIT INT TERM


###############################################################################
# Create directories
###############################################################################

mkdir -p "$RUN_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$OLLAMA_INSTALL_DIR"
mkdir -p "$OLLAMA_MODELS_DIR"


###############################################################################
# Job information
###############################################################################

log "============================================================"
log "JOB STARTED"
log "============================================================"

log "Job ID: ${OAR_JOB_ID:-manual}"
log "Host: $(hostname -f 2>/dev/null || hostname)"

log "Root directory:"
log "$ROOT_DIR"

log "Run directory:"
log "$RUN_DIR"

log "Results directory:"
log "$RESULTS_DIR"

log "Ollama models directory:"
log "$OLLAMA_MODELS_DIR"


###############################################################################
# System information
###############################################################################

log "============================================================"
log "SYSTEM INFORMATION"
log "============================================================"

uname -a || true

if command -v lscpu >/dev/null 2>&1; then

    lscpu | grep -E \
        'Model name|Socket|Core|Thread|CPU\(s\)' \
        || true

fi


###############################################################################
# GPU information
###############################################################################

if command -v nvidia-smi >/dev/null 2>&1; then

    log "NVIDIA GPU detected"

    nvidia-smi || true

else

    log "No NVIDIA GPU detected"

fi


###############################################################################
# 1. Install Ollama locally
#
# NO sudo
# NO systemctl
# NO /usr/local/bin
###############################################################################

log "============================================================"
log "1. INSTALLING OLLAMA LOCALLY"
log "============================================================"

cd "$OLLAMA_INSTALL_DIR"

ARCH="$(uname -m)"

case "$ARCH" in

    x86_64|amd64)

        OLLAMA_ARCH="amd64"
        ;;

    aarch64|arm64)

        OLLAMA_ARCH="arm64"
        ;;

    *)

        log "ERROR: Unsupported architecture: $ARCH"
        exit 1
        ;;

esac


OLLAMA_ARCHIVE="ollama-linux-${OLLAMA_ARCH}.tar.zst"

log "Architecture: $ARCH"
log "Downloading Ollama: $OLLAMA_ARCHIVE"


curl -fL \
    "https://ollama.com/download/$OLLAMA_ARCHIVE" \
    -o "$OLLAMA_ARCHIVE"


###############################################################################
# Extract Ollama
###############################################################################

log "Extracting Ollama"


if tar --help 2>/dev/null | grep -q zstd; then

    tar --zstd -xf "$OLLAMA_ARCHIVE"

elif command -v unzstd >/dev/null 2>&1; then

    unzstd -c "$OLLAMA_ARCHIVE" | tar -xf -

elif command -v zstd >/dev/null 2>&1; then

    zstd -dc "$OLLAMA_ARCHIVE" | tar -xf -

else

    log "ERROR: Cannot extract Ollama."
    log "zstd support was not found."
    exit 1

fi


###############################################################################
# Locate Ollama executable
###############################################################################

OLLAMA_BIN="$(find "$OLLAMA_INSTALL_DIR" \
    -type f \
    -name ollama \
    -perm -u+x \
    | head -n 1)"


if [ -z "${OLLAMA_BIN:-}" ]; then

    log "ERROR: Ollama executable not found"

    find "$OLLAMA_INSTALL_DIR" \
        -maxdepth 4 \
        -type f \
        -print \
        || true

    exit 1

fi


log "Ollama executable:"
log "$OLLAMA_BIN"

"$OLLAMA_BIN" --version || true


log "1-INSTALL OLLAMA OK"


###############################################################################
# 2. Configure Ollama
###############################################################################

log "============================================================"
log "2. CONFIGURING OLLAMA"
log "============================================================"


export OLLAMA_MODELS="$OLLAMA_MODELS_DIR"

export OLLAMA_HOST="127.0.0.1:11434"


log "OLLAMA_MODELS=$OLLAMA_MODELS"

log "OLLAMA_HOST=$OLLAMA_HOST"


log "2-CONFIGURE OLLAMA OK"


###############################################################################
# 3. Start Ollama
###############################################################################

log "============================================================"
log "3. STARTING OLLAMA SERVER"
log "============================================================"


"$OLLAMA_BIN" serve > "$OLLAMA_LOG" 2>&1 &

OLLAMA_PID=$!


log "Ollama PID: $OLLAMA_PID"

log "Ollama log:"
log "$OLLAMA_LOG"


###############################################################################
# Wait for Ollama API
###############################################################################

log "Waiting for Ollama API..."

OLLAMA_READY=0


for i in $(seq 1 120); do

    if curl -fsS \
        http://127.0.0.1:11434/api/tags \
        >/dev/null 2>&1; then

        OLLAMA_READY=1

        log "Ollama API is ready"

        break

    fi


    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then

        log "ERROR: Ollama server crashed"

        echo
        echo "================ OLLAMA LOG ================="

        cat "$OLLAMA_LOG" || true

        echo "============================================="

        exit 1

    fi


    sleep 1

done


if [ "$OLLAMA_READY" -ne 1 ]; then

    log "ERROR: Ollama did not start within 120 seconds"

    echo
    echo "================ OLLAMA LOG ================="

    cat "$OLLAMA_LOG" || true

    echo "============================================="

    exit 1

fi


log "Testing Ollama API"

curl -fsS \
    http://127.0.0.1:11434/api/tags \
    || true

echo


log "3-START OLLAMA OK"


###############################################################################
# 4. Pull benchmark models
###############################################################################

log "============================================================"
log "4. PULLING OLLAMA MODELS"
log "============================================================"


for model in "${OLLAMA_MODELS_TO_PULL[@]}"; do

    log "------------------------------------------------------------"
    log "Model: $model"
    log "------------------------------------------------------------"

    log "Checking whether model is already available..."


    if "$OLLAMA_BIN" list \
        | awk 'NR > 1 { print $1 }' \
        | grep -Fxq "$model"; then

        log "Model already downloaded:"
        log "$model"

        continue

    fi


    log "Pulling:"
    log "$model"


    if "$OLLAMA_BIN" pull "$model"; then

        log "Successfully pulled:"
        log "$model"

    else

        log "ERROR: Failed to pull model:"
        log "$model"

        exit 1

    fi

done


###############################################################################
# Show downloaded models
###############################################################################

log "============================================================"
log "INSTALLED OLLAMA MODELS"
log "============================================================"

"$OLLAMA_BIN" list


log "4-PULL MODELS OK"


# Pharo creation and benchmark installation are handled by pipeline-common.sh,
# called by the pipelines below. It snapshots the local benchmark checkout when
# available, otherwise downloads BENCHMARK_REF, and saves one shared split.


###############################################################################
# 7. Check Ollama before benchmark
###############################################################################

log "============================================================"
log "7. CHECKING OLLAMA BEFORE BENCHMARK"
log "============================================================"


if ! curl -fsS \
    http://127.0.0.1:11434/api/tags \
    >/dev/null; then

    log "ERROR: Ollama API is no longer responding"

    cat "$OLLAMA_LOG" || true

    exit 1

fi


log "Ollama is running"


log "Available models:"

"$OLLAMA_BIN" list


###############################################################################
# 8. Run normal benchmarks, then train and benchmark the re-ranker
###############################################################################

log "============================================================"
log "8. RUNNING BENCHMARKS"
log "============================================================"
log "Benchmark packages: $BENCHMARK_PACKAGE_COUNT; seed: $RANKING_SEED"
log "Re-ranker epochs: $RANKING_EPOCHS; training uses all remaining packages"
log "Shared experiment: $EXPERIMENT_DIR"

# Use child shells so pipeline cleanup traps do not replace Ollama's cleanup.
# Deploy all three pipeline-*.sh files alongside this job script.
bash "$ROOT_DIR/pipeline-normal-bench.sh"
log "NORMAL BENCHMARKS OK"

bash "$ROOT_DIR/pipeline-re-ranker-bench.sh"
log "RE-RANKER TRAINING AND BENCHMARKS OK"
log "Re-ranker performance image: $RESULTS_DIR/performance-re-ranker.png"

log "8-RUN BENCHMARKS OK"


###############################################################################
# 9. Show results
###############################################################################

log "============================================================"
log "9. BENCHMARK RESULTS"
log "============================================================"


if [ -d "$RESULTS_DIR" ]; then

    log "Generated files:"

    find "$RESULTS_DIR" \
        -maxdepth 4 \
        -type f \
        -print \
        || true

else

    log "WARNING: Results directory does not exist"

fi


###############################################################################
# 10. Ollama final status
###############################################################################

log "============================================================"
log "10. OLLAMA FINAL STATUS"
log "============================================================"


"$OLLAMA_BIN" list || true


###############################################################################
# Ollama log
###############################################################################

log "============================================================"
log "LAST 100 LINES OF OLLAMA LOG"
log "============================================================"


tail -n 100 "$OLLAMA_LOG" || true


###############################################################################
# Finished
###############################################################################

log "============================================================"
log "JOB FINISHED SUCCESSFULLY"
log "============================================================"

log "Job ID:"
log "${OAR_JOB_ID:-manual}"

log "Results:"
log "$RESULTS_DIR"

log "Ollama models:"
log "$OLLAMA_MODELS_DIR"

log "Ollama log:"
log "$OLLAMA_LOG"

log "Complete run directory:"
log "$RUN_DIR"