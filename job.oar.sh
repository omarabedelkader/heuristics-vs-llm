#!/usr/bin/env bash

#OAR -q default
#OAR -p larochette
#OAR -l host=1,walltime=55:00:00
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

# The benchmark pipeline owns the image, package split and package workers.
export EXPERIMENT_DIR="$RUN_DIR"
export RESULTS_DIR="$RUN_DIR/results"
# Edit these two defaults to choose the total benchmark packages and concurrency.
# Environment variables supplied at submission time override these defaults.
export BENCHMARK_PACKAGE_COUNT="${BENCHMARK_PACKAGE_COUNT:-50}"
export BENCHMARK_JOBS="${BENCHMARK_JOBS:-8}"
for setting in BENCHMARK_PACKAGE_COUNT BENCHMARK_JOBS; do
    if [[ ! "${!setting}" =~ ^[1-9][0-9]*$ ]]; then
        echo "$setting must be a positive integer" >&2
        exit 1
    fi
done
export RANKING_SEED="${RANKING_SEED:-42}"
export RANKING_EPOCHS="${RANKING_EPOCHS:-10}"
export BENCHMARK_REF="${BENCHMARK_REF:-main}"
# BENCHMARK_REPO_DIR and PIPELINE_PYTHON, if set, are inherited by the pipeline.

OLLAMA_INSTALL_DIR="$RUN_DIR/ollama-bin"

# Persistent model directory.
# Models downloaded by one job can be reused by future jobs.
OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-$ROOT_DIR/.ollama-models}"

OLLAMA_LOG="$RUN_DIR/ollama.log"

# BENCHMARK_SHARED=1: this job is one of several hosts sharing the benchmarks of one
# resumed run (submit-shared.sh). Each host has its own Ollama log, and all of them
# reuse the Ollama installation already in RUN_DIR instead of extracting it at once.
OLLAMA_REUSE=""
if [[ -n "${BENCHMARK_SHARED:-}" ]]; then
    OLLAMA_LOG="$RUN_DIR/ollama.$(hostname -s).log"
    if [[ -x "$OLLAMA_INSTALL_DIR/bin/ollama" ]] &&
            { [[ ! -e /dev/kfd ]] || compgen -G "$OLLAMA_INSTALL_DIR/lib/ollama/rocm*" >/dev/null; }; then
        OLLAMA_REUSE=1
    else
        OLLAMA_INSTALL_DIR="$RUN_DIR/ollama-bin-$(hostname -s)"
    fi
fi
OLLAMA_PID=""
PIPELINE_PID=""

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
    if [[ -n "${PIPELINE_PID:-}" ]]; then
        kill "$PIPELINE_PID" 2>/dev/null || true
        wait "$PIPELINE_PID" 2>/dev/null || true
    fi

    if [ -n "${OLLAMA_PID:-}" ]; then

        if kill -0 "$OLLAMA_PID" 2>/dev/null; then

            log "Stopping Ollama (PID $OLLAMA_PID)"

            kill "$OLLAMA_PID" || true

            wait "$OLLAMA_PID" 2>/dev/null || true

        fi

    fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM


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

elif [ -e /dev/kfd ]; then

    # larochette: 4 x AMD Instinct MI210 (64 GiB each)
    log "AMD GPU detected (/dev/kfd)"

    if command -v rocm-smi >/dev/null 2>&1; then
        rocm-smi || true
    elif command -v amd-smi >/dev/null 2>&1; then
        amd-smi list || true
    else
        ls -l /dev/kfd /dev/dri || true
    fi

else

    log "No GPU detected"

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
if [[ -n "$OLLAMA_REUSE" ]]; then
    log "Reusing the Ollama installation of this run: $OLLAMA_INSTALL_DIR"
else
    log "Downloading Ollama: $OLLAMA_ARCHIVE"
    curl -fL \
        "https://ollama.com/download/$OLLAMA_ARCHIVE" \
        -o "$OLLAMA_ARCHIVE"
fi


###############################################################################
# Extract Ollama
###############################################################################

[[ -n "$OLLAMA_REUSE" ]] || log "Extracting Ollama"


extract_zst() {

    if tar --help 2>/dev/null | grep -q zstd; then

        tar --zstd -xf "$1"

    elif command -v unzstd >/dev/null 2>&1; then

        unzstd -c "$1" | tar -xf -

    elif command -v zstd >/dev/null 2>&1; then

        zstd -dc "$1" | tar -xf -

    else

        log "ERROR: Cannot extract Ollama."
        log "zstd support was not found."
        exit 1

    fi

}


[[ -n "$OLLAMA_REUSE" ]] || extract_zst "$OLLAMA_ARCHIVE"


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


###############################################################################
# AMD GPUs: ROCm libraries of the same Ollama release
#
# The archive above only runs on NVIDIA GPUs or on the CPU. On an AMD node
# (larochette: 4 x Instinct MI210) Ollama also needs its ROCm add-on, extracted
# into the same directory; without it every model silently runs on the CPU.
###############################################################################

if [ -e /dev/kfd ] && [[ -z "$OLLAMA_REUSE" ]]; then

    # An unreachable host makes the binary report only its own (client) version.
    OLLAMA_VERSION="$(OLLAMA_HOST=127.0.0.1:1 "$OLLAMA_BIN" --version 2>&1 \
        | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | tail -n 1 || true)"

    OLLAMA_ROCM_ARCHIVE="ollama-linux-${OLLAMA_ARCH}-rocm.tar.zst"

    log "AMD GPU found: downloading $OLLAMA_ROCM_ARCHIVE (Ollama ${OLLAMA_VERSION:-latest})"

    if curl -fL --retry 5 --retry-delay 10 \
        "https://ollama.com/download/$OLLAMA_ROCM_ARCHIVE${OLLAMA_VERSION:+?version=$OLLAMA_VERSION}" \
        -o "$OLLAMA_ROCM_ARCHIVE"; then

        extract_zst "$OLLAMA_ROCM_ARCHIVE"
        rm -f "$OLLAMA_ROCM_ARCHIVE"

        log "ROCm libraries installed"

    else

        log "WARNING: ROCm download failed: the models will run on the CPU"

    fi

fi


log "1-INSTALL OLLAMA OK"


###############################################################################
# 2. Configure Ollama
###############################################################################

log "============================================================"
log "2. CONFIGURING OLLAMA"
log "============================================================"


export OLLAMA_MODELS="$OLLAMA_MODELS_DIR"
# The model directory may be shared by several jobs: never let one delete blobs.
export OLLAMA_NOPRUNE=1

export OLLAMA_HOST="127.0.0.1:11434"

# One request slot per benchmark worker, and all four models kept loaded: the
# concurrent workers are served in parallel instead of queuing or reloading models.
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-$BENCHMARK_JOBS}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-${#OLLAMA_MODELS_TO_PULL[@]}}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}"
# The benchmark requests set no num_ctx. Without a GPU Ollama answers them with a
# 4096-token context; with 64 GiB GPUs recent versions pick a far larger default,
# whose cache for 8 parallel slots would push the models out of GPU memory.
# 4096 keeps the requests exactly as on the CPU nodes.
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-4096}"

# The pipeline uses this server instead of starting its own.
export BENCHMARK_OLLAMA_HOST="$OLLAMA_HOST"


log "OLLAMA_MODELS=$OLLAMA_MODELS"

log "OLLAMA_HOST=$OLLAMA_HOST"

log "OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL OLLAMA_MAX_LOADED_MODELS=$OLLAMA_MAX_LOADED_MODELS"

log "OLLAMA_CONTEXT_LENGTH=$OLLAMA_CONTEXT_LENGTH"


log "2-CONFIGURE OLLAMA OK"


###############################################################################
# 3. Start Ollama
###############################################################################

log "============================================================"
log "3. STARTING OLLAMA SERVER"
log "============================================================"


# Per-request runner lines are filtered out and the log is capped (OLLAMA_LOG_MAX_MB),
# so a long run cannot fill the disk quota with it.
"$OLLAMA_BIN" serve > >(awk -v limit_mb="${OLLAMA_LOG_MAX_MB:-200}" \
    -f "$ROOT_DIR/scripts/ollama-log-filter.awk" > "$OLLAMA_LOG") 2>&1 &

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


# On a GPU node, Ollama reports each GPU it will use ("inference compute") once it
# has discovered them, shortly after it starts answering.
if [ -e /dev/kfd ] || command -v nvidia-smi >/dev/null 2>&1; then

    for i in $(seq 1 120); do
        grep -q 'inference compute' "$OLLAMA_LOG" 2>/dev/null && break
        sleep 1
    done

    if grep 'inference compute' "$OLLAMA_LOG" | grep -iE 'library=(rocm|cuda|vulkan)' >/dev/null; then

        log "Ollama uses these GPUs:"
        grep 'inference compute' "$OLLAMA_LOG" || true

    else

        log "WARNING: Ollama found no GPU: the models run on the CPU (much slower)"
        grep -iE 'inference compute|gpu|rocm|amdgpu' "$OLLAMA_LOG" | head -n 30 || true

    fi

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


# pipeline-benchmarks.sh builds the experiment image from the benchmark source,
# saves the benchmark package list, and owns package workers.


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


# Load every model into GPU memory now: a cold load of the 7B model from NFS
# takes longer than the benchmark client's 120 s HTTP timeout.
for model in "${OLLAMA_MODELS_TO_PULL[@]}"; do
    log "Loading model into memory: $model"
    if ! curl -fsS --max-time 1800 http://127.0.0.1:11434/api/generate \
        -d "{\"model\": \"$model\", \"prompt\": \"\", \"keep_alive\": -1}" \
        >/dev/null; then
        log "ERROR: Failed to load model: $model"
        tail -50 "$OLLAMA_LOG" || true
        exit 1
    fi
done

log "Loaded models:"

"$OLLAMA_BIN" ps


###############################################################################
# 8. Run baseline, dependency, LLM and hybrid benchmarks
###############################################################################

log "============================================================"
log "8. RUNNING BENCHMARKS"
log "============================================================"
log "Benchmark packages: $BENCHMARK_PACKAGE_COUNT; benchmarks running at the same time: $BENCHMARK_JOBS"
log "Seed: $RANKING_SEED"
log "Experiment: $EXPERIMENT_DIR"

# Run the same maintained pipeline used outside OAR: the experiment image, the saved
# benchmark packages, baseline/dependency/LLM/hybrid benchmarks and publication.
# Ollama stays owned by this wrapper. Deploy pipeline-benchmarks.sh and scripts/
# alongside job.oar.sh.
export OLLAMA_BIN OLLAMA_MODELS_DIR
# --foreground: the job must last as long as the benchmarks.
bash "$ROOT_DIR/pipeline-benchmarks.sh" --foreground "$@" &
PIPELINE_PID=$!
pipeline_status=0
wait "$PIPELINE_PID" || pipeline_status=$?
PIPELINE_PID=""
if [[ "$pipeline_status" != 0 ]]; then
    log "Benchmark pipeline failed (exit $pipeline_status)"
    exit "$pipeline_status"
fi
log "BENCHMARK PIPELINE OK"

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