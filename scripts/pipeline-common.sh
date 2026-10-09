# Shared setup for pipeline-benchmarks.sh and pipeline-reranker.sh. Source it after
# setting SCRIPT_DIR and RUN_PREFIX (benchmark-run or reranker-run).
#
# Both pipelines use the same dataset, frozen source, benchmark image and SAVED list of
# benchmark packages (EXPERIMENT_DIR/benchmark-selection.json). Whichever pipeline runs
# first creates the list; every later run of either pipeline reuses it, so both
# benchmark exactly the same packages and the re-ranker never trains on them.
# BENCHMARK_PACKAGES_FILE supplies your own list instead (see import-selection).

if [[ ! "${BENCHMARK_JOBS:-}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Set BENCHMARK_JOBS to the number of simultaneous package jobs you want (positive integer)." >&2
    exit 1
fi
export BENCHMARK_JOBS
if [[ -n "${BENCHMARK_RESUME_DIR:-}" ]]; then
    BENCHMARK_RESUME_DIR="$(cd "$BENCHMARK_RESUME_DIR" && pwd -P)"
fi
if [[ -n "${BENCHMARK_PACKAGES_FILE:-}" ]]; then
    if [[ ! -f "$BENCHMARK_PACKAGES_FILE" ]]; then
        echo "BENCHMARK_PACKAGES_FILE does not exist: $BENCHMARK_PACKAGES_FILE" >&2
        exit 1
    fi
    BENCHMARK_PACKAGES_FILE="$(cd "$(dirname "$BENCHMARK_PACKAGES_FILE")" && pwd -P)/$(basename "$BENCHMARK_PACKAGES_FILE")"
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
export BENCHMARK_SELECTION_FILE="$EXPERIMENT_DIR/benchmark-selection.json"
export BENCHMARK_PACKAGE_COUNT="${BENCHMARK_PACKAGE_COUNT:-}"
export RANKING_SEED="${RANKING_SEED:-}"
PIPELINE_PYTHON="${RERANKER_PYTHON:-python3}"
WORKER_PID=""
RESULTS_LOCKED=""
CLEANUP_PIDS=()

# One pipeline at a time per experiment: concurrent runs would share CPUs and distort
# the latency and memory measurements of both.
if ! mkdir "$EXPERIMENT_DIR/.running" 2>/dev/null; then
    echo "Experiment is already running: $EXPERIMENT_DIR (.running lock)" >&2
    exit 1
fi
pipeline_cleanup() {
    local pid
    for pid in ${WORKER_PID:-} "${CLEANUP_PIDS[@]:-}"; do
        [[ -n "$pid" ]] || continue
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
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
    "$PIPELINE_PYTHON" "$SCRIPT_DIR/scripts/download_snapshot.py" "$MINING_DIR" || snapshot_status=$?
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
"$PIPELINE_PYTHON" "$SCRIPT_DIR/scripts/ranking_corpus.py" verify \
    "$RANKING_CORPUS_DIR" --repository "$MINING_DIR/repository"
if [[ ! -d "$REPO_DIR" ]]; then
    if [[ -f "$EXPERIMENT_DIR/image-ready" || -f "$BENCHMARK_SELECTION_FILE" ]]; then
        echo "Frozen benchmark source is missing; restore it or use a new EXPERIMENT_DIR." >&2
        exit 1
    fi
    source_stage="$(mktemp -d "$EXPERIMENT_DIR/source.XXXXXX")"
    cp -R "$MINING_DIR/repository/." "$source_stage/"
    mv "$source_stage" "$REPO_DIR"
fi
# This small source check also rejects an incompatible repository on reruns.
"$PIPELINE_PYTHON" - "$SCRIPT_DIR/scripts" "$RANKING_CORPUS_DIR" "$REPO_DIR" <<'PYCODE'
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from ranking_corpus import load_manifest
load_manifest(Path(sys.argv[2]), repository=Path(sys.argv[3]))
PYCODE

if [[ ! -e "$EXPERIMENT_DIR/image" ]]; then
    if [[ -f "$BENCHMARK_SELECTION_FILE" || -f "$EXPERIMENT_DIR/image-ready" ]]; then
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
    "$PIPELINE_PYTHON" - "$SCRIPT_DIR/scripts" "$RANKING_CORPUS_DIR" "$image_stage/Pharo.image" <<'PYCODE'
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
"$PIPELINE_PYTHON" - "$SCRIPT_DIR/scripts" "$RANKING_CORPUS_DIR" "$EXPERIMENT_DIR/image/Pharo.image" <<'PYCODE'
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from ranking_corpus import load_manifest
load_manifest(Path(sys.argv[2]), image=Path(sys.argv[3]))
PYCODE
"$PIPELINE_PYTHON" "$SCRIPT_DIR/scripts/pharo_runtime.py" "$EXPERIMENT_DIR/image"
touch "$EXPERIMENT_DIR/image-ready"
cd "$EXPERIMENT_DIR/image"

# The saved benchmark package list: your own file, else the published snapshot's
# selection for a default experiment, else a new seeded selection made in Pharo.
if [[ -n "${BENCHMARK_PACKAGES_FILE:-}" ]]; then
    if [[ -n "$BENCHMARK_PACKAGE_COUNT" ]]; then
        echo "BENCHMARK_PACKAGES_FILE already lists the packages; unset BENCHMARK_PACKAGE_COUNT." >&2
        exit 1
    fi
    "$PIPELINE_PYTHON" "$SCRIPT_DIR/scripts/ranking_corpus.py" import-selection \
        "$RANKING_CORPUS_DIR" --packages "$BENCHMARK_PACKAGES_FILE" \
        --output "$BENCHMARK_SELECTION_FILE" --seed "${RANKING_SEED:-42}"
elif [[ ! -e "$BENCHMARK_SELECTION_FILE" &&
      -f "$MINING_DIR/snapshot-selection/benchmark-selection.json" &&
      -z "$BENCHMARK_PACKAGE_COUNT" && -z "$RANKING_SEED" ]]; then
    cp "$MINING_DIR/snapshot-selection/benchmark-selection.json" "$BENCHMARK_SELECTION_FILE"
    echo "Using the snapshot's saved benchmark selection"
fi

# Select (or check) and save benchmark names in the independent benchmark image.
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
echo "Benchmark packages (saved, shared by both pipelines): $BENCHMARK_SELECTION_FILE"

# Create a new run directory, or check an explicit resume. Its split.json is the
# authoritative four-way split derived from the saved list: benchmark packages are
# never in train, validation or test.
prepare_run_directory() {
    local split_options=("$@")
    if [[ -n "${BENCHMARK_RESUME_DIR:-}" ]]; then
        RUN_DIR="$BENCHMARK_RESUME_DIR"
        if [[ "$(dirname "$RUN_DIR")" != "$EXPERIMENT_DIR" || "$(basename "$RUN_DIR")" != "$RUN_PREFIX".* ||
              ! -f "$RUN_DIR/workers.json" ]]; then
            echo "BENCHMARK_RESUME_DIR must be a prepared $RUN_PREFIX.* directory inside EXPERIMENT_DIR." >&2
            exit 1
        fi
    else
        RUN_DIR="$(mktemp -d "$EXPERIMENT_DIR/$RUN_PREFIX.XXXXXX")"
    fi
    # On a resume this only checks that the saved split still matches the settings.
    "$PIPELINE_PYTHON" "$SCRIPT_DIR/scripts/ranking_corpus.py" split \
        "$RANKING_CORPUS_DIR" --selection "$BENCHMARK_SELECTION_FILE" \
        --output "$RUN_DIR/split.json" ${split_options[@]+"${split_options[@]}"}
    export RUN_DIR
    export PUBLICATION_DIR="$(mktemp -d "$RUN_DIR/publication.XXXXXX")"
    echo "Run directory: $RUN_DIR"
    echo "Package concurrency selected by you: $BENCHMARK_JOBS"
}

run_package_phase() {
    "$PIPELINE_PYTHON" "$SCRIPT_DIR/scripts/package_benchmarks.py" run \
        --run "$RUN_DIR" --image "$EXPERIMENT_DIR/image" \
        --phase "$1" --jobs "$BENCHMARK_JOBS" &
    WORKER_PID=$!
    local status=0
    wait "$WORKER_PID" || status=$?
    WORKER_PID=""
    if [[ "$status" != 0 ]]; then
        echo "Package phase failed. Resume with BENCHMARK_RESUME_DIR=\"$RUN_DIR\" and your BENCHMARK_JOBS setting." >&2
        return "$status"
    fi
}

# Aggregate one phase, export it in Pharo and publish only a complete set of files.
publish_phase() {
    local phase="$1"
    shift
    local name
    export BENCHMARK_AGGREGATE="$RUN_DIR/aggregate.json"
    "$PIPELINE_PYTHON" "$SCRIPT_DIR/scripts/package_benchmarks.py" aggregate \
        --run "$RUN_DIR" --image "$EXPERIMENT_DIR/image" --phase "$phase"
    ./pharo --headless Pharo.image --no-default-preferences eval "$(cat "$SCRIPT_DIR/scripts/benchmark-export.st")"
    # The upstream re-ranker exporter uses an extra hyphen; publish the requested filename.
    if [[ -f "$PUBLICATION_DIR/performance-re-ranker.png" ]]; then
        mv "$PUBLICATION_DIR/performance-re-ranker.png" "$PUBLICATION_DIR/performance-reranker.png"
    fi
    for name in "$@"; do
        if [[ ! -s "$PUBLICATION_DIR/$name" ]]; then
            echo "Missing or empty benchmark artifact: $PUBLICATION_DIR/$name" >&2
            exit 1
        fi
    done
    for name in "$@"; do
        cp "$PUBLICATION_DIR/$name" "$RESULTS_DIR/$name"
    done
    echo "Published $#: $* -> $RESULTS_DIR"
}

print_runtime_summary() {
    if [[ -f "$RUN_DIR/estimate.json" ]]; then
        "$PIPELINE_PYTHON" -c 'import json, sys; e = json.load(open(sys.argv[1])); print("Predicted total: %.1f h; this invocation took: %.1f h (per-package times: timings.jsonl)" % (e["totalSeconds"] / 3600, int(sys.argv[2]) / 3600))' \
            "$RUN_DIR/estimate.json" "$SECONDS"
    fi
}
