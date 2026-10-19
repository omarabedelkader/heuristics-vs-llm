# Setup for pipeline-benchmarks.sh. Source it after setting SCRIPT_DIR.
#
# An experiment (EXPERIMENT_DIR) holds a frozen copy of the benchmark source, a Pharo
# image built from it, and the SAVED list of benchmark packages
# (benchmark-selection.json). The first run creates all three; later runs reuse the
# image and the list, and load the source again only when it changed in the checkout.
# Nothing is mined or downloaded from a dataset: the benchmarks run on the image itself.
# BENCHMARK_PACKAGES_FILE supplies your own package list instead of a seeded selection.

if [[ ! "${BENCHMARK_JOBS:-}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Set BENCHMARK_JOBS to the number of simultaneous benchmarks you want (positive integer)." >&2
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
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$SCRIPT_DIR/experiment}"
mkdir -p "$EXPERIMENT_DIR"
EXPERIMENT_DIR="$(cd "$EXPERIMENT_DIR" && pwd -P)"
export RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/resutls}"
mkdir -p "$RESULTS_DIR"
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"
export REPO_DIR="$EXPERIMENT_DIR/repository"
export BENCHMARK_SELECTION_FILE="$EXPERIMENT_DIR/benchmark-selection.json"
export BENCHMARK_PACKAGE_COUNT="${BENCHMARK_PACKAGE_COUNT:-}"
export RANKING_SEED="${RANKING_SEED:-}"
PIPELINE_PYTHON="${PIPELINE_PYTHON:-python3}"
IMAGE_PIN="$EXPERIMENT_DIR/image-source.json"
WORKER_PID=""
RESULTS_LOCKED=""
CLEANUP_PIDS=()

# One pipeline at a time per experiment: concurrent runs would share CPUs and distort
# the latency and memory measurements of both. BENCHMARK_SHARED=1 is the exception:
# one pipeline per host (each its own CPUs and Ollama), sharing the benchmarks of one
# resumed run directory (scripts/shared_run.py).
BENCHMARK_SHARED="${BENCHMARK_SHARED:-}"
EXPERIMENT_LOCKED=""
if [[ -n "$BENCHMARK_SHARED" ]]; then
    if [[ -z "${BENCHMARK_RESUME_DIR:-}" ]]; then
        echo "BENCHMARK_SHARED needs BENCHMARK_RESUME_DIR: the run directory the hosts share." >&2
        exit 1
    fi
    if [[ -d "$EXPERIMENT_DIR/.running" ]]; then
        echo "Warning: $EXPERIMENT_DIR/.running exists; ignored with BENCHMARK_SHARED" \
             "(make sure no single-host pipeline runs this experiment)." >&2
    fi
elif ! mkdir "$EXPERIMENT_DIR/.running" 2>/dev/null; then
    echo "Experiment is already running: $EXPERIMENT_DIR (.running lock)" >&2
    exit 1
else
    EXPERIMENT_LOCKED=1
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
    if [[ -n "$EXPERIMENT_LOCKED" ]]; then
        rmdir "$EXPERIMENT_DIR/.running"
    fi
}
trap pipeline_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Shared hosts take this lock only to publish (publish_results).
if [[ -z "$BENCHMARK_SHARED" ]]; then
    if ! mkdir "$RESULTS_DIR/.running" 2>/dev/null; then
        echo "Results directory is already in use: $RESULTS_DIR" >&2
        exit 1
    fi
    RESULTS_LOCKED=1
fi

# The experiment's source is a frozen copy of the local checkout (or, without one, of
# the GitHub revision BENCHMARK_REF). When the code changes in the checkout, the copy
# and the experiment image follow it on the next run (scripts/benchmark_source.py).
BENCHMARK_CHECKOUT="${BENCHMARK_REPO_DIR:-$SCRIPT_DIR/../HeuristicCompletion-Benchmarks-Multiples}"
benchmark_source() {
    local action="$1"
    shift
    "$PIPELINE_PYTHON" "$SCRIPT_DIR/scripts/benchmark_source.py" "$action" --repository "$REPO_DIR" "$@"
}
if [[ ! -d "$REPO_DIR" && ( -e "$EXPERIMENT_DIR/image" || -f "$BENCHMARK_SELECTION_FILE" ) ]]; then
    echo "Frozen benchmark source is missing; restore it or use a new EXPERIMENT_DIR." >&2
    exit 1
fi
if [[ -n "$BENCHMARK_SHARED" ]]; then
    # Shared hosts resume the run with the frozen source as it is: one host changing it
    # would change the image under the benchmarks of the others.
    if [[ ! -d "$REPO_DIR" ]]; then
        echo "BENCHMARK_SHARED: the experiment has no frozen source: $REPO_DIR" >&2
        exit 1
    fi
elif [[ -d "$BENCHMARK_CHECKOUT/src" || -d "$REPO_DIR" || -n "${BENCHMARK_REPO_DIR:-}" ]]; then
    benchmark_source sync --checkout "$BENCHMARK_CHECKOUT"
else
    BENCHMARK_REF="${BENCHMARK_REF:-main}"
    source_download="$(mktemp -d "$EXPERIMENT_DIR/download.XXXXXX")"
    curl -fsSL "https://codeload.github.com/omarabedelkader/HeuristicCompletion-Benchmarks-Multiples/tar.gz/$BENCHMARK_REF" \
        -o "$source_download/repository.tar.gz"
    tar -xzf "$source_download/repository.tar.gz" --strip-components=1 -C "$source_download"
    benchmark_source sync --checkout "$source_download" --origin "GitHub ref: $BENCHMARK_REF"
    rm -rf "$source_download"
fi

if [[ ! -e "$EXPERIMENT_DIR/image" ]]; then
    if [[ -f "$BENCHMARK_SELECTION_FILE" || -f "$IMAGE_PIN" ]]; then
        echo "The experiment image is missing; restore it or use a new EXPERIMENT_DIR." >&2
        exit 1
    fi
    echo "Creating the experiment's Pharo image and installing the frozen benchmark source"
    # An interrupted attempt stays in image-pending.* and is never used as the image.
    rm -rf "$EXPERIMENT_DIR"/image-pending.*
    image_stage="$(mktemp -d "$EXPERIMENT_DIR/image-pending.XXXXXX")"
    (
        cd "$image_stage"
        curl -fsSL "${PHARO_DOWNLOAD_URL:-https://get.pharo.org/140+vm}" -o get-pharo.sh
        bash get-pharo.sh
        ./pharo --headless Pharo.image eval --save "
Metacello new
    repository: 'tonel://', ((OSEnvironment current at: 'REPO_DIR') asFileReference / 'src') fullName;
    baseline: 'ExtendedHeuristicCompletionBenchmarks';
    load
"
    )
    mv "$image_stage" "$EXPERIMENT_DIR/image"
    benchmark_source pin-image --image "$EXPERIMENT_DIR/image/Pharo.image" --pin "$IMAGE_PIN"
fi
if [[ ! -f "$EXPERIMENT_DIR/image/Pharo.image" || ! -x "$EXPERIMENT_DIR/image/pharo" ]]; then
    echo "The experiment image is incomplete; restore it or use a new EXPERIMENT_DIR." >&2
    exit 1
fi
image_status=0
benchmark_source check-image --image "$EXPERIMENT_DIR/image/Pharo.image" --pin "$IMAGE_PIN" || image_status=$?
if [[ "$image_status" != 0 && "$image_status" != 5 ]]; then
    exit "$image_status"
fi
if [[ "$image_status" == 5 && -n "$BENCHMARK_SHARED" ]]; then
    echo "BENCHMARK_SHARED: the experiment image does not hold the frozen source; it cannot be resumed." >&2
    exit 1
fi
if [[ "$image_status" == 5 ]]; then
    # The frozen source differs from the code in this image: load it, removing what it dropped.
    echo "Loading the frozen benchmark source into the experiment image"
    (cd "$EXPERIMENT_DIR/image" && ./pharo --headless Pharo.image eval --save "
| source |
source := (OSEnvironment current at: 'REPO_DIR') asFileReference / 'src'.
#('ExtendedHeuristicCompletion-Benchmarks' 'ExtendedHeuristicCompletion-Benchmarks-Tests') do: [ :name |
    (PackageOrganizer default hasPackage: name) ifTrue: [
        MCPackageLoader new
            updatePackage: (MCPackage named: name)
            withSnapshot: (TonelReader on: source fileName: name) snapshot;
            load ] ].
")
    benchmark_source pin-image --image "$EXPERIMENT_DIR/image/Pharo.image" --pin "$IMAGE_PIN"
fi
cd "$EXPERIMENT_DIR/image"

# The saved benchmark package list: your own file, else a new seeded selection made in Pharo.
if [[ -n "${BENCHMARK_PACKAGES_FILE:-}" ]]; then
    if [[ -n "$BENCHMARK_PACKAGE_COUNT" ]]; then
        echo "BENCHMARK_PACKAGES_FILE already lists the packages; unset BENCHMARK_PACKAGE_COUNT." >&2
        exit 1
    fi
    export ELIGIBLE_PACKAGES_FILE="$EXPERIMENT_DIR/eligible-packages.json"
    ./pharo --headless Pharo.image eval "
(OSEnvironment current at: 'ELIGIBLE_PACKAGES_FILE') asFileReference writeStreamDo: [ :out |
    out truncate; nextPutAll: (STONJSON toString: CooBenchRunner eligiblePackageNames asArray); lf ].
" > /dev/null
    "$PIPELINE_PYTHON" "$SCRIPT_DIR/scripts/package_benchmarks.py" select \
        --packages "$BENCHMARK_PACKAGES_FILE" --eligible "$ELIGIBLE_PACKAGES_FILE" \
        --output "$BENCHMARK_SELECTION_FILE" --seed "${RANKING_SEED:-42}"
fi

# Select (or check) and save benchmark names in the experiment image.
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
                Error signal: 'Package count differs from the saved selection; use a new EXPERIMENT_DIR' ] ].
        seedText ifNotEmpty: [
            seedText asNumber = split seed ifFalse: [
                Error signal: 'Seed differs from the saved selection; use a new EXPERIMENT_DIR' ] ] ]
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
    nextPutAll: 'Saved selection: ', split benchmarkPackageNames size asString, ' benchmark packages'; lf;
    nextPutAll: file fullName; lf.
"
echo "Benchmark packages (saved): $BENCHMARK_SELECTION_FILE"

# Create a new run directory, or check an explicit resume. Its selection.json is the
# experiment's saved list; a run never benchmarks other packages.
prepare_run_directory() {
    if [[ -n "${BENCHMARK_RESUME_DIR:-}" ]]; then
        RUN_DIR="$BENCHMARK_RESUME_DIR"
        if [[ "$(dirname "$RUN_DIR")" != "$EXPERIMENT_DIR" || "$(basename "$RUN_DIR")" != benchmark-run.* ||
              ! -f "$RUN_DIR/workers.json" ]]; then
            echo "BENCHMARK_RESUME_DIR must be a prepared benchmark-run.* directory inside EXPERIMENT_DIR." >&2
            exit 1
        fi
        if ! cmp -s "$BENCHMARK_SELECTION_FILE" "$RUN_DIR/selection.json"; then
            echo "The saved benchmark packages differ from this run's; use a new run." >&2
            exit 1
        fi
    else
        RUN_DIR="$(mktemp -d "$EXPERIMENT_DIR/benchmark-run.XXXXXX")"
        cp "$BENCHMARK_SELECTION_FILE" "$RUN_DIR/selection.json"
    fi
    export RUN_DIR
    export PUBLICATION_DIR="$(mktemp -d "$RUN_DIR/publication.XXXXXX")"
    echo "Run directory: $RUN_DIR"
    echo "Benchmarks running at the same time (BENCHMARK_JOBS): $BENCHMARK_JOBS"
}

run_packages() {
    local runner=package_benchmarks.py
    [[ -z "$BENCHMARK_SHARED" ]] || runner=shared_run.py
    "$PIPELINE_PYTHON" "$SCRIPT_DIR/scripts/$runner" run \
        --run "$RUN_DIR" --image "$EXPERIMENT_DIR/image" --jobs "$BENCHMARK_JOBS" &
    WORKER_PID=$!
    local status=0
    wait "$WORKER_PID" || status=$?
    WORKER_PID=""
    if [[ "$status" != 0 ]]; then
        echo "Benchmarks failed. Resume with BENCHMARK_RESUME_DIR=\"$RUN_DIR\" and your BENCHMARK_JOBS setting." >&2
        return "$status"
    fi
}

# Aggregate the packages, export in Pharo and publish only complete sets of files:
#   RESULTS_DIR/results-before-pharo-validation  LLM completion answers what the model generates
#   RESULTS_DIR/results-after-pharo-validation   LLM completion answers only real Pharo names
# Baseline, Dependency and Hybrid are the same measurements in both. pharo-validation.tex
# (before) counts the generated candidates that are real Pharo names.
publish_results() {
    local name set
    if [[ -n "$BENCHMARK_SHARED" ]]; then
        # Every host ends with the complete run; the first one to get here publishes.
        if ! mkdir "$RESULTS_DIR/.running" 2>/dev/null; then
            echo "Another host of the run is publishing the results to $RESULTS_DIR"
            return 0
        fi
        RESULTS_LOCKED=1
    fi
    "$PIPELINE_PYTHON" "$SCRIPT_DIR/scripts/package_benchmarks.py" aggregate \
        --run "$RUN_DIR" --image "$EXPERIMENT_DIR/image"
    local publication="$PUBLICATION_DIR"
    for set in before after; do
        export BENCHMARK_AGGREGATE="$RUN_DIR/aggregate-$set.json"
        export PUBLICATION_DIR="$publication/$set"
        mkdir -p "$PUBLICATION_DIR"
        ./pharo --headless Pharo.image --no-default-preferences eval "$(cat "$SCRIPT_DIR/scripts/benchmark-export.st")" > /dev/null
        for name in "$@"; do
            if [[ ! -s "$PUBLICATION_DIR/$name" ]]; then
                echo "Missing or empty benchmark artifact: $PUBLICATION_DIR/$name" >&2
                exit 1
            fi
        done
    done
    cp "$RUN_DIR/pharo-validation.tex" "$publication/before/pharo-validation.tex"
    for set in before after; do
        mkdir -p "$RESULTS_DIR/results-$set-pharo-validation"
        cp "$publication/$set"/* "$RESULTS_DIR/results-$set-pharo-validation/"
        echo "Published ($set Pharo validation): $(ls "$publication/$set" | tr '\n' ' ')-> $RESULTS_DIR/results-$set-pharo-validation"
    done
    export PUBLICATION_DIR="$publication"
}
