#!/usr/bin/env bash
# Resume an interrupted benchmark run on several larochette hosts at the same time.
#
# Each host is its own OAR reservation running job.oar.sh with BENCHMARK_SHARED=1 on
# the same run directory: the hosts share the benchmarks still to run, and completed
# benchmarks are never run again. Run it again with the same RUN_DIR to continue
# after the reservations end; the first host to find the run complete publishes it.
#
# Usage: ./submit-shared.sh RUN_DIR [HOSTS] [START] [END]
#   RUN_DIR  experiment of the interrupted job, e.g. oar-runs/336588
#   HOSTS    number of hosts (default 3)
#   START    reservation start (default today 19:00:00; "now" submits right away)
#   END      reservation end (default 09:00:00 the day after START)
# BENCHMARK_JOBS (default 8) is the number of benchmarks per host.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 1 || "$1" == -h || "$1" == --help ]]; then
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    exit $(( $# < 1 ))
fi
RUN_DIR="$(cd "$1" && pwd -P)"
HOSTS="${2:-3}"
START="${3:-$(date '+%Y-%m-%d') 19:00:00}"
BENCHMARK_JOBS="${BENCHMARK_JOBS:-8}"

runs=("$RUN_DIR"/benchmark-run.*/)
if [[ ${#runs[@]} -ne 1 || ! -f "${runs[0]}workers.json" ]]; then
    echo "Expected exactly one prepared benchmark-run.* directory in $RUN_DIR" >&2
    exit 1
fi
RESUME_DIR="${runs[0]%/}"

if [[ "$START" == now ]]; then
    start_epoch="$(date '+%s')"
    reservation=()
else
    start_epoch="$(date -d "$START" '+%s')"
    reservation=(-r "$(date -d "@$start_epoch" '+%Y-%m-%d %H:%M:%S')")
fi
if [[ -n "${4:-}" ]]; then
    end_epoch="$(date -d "$4" '+%s')"
else
    end_epoch="$(date -d "$(date -d "@$start_epoch" '+%Y-%m-%d') 09:00:00 tomorrow" '+%s')"
fi
seconds=$(( end_epoch - start_epoch ))
if (( seconds <= 0 )); then
    echo "The reservation ends before it starts" >&2
    exit 1
fi
walltime="$(printf '%d:%02d:00' $(( seconds / 3600 )) $(( seconds % 3600 / 60 )))"

# A job killed at its walltime leaves its locks behind; refuse while one still runs.
if oarstat -u "$USER" 2>/dev/null | grep -q 'heuristic-completion'; then
    echo "A heuristic-completion job of yours is still queued or running (oarstat -u):" >&2
    echo "wait for it to end, or delete it, before resuming this run." >&2
    exit 1
fi
rmdir "$RUN_DIR/.running" "$RUN_DIR/results/.running" 2>/dev/null || true

parts="$(find "$RESUME_DIR/packages" -name part.json | wc -l)"
echo "Run: $RESUME_DIR ($parts benchmarks already done; they are kept)"
echo "Hosts: $HOSTS, $BENCHMARK_JOBS benchmarks each; walltime $walltime${reservation:+ from ${reservation[1]}}"
for ((host = 1; host <= HOSTS; host++)); do
    oarsub -q default -p larochette -l "host=1,walltime=$walltime" "${reservation[@]}" \
        -n heuristic-completion-benchmark \
        -O "$ROOT_DIR/heuristic-completion.%jobid%.out" \
        -E "$ROOT_DIR/heuristic-completion.%jobid%.err" \
        "BENCHMARK_SHARED=1 BENCHMARK_JOBS=$BENCHMARK_JOBS RUN_DIR='$RUN_DIR' \
BENCHMARK_RESUME_DIR='$RESUME_DIR' '$ROOT_DIR/job.oar.sh'"
done
echo "Follow them: tail -f $ROOT_DIR/heuristic-completion.<jobid>.out"
