#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/pipeline-common.sh"

echo "Running normal benchmarks on the saved benchmark packages"
./pharo --headless Pharo.image eval "
| split comparison files |
split := CooBenchmarkSplit readFrom: (OSEnvironment current at: 'BENCHMARK_SPLIT_FILE').
comparison := CooBenchRunner normalBenchmarksForSplit: split.
files := CooBenchRunner export: comparison to: (OSEnvironment current at: 'RESULTS_DIR').
files do: [ :file | Stdio stdout nextPutAll: file fullName; lf ].
"
echo "Normal benchmarks complete: $RESULTS_DIR"
