#!/usr/bin/env bash
set -e

# 1. Download Pharo + VM
mkdir -p image
cd image
curl -fsSL https://get.pharo.org/140+vm | bash

echo "1-DOWNLOAD IMAGE OK"

# 2. Install your repository into the image
./pharo Pharo.image eval --save "
Metacello new
  githubUser: 'omarabedelkader' project: 'HeuristicCompletion-Benchmarks' commitish: 'main' path: 'src';
  baseline: 'ExtendedHeuristicCompletionBenchmarks';
  load

"

echo "2-DOWNLOAD REPO OK"


# 3. Run the benchmarks
./pharo Pharo.image eval --save "CooStaticBenchmarksVariablesSorter nec."