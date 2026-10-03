#!/usr/bin/env bash
set -e

echo "Creating Image"
# 1. Download Pharo + VM
mkdir -p image
cd image
curl -fsSL https://get.pharo.org/140+vm | bash

echo "1-DOWNLOAD IMAGE OK"


echo "Installation Repository"
# 2. Install your repository into the image
./pharo Pharo.image eval --save "
Metacello new
  githubUser: 'omarabedelkader' project: 'HeuristicCompletion-Benchmarks-Multiples' commitish: 'main' path: 'src';
  baseline: 'ExtendedHeuristicCompletionBenchmarks';
  load

"

echo "2-INSTALL REPO OK"


echo "Run Benchmark"
# 3. Run the benchmarks
./pharo Pharo.image eval --save "

| comparison files |
comparison := CooBenchRunner necTests.
files := CooBenchRunner
    export: comparison
    to: '/Users/omar/Desktop/HeuristicCompletion-Benchmarks-Multiples/benchmark-results'.
files inspect.
"


echo "3-RUN BENCHMARKS OK"