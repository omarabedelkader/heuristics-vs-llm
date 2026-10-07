# Mine once, then train and benchmark

There are exactly two shell scripts:

```bash
./pipeline-mine-training-data.sh
./pipeline-benchmarks.sh
```

The first downloads a Pharo image for mining. The second downloads **another,
independent Pharo image** for benchmarking and reads the saved dataset. It never
runs or copies the mining image.

## 1. Mining

`pipeline-mine-training-data.sh` downloads Pharo into `mining/image/`, installs a
frozen snapshot of the benchmark code, and mines examples from **every eligible
package**. It saves:

- `mining/corpus/all.jsonl`: all examples, with their package names.
- `mining/corpus/packages.json`: the complete package list.
- `mining/corpus/manifest.json`: package counts, corpus checksum, source checksum,
  and mining-image provenance.
- `mining/repository/`: the source snapshot needed by the benchmark pipeline.

Mining does not select benchmark packages or train the model. Re-running the script
verifies and reuses a completed dataset. Publication happens only after a complete
export succeeds; interrupted attempts remain under `corpus-pending.*` and are not
used. Mining needs Python 3, curl and tar; it does not need Ollama or ML dependencies.

Once mining succeeds, keep `corpus/` and `repository/`. **The mining image is no
longer needed to train or benchmark.** These two directories can also be transferred
to another machine under its `MINING_DIR`.

## 2. Complete comparison pipeline

`pipeline-benchmarks.sh` performs the whole comparison workflow:

1. Verifies the saved corpus and copies its frozen source snapshot.
2. Downloads its own Pharo image into `experiment/image/` and installs that source.
3. Randomly selects benchmark packages and saves their names, the seed, and the
   remaining training package names in `experiment/split.json`.
4. Filters the saved dataset: benchmark packages go only to `test.jsonl`; every
   other package goes to `training.jsonl`.
5. Installs Python dependencies, trains for fixed epochs, and evaluates held-out data.
6. Reuses or starts Ollama and downloads missing models named by the frozen source.
7. Runs baseline heuristics, dependency heuristics, LLM completion (0.5B, 1.5B,
   3B, 7B), and hybrid dependency + each LLM on the saved benchmark packages.
8. Starts the ONNX re-ranker, verifies its serving model, and benchmarks
   NeuralRank-10, 20, 30 and 50 on those exact same packages.
9. Verifies the two comparisons report identical package/class/method counts and
   publishes the five files below together. Servers started by the script are
   stopped; a pre-existing Ollama server is left running.

Training receives only the training file and the saved split. Evaluation happens
after training. Neither training data nor test data is mined during a benchmark
run. Actual benchmark measurements run in the independent benchmark image.

The first run defaults to 50 randomly selected packages and seed 42. Later runs
reuse the saved selection and benchmark image. Explicit count or seed changes
fail if they disagree with the saved split. Each training run gets a new
`experiment/reranker-run.*/` directory containing its model, split, filtered
training/test files, corpus provenance, evaluation and log.

The independently downloaded image must have the same eligible package names as
the dataset, and use the same frozen benchmark source. A mismatch fails before
training; it never silently drops packages or re-mines data. Both scripts default
to the Pharo 14 downloader. If the upstream image changes between downloads, use
`PHARO_DOWNLOAD_URL` for a compatible/pinned Pharo image installer, or prepare a
new dataset for that image. The image files themselves do not need identical hashes.

The single final output folder is **`resutls/` beside the scripts** (using the
requested spelling):

```text
resutls/
  results-table.tex
  results-table-re-ranker.tex
  performance.png
  performance-re-ranker.png
  dataset-summary.tex
```

`results-table.tex` and `performance.png` cover baseline, dependency, the four LLM
completion models, and their four hybrid variants. The `-re-ranker` files cover
the four neural re-rankers. Both figures show completion time and Pharo memory
delta, not total Python/Ollama server RAM.

There is **one** `dataset-summary.tex`, counting unique benchmark packages,
classes, and methods using the same traversal as the benchmark. It describes the
held-out benchmark selection, not the mining or training dataset, and does not
multiply counts by the number of strategies or completion kinds.

Outputs are staged in the run directory; all five must be nonempty before they
are copied into `resutls/`. A failed benchmark leaves the previous published files
in place. Models, logs, split and evaluation data stay under `experiment/`.

The previous standalone normal-benchmark, shared setup, and OAR job scripts have
been removed. OAR directives now live in
`pipeline-benchmarks.sh`, so it can be submitted directly using your usual OAR
command. Deploy the `scripts/` directory alongside both shell scripts.

Ollama uses `127.0.0.1:11434`. If no server is running, the pipeline uses an installed
`ollama` or `OLLAMA_BIN`; on Linux it can also download and extract the official
archive locally without sudo ([Ollama installation documentation](https://docs.ollama.com/linux)).
On macOS, install Ollama or set `OLLAMA_BIN` first. Missing models are fetched via
the [Ollama pull API](https://docs.ollama.com/api/pull). An owned server is stopped
after normal benchmarks, before re-ranker measurements.

## Reuse the dataset for another random selection

No image copying or additional mining is needed:

```bash
MINING_DIR="$PWD/mining" \
EXPERIMENT_DIR="$PWD/experiment-100" \
BENCHMARK_PACKAGE_COUNT=100 RANKING_SEED=123 \
./pipeline-benchmarks.sh
```

Each experiment has its own benchmark image and saved selection. `MINING_DIR` and
`EXPERIMENT_DIR` must be different directories. Only one run may write to a given
mining or experiment directory at a time.
The results directory also has a lock to prevent concurrent experiments from
mixing their output files. A successful later run replaces the five published files.

## Configuration

| Variable | Default / purpose |
| --- | --- |
| `MINING_DIR` | `mining/` beside the scripts; saved dataset, source and mining image |
| `EXPERIMENT_DIR` | `experiment/` beside the scripts; independent benchmark image and split |
| `BENCHMARK_PACKAGE_COUNT` | 50 on the first benchmark run; saved count thereafter |
| `RANKING_SEED` | 42 on the first benchmark run; saved seed thereafter |
| `RANKING_EPOCHS` | 10; choose before inspecting benchmark results |
| `RESULTS_DIR` | `resutls/` beside the scripts; one folder for all five publication files |
| `BENCHMARK_REPO_DIR` | Mining source checkout; defaults to the sibling `HeuristicCompletion-Benchmarks-Multiples` checkout |
| `BENCHMARK_REF` | `main`, when mining fetches source from GitHub instead |
| `PHARO_DOWNLOAD_URL` | `https://get.pharo.org/140+vm`; installer used independently by each script |
| `RERANKER_PYTHON` | Optional Python environment; otherwise training creates an experiment venv |
| `OLLAMA_BIN` | Optional Ollama executable; otherwise use PATH or a local Linux installation |
| `OLLAMA_MODELS_DIR` | `.ollama-models/` beside the scripts, for servers started by this pipeline |

“All packages” means packages with methods visited by the benchmark traversal.
Packages without such methods cannot produce examples. Zero-row eligible packages
remain recorded in the manifest and split. Package families can occur on both
sides; the holdout is by exact package name.

Datasets from the older shared-image workflow do not contain the new frozen-source
checksum. Prepare a dataset with the mining script for this two-image workflow.

## Checks

```bash
python3 -m unittest discover -s tests -v
bash -n pipeline-mine-training-data.sh pipeline-benchmarks.sh
```

The shell integration tests use stand-ins for downloads, Pharo and ML execution.
They verify separate image downloads, saved-data reuse without the mining image,
package exclusion, saved-split reuse across both comparisons, all five final
artifacts, failure handling, server cleanup, and rejection of incompatible source
or package pools. API tests check model reuse, downloads, and download failures.
