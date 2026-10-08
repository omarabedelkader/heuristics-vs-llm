# Mine once, then train and benchmark

Start the complete pipeline with your chosen concurrency:

```bash
BENCHMARK_JOBS=8 ./pipeline-benchmarks.sh
```

To skip only the live neural re-ranker benchmarks, append this flag:

```bash
BENCHMARK_JOBS=8 ./pipeline-benchmarks.sh --skip-reranker-benchmarks
```

Training, validation, independent testing and learning figures still run. The
re-ranker HTTP server and NeuralRank package jobs are skipped. The results folder
then contains `results-table.tex`, `performance.png` and `dataset-summary.tex`;
older neural benchmark tables/figures are removed only after successful publication.
Without the flag, all stages run and all five benchmark files are published.
You can later resume that same run without the flag to add the neural benchmarks,
reusing its normal benchmark results and trained model. Use `--help` for usage.

The pipeline first checks for `mining/corpus/manifest.json`. If a completed dataset
exists, it verifies and reuses it without accessing Hugging Face. Otherwise, it
downloads the published [dataset snapshot](https://huggingface.co/datasets/pharo-llm/pharo-reranker-dataset/tree/904b689aed068d765a0fac4bd664e8e60773cd4c),
pinned to commit `904b689aed068d765a0fac4bd664e8e60773cd4c`. If no snapshot is
published at that revision (the checksum list returns HTTP 404), it runs
`pipeline-mine-training-data.sh` and continues only after mining succeeds.
Network, permission, checksum and incomplete-snapshot errors stop the pipeline;
they never silently trigger mining.

The download uses Python's standard library and curl; no Hugging Face login or
Python package is needed for this public dataset. About 1.04 GB is transferred.
The compressed corpus and ZIPs are verified, extracted into a temporary directory,
and checked against the original image, source and corpus fingerprints before
publication. The completed corpus is installed last. Archive launch permissions
are preserved. The current pipeline scripts are never replaced by archived code.
Download preparation uses the same mining lock as the mining script.

The published VM is **macOS ARM64**. On **Linux x86_64/OAR**, the pipeline installs
the official Linux VM of the same version (`v12.0.5-beta`, commit `7884d28`), pinned
by SHA-256 in `scripts/pharo_runtime.py`. The corpus, source and `Pharo.image` bytes
remain unchanged. `image/runtime-origin.json` records the replacement VM URL,
checksum and image checksum. This also handles an existing Mac snapshot copied to
Linux. If the official stable download changes, its checksum mismatch stops the run
until a replacement is deliberately validated and pinned. Other incompatible
platforms are rejected. Benchmark timings can differ between platforms.
Allow at least 32 GB free for restoration and training partitions, plus space for
models and package workers. Interrupted/failed downloads clean up their staging
files; existing mining files are never overwritten by a download.

A fresh experiment downloaded with no explicit `BENCHMARK_PACKAGE_COUNT` or
`RANKING_SEED` and the default ratios reuses the snapshot's saved split: **5 benchmark,
581 training, 73 validation and 73 test packages, seed 42**. Explicit count/seed or
nondefault ratio settings create a new selection using the existing workflow.
Downloaded split files remain in `mining/snapshot-selection/`, with the dataset
commit recorded in `mining/snapshot-origin.json`. Existing experiment selections
are always retained.
Mining and benchmarking use **separate copies of the same Pharo image snapshot**.
The benchmark template copies the mining image, VM and supporting files without
downloading a newer build or reinstalling code. You can still run mining separately.

An existing mining lock prevents starting a second mining process. If mining is
already running, the pipeline stops with that explanation; rerun it after mining
finishes. Incomplete staging data is never used, and invalid saved data causes an
error instead of being overwritten or silently re-mined.

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

Once mining succeeds, keep `corpus/`, `repository/` and **`image/`**. Each new
experiment needs the original mining snapshot. Transfer all three directories
under `MINING_DIR` when moving to a machine with a compatible OS and architecture.

## 2. Complete comparison pipeline

`pipeline-benchmarks.sh` performs the whole comparison workflow:

1. Checks for the saved corpus, downloads the pinned snapshot if missing (or mines
   if no remote snapshot is published), then verifies the corpus
   and copies its frozen source snapshot.
2. Copies `mining/image/` into `experiment/image/` and verifies the image checksum
   against the corpus manifest. The installed source is already in that snapshot.
3. Randomly selects and reserves the benchmark packages. Saves the selection in
   `experiment/benchmark-selection.json` and the authoritative four-part package
   split in `experiment/split.json`.
4. Divides **only the remaining packages** into 80% training, 10% validation and
   10% test (by package count, with deterministic rounding). All four groups are
   mutually disjoint. Benchmark rows are excluded from **every** model-development
   JSONL file; the saved mining corpus itself remains unchanged.
5. Copies the prepared benchmark image into a separate directory for each package
   as its job starts. Runs baseline, dependency, all four LLMs, and all four hybrid
   variants, with at most `BENCHMARK_JOBS` package processes at once.
6. After all normal benchmarks finish, trains the neural re-ranker using only
   `training.jsonl`. Measures `validation.jsonl` after every epoch and retains the
   checkpoint with the highest validation MRR@10 (first epoch wins ties).
7. Evaluates that selected checkpoint on the separate `test.jsonl`, then generates
   learning curves, test metrics and a rank-transition heatmap in PNG and PDF.
8. Starts the ONNX re-ranker and runs NeuralRank-10, 20, 30 and 50 on the reserved
   benchmark packages, using the same worker limit and independent package images.
9. Validates all package checkpoints, combines observation counts and unrounded
   measurement totals, and exports the five consolidated benchmark files below.
   Servers started by this script are stopped; an existing Ollama server is retained.

The four groups have distinct purposes:

| Group | Purpose | Used to select the model? |
| --- | --- | --- |
| Training | Fit model weights | Yes, through optimization |
| Validation | Select the epoch/checkpoint | Yes, by validation MRR@10 |
| Test | Evaluate the already-selected model offline | No |
| Benchmark | Final live comparison of completion strategies | No |

The 80/10/10 percentages apply **after** reserving benchmark packages, not to the
whole corpus. For example, 250 benchmark packages and 1,000 remaining packages
produce 800 training, 100 validation, 100 test and 250 benchmark packages. The
partition is seeded and saved before any measurements; results never cause a
redraw. At least three non-benchmark packages are required. Each model-development
partition must contain rows, and training/validation must contain positive
candidates; otherwise the run fails with an explanation.

This follows the [grouped holdout and independent test-set principles in the
scikit-learn documentation](https://scikit-learn.org/stable/modules/cross_validation.html).
80/10/10 is the chosen allocation, not a guarantee of the best accuracy. Because
splitting is by exact package name, related package families can still cross
partitions; this is a package-level generalization experiment, not a claim of
project-level independence. The test and benchmark results must not be used to
choose epochs, seeds, features or other model settings.

The local `scripts/reranker_workflow.py` imports the frozen model, feature encoder
and ONNX runtime. It passes no test or benchmark rows to training or checkpoint
selection. No benchmark-source checkout or mining snapshot is modified. Features
are encoded from the previously mined records; the pipeline does not re-mine data.

You must explicitly set `BENCHMARK_JOBS` to a positive integer. There is no
automatic CPU-based selection or default. For example, to benchmark 250 packages
with at most 8 packages running simultaneously:

```bash
BENCHMARK_PACKAGE_COUNT=250 BENCHMARK_JOBS=8 ./pipeline-benchmarks.sh
```

Choose your own job count. It controls package processes on the allocated host,
not the number of OAR allocations. The normal and re-ranker phases run separately;
the limit applies to each phase. Images retain the same loaded packages and
dependencies, but each worker benchmarks only its assigned package. VM binaries
and source archives are shared; writable images, change files, logs and working
directories are separate. Disk usage grows by roughly one image per package.

Ollama and the re-ranker service are shared by workers. Their throughput and your
CPU/GPU/RAM capacity determine the speedup; 8 jobs does not guarantee an 8× speedup.
Timing measurements include contention at the selected concurrency. Each phase
records the chosen job count in `worker-invocations.jsonl`, including on resume.

The first run uses the downloaded snapshot's saved selection when available;
otherwise it defaults to 50 randomly selected packages and seed 42. Later runs
reuse the saved selection and benchmark image. Explicit count, seed or ratio changes
fail if they disagree with the saved split. Each new training run gets a new
`experiment/reranker-run.*/` directory containing its model, split, filtered
training/validation/test files, corpus provenance, evaluation, figures and logs.

The benchmark template must match the mining-image checksum recorded in the corpus
manifest, including on reruns. Workers receive their own writable copies of that
template. A mismatch fails before selection or training; packages are never silently
dropped and data is never re-mined. Existing experiments from the former download
workflow need a new `EXPERIMENT_DIR` using the original mining snapshot.
`PHARO_DOWNLOAD_URL` only controls the initial mining-image download.

The single final output folder is **`resutls/` beside the scripts** (using the
requested spelling):

```text
resutls/
  results-table.tex
  results-table-re-ranker.tex
  performance.png
  performance-reranker.png
  dataset-summary.tex
```

`results-table.tex` and `performance.png` cover baseline, dependency, the four LLM
completion models, and their four hybrid variants. `results-table-re-ranker.tex`
and `performance-reranker.png` cover the four neural re-rankers. Both figures show completion time and Pharo memory
delta, not total Python/Ollama server RAM.

There is **one** `dataset-summary.tex`, counting unique benchmark packages,
classes, and methods using the same traversal as the benchmark. It describes the
reserved benchmark selection, not the mining, training, validation or test dataset, and does not
multiply counts by the number of strategies or completion kinds.

Outputs are staged in the run directory; all five must be nonempty before they
are copied into `resutls/`. A failed benchmark leaves the previous published files
in place. Models, logs, split and evaluation data stay under `experiment/`.
After successful publication, the former `performance-re-ranker.png` filename is
removed from the results folder in favor of `performance-reranker.png`.

`job.oar.sh` preserves the original local Ollama installation, persistent model
cache, four model downloads, server startup and job logs. Its benchmark section
now calls `pipeline-benchmarks.sh` once, which handles the complete workflow above.
Edit these two defaults near the top of `job.oar.sh`:

```bash
export BENCHMARK_PACKAGE_COUNT="${BENCHMARK_PACKAGE_COUNT:-250}"
export BENCHMARK_JOBS="${BENCHMARK_JOBS:-8}"
```

The first is the total number of reserved benchmark packages; the second is the
maximum number of simultaneous package workers on the single allocated host.
Environment settings override these defaults. For example, with a Linux host:

```bash
BENCHMARK_PACKAGE_COUNT=100 BENCHMARK_JOBS=4 bash ./job.oar.sh
```

Submit `job.oar.sh` with your usual OAR command. Deploy both pipeline scripts and
`scripts/` alongside it. Each OAR job uses `oar-runs/$OAR_JOB_ID` for its experiment
and `oar-runs/$OAR_JOB_ID/results` for the final tables and figures. Mining inputs
and downloaded models are shared across jobs through `MINING_DIR` and
`OLLAMA_MODELS_DIR`. Changing the package count requires a fresh run directory;
an existing saved selection is never silently replaced. Optional pipeline flags,
including `--skip-reranker-benchmarks`, are forwarded by the job wrapper.
An error in the pipeline propagates to the OAR job, and its Ollama server is stopped.
The wrapper keeps its Ollama server running through the pipeline; the standalone
pipeline retains its own server lifecycle as described below.

Ollama uses `127.0.0.1:11434`. If no server is running, the pipeline uses an installed
`ollama` or `OLLAMA_BIN`; on Linux it can also download and extract the official
archive locally without sudo ([Ollama installation documentation](https://docs.ollama.com/linux)).
On macOS, install Ollama or set `OLLAMA_BIN` first. Missing models are fetched via
the [Ollama pull API](https://docs.ollama.com/api/pull). An owned server is stopped
after normal benchmarks, before neural training and re-ranker measurements.

## Learning and independent test figures

The five files in `resutls/` remain the consolidated live benchmark artifacts.
Additional paper figures live in `experiment/reranker-run.*/learning/`:

- `learning-curves.png` / `.pdf`: training and validation cross-entropy, plus
  validation MRR@10 from the untrained model (epoch 0) through every training epoch.
  A dashed line marks the selected checkpoint. Loss excludes rows whose target is
  absent from the candidates; MRR includes these misses as zero. Training loss is
  the mean during each epoch; validation loss is measured at its end.
- `test-performance.png` / `.pdf`: offline test MRR@10 and accuracy@1 for the
  dependency candidate order and the selected re-ranker, across measured K values.
- `test-rank-transitions.png` / `.pdf`: counts of test targets moving between rank
  buckets before and after re-ranking, using the largest K supported by the saved
  data. This is a rank-transition heatmap, not a classifier confusion matrix.

`model/learning-history.json` stores the underlying epoch measurements and
`evaluation.json` stores test metrics, candidate recall and the heatmap counts. Resumes regenerate the figures from
these saved measurements without repeating successful training or test evaluation.
PDFs are vector exports; PNGs use 300 DPI. The figures report measured outcomes,
including no improvement or overfitting when present; they do not assume learning
succeeded. No test metric influences checkpoint selection.

## Resume an interrupted package run

The script prints its run directory before benchmarking. A package checkpoint is
published only after its process exits successfully and its full measurements pass
validation. Completed packages survive a later failure. Resume explicitly:

```bash
BENCHMARK_JOBS=8 \
BENCHMARK_RESUME_DIR="$PWD/experiment/reranker-run.ABC123" \
./pipeline-benchmarks.sh
```

Use the actual directory printed by your run, plus its original `EXPERIMENT_DIR`,
`MINING_DIR` and `RANKING_EPOCHS` if you customized them. You can choose a different
job count for the resume. It skips completed package phases and reuses a completed training/evaluation stage.
An incomplete training stage is retried with the same saved partitions and settings. A regular invocation without
`BENCHMARK_RESUME_DIR` starts a fresh training/benchmark run.

The retained layout is:

```text
experiment/reranker-run.*/
  split.json, corpus.json, training.jsonl, validation.jsonl, test.jsonl
  training-config.json, training.log, evaluation.json
  model/
  learning/                    # learning curves and independent test figures
  workers.json                 # input fingerprints and ordered package list
  model-inputs.json            # model fingerprint pinned before re-ranker jobs
  worker-invocations.jsonl      # phase/concurrency history
  packages/0001/
    package.json               # assigned package name
    image/Pharo.image          # independent writable image
    normal.json, normal.log
    reranker.json, reranker.log
  packages/0002/...
  aggregate.json               # combined counts and measurement totals
  publication.*/               # freshly staged final tables and figures
```

Resume rejects changed model, split, partition files, training settings, image or workflow scripts.
Older two-part experiment splits are rejected: use a new `EXPERIMENT_DIR`; the
existing mining corpus can still be reused.
Missing, duplicated or mismatched measurements block aggregation. Each package's
corpus is counted once, and the exporter also checks it against the full selected
corpus in the benchmark image. No package-level tables are concatenated or rounded
before aggregation. The benchmark-source repository is unchanged.

## Reuse the dataset for another random selection

No additional mining is needed; worker images are copied from the benchmark template:

```bash
MINING_DIR="$PWD/mining" \
EXPERIMENT_DIR="$PWD/experiment-100" \
BENCHMARK_PACKAGE_COUNT=100 RANKING_SEED=123 \
BENCHMARK_JOBS=8 ./pipeline-benchmarks.sh
```

Each experiment has its own benchmark image and saved selection. `MINING_DIR` and
`EXPERIMENT_DIR` must be different directories. Only one coordinator may write to a given
mining or experiment directory at a time; its package workers use separate directories.
The results directory also has a lock to prevent concurrent experiments from
mixing their output files. A successful later run replaces the five published files.

## Configuration

| Variable | Default / purpose |
| --- | --- |
| `MINING_DIR` | `mining/` beside the scripts; saved dataset, source and mining image |
| `RANKING_DATASET_REPO` | `pharo-llm/pharo-reranker-dataset`; public Hugging Face dataset checked when local corpus is missing |
| `RANKING_DATASET_REVISION` | `904b689aed068d765a0fac4bd664e8e60773cd4c`; full immutable commit SHA, never a moving branch |
| `EXPERIMENT_DIR` | `experiment/` beside the scripts; independent benchmark image and split |
| `BENCHMARK_PACKAGE_COUNT` | Snapshot's saved count (5) after download unless explicitly set; otherwise 50 for a new selection; saved count thereafter |
| `BENCHMARK_JOBS` | Required positive integer chosen by you; maximum simultaneous package processes |
| `BENCHMARK_RESUME_DIR` | Optional existing `reranker-run.*` directory; reuse its trained model and completed package phases |
| `RANKING_SEED` | 42 on the first benchmark run; saved seed thereafter |
| `RANKING_EPOCHS` | 10 training epochs; the best checkpoint is selected by validation MRR@10 |
| `RANKING_SPLIT_RATIOS` | `80,10,10`; train/validation/test percentages of non-benchmark packages, chosen before measurements |
| `RESULTS_DIR` | `resutls/` beside the scripts; one folder for all five publication files |
| `BENCHMARK_REPO_DIR` | Mining source checkout; defaults to the sibling `HeuristicCompletion-Benchmarks-Multiples` checkout |
| `BENCHMARK_REF` | `main`, when mining fetches source from GitHub instead |
| `PHARO_DOWNLOAD_URL` | `https://get.pharo.org/140+vm`; installer used only when mining locally |
| `RERANKER_PYTHON` | Optional Python environment; otherwise training creates an experiment venv |
| `OLLAMA_BIN` | Optional Ollama executable; otherwise use PATH or a local Linux installation |
| `OLLAMA_MODELS_DIR` | `.ollama-models/` beside the scripts, for servers started by this pipeline |

“All packages” means packages with methods visited by the benchmark traversal.
Packages without such methods cannot produce examples. Zero-row eligible packages
remain recorded in the manifest and split; they are never silently dropped. Package families can occur on both
sides; the holdout is by exact package name.

Datasets from the older shared-image workflow do not contain the new frozen-source
checksum. Prepare a dataset with the mining script for this two-image workflow.

## Checks

```bash
python3 -m unittest discover -s tests -v
bash -n job.oar.sh pipeline-mine-training-data.sh pipeline-benchmarks.sh
```

The shell integration tests use stand-ins for downloads, Pharo and ML execution.
They verify exact image copying, rejection of changed snapshots, saved-data reuse,
package exclusion, saved-split reuse across both comparisons, all five final
artifacts, failure handling, server cleanup, and rejection of incompatible source
or package pools. API tests check model reuse, downloads, and download failures.
Worker tests check the concurrency cap, immediate slot reuse, isolated images,
checkpoint resume, process-tree cleanup, observation-weighted aggregation, and
rejection of incomplete, duplicated or incompatible package results.

To also run the real PyTorch → ONNX → evaluation → figure smoke test, use a Python
environment with the frozen re-ranker requirements and `scripts/reranker-requirements.txt`:

```bash
RERANKER_TEST_REPOSITORY="$PWD/mining/repository" \
python3 -m unittest discover -s tests -p test_reranker_workflow.py -v
```

This uses tiny synthetic fixtures in a temporary directory, not the mined corpus
or the live benchmark packages.
