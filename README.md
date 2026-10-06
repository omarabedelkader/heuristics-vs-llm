# Shared-package benchmark experiment

Choose the benchmark package count once, then run both pipelines:

```bash
BENCHMARK_PACKAGE_COUNT=50 ./pipeline-normal-bench.sh
./pipeline-re-ranker-bench.sh
```

The first run creates `experiment/split.json` before collecting examples or
running benchmarks. It randomly selects exactly 50 eligible packages (seed 42
by default). The normal baseline, dependency, LLM completion, and LLM hybrid
benchmarks all use those packages. The second command reads the same selection,
trains the re-ranker on **every remaining eligible package**, and benchmarks the
four neural ranking strategies on the original 50 packages.

Replace `50` with the number of packages you want. Re-ranking produces both
`results-table-re-ranker.tex` and **`performance-re-ranker.png`** in
`experiment/results/`. The separate image plots average completion time (ms)
against average Pharo memory change per completion (MB) for the four re-rankers,
for both Methods and Classes. It uses the already collected measurements and
keeps the normal benchmark's `performance.png` separate. Memory here is the
Pharo VM delta, not the Python scoring server's total RAM.

Training uses a fixed number of epochs. Benchmark packages never enter training
or validation, and test examples are collected into a separate file only after
training. Training rejects rows from packages outside the saved training set.
The re-ranking runner also checks that the serving model has the same split.
Packages with no completion examples are listed in model metadata.

Both scripts reuse the same image and a frozen copy of the benchmark source.
By default, source comes from the sibling
`../HeuristicCompletion-Benchmarks-Multiples` checkout. If it is absent, source
is fetched from GitHub's `neural-ranking` branch. The local checkout works before
publishing changes; remote use requires the updated branch on GitHub.

Results are written to `experiment/results/`. Each re-ranker training run keeps
its model, split, training/test corpora, evaluation and log in a fresh
`experiment/reranker-run.*/` directory. Re-running a pipeline reuses the split;
changing the requested count or seed fails instead of silently redrawing it.
Each experiment must run one pipeline at a time.

For a new selection, use a new experiment directory for **both** commands:

```bash
export EXPERIMENT_DIR="$PWD/experiment-100"
BENCHMARK_PACKAGE_COUNT=100 RANKING_SEED=123 ./pipeline-normal-bench.sh
RANKING_EPOCHS=10 ./pipeline-re-ranker-bench.sh
```

Configuration:

| Variable | Default / purpose |
| --- | --- |
| `BENCHMARK_PACKAGE_COUNT` | 50 for a new experiment; reuse the saved count thereafter |
| `RANKING_SEED` | 42 for a new experiment; reuse the saved seed thereafter |
| `RANKING_EPOCHS` | 10; choose before inspecting benchmark results |
| `EXPERIMENT_DIR` | `experiment/` beside the scripts |
| `RESULTS_DIR` | `$EXPERIMENT_DIR/results` |
| `BENCHMARK_REPO_DIR` | Local benchmark checkout to snapshot for a new experiment |
| `BENCHMARK_REF` | `neural-ranking`, when fetching source from GitHub |
| `RERANKER_PYTHON` | Optional existing Python environment; otherwise create an experiment venv |

`TRAIN_PACKAGE_COUNT` and `VALIDATION_PACKAGE_COUNT` no longer control this
workflow: all non-benchmark packages are used for fixed-epoch training. Only
packages containing methods visited by the benchmark traversal are eligible.
This is a package holdout; related package families may occur on both sides.

The normal benchmarks require the existing four Ollama models. The re-ranker
pipeline installs its Python requirements and starts/stops its local ONNX
service automatically. For interactive Pharo use after Metacello loading, see
the benchmark repository's **Shared package holdout** instructions.
