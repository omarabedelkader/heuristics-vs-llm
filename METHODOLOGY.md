# Methodology: heuristic, LLM, and hybrid completion in Pharo

This document describes the experiment implemented in this workspace, using the frozen source in `oar-runs/336588/repository/src` and the Python, shell, and Smalltalk worker scripts. It describes what the implementation actually measures, distinguishes completed results from planned measurements, and identifies limitations that matter when interpreting the results.

Writing this document does not execute benchmarks or change their code, saved image, configuration, or checkpoints. The experiment may continue running independently.

## 1. Objective and research questions

The experiment evaluates how accurately different systems recover an existing Pharo name from a typed prefix and its code context, and what execution time and Pharo memory change accompany that completion.

The comparisons address four questions:

1. How do a simple heuristic baseline and a dependency-aware heuristic configuration compare with language-model completion?
2. How does performance vary across the 0.5B, 1.5B, 3B, and 7B model variants?
3. What changes when LLM search is constrained to names present in the Pharo image?
4. Can an LLM improve the ordering of the dependency heuristic's suggestions without changing their candidate set?

The evaluation is a static reconstruction experiment over existing source code. It does not measure developer productivity, acceptance of suggestions in an IDE, or correctness of newly generated programs. The ground truth is the name already present at the selected source location.

## 2. Experimental units and design

Three different units must be distinguished:

| Unit | Meaning |
| --- | --- |
| Package | A selected package from the frozen Pharo image |
| Benchmark task | One package × one completion kind × one strategy |
| Observation | One eligible AST occurrence × one tested prefix length, evaluated by a strategy |

A task can contain many observations. Finishing a task creates one checkpoint; it does not necessarily finish the package.

There are two completion kinds:

- **Methods**, stored as `messages`: complete the selector of a message-send AST node.
- **Classes**, stored as `variables`: complete a class name referenced by a variable AST node with a resolved global binding whose value is a class.

Despite the internal name `variables`, the second category is not a general local-variable-name completion benchmark. Its eligibility check also requires an uppercase initial character and excludes non-class globals and uppercase locals/class variables.

For each kind, the worker measures 14 strategy configurations:

| Family | Configurations | Number |
| --- | --- | ---: |
| Heuristics | Baseline; Dependency | 2 |
| LLM before Pharo validation | 0.5B; 1.5B; 3B; 7B | 4 |
| LLM after Pharo validation | 0.5B; 1.5B; 3B; 7B | 4 |
| Dependency candidates reranked by an LLM | 0.5B; 1.5B; 3B; 7B | 4 |
| **Total per kind** | | **14** |

Consequently, each package requires `2 × 14 = 28` tasks. With 50 packages, the complete experiment requires **1,400 tasks**. Each finished task retains seven rows, corresponding to prefix lengths 2–8, so a complete run contains 9,800 task/prefix rows. These rows are aggregates, not individual observations; some rows can have zero observations.

The final before-validation and after-validation reports each show 10 strategies: two heuristics, four LLMs, and four hybrids. The two reports reuse the same heuristic and hybrid measurements. Only the four standalone LLM configurations change between them.

## 3. Corpus construction and package selection

### 3.1 Source of the corpus

The corpus comes from packages already loaded in the experiment's Pharo image. This pipeline does not mine or download a separate evaluation dataset. It prepares an image containing the benchmark implementation and evaluates that image's own packages.

The authoritative traversal is `CoBenchmarkPackage>>methodsDo:`. The frozen benchmark source calls this image-provided implementation rather than defining an independent traversal. Corpus reporting follows the same traversal, so its method count is not a claim about every method present in the repository or every extension method registered in the image.

The corpus summary counts selected packages, their defined non-trait classes, and the unique methods visited by the benchmark traversal. The saved per-package corpus summaries for this experiment give:

| Entity | Count |
| --- | ---: |
| Selected packages | 50 |
| Defined non-trait classes | 585 |
| Traversed methods | 5,538 |

These are corpus entities, not completion-observation counts. A method can contain multiple target occurrences, and each target can produce several prefix observations.

### 3.2 Eligible pool and sampling

`CooBenchRunner>>eligiblePackageNames` selects loaded packages for which `CoBenchmarkPackage>>methodsDo:` visits at least one method. Package names are deduplicated and sorted before sampling.

The selection algorithm repeatedly chooses a random index from the remaining sorted names and removes that name. This samples packages without replacement. The random-number generator is Pharo's `Random`, initialized with the saved seed.

For this experiment, `benchmark-selection.json` records:

| Setting | Value |
| --- | ---: |
| Eligible packages | 733 |
| Selected benchmark packages | 50 |
| Remaining packages in the `train` partition | 683 |
| Selection seed | 42 |

Sampling is at package level, not stratified by package size, application domain, test status, or number of completion sites. Tests and baseline packages can therefore appear in the selected corpus.

### 3.3 Meaning of the saved `train` partition

The selection file partitions eligible packages into `benchmark` and `train`, with no overlap. It verifies that their union is the eligible pool. However, **the active 1,400-task pipeline does not train a model on the 683 remaining packages**. It uses already available Ollama models.

The partition name does not establish that these benchmark packages were excluded from the language models' pretraining or supervised fine-tuning data. That requires separate model-training provenance, which this pipeline does not provide.

### 3.4 Reuse of the selection

The selection is saved once and reused on resume. Workers check that the saved eligible set matches the image and that their assigned package belongs to the benchmark set. Changing the package pool or selection is not treated as a continuation of the same experiment.

## 4. Creating completion observations

The benchmark traverses method ASTs and selects message nodes or eligible class-reference nodes, depending on the completion kind. It keeps the original selector or class name as the expected answer.

For a target of length `L`, the prefix lengths are:

```text
2, 3, ..., min(L, 8)
```

Thus a target contributes `max(0, min(L, 8) - 1)` observations per strategy. Names shorter than two characters contribute none. For a name of length at most eight, the longest tested prefix is the entire name; the benchmark therefore includes fully typed-name cases, not only incomplete names.

For example, the target `OrderedCollection` contributes the prefixes `Or`, `Ord`, `Orde`, `Order`, `Ordere`, `Ordered`, and `OrderedC`. A three-character target contributes only its two- and three-character prefixes.

The benchmark takes at most ten candidates and searches for the exact expected string. Matching is case-sensitive. A missing answer receives rank zero internally. The same selector appearing at several call sites is evaluated several times; observations are not deduplicated by target string.

For heuristic completion, the target AST field is temporarily replaced with the prefix, the completion builder is called, and the original field is restored. Message positions use the first keyword interval; class references use the variable-node position. LLM completion constructs a source snapshot instead of using the expected answer as an input field.

The common target sites and prefix rules support comparisons across strategies. They do not imply that heuristic AST context and LLM textual context carry identical information.

## 5. Heuristic strategies

Both heuristic families use `CoASTHeuristicsResultSetBuilder`. Results are filtered with `CoCaseSensitiveBeginsWithFilter` and limited to the first ten entries.

### 5.1 Baseline

The message baseline installs `CoUnknownMessageHeuristic` as its message heuristic. The class-name baseline installs `CoGlobalVariablesHeuristic` as its variables heuristic.

These are concrete Pharo heuristic configurations. The benchmark does not define the baseline as random ranking, an alphabetical list, or an LLM fallback. The internal implementations of the `Co...` heuristics are dependencies supplied by the image; the frozen experiment source explicitly specifies which ones it composes.

### 5.2 Dependency-aware messages

The message configuration composes the following heuristics in this order:

1. `CoSelfMessageHeuristic`
2. `CoSuperMessageHeuristic`
3. `CoTypedReceiverMessageHeuristic`
4. `CoInitializeInferencedMessageHeuristic`
5. `CoLiteralMessageHeuristic`
6. `CoGlobalVariableMessageHeuristic`
7. `CoVariableWithTypeNameMessageHeuristic`
8. `CoDependencyMessageHeuristics`
9. `CoUnknownMessageHeuristic`

This configuration combines receiver/context-sensitive heuristics with dependency-aware completion and an unknown-message fallback. Its ordering above is the builder configuration, not a separately learned ranking function.

### 5.3 Dependency-aware class names

The variables builder composes:

1. `CoLocalScopeVariablesHeuristic`
2. `CoWorkspaceVariablesHeuristic`
3. `CoInstanceAccessibleVariablesHeuristic`
4. `CoGlobalVariablesHeuristic` configured with `CoDependencyGlobalVariableFetcher`
5. A final ordinary `CoGlobalVariablesHeuristic`

Although this builder can propose names from different scopes, the benchmark's expected answers in this category are class references. A returned candidate only scores as correct if it exactly matches the expected class name.

## 6. LLM models and request protocol

### 6.1 Model identities

The runner overrides the client's generic default model with these explicit tags:

| Configuration suffix | Ollama model tag |
| --- | --- |
| `05B` | `pharo-llm/Qwen2.5-Coder-SFT:0.5B` |
| `15B` | `pharo-llm/Qwen2.5-Coder-SFT:1.5B` |
| `3B` | `pharo-llm/Qwen2.5-Coder-SFT:3b` |
| `7B` | `pharo-llm/Qwen2.5-Coder-SFT:7b` |

These are nominal size labels. The original job logs report F16 model artifacts. For strict reproduction, preserve the actual model blobs and logged digests, not only tag strings: the checkpoint manifest does not pin Ollama model digests.

### 6.2 Fill-in-the-middle snapshot

The context policy is `fim-structural-v1`. Each snapshot contains:

- Source text before the target selector/name, followed by the typed prefix.
- A suffix beginning after the original AST node.
- Structural class context: class name, superclass, instance-variable names, and class-variable names.

For a message node, the suffix begins after the entire message node, excluding its held-out selector text and arguments in that node. For a class-reference node, it begins after the variable node. Structural context does not include an additional copy of the method body.

The prompt template is:

```text
{{ .Context }}<|fim_prefix|>{{ .Prompt }}<|fim_suffix|>{{ .Suffix }}<|fim_middle|>
```

Line endings are normalized to LF. Template substitution is single-pass so template-like text inside source is treated as literal data.

The expected answer is retained in benchmark records for scoring, but it is not supplied as a privileged answer to the completer or reranker. This removes direct inclusion of the held-out target span; it does not guarantee the same string cannot occur naturally elsewhere in surrounding code, a declaration, or the model's training data.

### 6.3 Ollama requests

`CooLLMClient` uses synchronous HTTP requests to `/api/generate`. The search repeatedly asks for next-token alternatives; it does not request a natural-language list of ten suggestions.

| Request setting | Implemented value |
| --- | --- |
| `raw` | `true` |
| `stream` | `false` |
| `num_predict` | 1 token per request |
| `temperature` | 0 |
| `seed` | 42 |
| `num_ctx` | 8192 |
| `keep_alive` | `5m` in the request payload |
| `logprobs` | `true` for search requests |
| `top_logprobs` | 20 by default |
| HTTP timeout | 120 seconds |
| HTTP retries | 0 |

The stop list contains end-of-text, FIM, repository, and file-separator markers. The client requires a completed response and rejects responses reporting more than one generated token.

The job scripts also configure server-level context length and keep-alive defaults. In particular, a logged server context default of 4096 is not the request payload: the client explicitly requests `num_ctx = 8192`. Effective behavior should be checked against the actual Ollama version and logs rather than conflating those two settings.

If Ollama returns an invisible end token without log-probabilities, the client represents it as a certain end, with log-probability zero. This is an implementation approximation. Other missing or invalid responses raise errors rather than silently becoming incorrect completions.

## 7. Standalone LLM search

### 7.1 Search parameters

`CooLLMBeamCompleter` uses:

| Parameter | Default |
| --- | ---: |
| Beam width | 10 |
| Next-token alternatives per request | 20 |
| Maximum search steps | 12 |
| Maximum returned candidates | 10 |

Each beam represents text generated after the typed prefix. A step queries the model after the FIM prompt plus that beam's continuation. Token log-probabilities are exponentiated and multiplied into path probabilities. Paths reaching the same continuation are combined by summing their probability mass.

Name characters are letters, digits, and underscores, with colons additionally permitted for message selectors. Binary selectors use special characters. A candidate is finished when an end marker is returned or a token reaches a non-name character. The score therefore accounts for both generating the remaining name and ending it, avoiding an automatic preference for a shorter name simply because it is a prefix of another name.

Only the strongest ten continuing beams survive a step. Once ten completed candidates exist, beams unable to exceed the tenth candidate's probability are dropped. Search also stops when no beam remains or the 12-step limit is reached. Candidate ties use lexical ordering.

The output can contain fewer than ten suggestions. A short candidate list is still scored on the original observation, including a miss if the target is absent. Bounded search and the limited top-token interface mean this is not an exhaustive ranking of all possible names.

### 7.2 Before Pharo validation

Strategies `llmUnvalidated05B`, `llmUnvalidated15B`, `llmUnvalidated3B`, and `llmUnvalidated7B` set `validating: false`.

The image vocabulary does not restrict beam expansion or acceptance of a finished name. Search follows syntactically permitted name characters and can return names absent from the image. The vocabulary is still used afterward to count how many returned candidates are real image names; that diagnostic does not alter their ranking.

These are raw name-generation results under the implemented beam search, not unrestricted generation of arbitrary program text.

### 7.3 After Pharo validation

Strategies `llm05B`, `llm15B`, `llm3B`, and `llm7B` set `validating: true`.

The vocabulary consists of class names from `Smalltalk allClasses`, or selectors defined on their instance and class sides. Names are indexed by their first two characters. During search, a continuation is retained only if it can still lead to a matching image name; finished candidates must belong to that vocabulary.

**The after-validation experiment is a separate constrained search, not merely removal of invalid entries from the raw top ten.** Vocabulary constraints can change beam survival, search requests, time, and the final candidate list. The two variants execute separately and have distinct checkpoints.

Validation establishes name existence in the image. It does not prove that a selector is understood by the particular receiver, that a class is appropriate at the call site, or that the resulting code is semantically correct. It does not execute the suggestion or compile a generated program.

## 8. Hybrid reranking

For each observation, `CooHybridBenchmarks` first obtains the dependency configuration's top ten prefix-filtered candidates. It then passes those names and the FIM snapshot to `CooLLMBeamReranker`.

The reranker adds a Smalltalk comment listing the candidates to the structural context. Its context policy is `fim-structural-v1+candidates`. It scores candidate continuations from next-token log-probabilities, including an end-of-name factor.

Unlike the standalone beam completer, the reranker has no beam-width pruning of candidate-consistent paths. It still sees only 20 token alternatives per request and stops after at most 12 steps. If a required next token or ending is absent from the returned alternatives, the implementation assigns a bound-based score using the lowest listed log-probability and records the candidate as bounded. Remaining paths at the step limit also produce bounds. These scores should not be described as exact full-sequence likelihoods for every candidate.

Names are deduplicated while preserving their original order. Candidates with scores are ordered by descending score; on equal scores, exact results precede bounds, with input order resolving remaining ties. Names that do not receive a score are retained afterward in original order.

The essential invariant is that reranking retains the input candidate set. Therefore, on matching observations and candidate lists:

- It can improve or worsen MRR and Top-1/2/3.
- It cannot rescue a target missing from the dependency candidate set.
- Top-10 recall should equal dependency Top-10 recall when the same unique top-ten set is used.

Different package coverage in a partial report can obscure this invariant. Aggregate values from different subsets should not be compared as if they were paired observations.

The hybrid records promotions, demotions, unchanged ranks, and heuristic misses in memory. The ordinary checkpoint export retains aggregate ranking/timing statistics rather than those full diagnostic records.

## 9. Metrics and aggregation

### 9.1 Exact-match ranking

Let `r_i` be the position of the expected answer among the returned top ten for observation `i`. A missing target has no successful rank.

```text
RR_i = 1 / r_i    if the target is at rank 1..10
RR_i = 0          otherwise

MRR = sum(RR_i) / N
```

This is reciprocal rank truncated by the ten-candidate evaluation window. The implementation selector is named `mmrForPrefixSize:`, but it computes mean reciprocal rank, conventionally abbreviated **MRR**.

A rank-one answer contributes 1, rank two contributes 0.5, rank ten contributes 0.1, and a miss contributes zero.

### 9.2 Top-k accuracy / Recall@k

```text
Top-k = 100 × number of observations with rank in 1..k / N
```

The exported thresholds are 1, 2, 3, and 10. With one expected name per observation, this Top-k accuracy equals Recall@k. It does not measure precision over multiple acceptable completions or reward alternative semantically reasonable answers.

### 9.3 Candidate validity

For each prefix, standalone LLM tasks retain the number of candidates returned and the number present in the image vocabulary:

```text
Validity (%) = 100 × valid returned candidates / returned candidates
```

This denominator is returned candidate names, not observations, internal beam expansions, tokens, or HTTP requests. At most ten names are returned per observation. An image-valid candidate can still be the wrong answer.

The validation table uses unvalidated LLM measurements. In validated tasks, every returned candidate must be image-valid, and checkpoint validation requires `valid == generated`. If no candidates are returned, the validity ratio is undefined and should be shown as unavailable, not interpreted as successful validation.

### 9.4 Latency

Each observation starts a millisecond-clock measurement before prefix/context preparation and completion. The timer ends after ranking and accounting. Depending on strategy, it therefore includes AST/prompt preparation, heuristic candidate construction, synchronous HTTP waiting, multiple token-search requests, and bookkeeping.

Mean latency is total recorded milliseconds divided by observation count. It is not pure model GPU compute time or total task wall time. Process startup, some traversal/setup work, checkpoint writing, and publication are outside the per-observation measure. The log's task duration measures a broader interval.

No dedicated warm-up phase or repeated-trial schedule is implemented in this worker pipeline. Model loading, cache state, worker contention, and server scheduling can influence measured latency. Temperature zero and a fixed seed do not establish identical timing or identical numerical behavior across hardware and runtimes.

### 9.5 Memory

The memory measurement is:

```text
memory delta_i = Smalltalk vm memorySize after completion
                 - Smalltalk vm memorySize before completion

mean memory delta (MB) = sum(memory delta_i) / N / 1,000,000
```

It is a change in the Pharo VM's reported memory size, not a peak resident-set measurement, allocation profiler, or total system-memory footprint. Zero and negative deltas are possible and are preserved. The measure excludes the separate Ollama server's model memory and GPU memory. In-memory benchmark diagnostic records can contribute to the Pharo-side change.

### 9.6 Weighting

Workers retain sufficient statistics per package, kind, strategy, and prefix. Aggregation sums these statistics before dividing by total observation count. Thus MRR, Top-k, mean time, and mean memory are **observation-weighted**, not equally weighted averages of package percentages.

For example, packages with 100 and 1,000 observations have weights proportional to 100 and 1,000. Prefix-length averages likewise pool observations; shorter targets do not contribute to longer-prefix rows.

Zero-observation rows do not supply evidence of poor or perfect accuracy. Missing tasks are different from completed tasks with zero observations and must never be silently substituted with zeros.

### 9.7 Statistics not supplied by the final checkpoints

The standard pipeline does not export confidence intervals, significance tests, per-observation paired ranks, or latency distributions. Although some benchmark objects provide additional metrics or keep latency samples in memory, `part.json` stores totals and hit counts. Exact medians, percentiles, and observation-level resampling cannot be reconstructed from those totals alone.

## 10. Execution, isolation, and resume

### 10.1 Frozen experiment inputs

The experiment directory contains a frozen source repository, a prepared Pharo image, an image/source hash file, a package selection, and the benchmark-run directory.

`image-source.json` associates the image hash with the frozen source hash. `workers.json` pins the selection, `Pharo.image`, and three worker/export scripts using SHA-256. Resume rejects changes in these pinned inputs or the selected packages.

The copied source's recorded origin is `GitHub ref: main`; that label alone is not an immutable Git revision. The saved files and content hashes provide the stronger identity for this run.

The resume manifest does not pin everything that can affect an experiment. In particular, it does not hash the Pharo VM executable or launcher, Ollama executable, model artifacts, GPU/CPU, operating system, or all shell orchestration scripts. A successful resume integrity check is therefore not proof of an identical execution environment.

### 10.2 Worker isolation

Every task runs in its own Pharo process with a private copy of the image and changes file. Source files are linked for access. Workers do not save back to the template image. Each task has its own log and checkpoint location.

`BENCHMARK_JOBS` controls the number of simultaneous tasks. Each worker queries the local Ollama endpoint configured for that invocation. The pipeline can start its own server, with parallel request slots derived from the worker count, and can reuse already downloaded models.

The task order prioritizes 7B, then 3B, 1.5B, and 0.5B; within each size it visits unvalidated completion, validated completion, then hybrid. Baseline and dependency tasks follow. This is an operational schedule, not a randomized experimental order.

### 10.3 Checkpoint granularity

Each task writes `part.pending.json`. After the child exits successfully, Python validates the task identity, corpus, seven prefix rows, totals, and hit counts, then promotes the file to `part.json`.

Resume checks completed checkpoints and skips them. If interrupted midway through a task, observations inside that task are not independently checkpointed: that unfinished task is rerun. The pipeline does not need to rerun the completed tasks for the same package.

The message `Packages left in this run: 50/50` means no package has completed all 28 tasks yet. It does not imply that no checkpoints have completed.

### 10.4 Shared Grid5000 execution

`shared_run.py` allows several hosts to work on the same run directory. A task is claimed with an atomic claim-file creation, refreshed every 60 seconds. The default lease is 900 seconds; expired claims can be taken over after a host stops refreshing them. Completed checkpoints remain the basis for deciding what work is still necessary.

`submit-shared.sh` submits OAR jobs against an existing experiment. It checks for queued/running benchmark jobs before its resume submission and handles the old top-level locks. This scheduler script is distinct from the local macOS pipeline.

### 10.5 Errors and stopping

HTTP and search errors propagate. A failed completion is not silently converted to rank zero and no heuristic fallback disguises the failure. A failing task does not become a completed checkpoint; other saved task results remain reusable.

The default local launcher detaches the pipeline from the terminal and writes to a log. Ctrl+C in the log-viewing wrapper stops viewing rather than stopping the detached benchmark. The printed `kill <PID>` command requests termination; the Python worker runner handles termination and stops active worker process groups. Completed checkpoints remain available.

## 11. Aggregation and final artifacts

Once all 28 parts for a package exist, they are merged into `result.json`. Package merging verifies consistent corpus counts. Full aggregation requires valid complete results for every selected package.

The pipeline creates `aggregate-before.json`, `aggregate-after.json`, and `aggregate.json` in the benchmark-run directory. In the before view, unvalidated LLM strategies are renamed to the corresponding publication labels; in the after view, the validated LLM strategies are selected. Heuristic and hybrid rows are shared.

The Smalltalk export stage consumes these aggregate totals without issuing new inference requests. It independently checks the corpus against the image, generates publication artifacts in a staging directory, and publishes the expected files to the results folders.

```text
results/
  results-before-pharo-validation/
    results-table.tex
    recall-top.tex
    performance.png
    dataset-summary.tex
    pharo-validation.tex
  results-after-pharo-validation/
    results-table.tex
    recall-top.tex
    performance.png
    dataset-summary.tex
```

| Artifact | Contents |
| --- | --- |
| `results-table.tex` | MRR by strategy, completion kind, and prefix length, with an observation-weighted average |
| `recall-top.tex` | Top-1, Top-2, Top-3, and Recall@10 percentages pooled over packages/prefixes |
| `performance.png` | Mean Pharo memory change versus mean completion latency |
| `dataset-summary.tex` | Counts of packages, classes, and traversed methods |
| `pharo-validation.tex` | Raw-generation candidate counts and image-validity percentages |

In the standard pipeline, `pharo-validation.tex` is published only in the **before-validation** folder. The separately requested partial export supplied earlier included that diagnostic in both folders; this does not mean the after-validation pipeline performs an additional validity experiment.

An empty results directory while tasks remain unfinished is expected: intermediate data live under the run's `packages` directory. Producing a partial export is possible from completed checkpoints, but it is a separate report and must identify incomplete coverage.

## 12. Interpretation and validity limits

### 12.1 Partial coverage

A task-completion percentage is not a percentage of completed runtime or observations. Packages and strategies have unequal costs and unequal numbers of sites. A partial table can overrepresent smaller or faster tasks. Report package coverage for each strategy/kind and avoid treating unmatched subsets as a fair head-to-head comparison.

### 12.2 Hardware migration in this run

The original logs describe Linux Grid5000 hosts using AMD Instinct MI210 GPUs and eight concurrent workers per host. The local continuation shown in this session uses a macOS ARM64 Pharo VM, a local Ollama server, and two workers.

Continuing from checkpoints preserves saved work, but combines observations collected in different environments. Since strategies are scheduled in a fixed size-based order, hardware can become confounded with strategy: earlier models may be measured mainly on Grid5000 and later work mainly on the Mac.

Consequently, a combined timing or memory chart must not be presented as a controlled same-machine comparison. For such a claim, equivalent workloads need measurements on a common environment, or results must be separated by environment using available provenance. The ordinary `part.json` schema does not include a hardware identity per observation; invocation logs and timing files are supplementary evidence, not a replacement for detailed provenance.

### 12.3 Dataset and target limitations

The corpus is one image's selected packages, including tests and infrastructure. It is not a random sample of all Pharo projects or interactive editing sessions. Repeated call sites and multiple prefixes from the same occurrence are correlated. A large raw observation count should not be interpreted as that many independent experimental subjects.

Exact-match scoring credits the original name only. Alternative useful suggestions receive no credit. Fully typed-name cases are included, and one-character names are excluded by the minimum prefix length. Both choices affect the population being measured.

### 12.4 Vocabulary and context limitations

Validated completion knows the whole image's names, including names in the benchmark packages. This is an explicit closed-vocabulary completion setting, not an unseen-name generalization test. Image existence does not establish contextual applicability.

The held-out source span is removed from the FIM input, but declarations and surrounding source may naturally contain the answer. Heuristics operate on parsed/resolved ASTs, while LLMs receive a textual snapshot. Do not describe these as identical-context systems or claim absence of all possible leakage.

### 12.5 Search and model limitations

Top-20 token access, beam width, the 12-step bound, end-marker approximation, and hybrid bound-based scores influence results. “LLM completion” here means this particular decoding procedure with these model artifacts, not every possible decoding strategy for the same model family.

The code contains neural-ranker, adaptive-hybrid, and training-related classes, but they are not included in the active 14-strategy worker list. Their presence is not evidence that they were trained or evaluated in this experiment.

### 12.6 Claims supported by these artifacts

Complete results support descriptive comparisons of exact-name ranking and the measured runtime/memory quantities under the recorded setup. They do not by themselves establish statistical significance, real-world productivity gains, peak model-memory efficiency, causal benefits of parameter count alone, or absence of model-training contamination.

## 13. Reproducibility record

The stable experiment identifiers inspected for this document are:

| Item | Value |
| --- | --- |
| Experiment directory | `oar-runs/336588` |
| Run directory | `benchmark-run.zrN2cB` |
| Run ID | `477f009225f7484b9abe840395059d2a` |
| Split schema | `coo-package-split-v1` |
| Worker schema | `coo-package-workers-v4` |
| Checkpoint schema | `coo-package-part-v1` |
| Package-result schema | `coo-package-result-v3` |
| Seed | 42 |
| Frozen image SHA-256 | `5af77f5f30b5cb5079177708868e3b046c8d6543b16765ca26cece3d77a95859` |
| Frozen source SHA-256 | `881597d7d9a2be16e6d238685ce5464b7cabb719b4656dd87eb1b28c15db41f0` |

A reproducibility archive should retain the selection, frozen source, image and source files, manifests, model tags and digests, Ollama and VM versions, environment settings, hardware information, logs, timings, and completed parts. The current source hashes protect important inputs but do not replace this broader environment record.

## 14. Source map

The following local files establish the implementation details above. These references point to the frozen source used by this experiment rather than a possibly changing upstream branch.

| Topic | Source |
| --- | --- |
| Selection, model mapping, corpus | [CooBenchRunner](oar-runs/336588/repository/src/ExtendedHeuristicCompletion-Benchmarks/CooBenchRunner.class.st), [CooBenchmarkSplit](oar-runs/336588/repository/src/ExtendedHeuristicCompletion-Benchmarks/CooBenchmarkSplit.class.st) |
| Message heuristic configurations | [CooBenchRunnerMessage](oar-runs/336588/repository/src/ExtendedHeuristicCompletion-Benchmarks/CooBenchRunnerMessage.class.st) |
| Class-name heuristic configurations | [CooBenchRunnerVariables](oar-runs/336588/repository/src/ExtendedHeuristicCompletion-Benchmarks/CooBenchRunnerVariables.class.st) |
| Observation construction | [CooStaticBenchmarksMessage](oar-runs/336588/repository/src/ExtendedHeuristicCompletion-Benchmarks/CooStaticBenchmarksMessage.class.st), [CooStaticBenchmarksVariables](oar-runs/336588/repository/src/ExtendedHeuristicCompletion-Benchmarks/CooStaticBenchmarksVariables.class.st) |
| Metric accounting | [CooStaticBenchmarks](oar-runs/336588/repository/src/ExtendedHeuristicCompletion-Benchmarks/CooStaticBenchmarks.class.st) |
| FIM context | [CooLLMContextSnapshot](oar-runs/336588/repository/src/ExtendedHeuristicCompletion-Benchmarks/CooLLMContextSnapshot.class.st) |
| HTTP protocol | [CooLLMClient](oar-runs/336588/repository/src/ExtendedHeuristicCompletion-Benchmarks/CooLLMClient.class.st) |
| LLM observation loop and search | [CooLLMBenchmarks](oar-runs/336588/repository/src/ExtendedHeuristicCompletion-Benchmarks/CooLLMBenchmarks.class.st), [CooLLMBeamCompleter](oar-runs/336588/repository/src/ExtendedHeuristicCompletion-Benchmarks/CooLLMBeamCompleter.class.st) |
| Hybrid observation loop and reranking | [CooHybridBenchmarks](oar-runs/336588/repository/src/ExtendedHeuristicCompletion-Benchmarks/CooHybridBenchmarks.class.st), [CooLLMBeamReranker](oar-runs/336588/repository/src/ExtendedHeuristicCompletion-Benchmarks/CooLLMBeamReranker.class.st) |
| Task-level retained statistics | [benchmark-package.st](scripts/benchmark-package.st) |
| Integrity, execution, aggregation | [package_benchmarks.py](scripts/package_benchmarks.py), [benchmark_source.py](scripts/benchmark_source.py) |
| Multi-host task claims | [shared_run.py](scripts/shared_run.py), [submit-shared.sh](submit-shared.sh) |
| Local setup, Ollama, publication | [pipeline-benchmarks.sh](pipeline-benchmarks.sh), [pipeline-common.sh](scripts/pipeline-common.sh), [ollama_models.py](scripts/ollama_models.py) |
| Export tables and charts | [benchmark-export.st](scripts/benchmark-export.st), [CooBenchmarkChart](oar-runs/336588/repository/src/ExtendedHeuristicCompletion-Benchmarks/CooBenchmarkChart.class.st) |

## Appendix: selected packages

The saved selection order is:

```text
Network-Mail
Spec2-ListView
System-Settings-Core
NewTools-Core
SystemCommands-MethodCommands-Tests
Spec2-Code-Commands
NewTools-RewriterTools-Backend-Tests
Refactoring-Conditions
Tool-Profilers-Tests
Zinc-Resource-Meta-Tests
Iceberg-Libgit-Filetree
Slot-Tests
BaselineOfEnlumineurFormatter
Text-Scanning
SUnit-Rules-Tests
BaselineOfSlot
HeuristicCompletion-Benchmarks-Tests
FFI-Kernel
Calypso-SystemTools-QueryBrowser
Morphic-Widgets-Scrolling
NewTools-DependencyAnalyser-Tests
Reflectivity-Tests
Epicea
System-Localization-Tests
Tool-DependencyAnalyser-UI-Tab
Commander2
FormCanvas-Tests
BaselineOfSystemFinalization
BaselineOfCommander2
Debugging-Utils-Tests
Roassal-Animation
NewTools-WindowManager-Tests
Calypso-NavigationModel-Tests
TaskIt
RottenTestsFinder-FakeTests
Keymapping-Tests
BaselineOfBaseLibraries
Hiedra
BaselineOfEquals
Hiedra-Tests
Refactoring-Environment-Tests
Athens-Cairo
BaselineOfReflectionMirrors
System-Finalization-Tests
BaselineOfGeometry
BaselineOfMonticello
DrTests-TestCoverage-Tests
System-OSEnvironments
Morphic-Widgets-Windows
Collections-Support
```
