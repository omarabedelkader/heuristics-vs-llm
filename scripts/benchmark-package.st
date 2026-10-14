"Run ONE benchmark (BENCHMARK_KIND x BENCHMARK_STRATEGY) of one package and retain
sufficient statistics, without rounding. The pipeline runs the 28 benchmarks of every
package as separate, concurrent workers and merges their parts.
LLM completion runs twice: before Pharo validation (the candidates are what the model
generates, real names or not) and after it (only real names of the image)."
| selection packageName package strategies unvalidated labelFor kinds kind strategy benchmark started rows output |
selection := STONJSON fromString: (OSEnvironment current at: 'BENCHMARK_RUN_SELECTION') asFileReference contents.
(selection at: 'schema') = 'coo-package-split-v1' ifFalse: [ Error signal: 'Saved benchmark selection required' ].
(selection at: 'eligible') asSet = CooBenchRunner eligiblePackageNames asSet ifFalse: [
    Error signal: 'Worker package pool differs from the saved selection' ].
packageName := OSEnvironment current at: 'BENCHMARK_PACKAGE'.
((selection at: 'benchmark') includes: packageName) ifFalse: [
    Error signal: 'Worker package is not in the saved benchmark selection' ].
unvalidated := Dictionary newFrom: {
    #llmUnvalidated05B -> #llm05B. #llmUnvalidated15B -> #llm15B.
    #llmUnvalidated3B -> #llm3B. #llmUnvalidated7B -> #llm7B }.
strategies := #(heuristicsBaseline heuristicsDependency
    llmUnvalidated05B llmUnvalidated15B llmUnvalidated3B llmUnvalidated7B
    llm05B llm15B llm3B llm7B hybrid05B hybrid15B hybrid3B hybrid7B).
(strategies reject: [ :each | unvalidated includesKey: each ]) = CooBenchmarkChart publicationStrategies ifFalse: [
    Error signal: 'The published strategies differ from the ones this script measures' ].
labelFor := [ :strategy |
    (unvalidated includesKey: strategy)
        ifTrue: [ (CooBenchmarkChart labelFor: (unvalidated at: strategy)) , ' BEFORE Pharo validation' ]
        ifFalse: [ (CooBenchmarkChart labelFor: strategy) ,
            ((unvalidated includes: strategy) ifTrue: [ ' AFTER Pharo validation' ] ifFalse: [ '' ]) ] ].
package := PackageOrganizer default packageNamed: packageName.
kinds := Dictionary newFrom: { #messages -> CooBenchRunnerMessage. #variables -> CooBenchRunnerVariables }.
kind := (OSEnvironment current at: 'BENCHMARK_KIND') asSymbol.
strategy := (OSEnvironment current at: 'BENCHMARK_STRATEGY') asSymbol.
(kinds includesKey: kind) ifFalse: [ Error signal: 'Unknown benchmark kind: ', kind ].
(strategies includes: strategy) ifFalse: [ Error signal: 'Unknown benchmark strategy: ', strategy ].
"The pipeline's own Ollama server (tuned for concurrent requests), when it names one."
(OSEnvironment current at: 'BENCHMARK_OLLAMA_URL' ifAbsent: [ '' ]) ifNotEmpty: [ :url |
    CooLLMClient default baseUrl: url ].
started := Time millisecondClockValue.
benchmark := ((kinds at: kind) new package: package; baseline: strategy; others: #(); run; yourself)
    results at: strategy.
Stdio stdout
    nextPutAll: 'Progress: ';
    nextPutAll: (CooBenchmarkChart labelForKind: kind); nextPutAll: ' - ';
    nextPutAll: (labelFor value: strategy); nextPutAll: ' done in ';
    nextPutAll: ((Time millisecondClockValue - started) / 60000.0 printShowingDecimalPlaces: 1);
    nextPutAll: ' min'.
(unvalidated includesKey: strategy) ifTrue: [
    | generated valid |
    generated := (2 to: 8) inject: 0 into: [ :sum :prefix | sum + (benchmark generatedForPrefixSize: prefix) ].
    valid := (2 to: 8) inject: 0 into: [ :sum :prefix | sum + (benchmark validForPrefixSize: prefix) ].
    Stdio stdout nextPutAll: ': '; print: valid; nextPutAll: ' of '; print: generated;
        nextPutAll: ' generated candidates are real Pharo names'.
    generated > 0 ifTrue: [
        Stdio stdout nextPutAll: ' ('; nextPutAll: (valid * 100 / generated printShowingDecimalPlaces: 1); nextPutAll: '%)' ] ].
Stdio stdout lf; flush.
rows := OrderedCollection new.
2 to: 8 do: [ :prefix |
    | count |
    count := benchmark totalEntriesPerPrefixSize: prefix.
    rows add: (Dictionary new
        at: 'kind' put: kind asString;
        at: 'strategy' put: strategy asString;
        at: 'prefix' put: prefix;
        at: 'count' put: count;
        at: 'reciprocalRankSum' put: (count * (benchmark mmrForPrefixSize: prefix)) asFloat;
        at: 'timeMs' put: (benchmark totalTimeForPrefix: prefix) asFloat;
        at: 'memoryBytes' put: (benchmark totalMemoryForPrefix: prefix) asFloat;
        yourself).
    (benchmark respondsTo: #generatedForPrefixSize:) ifTrue: [
        rows last
            at: 'generated' put: (benchmark generatedForPrefixSize: prefix);
            at: 'valid' put: (benchmark validForPrefixSize: prefix) ].
    "Hits: observations whose target is ranked within the first k (Top-k / Recall@k)."
    #(1 2 3 10) do: [ :k |
        rows last at: 'top', k printString put:
            (count * (benchmark accuracyForCompletionIndex: (1 to: k) withPrefixSize: prefix)) rounded ] ].
output := Dictionary new
    at: 'schema' put: 'coo-package-part-v1';
    at: 'package' put: packageName;
    at: 'kind' put: kind asString;
    at: 'strategy' put: strategy asString;
    at: 'runId' put: (OSEnvironment current at: 'BENCHMARK_RUN_ID');
    at: 'corpus' put: (CooBenchRunner corpusSummaryForPackageNames: { packageName });
    at: 'rows' put: rows asArray;
    yourself.
(OSEnvironment current at: 'BENCHMARK_OUTPUT') asFileReference writeStreamDo: [ :out |
    out truncate; nextPutAll: (STONJSON toString: output); lf ].
