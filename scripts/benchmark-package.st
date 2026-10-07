"Run one held-out package and retain sufficient statistics, without rounding."
| split packageName phase strategies comparison rows output |
split := CooBenchmarkSplit readFrom: (OSEnvironment current at: 'BENCHMARK_SPLIT_FILE').
split validateAgainst: CooBenchRunner eligiblePackageNames.
packageName := OSEnvironment current at: 'BENCHMARK_PACKAGE'.
(split benchmarkPackageNames includes: packageName) ifFalse: [
    Error signal: 'Worker package is not in the saved benchmark split' ].
phase := OSEnvironment current at: 'BENCHMARK_PHASE'.
strategies := phase = 'normal'
    ifTrue: [ CooBenchmarkChart publicationStrategies ]
    ifFalse: [
        phase = 'reranker' ifFalse: [ Error signal: 'Unknown benchmark phase' ].
        "Validate the full training split, never a synthetic one-package split."
        CooNeuralRankerClient new validateSplit: split.
        CooBenchmarkChart rerankerStrategies ].
comparison := CooBenchRunner compareMessagesAndVariablesFor: { packageName } strategies: strategies.
rows := OrderedCollection new.
#(messages variables) do: [ :kind |
    (comparison at: kind) first do: [ :run |
        strategies do: [ :strategy |
            | benchmark |
            benchmark := run results at: strategy.
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
                    yourself) ] ] ] ].
output := Dictionary new
    at: 'schema' put: 'coo-package-result-v1';
    at: 'package' put: packageName;
    at: 'phase' put: phase;
    at: 'runId' put: (OSEnvironment current at: 'BENCHMARK_RUN_ID');
    at: 'corpus' put: (comparison at: #corpus);
    at: 'rows' put: rows asArray;
    yourself.
(OSEnvironment current at: 'BENCHMARK_OUTPUT') asFileReference writeStreamDo: [ :out |
    out truncate; nextPutAll: (STONJSON toString: output); lf ].
