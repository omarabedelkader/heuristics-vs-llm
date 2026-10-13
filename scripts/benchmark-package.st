"Run one benchmark package and retain sufficient statistics, without rounding."
| selection packageName strategies comparison rows output |
selection := STONJSON fromString: (OSEnvironment current at: 'BENCHMARK_RUN_SELECTION') asFileReference contents.
(selection at: 'schema') = 'coo-package-split-v1' ifFalse: [ Error signal: 'Saved benchmark selection required' ].
(selection at: 'eligible') asSet = CooBenchRunner eligiblePackageNames asSet ifFalse: [
    Error signal: 'Worker package pool differs from the saved selection' ].
packageName := OSEnvironment current at: 'BENCHMARK_PACKAGE'.
((selection at: 'benchmark') includes: packageName) ifFalse: [
    Error signal: 'Worker package is not in the saved benchmark selection' ].
strategies := CooBenchmarkChart publicationStrategies.
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
                    yourself).
                "Hits: observations whose target is ranked within the first k (Top-k / Recall@k)."
                #(1 2 3 10) do: [ :k |
                    rows last at: 'top', k printString put:
                        (count * (benchmark accuracyForCompletionIndex: (1 to: k) withPrefixSize: prefix)) rounded ] ] ] ] ].
output := Dictionary new
    at: 'schema' put: 'coo-package-result-v2';
    at: 'package' put: packageName;
    at: 'runId' put: (OSEnvironment current at: 'BENCHMARK_RUN_ID');
    at: 'corpus' put: (comparison at: #corpus);
    at: 'rows' put: rows asArray;
    yourself.
(OSEnvironment current at: 'BENCHMARK_OUTPUT') asFileReference writeStreamDo: [ :out |
    out truncate; nextPutAll: (STONJSON toString: output); lf ].
