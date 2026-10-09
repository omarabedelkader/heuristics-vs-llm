"Run one held-out package and retain sufficient statistics, without rounding."
| split packageName phase strategies comparison rows output seen http metadata |
split := STONJSON fromString: (OSEnvironment current at: 'BENCHMARK_SPLIT_FILE') asFileReference contents.
(split at: 'schema') = 'coo-package-split-v2' ifFalse: [ Error signal: 'Four-part package split required' ].
(split at: 'eligible') asSet = CooBenchRunner eligiblePackageNames asSet ifFalse: [
    Error signal: 'Worker package pool differs from the saved split' ].
seen := Set new.
#('train' 'validation' 'test' 'benchmark') do: [ :key |
    | names |
    names := split at: key.
    (names isEmpty or: [ names asSet size ~= names size or: [ (seen intersection: names asSet) notEmpty ] ])
        ifTrue: [ Error signal: 'The four package partitions must be nonempty and disjoint' ].
    seen addAll: names ].
"Packages left out by TRAINING_PACKAGE_COUNT; absent when every training package is used."
(split at: 'unused' ifAbsent: [ #() ]) do: [ :name |
    (seen includes: name) ifTrue: [ Error signal: 'Unused packages must not appear in another partition' ].
    seen add: name ].
seen = (split at: 'eligible') asSet ifFalse: [ Error signal: 'Incomplete package partitions' ].
packageName := OSEnvironment current at: 'BENCHMARK_PACKAGE'.
((split at: 'benchmark') includes: packageName) ifFalse: [
    Error signal: 'Worker package is not in the saved benchmark split' ].
phase := OSEnvironment current at: 'BENCHMARK_PHASE'.
strategies := phase = 'normal'
    ifTrue: [ CooBenchmarkChart publicationStrategies ]
    ifFalse: [
        phase = 'reranker' ifFalse: [ Error signal: 'Unknown benchmark phase' ].
        "The upstream service exposes metadata unchanged; enforce the v2 split here."
        http := ZnClient new.
        [ http url: 'http://127.0.0.1:8765/metadata'; timeout: 5; numberOfRetries: 0;
            enforceHttpSuccess: true; get.
          metadata := STONJSON fromString: http contents
        ] ensure: [ http close ].
        (metadata at: 'packageSplit') = split ifFalse: [
            Error signal: 'Re-ranker model and four-part package split differ' ].
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
                    yourself).
                "Hits: observations whose target is ranked within the first k (Top-k / Recall@k)."
                #(1 2 3 10) do: [ :k |
                    rows last at: 'top', k printString put:
                        (count * (benchmark accuracyForCompletionIndex: (1 to: k) withPrefixSize: prefix)) rounded ] ] ] ] ].
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
