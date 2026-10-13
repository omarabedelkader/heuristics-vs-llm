"Run one benchmark package and retain sufficient statistics, without rounding."
| selection packageName package strategies kinds results done rows output |
selection := STONJSON fromString: (OSEnvironment current at: 'BENCHMARK_RUN_SELECTION') asFileReference contents.
(selection at: 'schema') = 'coo-package-split-v1' ifFalse: [ Error signal: 'Saved benchmark selection required' ].
(selection at: 'eligible') asSet = CooBenchRunner eligiblePackageNames asSet ifFalse: [
    Error signal: 'Worker package pool differs from the saved selection' ].
packageName := OSEnvironment current at: 'BENCHMARK_PACKAGE'.
((selection at: 'benchmark') includes: packageName) ifFalse: [
    Error signal: 'Worker package is not in the saved benchmark selection' ].
strategies := CooBenchmarkChart publicationStrategies.
package := PackageOrganizer default packageNamed: packageName.
kinds := { #messages -> CooBenchRunnerMessage. #variables -> CooBenchRunnerVariables }.
"One benchmark at a time, so each one is reported as soon as it is done (the pipeline shows these lines)."
results := Dictionary new.
done := 0.
kinds do: [ :kind |
    strategies do: [ :strategy |
        | runner started |
        started := Time millisecondClockValue.
        runner := kind value new package: package; baseline: strategy; others: #(); run; yourself.
        results at: kind key -> strategy put: (runner results at: strategy).
        done := done + 1.
        Stdio stdout
            nextPutAll: 'Progress: ';
            nextPutAll: (CooBenchmarkChart labelForKind: kind key); nextPutAll: ' - ';
            nextPutAll: (CooBenchmarkChart labelFor: strategy); nextPutAll: ' done in ';
            nextPutAll: ((Time millisecondClockValue - started) / 60000.0 printShowingDecimalPlaces: 1);
            nextPutAll: ' min ('; print: done; nextPut: $/; print: kinds size * strategies size; nextPut: $);
            lf; flush ] ].
rows := OrderedCollection new.
kinds do: [ :entry |
    | kind |
    kind := entry key.
    strategies do: [ :strategy |
        | benchmark |
        benchmark := results at: kind -> strategy.
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
                    (count * (benchmark accuracyForCompletionIndex: (1 to: k) withPrefixSize: prefix)) rounded ] ] ] ].
output := Dictionary new
    at: 'schema' put: 'coo-package-result-v2';
    at: 'package' put: packageName;
    at: 'runId' put: (OSEnvironment current at: 'BENCHMARK_RUN_ID');
    at: 'corpus' put: (CooBenchRunner corpusSummaryForPackageNames: { packageName });
    at: 'rows' put: rows asArray;
    yourself.
(OSEnvironment current at: 'BENCHMARK_OUTPUT') asFileReference writeStreamDo: [ :out |
    out truncate; nextPutAll: (STONJSON toString: output); lf ].
