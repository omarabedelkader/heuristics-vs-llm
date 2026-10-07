"Adapt retained totals to the existing publication exporter. No inference.
These temporary classes live only in this unsaved export process."
| data statsClass runClass makeComparison normal reranker corpus |
data := STONJSON fromString: (OSEnvironment current at: 'BENCHMARK_AGGREGATE') asFileReference contents.
corpus := CooBenchRunner corpusSummaryForPackageNames: (data at: 'packages').
#(packages classes methods) do: [ :key |
    (corpus at: key) = ((data at: 'corpus') at: key asString) ifFalse: [
        Error signal: 'Aggregated corpus differs from the benchmark image' ] ].
statsClass := (Object << #CooPipelineStatistics)
    slots: { #rows }; package: 'CooPipeline-Export'; install.
statsClass compile: 'rows: aDictionary rows := aDictionary'.
statsClass compile: 'totalEntriesPerPrefixSize: prefix ^ (rows at: prefix) at: ''count'''.
statsClass compile: 'mmrForPrefixSize: prefix
    | row count | row := rows at: prefix. count := row at: ''count''.
    ^ count = 0 ifTrue: [ 0 ] ifFalse: [ (row at: ''reciprocalRankSum'') / count ]'.
statsClass compile: 'totalTimeForPrefix: prefix ^ (rows at: prefix) at: ''timeMs'''.
statsClass compile: 'totalMemoryForPrefix: prefix ^ (rows at: prefix) at: ''memoryBytes'''.
runClass := (Object << #CooPipelineRun)
    slots: { #results }; package: 'CooPipeline-Export'; install.
runClass compile: 'results: aDictionary results := aDictionary'.
runClass compile: 'results ^ results'.
makeComparison := [ :phase :strategies |
    | comparison |
    comparison := Dictionary new
        at: #corpus put: corpus;
        at: #publicationStrategies put: strategies;
        yourself.
    #(messages variables) do: [ :kind |
        | results |
        results := Dictionary new.
        strategies do: [ :strategy |
            | rows |
            rows := Dictionary new.
            (data at: phase) do: [ :row |
                ((row at: 'kind') = kind asString and: [ (row at: 'strategy') = strategy asString ])
                    ifTrue: [ rows at: (row at: 'prefix') put: row ] ].
            results at: strategy put: (statsClass new rows: rows; yourself) ].
        comparison at: kind put: { { runClass new results: results; yourself } } ].
    comparison ].
normal := makeComparison value: 'normal' value: CooBenchmarkChart publicationStrategies.
reranker := makeComparison value: 'reranker' value: CooBenchmarkChart rerankerStrategies.
CooBenchRunner export: normal to: (OSEnvironment current at: 'PUBLICATION_DIR').
CooBenchRunner exportReranker: reranker to: (OSEnvironment current at: 'PUBLICATION_DIR').
