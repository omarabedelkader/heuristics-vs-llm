"Adapt retained totals to the existing publication exporter. No inference.
These temporary classes live only in this unsaved export process."
| data statsClass runClass makeComparison normal reranker corpus writeTopK |
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
"Top-k table: share of completions whose expected name is ranked within the first k,
weighted over every package and prefix 2-8 like the Average column. With a single
expected name, Top-k accuracy and Recall@k are the same number."
writeTopK := [ :phase :strategies :fileName |
    ((OSEnvironment current at: 'PUBLICATION_DIR') asFileReference / fileName) writeStreamDo: [ :out |
        out truncate.
        out nextPutAll: '\begin{table}[ht]'; lf;
            nextPutAll: '\centering'; lf;
            nextPutAll: '\caption{Top-k accuracy (\% of completions whose expected name is ranked within the first k suggestions; with one expected name, Top-k equals Recall@k), over all packages and prefixes 2--8.}'; lf;
            nextPutAll: '\renewcommand{\arraystretch}{1.3}'; lf;
            nextPutAll: '\begin{tabular}{|c|l|c|c|c|c|}'; lf;
            nextPutAll: '\hline'; lf;
            nextPutAll: '\textbf{} & \textbf{Strategy} & \textbf{Top-1} & \textbf{Top-2} & \textbf{Top-3} & \textbf{Recall@10} \\'; lf;
            nextPutAll: '\hline'; lf.
        #(messages variables) do: [ :kind |
            out nextPutAll: '\multirow{'; print: strategies size; nextPutAll: '}{*}{\textbf{';
                nextPutAll: (CooBenchmarkChart labelForKind: kind); nextPutAll: '}}'; lf.
            strategies withIndexDo: [ :strategy :index |
                | rows count |
                rows := (data at: phase) select: [ :row |
                    (row at: 'kind') = kind asString and: [ (row at: 'strategy') = strategy asString ] ].
                count := rows inject: 0 into: [ :sum :row | sum + (row at: 'count') ].
                out nextPutAll: '& '; nextPutAll: (CooBenchmarkChart labelFor: strategy).
                #('top1' 'top2' 'top3' 'top10') do: [ :field |
                    out nextPutAll: ' & '.
                    count = 0
                        ifTrue: [ out nextPutAll: '--' ]
                        ifFalse: [ out nextPutAll: (((rows inject: 0 into: [ :sum :row | sum + (row at: field) ])
                            * 100 / count) asFloat printShowingDecimalPlaces: 1) ] ].
                out nextPutAll: ' \\'.
                index < strategies size ifTrue: [ out nextPutAll: ' \cline{2-6}' ].
                out lf ].
            out nextPutAll: '\hline'; lf ].
        out nextPutAll: '\end{tabular}'; lf; nextPutAll: '\end{table}'; lf ] ].
"Each pipeline aggregates only its own phase: normal or re-ranker."
(data includesKey: 'normal') ifTrue: [
    normal := makeComparison value: 'normal' value: CooBenchmarkChart publicationStrategies.
    CooBenchRunner export: normal to: (OSEnvironment current at: 'PUBLICATION_DIR').
    writeTopK value: 'normal' value: CooBenchmarkChart publicationStrategies value: 'recall-top.tex' ].
(data includesKey: 'reranker') ifTrue: [
    reranker := makeComparison value: 'reranker' value: CooBenchmarkChart rerankerStrategies.
    CooBenchRunner exportReranker: reranker to: (OSEnvironment current at: 'PUBLICATION_DIR').
    writeTopK value: 'reranker' value: CooBenchmarkChart rerankerStrategies value: 'recall-top-re-ranker.tex' ].
