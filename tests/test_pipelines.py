"""Exercise both entry points with download, Pharo and ML stand-ins."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pipeline test ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.mining = self.root / "mining"
        self.experiment = self.root / "experiment"
        source = self.root / "local-source"
        for name in ("src/ExtendedHeuristicCompletion-Benchmarks", "scripts", "reranker"):
            (source / name).mkdir(parents=True)
        (source / 'src/ExtendedHeuristicCompletion-Benchmarks/CooBenchmarkSplit.class.st').touch()
        (source / 'reranker/requirements.txt').touch()
        binaries = self.root / "bin"
        binaries.mkdir()
        self.env = {**os.environ, "MINING_DIR": str(self.mining),
                    "EXPERIMENT_DIR": str(self.experiment), "BENCHMARK_REPO_DIR": str(source),
                    "BENCHMARK_PACKAGE_COUNT": "1", "RANKING_SEED": "42",
                    "RESULTS_DIR": str(self.root / "resutls"),
                    "TEST_ROOT": str(self.root), "PATH": str(binaries) + os.pathsep + os.environ['PATH']}
        self.env.pop("RERANKER_PYTHON", None)
        self.executable(binaries / 'curl', '''
import json, os, sys
from pathlib import Path
root = Path(os.environ['TEST_ROOT'])
if 'http://127.0.0.1:11434/api/tags' in sys.argv:
    if os.environ.get('TEST_START_OLLAMA') and not (root / 'ollama-ready').exists():
        sys.exit(7)
    print(json.dumps({'models': [{'name': 'model-' + size} for size in ['05', '15', '3', '7']]}))
    sys.exit(0)
with (root / 'downloads').open('a') as out:
    out.write(str(Path.cwd()) + '\\n')
Path(sys.argv[sys.argv.index('-o') + 1]).write_text(
    '#!/usr/bin/env bash\\ncp "$TEST_ROOT/pharo-stub" ./pharo\\nprintf "%s" "$PWD" > Pharo.image\\n')
''')
        self.executable(self.root / "pharo-stub", '''
import json, os, random, sys
from pathlib import Path
root = Path(os.environ['TEST_ROOT'])
code = sys.argv[-1]
with (root / 'pharo-calls.jsonl').open('a') as out:
    out.write(json.dumps(dict(cwd=str(Path.cwd()), code=code)) + '\\n')
packages = ['Package-A', 'Package-B', 'Package-C']
if 'exportRankingCorpusForPackages:' in code:
    directory = Path(os.environ['CORPUS_STAGE_DIR'])
    (directory / 'packages.json').write_text(json.dumps(packages))
    with (directory / 'all.jsonl').open('w') as out:
        for name in packages:
            out.write(json.dumps(dict(schema='coo-ranking-v1', group=name)) + '\\n')
elif 'benchmarkPackages:' in code:
    if os.environ.get('TEST_DIFFERENT_POOL'):
        packages.append('New-Package')
    path = Path(os.environ['BENCHMARK_SPLIT_FILE'])
    (path.parent / 'llm-models.json').write_text(json.dumps(['model-05', 'model-15', 'model-3', 'model-7']))
    if not path.exists():
        benchmark = random.Random(42).sample(packages, 1)
        path.write_text(json.dumps(dict(schema='coo-package-split-v1', seed=42,
            eligible=packages, benchmark=benchmark,
            train=[name for name in packages if name not in benchmark])))
elif 'normalBenchmarksForSplit:' in code:
    run = Path(os.environ['RERANKER_RUN_DIR'])
    assert (run / 'split.json').read_bytes() == Path(os.environ['BENCHMARK_SPLIT_FILE']).read_bytes()
    (root / 'normal-split.json').write_bytes((run / 'split.json').read_bytes())
    publication = Path(os.environ['PUBLICATION_DIR'])
    (publication / 'results-table.tex').write_text('Baseline Dependency LLM 05 15 3 7 Hybrid 05 15 3 7')
    (publication / 'performance.png').write_bytes(b'normal performance image')
    (publication / 'dataset-summary.tex').write_text('Packages: 1 Classes: 2 Methods: 3')
    (run / 'benchmark-corpus.json').write_text(json.dumps(dict(packages=1, classes=2, methods=3)))
elif 'rerankerBenchmarksForSplit:' in code:
    assert (root / 'trainer-checked').exists()
    assert (root / 'normal-split.json').read_bytes() == Path(os.environ['BENCHMARK_SPLIT_FILE']).read_bytes()
    if os.environ.get('TEST_FAIL_RERANKER'):
        sys.exit(24)
    publication = Path(os.environ['PUBLICATION_DIR'])
    (publication / 'results-table-re-ranker.tex').write_text('NeuralRank 10 20 30 50')
    if not os.environ.get('TEST_MISSING_FIGURE'):
        (publication / 'performance-re-ranker.png').write_bytes(b'reranker performance image')
    (root / 'benchmark-complete').touch()
elif 'Metacello new' not in code:
    raise SystemExit('Unexpected Pharo call: ' + code)
''')
        python = self.root / 'python-stand-in'
        self.executable(python, '''
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
root = Path(os.environ['TEST_ROOT'])
if args[:2] == ['-m', 'pip']:
    sys.exit(0)
if Path(args[0]).name == 'train.py':
    split = json.loads(Path(args[args.index('--package-split') + 1]).read_text())
    rows = [json.loads(line) for line in Path(args[1]).read_text().splitlines()]
    assert {row['group'] for row in rows} == set(split['train'])
    assert not ({row['group'] for row in rows} & set(split['benchmark']))
    (root / 'trainer-checked').touch()
    sys.exit(0)
if Path(args[0]).name == 'evaluate.py':
    assert (root / 'trainer-checked').exists()
    split = json.loads((Path(os.environ['EXPERIMENT_DIR']) / 'split.json').read_text())
    rows = [json.loads(line) for line in Path(args[2]).read_text().splitlines()]
    assert {row['group'] for row in rows} == set(split['benchmark'])
    print('[]')
    sys.exit(0)
if Path(args[0]).name == 'serve.py':
    time.sleep(30)
    sys.exit(0)
# Stub only the HTTP readiness test, leaving the source validation script real.
if args[0] == '-' and len(args) == 3:
    sys.exit(0)
os.execv(sys.executable, [sys.executable, *args])
''')
        self.env['RERANKER_PYTHON'] = str(python)
        ollama = self.root / 'ollama-stand-in'
        self.executable(ollama, '''
import os, time
from pathlib import Path
root = Path(os.environ['TEST_ROOT'])
(root / 'ollama-pid').write_text(str(os.getpid()))
(root / 'ollama-ready').touch()
time.sleep(30)
''')
        self.env['OLLAMA_BIN'] = str(ollama)
        self.env['OLLAMA_MODELS_DIR'] = str(self.root / 'models')

    def executable(self, path, body):
        path.write_text(f"#!{sys.executable}\n" + body)
        path.chmod(0o755)

    def run_script(self, name, expected=0):
        result = subprocess.run(["bash", str(ROOT / name)], env=self.env,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        for directory in (self.mining, self.experiment):
            self.assertFalse((directory / ".running").exists())
        self.assertFalse((self.root / 'resutls/.running').exists())
        return result

    def test_exactly_two_shell_entry_points(self):
        self.assertEqual(sorted(p.name for p in ROOT.glob('*.sh')),
                         ['pipeline-benchmarks.sh', 'pipeline-mine-training-data.sh'])

    def test_missing_corpus_fails_before_downloads(self):
        result = self.run_script('pipeline-benchmarks.sh', 1)
        self.assertIn('pipeline-mine-training-data.sh', result.stderr)
        self.assertFalse((self.root / 'downloads').exists())

    def test_two_downloads_and_saved_data_survives_removing_mining_image(self):
        self.run_script('pipeline-mine-training-data.sh')
        self.assertFalse((self.mining / 'split.json').exists())
        mining_image = (self.mining / 'image/Pharo.image').read_bytes()
        calls_before = (self.root / 'pharo-calls.jsonl').read_bytes()
        shutil.rmtree(self.mining / 'image')
        self.run_script('pipeline-mine-training-data.sh')
        self.assertEqual((self.root / 'pharo-calls.jsonl').read_bytes(), calls_before)
        self.run_script('pipeline-benchmarks.sh')
        self.assertNotEqual((self.experiment / 'image/Pharo.image').read_bytes(), mining_image)
        split = (self.experiment / 'split.json').read_bytes()
        self.run_script('pipeline-benchmarks.sh')
        self.assertEqual((self.experiment / 'split.json').read_bytes(), split)
        self.assertEqual((self.root / 'downloads').read_text().splitlines(),
                         [str(self.mining / 'image'), str(self.experiment / 'image')])
        calls = [json.loads(line) for line in (self.root / 'pharo-calls.jsonl').read_text().splitlines()]
        exports = [c for c in calls if 'exportRankingCorpusForPackages:' in c['code']]
        self.assertEqual(len(exports), 1)
        self.assertEqual(exports[0]['cwd'], str(self.mining / 'image'))
        for call in calls:
            if 'benchmarkPackages:' in call['code'] or 'rerankerBenchmarksForSplit:' in call['code']:
                self.assertEqual(call['cwd'], str(self.experiment / 'image'))
            self.assertNotIn('exportTrainingForSplit:', call['code'])
            self.assertNotIn('exportTestForSplit:', call['code'])
        self.assertTrue((self.root / 'benchmark-complete').exists())
        self.assertFalse((self.root / 'ollama-pid').exists(), 'Existing server must not be restarted')
        expected = {'results-table.tex', 'results-table-re-ranker.tex',
                    'performance.png', 'performance-re-ranker.png', 'dataset-summary.tex'}
        self.assertEqual({p.name for p in (self.root / 'resutls').iterdir()}, expected)
        self.assertEqual((self.root / 'resutls/dataset-summary.tex').read_text(),
                         'Packages: 1 Classes: 2 Methods: 3')
        self.assertEqual(sum('normalBenchmarksForSplit:' in c['code'] for c in calls), 2)
        self.assertEqual(sum('rerankerBenchmarksForSplit:' in c['code'] for c in calls), 2)

    def test_owned_ollama_stopped_after_run(self):
        self.run_script('pipeline-mine-training-data.sh')
        self.env['TEST_START_OLLAMA'] = '1'
        self.run_script('pipeline-benchmarks.sh')
        pid = int((self.root / 'ollama-pid').read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_failed_run_does_not_publish_partial_or_stale_outputs(self):
        self.run_script('pipeline-mine-training-data.sh')
        self.run_script('pipeline-benchmarks.sh')
        saved = {p.name: p.read_bytes() for p in (self.root / 'resutls').iterdir()}
        for flag, status in [('TEST_FAIL_RERANKER', 24), ('TEST_MISSING_FIGURE', 1)]:
            with self.subTest(flag=flag):
                self.env[flag] = '1'
                self.run_script('pipeline-benchmarks.sh', status)
                self.env.pop(flag)
                self.assertEqual({p.name: p.read_bytes() for p in (self.root / 'resutls').iterdir()}, saved)

    def test_same_directory_rejected(self):
        self.run_script('pipeline-mine-training-data.sh')
        self.env['EXPERIMENT_DIR'] = str(self.mining)
        result = self.run_script('pipeline-benchmarks.sh', 1)
        self.assertIn('must differ', result.stderr)

    def test_different_package_pool_rejected_before_training(self):
        self.run_script('pipeline-mine-training-data.sh')
        self.env['TEST_DIFFERENT_POOL'] = '1'
        result = self.run_script('pipeline-benchmarks.sh', 1)
        self.assertIn('different package pools', result.stderr)
        self.assertFalse((self.root / 'trainer-checked').exists())

    def test_changed_source_rejected_before_download(self):
        self.run_script('pipeline-mine-training-data.sh')
        (self.mining / 'repository/reranker/train.py').write_text('modified')
        result = self.run_script('pipeline-benchmarks.sh', 1)
        self.assertIn('Frozen source does not match', result.stderr)
        self.assertEqual(len((self.root / 'downloads').read_text().splitlines()), 1)


if __name__ == '__main__':
    unittest.main()
