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
                    "BENCHMARK_PACKAGE_COUNT": "1", "BENCHMARK_JOBS": "2", "RANKING_SEED": "42",
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
packages = ['Package-A', 'Package-B', 'Package-C', 'Package-D', 'Package-E', 'Package-F']
if 'exportRankingCorpusForPackages:' in code:
    if os.environ.get('TEST_FAIL_MINING'):
        sys.exit(23)
    directory = Path(os.environ['CORPUS_STAGE_DIR'])
    (directory / 'packages.json').write_text(json.dumps(packages))
    with (directory / 'all.jsonl').open('w') as out:
        for name in packages:
            out.write(json.dumps(dict(schema='coo-ranking-v1', group=name)) + '\\n')
elif 'benchmarkPackages:' in code:
    if os.environ.get('TEST_DIFFERENT_POOL'):
        packages.append('New-Package')
    path = Path(os.environ['BENCHMARK_SELECTION_FILE'])
    (path.parent / 'llm-models.json').write_text(json.dumps(['model-05', 'model-15', 'model-3', 'model-7']))
    if not path.exists():
        benchmark = random.Random(42).sample(packages, 1)
        path.write_text(json.dumps(dict(schema='coo-package-split-v1', seed=42,
            eligible=packages, benchmark=benchmark,
            train=[name for name in packages if name not in benchmark])))
elif 'coo-package-result-v1' in code:
    sys.path.insert(0, os.environ['PIPELINE_SCRIPTS'])
    from package_benchmarks import STRATEGIES
    phase = os.environ['BENCHMARK_PHASE']
    package = os.environ['BENCHMARK_PACKAGE']
    split = json.loads(Path(os.environ['BENCHMARK_SPLIT_FILE']).read_text())
    assert package in split['benchmark']
    assert all(package not in split[key] for key in ('train', 'validation', 'test'))
    assert Path.cwd().name == 'image' and Path.cwd().parent.parent.name == 'packages'
    assert (Path.cwd() / 'Pharo.image').exists()
    if phase == 'reranker' and os.environ.get('TEST_FAIL_RERANKER'):
        sys.exit(24)
    rows = [dict(kind=kind, strategy=strategy, prefix=prefix, count=2,
                 reciprocalRankSum=1.0, timeMs=4.0, memoryBytes=-2.0)
            for kind in ('messages', 'variables') for strategy in STRATEGIES[phase]
            for prefix in range(2, 9)]
    Path(os.environ['BENCHMARK_OUTPUT']).write_text(json.dumps(dict(
        schema='coo-package-result-v1', package=package, phase=phase,
        runId=os.environ['BENCHMARK_RUN_ID'], corpus=dict(packages=1, classes=2, methods=3), rows=rows)))
elif 'CooPipelineStatistics' in code:
    data = json.loads(Path(os.environ['BENCHMARK_AGGREGATE']).read_text())
    assert data['corpus'] == dict(packages=1, classes=2, methods=3)
    publication = Path(os.environ['PUBLICATION_DIR'])
    (publication / 'results-table.tex').write_text('Baseline Dependency LLM 05 15 3 7 Hybrid 05 15 3 7')
    (publication / 'performance.png').write_bytes(b'normal performance image')
    (publication / 'dataset-summary.tex').write_text('Packages: 1 Classes: 2 Methods: 3')
    if 'reranker' in data:
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
if Path(args[0]).name == 'reranker_workflow.py':
    action = args[1]
    if action == 'train':
        split = json.loads(Path(args[args.index('--split') + 1]).read_text())
        assert '--data' not in args
        for key, option in [('train', '--training'), ('validation', '--validation')]:
            rows = [json.loads(line) for line in Path(args[args.index(option) + 1]).read_text().splitlines()]
            assert {row['group'] for row in rows} == set(split[key])
            assert not ({row['group'] for row in rows} & set(split['benchmark'] + split['test']))
        run = Path(os.environ['RERANKER_RUN_DIR'])
        assert (run / 'packages/0001/normal.json').exists(), 'Normal benchmarks must precede training'
        if os.environ.get('TEST_FAIL_TRAINING'):
            sys.exit(25)
        model = Path(args[args.index('--output') + 1])
        model.mkdir(exist_ok=True)
        (model / 'ranker.onnx').write_bytes(b'model')
        (model / 'metadata.json').write_text(json.dumps(dict(packageSplit=split)))
        (model / 'learning-history.json').write_text('{}')
        with (root / 'training-calls').open('a') as out:
            out.write('train\\n')
        (root / 'trainer-checked').touch()
    elif action == 'evaluate':
        assert (root / 'trainer-checked').exists()
        split = json.loads(Path(args[args.index('--split') + 1]).read_text())
        rows = [json.loads(line) for line in Path(args[args.index('--data') + 1]).read_text().splitlines()]
        assert {row['group'] for row in rows} == set(split['test'])
        assert not ({row['group'] for row in rows} & set(split['benchmark']))
        Path(args[args.index('--output') + 1]).write_text('{}')
    elif action == 'report':
        directory = Path(args[args.index('--output') + 1])
        directory.mkdir(exist_ok=True)
        for name in ('learning-curves', 'test-performance', 'test-rank-transitions'):
            for extension in ('png', 'pdf'):
                (directory / (name + '.' + extension)).write_bytes(b'figure')
    else:
        raise AssertionError(action)
    sys.exit(0)
if Path(args[0]).name == 'serve.py':
    with (root / 'reranker-server-calls').open('a') as out:
        out.write('serve\\n')
    time.sleep(30)
    sys.exit(0)
# Stub only the HTTP readiness test, leaving the source validation script real.
if args[0] == '-' and len(args) == 3:
    sys.exit(0)
os.execv(sys.executable, [sys.executable, *args])
''')
        self.env['RERANKER_PYTHON'] = str(python)
        self.env['PIPELINE_SCRIPTS'] = str(ROOT / 'scripts')
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

    def run_script(self, name, expected=0, args=()):
        result = subprocess.run(["bash", str(ROOT / name), *args], env=self.env,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        for directory in (self.mining, self.experiment):
            self.assertFalse((directory / ".running").exists())
        self.assertFalse((self.root / 'resutls/.running').exists())
        return result

    def test_exactly_two_shell_entry_points(self):
        self.assertEqual(sorted(p.name for p in ROOT.glob('*.sh')),
                         ['pipeline-benchmarks.sh', 'pipeline-mine-training-data.sh'])

    def test_help_and_unknown_options_before_any_work(self):
        self.env.pop('BENCHMARK_JOBS')
        result = self.run_script('pipeline-benchmarks.sh', args=['--help'])
        self.assertIn('--skip-reranker-benchmarks', result.stdout)
        result = self.run_script('pipeline-benchmarks.sh', 2, args=['--unknown'])
        self.assertIn('Unknown option', result.stderr)
        self.assertFalse((self.root / 'downloads').exists())

    def test_skip_reranker_benchmarks_keeps_training_and_removes_stale_neural_outputs(self):
        self.run_script('pipeline-benchmarks.sh')
        runs = set(self.experiment.glob('reranker-run.*'))
        self.env['TEST_FAIL_RERANKER'] = '1'
        result = self.run_script('pipeline-benchmarks.sh', args=['--skip-reranker-benchmarks'])
        self.assertIn('Skipping live neural', result.stdout)
        run = (set(self.experiment.glob('reranker-run.*')) - runs).pop()
        self.assertTrue((run / 'training-ready').exists())
        self.assertTrue((run / 'evaluation.json').is_file())
        self.assertEqual(len(list((run / 'learning').glob('*.png'))), 3)
        self.assertFalse((run / 'packages/0001/reranker.json').exists())
        self.assertEqual((self.root / 'reranker-server-calls').read_text().splitlines(), ['serve'])
        self.assertEqual({p.name for p in (self.root / 'resutls').iterdir()},
                         {'results-table.tex', 'performance.png', 'dataset-summary.tex'})

    def test_resume_without_skip_flag_adds_neural_benchmarks_without_retraining(self):
        self.run_script('pipeline-benchmarks.sh', args=['--skip-reranker-benchmarks'])
        run = next(self.experiment.glob('reranker-run.*'))
        saved = (run / 'packages/0001/normal.json').read_bytes()
        self.assertFalse((self.root / 'reranker-server-calls').exists())
        self.env['BENCHMARK_RESUME_DIR'] = str(run)
        self.run_script('pipeline-benchmarks.sh')
        self.assertEqual((run / 'packages/0001/normal.json').read_bytes(), saved)
        self.assertEqual((self.root / 'training-calls').read_text().splitlines(), ['train'])
        self.assertTrue((run / 'packages/0001/reranker.json').exists())
        self.assertEqual(len(list((self.root / 'resutls').iterdir())), 5)

    def test_job_count_must_be_explicit_and_positive(self):
        for value in ('', '0', '-1', 'auto', '2.5'):
            self.env['BENCHMARK_JOBS'] = value
            result = self.run_script('pipeline-benchmarks.sh', 1)
            self.assertIn('BENCHMARK_JOBS', result.stderr)
            self.assertFalse((self.root / 'downloads').exists())

    def test_resume_uses_same_model_and_completed_package_results(self):
        self.run_script('pipeline-mine-training-data.sh')
        self.env['TEST_FAIL_RERANKER'] = '1'
        self.run_script('pipeline-benchmarks.sh', 1)
        run = next(self.experiment.glob('reranker-run.*'))
        saved = (run / 'packages/0001/normal.json').read_bytes()
        self.env.pop('TEST_FAIL_RERANKER')
        self.env['BENCHMARK_RESUME_DIR'] = str(run)
        self.env['BENCHMARK_JOBS'] = '1'
        self.run_script('pipeline-benchmarks.sh')
        self.assertEqual((run / 'packages/0001/normal.json').read_bytes(), saved)
        self.assertEqual((self.root / 'training-calls').read_text().splitlines(), ['train'])
        calls = [json.loads(line) for line in (self.root / 'pharo-calls.jsonl').read_text().splitlines()]
        self.assertEqual(sum('coo-package-result-v1' in c['code'] for c in calls), 3)

    def test_training_failure_preserves_normal_checkpoints_for_resume(self):
        self.run_script('pipeline-mine-training-data.sh')
        self.env['TEST_FAIL_TRAINING'] = '1'
        self.run_script('pipeline-benchmarks.sh', 1)
        run = next(self.experiment.glob('reranker-run.*'))
        saved = (run / 'packages/0001/normal.json').read_bytes()
        self.assertFalse((run / 'training-ready').exists())
        self.assertFalse((run / 'model-inputs.json').exists())
        self.env.pop('TEST_FAIL_TRAINING')
        self.env['BENCHMARK_RESUME_DIR'] = str(run)
        self.run_script('pipeline-benchmarks.sh')
        self.assertEqual((run / 'packages/0001/normal.json').read_bytes(), saved)
        self.assertEqual((self.root / 'training-calls').read_text().splitlines(), ['train'])
        self.assertEqual(len(list((run / 'learning').glob('*.png'))), 3)
        self.assertEqual(len(list((run / 'learning').glob('*.pdf'))), 3)

    def test_missing_corpus_is_mined_automatically_then_reused(self):
        result = self.run_script('pipeline-benchmarks.sh')
        self.assertIn('Running pipeline-mine-training-data.sh', result.stdout)
        self.assertTrue((self.mining / 'corpus/manifest.json').is_file())
        self.assertTrue((self.root / 'benchmark-complete').exists())
        saved = (self.mining / 'corpus/all.jsonl').read_bytes()
        self.run_script('pipeline-benchmarks.sh')
        self.assertEqual((self.mining / 'corpus/all.jsonl').read_bytes(), saved)
        calls = [json.loads(line) for line in (self.root / 'pharo-calls.jsonl').read_text().splitlines()]
        self.assertEqual(sum('exportRankingCorpusForPackages:' in c['code'] for c in calls), 1)
        self.assertEqual((self.root / 'downloads').read_text().splitlines(),
                         [str(self.mining / 'image')])

    def test_failed_automatic_mining_stops_before_benchmarks(self):
        self.env['TEST_FAIL_MINING'] = '1'
        self.run_script('pipeline-benchmarks.sh', 23)
        self.assertFalse((self.mining / 'corpus/manifest.json').exists())
        self.assertFalse((self.experiment / 'image').exists())
        self.assertFalse((self.root / 'trainer-checked').exists())

    def test_existing_mining_lock_is_preserved_without_duplicate_work(self):
        lock = self.mining / '.running'
        lock.mkdir(parents=True)
        result = subprocess.run(['bash', str(ROOT / 'pipeline-benchmarks.sh')], env=self.env,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('Mining is already running', result.stderr)
        self.assertTrue(lock.is_dir())
        self.assertFalse((self.experiment / '.running').exists())
        self.assertFalse((self.root / 'resutls/.running').exists())
        self.assertFalse((self.root / 'downloads').exists())

    def test_invalid_saved_corpus_is_not_remined_or_overwritten(self):
        self.run_script('pipeline-mine-training-data.sh')
        path = self.mining / 'corpus/all.jsonl'
        path.write_text('corrupted data\n')
        calls = (self.root / 'pharo-calls.jsonl').read_bytes()
        result = self.run_script('pipeline-benchmarks.sh', 1)
        self.assertIn('checksum mismatch', result.stderr)
        self.assertEqual(path.read_text(), 'corrupted data\n')
        self.assertEqual((self.root / 'pharo-calls.jsonl').read_bytes(), calls)

    def test_exact_image_copy_and_existing_experiment_survives_removing_mining_image(self):
        self.run_script('pipeline-mine-training-data.sh')
        self.assertFalse((self.mining / 'split.json').exists())
        mining_image = (self.mining / 'image/Pharo.image').read_bytes()
        (self.mining / 'image/Pharo.changes').write_text('mining changes')
        (self.mining / 'image/Pharo.sources').write_text('matching sources')
        (self.mining / 'image/pharo-vm').mkdir()
        (self.mining / 'image/pharo-vm/runtime').write_text('matching VM')
        self.run_script('pipeline-benchmarks.sh')
        self.assertEqual((self.experiment / 'image/Pharo.image').read_bytes(), mining_image)
        self.assertFalse(os.path.samefile(self.mining / 'image/Pharo.image',
                                         self.experiment / 'image/Pharo.image'))
        for name in ('Pharo.changes', 'Pharo.sources', 'pharo-vm/runtime'):
            self.assertEqual((self.experiment / 'image' / name).read_bytes(),
                             (self.mining / 'image' / name).read_bytes())
        calls_before = (self.root / 'pharo-calls.jsonl').read_bytes()
        shutil.rmtree(self.mining / 'image')
        self.run_script('pipeline-mine-training-data.sh')
        self.assertEqual((self.root / 'pharo-calls.jsonl').read_bytes(), calls_before)
        split = (self.experiment / 'split.json').read_bytes()
        self.run_script('pipeline-benchmarks.sh')
        self.assertEqual((self.experiment / 'split.json').read_bytes(), split)
        self.assertEqual((self.root / 'downloads').read_text().splitlines(),
                         [str(self.mining / 'image')])
        calls = [json.loads(line) for line in (self.root / 'pharo-calls.jsonl').read_text().splitlines()]
        self.assertEqual(sum('Metacello new' in c['code'] for c in calls), 1)
        exports = [c for c in calls if 'exportRankingCorpusForPackages:' in c['code']]
        self.assertEqual(len(exports), 1)
        self.assertEqual(exports[0]['cwd'], str(self.mining / 'image'))
        for call in calls:
            if 'benchmarkPackages:' in call['code']:
                self.assertEqual(call['cwd'], str(self.experiment / 'image'))
            self.assertNotIn('exportTrainingForSplit:', call['code'])
            self.assertNotIn('exportTestForSplit:', call['code'])
        self.assertTrue((self.root / 'benchmark-complete').exists())
        self.assertFalse((self.root / 'ollama-pid').exists(), 'Existing server must not be restarted')
        expected = {'results-table.tex', 'results-table-re-ranker.tex',
                    'performance.png', 'performance-reranker.png', 'dataset-summary.tex'}
        self.assertEqual({p.name for p in (self.root / 'resutls').iterdir()}, expected)
        self.assertEqual((self.root / 'resutls/dataset-summary.tex').read_text(),
                         'Packages: 1 Classes: 2 Methods: 3')
        self.assertEqual(sum('coo-package-result-v1' in c['code'] for c in calls), 4)
        self.assertEqual(sum('CooPipelineStatistics' in c['code'] for c in calls), 2)

    def test_new_experiment_requires_original_mining_image(self):
        self.run_script('pipeline-mine-training-data.sh')
        shutil.rmtree(self.mining / 'image')
        result = self.run_script('pipeline-benchmarks.sh', 1)
        self.assertIn('original mining image is required', result.stderr)
        self.assertEqual(len((self.root / 'downloads').read_text().splitlines()), 1)
        self.assertFalse((self.experiment / 'image').exists())

    def test_changed_mining_image_rejected_before_selection(self):
        self.run_script('pipeline-mine-training-data.sh')
        (self.mining / 'image/Pharo.image').write_bytes(b'different Pharo build')
        result = self.run_script('pipeline-benchmarks.sh', 1)
        self.assertIn('Image does not match the prepared corpus', result.stderr)
        self.assertFalse((self.experiment / 'image').exists())
        self.assertFalse((self.experiment / 'benchmark-selection.json').exists())

    def test_changed_benchmark_image_rejected_on_rerun(self):
        self.run_script('pipeline-benchmarks.sh')
        (self.experiment / 'image/Pharo.image').write_bytes(b'different Pharo build')
        calls = (self.root / 'pharo-calls.jsonl').read_bytes()
        result = self.run_script('pipeline-benchmarks.sh', 1)
        self.assertIn('Image does not match the prepared corpus', result.stderr)
        self.assertEqual((self.root / 'pharo-calls.jsonl').read_bytes(), calls)

    def test_owned_ollama_stopped_after_run(self):
        self.run_script('pipeline-mine-training-data.sh')
        self.env['TEST_START_OLLAMA'] = '1'
        self.run_script('pipeline-benchmarks.sh')
        pid = int((self.root / 'ollama-pid').read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_success_replaces_legacy_figure_filename(self):
        self.run_script('pipeline-mine-training-data.sh')
        results = self.root / 'resutls'
        results.mkdir()
        legacy = results / 'performance-re-ranker.png'
        legacy.write_bytes(b'old figure')
        self.run_script('pipeline-benchmarks.sh')
        self.assertFalse(legacy.exists())
        self.assertTrue((results / 'performance-reranker.png').is_file())

    def test_failed_run_does_not_publish_partial_or_stale_outputs(self):
        self.run_script('pipeline-mine-training-data.sh')
        self.run_script('pipeline-benchmarks.sh')
        saved = {p.name: p.read_bytes() for p in (self.root / 'resutls').iterdir()}
        for flag, status in [('TEST_FAIL_RERANKER', 1), ('TEST_MISSING_FIGURE', 1)]:
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
