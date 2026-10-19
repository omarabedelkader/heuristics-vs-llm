"""Exercise pipeline-benchmarks.sh with download, Pharo and Ollama stand-ins."""
import hashlib
import json
import signal
import time
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = 'src/ExtendedHeuristicCompletion-Benchmarks'


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pipeline test ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.experiment = self.root / "experiment"
        self.checkout = self.root / "local-source"
        (self.checkout / PACKAGE).mkdir(parents=True)
        (self.checkout / PACKAGE / 'CooLLMBeamCompleter.class.st').write_text('first')
        binaries = self.root / "bin"
        binaries.mkdir()
        self.env = {**os.environ, "EXPERIMENT_DIR": str(self.experiment),
                    "BENCHMARK_REPO_DIR": str(self.checkout),
                    "BENCHMARK_PACKAGE_COUNT": "1", "BENCHMARK_JOBS": "2", "RANKING_SEED": "42",
                    "BENCHMARK_OLLAMA_HOST": "127.0.0.1:11434",
                    "RESULTS_DIR": str(self.root / "resutls"),
                    "TEST_ROOT": str(self.root), "PATH": str(binaries) + os.pathsep + os.environ['PATH']}
        for name in ("MINING_DIR", "PIPELINE_PYTHON", "BENCHMARK_PACKAGES_FILE", "BENCHMARK_RESUME_DIR",
                     "BENCHMARK_SHARED"):
            self.env.pop(name, None)
        self.executable(binaries / 'curl', '''
import json, os, sys
from pathlib import Path
root = Path(os.environ['TEST_ROOT'])
if any(arg.endswith('/api/tags') for arg in sys.argv):
    if os.environ.get('TEST_START_OLLAMA') and not (root / 'ollama-ready').exists():
        sys.exit(7)
    print(json.dumps({'models': [{'name': 'model-' + size} for size in ['05', '15', '3', '7']]}))
    sys.exit(0)
if any(arg.endswith('/api/version') for arg in sys.argv):
    if os.environ.get('TEST_START_OLLAMA') and not (root / 'ollama-ready').exists():
        sys.exit(7)
    print(json.dumps({'version': '0.40.4'}))
    sys.exit(0)
url = next(arg for arg in sys.argv if arg.startswith('http'))
with (root / 'downloads').open('a') as out:
    out.write(url + ' ' + str(Path.cwd()) + '\\n')
assert 'huggingface' not in url, 'No dataset is downloaded'
Path(sys.argv[sys.argv.index('-o') + 1]).write_text(
    '#!/usr/bin/env bash\\ncp "$TEST_ROOT/pharo-stub" ./pharo\\nprintf "%s" "fresh Pharo" > Pharo.image\\n')
''')
        self.executable(self.root / "pharo-stub", '''
import json, os, random, sys
from pathlib import Path
root = Path(os.environ['TEST_ROOT'])
code = sys.argv[-1]
with (root / 'pharo-calls.jsonl').open('a') as out:
    out.write(json.dumps(dict(cwd=str(Path.cwd()), code=code, session=os.getsid(0))) + '\\n')
packages = ['Package-A', 'Package-B', 'Package-C', 'Package-D', 'Package-E', 'Package-F']
if os.environ.get('TEST_DIFFERENT_POOL'):
    packages.append('New-Package')
source = Path(os.environ['REPO_DIR']) / 'src/ExtendedHeuristicCompletion-Benchmarks'
loaded = ' '.join(sorted(p.name + '=' + p.read_text() for p in source.iterdir()))
assert 'exportRankingCorpusForPackages:' not in code, 'Nothing is mined'
if 'Metacello new' in code:
    assert Path.cwd().name.startswith('image-pending.')
    Path('Pharo.image').write_text('image with ' + loaded)
elif 'MCPackageLoader' in code:
    Path('Pharo.image').write_text('image with ' + loaded)
elif 'ELIGIBLE_PACKAGES_FILE' in code:
    Path(os.environ['ELIGIBLE_PACKAGES_FILE']).write_text(json.dumps(packages))
elif 'benchmarkPackages:' in code:
    path = Path(os.environ['BENCHMARK_SELECTION_FILE'])
    (path.parent / 'llm-models.json').write_text(json.dumps(['model-05', 'model-15', 'model-3', 'model-7']))
    if path.exists():
        if set(json.loads(path.read_text())['eligible']) != set(packages):
            sys.exit('The image package pool changed')
    else:
        benchmark = random.Random(42).sample(packages, int(os.environ['BENCHMARK_PACKAGE_COUNT'] or 50))
        path.write_text(json.dumps(dict(schema='coo-package-split-v1', seed=42,
            eligible=packages, benchmark=benchmark,
            train=[name for name in packages if name not in benchmark])))
elif 'coo-package-part-v1' in code:
    sys.path.insert(0, os.environ['PIPELINE_SCRIPTS'])
    from package_benchmarks import MEASURED, LLM, BEFORE_VALIDATION
    package = os.environ['BENCHMARK_PACKAGE']
    selection = json.loads(Path(os.environ['BENCHMARK_RUN_SELECTION']).read_text())
    assert package in selection['benchmark']
    kind, strategy = os.environ['BENCHMARK_KIND'], os.environ['BENCHMARK_STRATEGY']
    assert Path.cwd().name == 'image' and Path.cwd().parent.name == kind + '-' + strategy
    assert Path.cwd().parents[3].name == 'packages'
    assert os.environ['BENCHMARK_OLLAMA_URL'] == 'http://' + os.environ['OLLAMA_HOST']
    assert (Path.cwd() / 'Pharo.image').exists()
    import time
    time.sleep(float(os.environ.get('TEST_SLOW_PACKAGE', 0)))
    if os.environ.get('TEST_FAIL_PACKAGE') == package:
        sys.exit(24)
    rows = [dict(kind=kind, strategy=strategy, prefix=prefix, count=2,
                 reciprocalRankSum=1.0, timeMs=4.0, memoryBytes=-2.0, top1=1, top2=1, top3=1, top10=1,
                 **(dict(generated=6, valid=6) if strategy in LLM else
                    dict(generated=8, valid=2) if strategy in BEFORE_VALIDATION.values() else {}))
            for prefix in range(2, 9)]
    if strategy == 'heuristicsBaseline' and kind == 'messages':
        print('Progress: Methods - Baseline done in 0.0 min', flush=True)
    Path(os.environ['BENCHMARK_OUTPUT']).write_text(json.dumps(dict(
        schema='coo-package-part-v1', package=package, kind=kind, strategy=strategy,
        runId=os.environ['BENCHMARK_RUN_ID'], corpus=dict(packages=1, classes=2, methods=3), rows=rows)))
elif 'CooPipelineStatistics' in code:
    data = json.loads(Path(os.environ['BENCHMARK_AGGREGATE']).read_text())
    count = len(data['packages'])
    assert data['corpus'] == dict(packages=count, classes=2 * count, methods=3 * count)
    assert len(data['rows']) == 2 * 10 * 7
    publication = Path(os.environ['PUBLICATION_DIR'])
    which = Path(os.environ['BENCHMARK_AGGREGATE']).name
    assert which == 'aggregate-%s.json' % publication.name, which
    generated = {row['generated'] for row in data['rows'] if row['strategy'].startswith('llm')}
    assert generated == ({8 * count} if publication.name == 'before' else {6 * count}), generated
    (publication / 'results-table.tex').write_text('Baseline Dependency LLM 05 15 3 7 Hybrid 05 15 3 7')
    (publication / 'dataset-summary.tex').write_text('Packages: %d' % count)
    (publication / 'recall-top.tex').write_text('Top-1 Top-2 Top-3 Recall@10')
    if not (os.environ.get('TEST_MISSING_FIGURE') and publication.name == 'after'):
        (publication / 'performance.png').write_bytes(b'performance image')
    (root / 'benchmark-complete').touch()
else:
    raise SystemExit('Unexpected Pharo call: ' + code)
''')
        self.env['PIPELINE_SCRIPTS'] = str(ROOT / 'scripts')
        ollama = self.root / 'ollama-stand-in'
        self.executable(ollama, '''
import os, time
from pathlib import Path
root = Path(os.environ['TEST_ROOT'])
(root / 'ollama-pid').write_text(str(os.getpid()))
(root / 'ollama-env.json').write_text(__import__('json').dumps({k: os.environ.get(k) for k in (
    'OLLAMA_HOST', 'OLLAMA_NUM_PARALLEL', 'OLLAMA_MAX_LOADED_MODELS', 'OLLAMA_MODELS')}))
(root / 'ollama-ready').touch()
time.sleep(30)
''')
        self.env['OLLAMA_BIN'] = str(ollama)
        self.env['OLLAMA_MODELS_DIR'] = str(self.root / 'models')

    def executable(self, path, body):
        path.write_text(f"#!{sys.executable}\n" + body)
        path.chmod(0o755)

    def run_pipeline(self, expected=0, args=()):
        result = subprocess.run(["bash", str(ROOT / 'pipeline-benchmarks.sh'), '--foreground', *args],
                                env=self.env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        self.assertFalse((self.experiment / ".running").exists())
        self.assertFalse((self.root / 'resutls/.running').exists())
        return result

    def calls(self, text=None):
        calls = [json.loads(line) for line in (self.root / 'pharo-calls.jsonl').read_text().splitlines()]
        return calls if text is None else [c for c in calls if text in c['code']]

    def results(self):
        """Published files by path below the results directory."""
        root = self.root / 'resutls'
        return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}

    def runs(self):
        return sorted(self.experiment.glob('benchmark-run.*'))

    def test_only_the_benchmark_and_cluster_entry_points_remain(self):
        self.assertEqual(sorted(p.name for p in ROOT.glob('*.sh')), ['job.fir.sh', 'job.oar.sh', 'pipeline-benchmarks.sh', 'submit-shared.sh'])
        for name in ('ranking_corpus.py', 'reranker_workflow.py', 'download_snapshot.py', 'runtime_estimate.py'):
            self.assertFalse((ROOT / 'scripts' / name).exists(), name)

    def test_default_run_is_detached_from_the_terminal_and_shows_its_log(self):
        log = self.root / 'my run.log'
        self.env['BENCHMARK_LOG'] = str(log)
        result = subprocess.run(["bash", str(ROOT / 'pipeline-benchmarks.sh')], env=self.env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('keep running if you press Ctrl-C or close this session', result.stdout)
        self.assertIn(f'tail -f {log}', result.stdout)
        # The log is shown while it runs, and kept.
        self.assertIn('Published (after Pharo validation)', result.stdout)
        self.assertIn('Published (after Pharo validation)', log.read_text())
        self.assertIn('results-after-pharo-validation/results-table.tex', self.results())
        # The benchmarks ran in a session of their own, not the launcher's.
        self.assertNotIn(os.getsid(0), {call['session'] for call in self.calls()})

    def test_detached_run_survives_its_launcher_and_reports_failure(self):
        log = self.root / 'my run.log'
        self.env['BENCHMARK_LOG'] = str(log)
        self.env['TEST_SLOW_PACKAGE'] = '0.5'
        launcher = subprocess.Popen(["bash", str(ROOT / 'pipeline-benchmarks.sh')], env=self.env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    start_new_session=True)
        pid = int(launcher.stdout.readline().split('PID ')[1].split(')')[0])
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not log.read_text():
            time.sleep(0.05)
        # The terminal goes away: everything in the launcher's session is hung up.
        os.killpg(launcher.pid, signal.SIGHUP)
        launcher.wait(timeout=10)
        self.assertFalse((self.root / 'benchmark-complete').exists(), 'The launcher left before the end')
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and 'This invocation took' not in log.read_text():
            time.sleep(0.1)
        self.assertTrue((self.root / 'benchmark-complete').exists(), log.read_text())
        self.assertIn('Published (after Pharo validation)', log.read_text())
        # A failing run gives its status to an attached launcher.
        self.env.pop('TEST_SLOW_PACKAGE')
        self.env['EXPERIMENT_DIR'] = str(self.root / 'missing parent' / 'experiment')
        self.env['BENCHMARK_JOBS'] = '0'
        result = subprocess.run(["bash", str(ROOT / 'pipeline-benchmarks.sh')], env=self.env,
                                capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)

    def test_help_and_unknown_options_before_any_work(self):
        self.env.pop('BENCHMARK_JOBS')
        result = self.run_pipeline(args=['--help'])
        self.assertIn('BENCHMARK_PACKAGES_FILE', result.stdout)
        result = self.run_pipeline(2, args=['--estimate-only'])
        self.assertIn('Unknown option', result.stderr)
        self.assertFalse(self.experiment.exists())

    def test_job_count_must_be_explicit_and_positive(self):
        for value in ('', '0', '-1', 'auto', '2.5'):
            self.env['BENCHMARK_JOBS'] = value
            result = self.run_pipeline(1)
            self.assertIn('BENCHMARK_JOBS', result.stderr)
            self.assertFalse((self.root / 'downloads').exists())

    def test_first_run_builds_the_image_from_the_checkout_without_any_dataset(self):
        result = self.run_pipeline()
        self.assertIn('Benchmark source: frozen from', result.stdout)
        self.assertNotIn('mining', (result.stdout + result.stderr).lower())
        self.assertFalse((self.root / 'mining').exists())
        self.assertFalse((ROOT / 'mining/.running').exists())
        # The only download is Pharo itself, into the image being built.
        downloads = (self.root / 'downloads').read_text().splitlines()
        self.assertEqual(len(downloads), 1)
        self.assertIn('get.pharo.org', downloads[0])
        frozen = self.experiment / 'repository'
        self.assertEqual((frozen / PACKAGE / 'CooLLMBeamCompleter.class.st').read_text(), 'first')
        self.assertIn('Local snapshot', (frozen / 'origin.txt').read_text())
        image = (self.experiment / 'image/Pharo.image').read_bytes()
        self.assertEqual(image, b'image with CooLLMBeamCompleter.class.st=first')
        pin = json.loads((self.experiment / 'image-source.json').read_text())
        self.assertEqual(pin['imageSHA256'], hashlib.sha256(image).hexdigest())
        self.assertEqual(list(self.experiment.glob('image-pending.*')), [])
        run, = self.runs()
        selection = json.loads((self.experiment / 'benchmark-selection.json').read_text())
        self.assertEqual(len(selection['benchmark']), 1)
        self.assertEqual(json.loads((run / 'selection.json').read_text()), selection)
        self.assertTrue((run / 'packages/0001/result.json').exists())
        timings = [json.loads(line) for line in (run / 'timings.jsonl').read_text().splitlines()]
        self.assertEqual([t['package'] for t in timings], selection['benchmark'])
        files = {'results-table.tex', 'performance.png', 'dataset-summary.tex', 'recall-top.tex'}
        self.assertEqual(set(self.results()), {
            *('results-after-pharo-validation/' + name for name in files),
            *('results-before-pharo-validation/' + name for name in files | {'pharo-validation.tex'})})
        self.assertIn('& 0.5B & 14 & 56 & 14 & 25.0',
                      self.results()['results-before-pharo-validation/pharo-validation.tex'].decode())
        self.assertIn('[messages heuristicsBaseline]: Methods - Baseline done in 0.0 min', result.stdout)
        self.assertFalse((self.root / 'ollama-pid').exists(), 'Existing server must not be restarted')

    def test_rerun_reuses_image_source_and_saved_packages(self):
        self.run_pipeline()
        selection = (self.experiment / 'benchmark-selection.json').read_bytes()
        image = (self.experiment / 'image/Pharo.image').read_bytes()
        self.run_pipeline()
        self.assertEqual((self.experiment / 'benchmark-selection.json').read_bytes(), selection)
        self.assertEqual((self.experiment / 'image/Pharo.image').read_bytes(), image)
        self.assertEqual(len((self.root / 'downloads').read_text().splitlines()), 1)
        self.assertEqual(len(self.calls('Metacello new')), 1)
        self.assertEqual(len(self.calls('MCPackageLoader')), 0)
        self.assertEqual(len(self.runs()), 2)
        # The frozen copy is enough: the checkout may be gone.
        shutil.rmtree(self.checkout)
        self.run_pipeline()
        self.assertEqual(len(self.calls('Metacello new')), 1)

    def test_changed_checkout_code_is_benchmarked_on_the_next_run(self):
        self.run_pipeline()
        first_run, = self.runs()
        first_pin = json.loads((self.experiment / 'image-source.json').read_text())
        # The checkout drops one class and adds another: the next run measures exactly that.
        (self.checkout / PACKAGE / 'CooLLMBeamCompleter.class.st').unlink()
        (self.checkout / PACKAGE / 'CooBenchRunner.class.st').write_text('later')
        result = self.run_pipeline()
        self.assertIn('code changed in', result.stdout)
        frozen = self.experiment / 'repository' / PACKAGE
        self.assertEqual([p.name for p in frozen.iterdir()], ['CooBenchRunner.class.st'])
        self.assertEqual([p.name for p in self.experiment.iterdir() if p.name.startswith('source.')], [])
        image = (self.experiment / 'image/Pharo.image').read_bytes()
        self.assertEqual(image, b'image with CooBenchRunner.class.st=later')
        pin = json.loads((self.experiment / 'image-source.json').read_text())
        self.assertNotEqual(pin['sourceSHA256'], first_pin['sourceSHA256'])
        self.assertEqual(pin['imageSHA256'], hashlib.sha256(image).hexdigest())
        self.assertEqual(len(self.calls('MCPackageLoader')), 1)
        self.assertEqual(len(self.calls('Metacello new')), 1)
        self.assertEqual(len((self.root / 'downloads').read_text().splitlines()), 1)
        # Unchanged code is not loaded again; a run made with the older code cannot continue.
        self.run_pipeline()
        self.assertEqual(len(self.calls('MCPackageLoader')), 1)
        self.env['BENCHMARK_RESUME_DIR'] = str(first_run)
        result = self.run_pipeline(1)
        self.assertIn('Resume inputs changed', result.stderr)

    def test_missing_checkout_is_reported_before_any_download(self):
        self.env['BENCHMARK_REPO_DIR'] = str(self.root / 'nowhere')
        result = self.run_pipeline(1)
        self.assertIn('Set BENCHMARK_REPO_DIR', result.stderr)
        self.assertFalse((self.root / 'downloads').exists())
        self.assertFalse((self.experiment / 'image').exists())

    def test_interrupted_image_build_is_never_used(self):
        self.executable(self.root / "pharo-stub", 'import sys\nsys.exit(31)\n')
        self.run_pipeline(31)
        self.assertFalse((self.experiment / 'image').exists())
        self.assertFalse((self.experiment / 'image-source.json').exists())
        self.assertEqual(len(list(self.experiment.glob('image-pending.*'))), 1)

    def test_changed_or_missing_image_rejected_on_rerun(self):
        self.run_pipeline()
        calls = (self.root / 'pharo-calls.jsonl').read_bytes()
        (self.experiment / 'image/Pharo.image').write_bytes(b'different Pharo build')
        result = self.run_pipeline(1)
        self.assertIn("Image does not match the experiment's frozen source", result.stderr)
        shutil.rmtree(self.experiment / 'image')
        result = self.run_pipeline(1)
        self.assertIn('experiment image is missing', result.stderr)
        self.assertEqual((self.root / 'pharo-calls.jsonl').read_bytes(), calls)
        self.assertEqual(len((self.root / 'downloads').read_text().splitlines()), 1)

    def test_package_count_and_pool_must_match_the_saved_selection(self):
        self.run_pipeline()
        self.env['TEST_DIFFERENT_POOL'] = '1'
        result = self.run_pipeline(1)
        self.assertIn('package pool changed', result.stderr)
        self.assertEqual(len(self.runs()), 1)

    def test_user_package_list_is_saved_and_reused(self):
        self.env.pop('BENCHMARK_PACKAGE_COUNT')
        packages = self.root / 'my packages.txt'
        packages.write_text('# packages to benchmark\n\nPackage-C  # one\nPackage-E\n')
        self.env['BENCHMARK_PACKAGES_FILE'] = str(packages)
        self.run_pipeline()
        selection = json.loads((self.experiment / 'benchmark-selection.json').read_text())
        self.assertEqual(selection['benchmark'], ['Package-C', 'Package-E'])
        run, = self.runs()
        self.assertEqual(len(list(run.glob('packages/*/result.json'))), 2)
        self.assertEqual((self.root / 'resutls/results-after-pharo-validation/dataset-summary.tex').read_text(), 'Packages: 2')
        # The saved list is reused without the file, and a different list is refused.
        self.env.pop('BENCHMARK_PACKAGES_FILE')
        self.run_pipeline()
        self.assertEqual(json.loads((self.runs()[-1] / 'selection.json').read_text())['benchmark'],
                         ['Package-C', 'Package-E'])
        packages.write_text('["Package-D"]')
        self.env['BENCHMARK_PACKAGES_FILE'] = str(packages)
        result = self.run_pipeline(1)
        self.assertIn('different benchmark package list', result.stderr)
        packages.write_text('Not-A-Package\n')
        self.env['EXPERIMENT_DIR'] = str(self.root / 'other-experiment')
        result = self.run_pipeline(1)
        self.assertIn('not benchmark-eligible', result.stderr)
        self.env['BENCHMARK_PACKAGE_COUNT'] = '1'
        result = self.run_pipeline(1)
        self.assertIn('unset BENCHMARK_PACKAGE_COUNT', result.stderr)

    def test_failed_package_is_resumed_without_repeating_finished_ones(self):
        self.env['BENCHMARK_PACKAGE_COUNT'] = '3'
        self.env['BENCHMARK_JOBS'] = '1'
        self.run_pipeline()
        failing = json.loads((self.experiment / 'benchmark-selection.json').read_text())['benchmark'][1]
        shutil.rmtree(self.runs()[0])
        (self.root / 'pharo-calls.jsonl').unlink()
        saved = self.results()
        (self.root / 'benchmark-complete').unlink()
        self.env['TEST_FAIL_PACKAGE'] = failing
        result = self.run_pipeline(1)
        self.assertIn('Resume with BENCHMARK_RESUME_DIR', result.stderr)
        self.assertFalse((self.root / 'benchmark-complete').exists())
        self.assertEqual(self.results(), saved)
        run, = self.runs()
        done = len(list(run.glob('packages/*/benchmarks/*/part.json')))
        self.assertGreater(done, 0)
        self.assertFalse(list(run.glob('packages/*/result.json')))
        self.env.pop('TEST_FAIL_PACKAGE')
        self.env['BENCHMARK_RESUME_DIR'] = str(run)
        (self.root / 'pharo-calls.jsonl').unlink()
        result = self.run_pipeline()
        # Only the benchmarks that had not finished run again.
        self.assertEqual(len(self.calls('coo-package-part-v1')), 3 * 28 - done)
        self.assertEqual(len(list(run.glob('packages/*/result.json'))), 3)
        self.assertEqual(self.runs(), [run])
        self.assertTrue((self.root / 'benchmark-complete').exists())

    def test_hosts_sharing_a_resumed_run_finish_it_once(self):
        self.env['BENCHMARK_PACKAGE_COUNT'] = '3'
        self.env['BENCHMARK_JOBS'] = '1'
        self.run_pipeline()
        failing = json.loads((self.experiment / 'benchmark-selection.json').read_text())['benchmark'][1]
        shutil.rmtree(self.runs()[0])
        (self.root / 'benchmark-complete').unlink()
        self.env['TEST_FAIL_PACKAGE'] = failing
        self.run_pipeline(1)
        run, = self.runs()
        done = len(list(run.glob('packages/*/benchmarks/*/part.json')))
        self.env.pop('TEST_FAIL_PACKAGE')
        # A job killed at its walltime leaves its locks behind.
        (self.experiment / '.running').mkdir()
        (self.root / 'pharo-calls.jsonl').unlink()
        env = dict(self.env, BENCHMARK_SHARED='1', BENCHMARK_RESUME_DIR=str(run))
        hosts = [subprocess.Popen(["bash", str(ROOT / 'pipeline-benchmarks.sh'), '--foreground'], env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                 for _ in range(2)]
        for host in hosts:
            output, _ = host.communicate(timeout=60)
            self.assertEqual(host.returncode, 0, output)
        (self.experiment / '.running').rmdir()
        self.assertFalse((self.root / 'resutls/.running').exists())
        # Between them, the hosts run each unfinished benchmark exactly once.
        self.assertEqual(len(self.calls('coo-package-part-v1')), 3 * 28 - done)
        self.assertEqual(len(list(run.glob('packages/*/result.json'))), 3)
        self.assertEqual(self.runs(), [run])
        self.assertTrue((self.root / 'benchmark-complete').exists())

    def test_shared_hosts_need_a_run_to_resume(self):
        self.env['BENCHMARK_SHARED'] = '1'
        result = self.run_pipeline(1)
        self.assertIn('BENCHMARK_SHARED needs BENCHMARK_RESUME_DIR', result.stderr)

    def test_resume_directory_must_be_a_prepared_run_of_this_experiment(self):
        self.run_pipeline()
        other = self.root / 'benchmark-run.elsewhere'
        other.mkdir()
        self.env['BENCHMARK_RESUME_DIR'] = str(other)
        result = self.run_pipeline(1)
        self.assertIn('prepared benchmark-run.* directory', result.stderr)

    def test_missing_figure_does_not_publish_partial_outputs(self):
        self.env['TEST_MISSING_FIGURE'] = '1'
        result = self.run_pipeline(1)
        self.assertIn('Missing or empty benchmark artifact', result.stderr)
        self.assertEqual(self.results(), {})

    def test_owned_ollama_stopped_after_run(self):
        self.env['TEST_START_OLLAMA'] = '1'
        self.env.pop('BENCHMARK_OLLAMA_HOST')
        self.run_pipeline()
        pid = int((self.root / 'ollama-pid').read_text())
        # Its own port, one request slot per worker, all four models kept loaded.
        env = json.loads((self.root / 'ollama-env.json').read_text())
        self.assertRegex(env['OLLAMA_HOST'], r'^127\.0\.0\.1:[1-9][0-9]*$')
        self.assertEqual(env['OLLAMA_NUM_PARALLEL'], '2')
        self.assertEqual(env['OLLAMA_MAX_LOADED_MODELS'], '4')
        self.assertEqual(env['OLLAMA_MODELS'], str(self.root / 'models'))
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)


if __name__ == '__main__':
    unittest.main()
