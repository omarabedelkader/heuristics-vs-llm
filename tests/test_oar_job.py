"""Exercise the OAR wrapper without downloading models or starting benchmarks."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OarJobTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='oar job ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        shutil.copy2(ROOT / 'job.oar.sh', self.root)
        binaries = self.root / 'bin'
        binaries.mkdir()
        self.env = {**os.environ, 'TEST_ROOT': str(self.root), 'OAR_JOB_ID': '123',
                    'PATH': str(binaries) + os.pathsep + os.environ['PATH']}
        for key in ('RUN_DIR', 'BENCHMARK_PACKAGE_COUNT', 'BENCHMARK_JOBS', 'OLLAMA_MODELS_DIR'):
            self.env.pop(key, None)
        self.executable(binaries / 'curl', '''
import os,sys
from pathlib import Path
root=Path(os.environ['TEST_ROOT'])
if '-o' in sys.argv:
    (root/'ollama-download').write_text(sys.argv[sys.argv.index('-o')-1])
    Path(sys.argv[sys.argv.index('-o')+1]).write_bytes(b'archive')
elif not (root/'server-ready').exists():
    sys.exit(7)
else:
    print('{"models": []}')
''')
        self.executable(binaries / 'tar', '''
import os,sys,shutil
from pathlib import Path
if '--help' in sys.argv: print('zstd');sys.exit(0)
Path('bin').mkdir(exist_ok=True)
shutil.copy2(Path(os.environ['TEST_ROOT'])/'fake-ollama','bin/ollama')
''')
        self.executable(self.root / 'fake-ollama', '''
import os,sys,time
from pathlib import Path
root=Path(os.environ['TEST_ROOT'])
if sys.argv[1]=='serve':
    (root/'server-pid').write_text(str(os.getpid()))
    (root/'server-ready').touch()
    time.sleep(30)
elif sys.argv[1]=='list':
    print('NAME ID SIZE')
    print('pharo-llm/Qwen2.5-Coder-SFT:0.5B cached 1GB')
elif sys.argv[1]=='pull':
    with (root/'pulls').open('a') as out: out.write(sys.argv[2]+'\\n')
elif sys.argv[1]=='--version': print('test Ollama')
''')
        (self.root / 'pipeline-benchmarks.sh').write_text(f'#!/bin/bash\nexec "{sys.executable}" "$TEST_ROOT/pipeline-stub" "$@"\n')
        self.executable(self.root / 'pipeline-stub', '''
import os,sys,json
from pathlib import Path
root=Path(os.environ['TEST_ROOT'])
os.kill(int((root/'server-pid').read_text()),0)
(root/'invocation.json').write_text(json.dumps(dict(args=sys.argv[1:],env={k:os.environ[k] for k in ['BENCHMARK_PACKAGE_COUNT','BENCHMARK_JOBS','EXPERIMENT_DIR','RESULTS_DIR','OLLAMA_BIN','OLLAMA_MODELS_DIR']})))
sys.exit(int(os.environ.get('TEST_PIPELINE_STATUS','0')))
''')

    def executable(self, path, body):
        path.write_text(f'#!{sys.executable}\n' + body)
        path.chmod(0o755)

    def run_job(self, *args, expected=0):
        result = subprocess.run(['bash', str(self.root / 'job.oar.sh'), *args], env=self.env,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        pid_file = self.root / 'server-pid'
        if pid_file.exists():
            with self.assertRaises(ProcessLookupError):
                os.kill(int(pid_file.read_text()), 0)
        return result

    def test_original_model_downloads_and_new_pipeline_defaults(self):
        self.run_job('--skip-reranker-benchmarks')
        invocation = json.loads((self.root / 'invocation.json').read_text())
        self.assertEqual(invocation['env']['BENCHMARK_PACKAGE_COUNT'], '250')
        self.assertEqual(invocation['env']['BENCHMARK_JOBS'], '8')
        self.assertEqual(invocation['args'], ['--skip-reranker-benchmarks'])
        self.assertEqual(invocation['env']['RESULTS_DIR'], str(self.root / 'oar-runs/123/results'))
        self.assertIn('https://ollama.com/download/ollama-linux-', (self.root / 'ollama-download').read_text())
        self.assertEqual((self.root / 'pulls').read_text().splitlines(), [
            'pharo-llm/Qwen2.5-Coder-SFT:1.5B', 'pharo-llm/Qwen2.5-Coder-SFT:3b',
            'pharo-llm/Qwen2.5-Coder-SFT:7b'])

    def test_user_counts_are_forwarded_and_pipeline_failure_cleans_up(self):
        self.env.update(BENCHMARK_PACKAGE_COUNT='100', BENCHMARK_JOBS='3', TEST_PIPELINE_STATUS='23')
        self.run_job(expected=23)
        invocation = json.loads((self.root / 'invocation.json').read_text())
        self.assertEqual(invocation['env']['BENCHMARK_PACKAGE_COUNT'], '100')
        self.assertEqual(invocation['env']['BENCHMARK_JOBS'], '3')

    def test_bad_counts_fail_before_downloads(self):
        for key in ('BENCHMARK_PACKAGE_COUNT', 'BENCHMARK_JOBS'):
            self.env[key] = '0'
            result = self.run_job(expected=1)
            self.assertIn('positive integer', result.stderr)
            self.assertFalse((self.root / 'ollama-download').exists())
            self.env.pop(key)


if __name__ == '__main__':
    unittest.main()
