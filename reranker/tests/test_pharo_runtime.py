"""Check native VM adaptation leaves the saved image untouched."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pharo_runtime


class PharoRuntimeTests(unittest.TestCase):
    def test_linux_runtime_preserves_image_and_records_pinned_binary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / 'image'
            (image / 'pharo-vm/Pharo.app').mkdir(parents=True)
            (image / 'Pharo.image').write_bytes(b'frozen image bytes')
            fixture = root / 'linux.zip'
            with zipfile.ZipFile(fixture, 'w') as archive:
                for name in ('pharo', 'lib/pharo', 'lib/libPharoVMCore.so'):
                    info = zipfile.ZipInfo(name)
                    info.external_attr = 0o100755 << 16
                    archive.writestr(info, b'Linux executable')
            digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
            def run(args, **kwargs):
                if args[0] == 'curl': shutil.copyfile(fixture, args[-1])
            with patch.object(pharo_runtime.platform, 'system', return_value='Linux'), \
                 patch.object(pharo_runtime.platform, 'machine', return_value='x86_64'), \
                 patch.object(pharo_runtime.subprocess, 'run', side_effect=run), \
                 patch.object(pharo_runtime, 'LINUX_VM_SHA256', digest):
                pharo_runtime.ensure_runtime(image)
                pharo_runtime.ensure_runtime(image)
            self.assertEqual((image / 'Pharo.image').read_bytes(), b'frozen image bytes')
            self.assertFalse((image / 'pharo-vm/Pharo.app').exists())
            self.assertTrue((image / 'pharo-vm/lib/pharo').exists())
            self.assertEqual((image / 'pharo').stat().st_mode & 0o777, 0o755)
            self.assertIn('exec "$DIR/pharo-vm/pharo" --headless "$@"', (image / 'pharo').read_text())
            self.assertEqual(json.loads((image / 'runtime-origin.json').read_text())['archiveSHA256'], digest)

    def test_unexpected_vm_bytes_leave_existing_runtime_intact(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / 'image'
            (image / 'pharo-vm/Pharo.app').mkdir(parents=True)
            (image / 'Pharo.image').write_bytes(b'frozen')
            def run(args, **kwargs): Path(args[-1]).write_bytes(b'wrong download')
            with patch.object(pharo_runtime.platform, 'system', return_value='Linux'), \
                 patch.object(pharo_runtime.platform, 'machine', return_value='x86_64'), \
                 patch.object(pharo_runtime.subprocess, 'run', side_effect=run):
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    pharo_runtime.ensure_runtime(image)
            self.assertTrue((image / 'pharo-vm/Pharo.app').exists())
            self.assertEqual((image / 'Pharo.image').read_bytes(), b'frozen')


if __name__ == '__main__':
    unittest.main()
