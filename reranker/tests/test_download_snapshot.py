"""Check safe extraction of remote ZIP input and immutable revision validation."""
import hashlib
from pathlib import Path
import stat
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from download_snapshot import extract, restore


class SnapshotExtractionTests(unittest.TestCase):
    def test_archive_cannot_escape_destination_or_create_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, mode in [('../outside', 0o644), ('/absolute', 0o644),
                               ('mining/image/link', stat.S_IFLNK | 0o777)]:
                with self.subTest(name=name):
                    info = zipfile.ZipInfo(name)
                    info.external_attr = mode << 16
                    with zipfile.ZipFile(root / 'bad.zip', 'w') as archive:
                        archive.writestr(info, b'outside')
                    with self.assertRaisesRegex(ValueError, 'Unsafe ZIP member'):
                        extract(root / 'bad.zip', root / 'output', {})
            self.assertFalse((root / 'output').exists())

    def test_only_requested_inputs_are_extracted_with_executable_modes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = b'#!/bin/sh\n'
            info = zipfile.ZipInfo('mining/image/pharo')
            info.external_attr = (stat.S_IFREG | 0o755) << 16
            with zipfile.ZipFile(root / 'image.zip', 'w') as archive:
                archive.writestr(info, payload)
                archive.writestr('pipeline-benchmarks.sh', b'must not overwrite current code')
            member = dict(size=len(payload), sha256=hashlib.sha256(payload).hexdigest(), mode='0o755')
            extract(root / 'image.zip', root / 'output', {info.filename: member})
            self.assertEqual((root / 'output' / info.filename).stat().st_mode & 0o777, 0o755)
            self.assertFalse((root / 'output/pipeline-benchmarks.sh').exists())
            with self.assertRaisesRegex(ValueError, 'missing required members'):
                extract(root / 'image.zip', root / 'missing', {'mining/image/missing': member})
            member['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                extract(root / 'image.zip', root / 'corrupt', {info.filename: member})

    def test_moving_revision_is_rejected_before_local_state_is_created(self):
        with tempfile.TemporaryDirectory() as temporary:
            mining = Path(temporary) / 'mining'
            with self.assertRaisesRegex(ValueError, 'full commit SHA'):
                restore(mining, 'owner/dataset', 'main')
            self.assertFalse(mining.exists())


if __name__ == '__main__':
    unittest.main()
