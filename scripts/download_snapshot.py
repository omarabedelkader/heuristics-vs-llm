"""Restore the pinned public Hugging Face snapshot; exit 3 only if it is absent."""
import argparse
import gzip
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from urllib.parse import quote
import zipfile

from ranking_corpus import load_split, read_json, sha256, verify
from pharo_runtime import can_adapt, ensure_runtime


DEFAULT_REPO = "pharo-llm/pharo-reranker-dataset"
DEFAULT_REVISION = "904b689aed068d765a0fac4bd664e8e60773cd4c"
DEFAULT_MANIFEST_SHA256 = "1bada374b6f5ea050da2f792407e817671a8628f3bae0c95f7f279b9daaed1f9"
IMAGE_ARCHIVE = "artifacts/pharo-image-macos-arm64.zip"
SOURCE_ARCHIVE = "artifacts/workspace.zip"
CORPUS_FILES = ("mining/corpus/manifest.json", "mining/corpus/packages.json",
                "mining/corpus/all.jsonl.gz")


def download(base, name, directory, optional=False):
    target = directory / name
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["curl", "--silent", "--show-error", "--location", "--fail",
         "--retry", "3", "--connect-timeout", "30", "--speed-limit", "1024",
         "--speed-time", "120", "--output", str(target), "--write-out", "%{http_code}",
         base + "/" + quote(name, safe="/")], capture_output=True, text=True)
    status = result.stdout.strip()
    if optional and status == "404" and result.returncode in (0, 22):
        return None
    if result.returncode or status != "200":
        raise ValueError(f"Snapshot download failed for {name} (HTTP {status or 'unknown'}, "
                         f"curl exit {result.returncode}); no mining fallback. Retry the download.")
    return target


def extract(archive, destination, members):
    """Extract only required inputs, preserving executable modes, never archived scripts."""
    found = set()
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            name = entry.filename
            path = PurePosixPath(name)
            mode = entry.external_attr >> 16
            if path.is_absolute() or ".." in path.parts or "\\" in name or stat.S_ISLNK(mode):
                raise ValueError(f"Unsafe ZIP member: {name}")
            if name not in members:
                continue
            if name in found or entry.is_dir():
                raise ValueError(f"Duplicate or invalid ZIP member: {name}")
            found.add(name)
            expected = members[name]
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(entry) as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            if target.stat().st_size != expected["size"] or sha256(target) != expected["sha256"]:
                raise ValueError(f"ZIP member checksum mismatch: {name}")
            target.chmod(int(expected["mode"], 8) & 0o777)
    if found != set(members):
        raise ValueError(f"Snapshot ZIP is missing required members: {sorted(set(members) - found)}")


def restore(mining, repo, revision):
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("RANKING_DATASET_REPO must be owner/name and RANKING_DATASET_REVISION a full commit SHA")
    mining.mkdir(parents=True, exist_ok=True)
    lock = mining / ".running"
    try:
        lock.mkdir()
    except FileExistsError:
        raise ValueError(f"Mining is already running: {mining} (.running lock)") from None
    try:
        if (mining / "corpus/manifest.json").is_file():
            verify(mining / "corpus", repository=mining / "repository")
            return 0
        base = f"https://huggingface.co/datasets/{repo}/resolve/{revision}"
        with tempfile.TemporaryDirectory(prefix=".snapshot-pending.", dir=mining) as temporary:
            stage = Path(temporary)
            checksum_file = download(base, "SHA256SUMS", stage, optional=True)
            if checksum_file is None:
                print(f"No published snapshot at {repo}@{revision}", flush=True)
                return 3
            # No snapshot may overwrite unfinished mining or a partial restore.
            for name in ("corpus", "image", "repository", "image-ready", "snapshot-selection", "snapshot-origin.json"):
                if (mining / name).exists() or (mining / name).is_symlink():
                    raise ValueError(f"Incomplete local mining state at {mining / name}; "
                                     "restore it or use a new MINING_DIR. Nothing was overwritten.")
            checksums = {}
            for line in checksum_file.read_text().splitlines():
                digest, name = line.split("  ", 1)
                if not re.fullmatch(r"[0-9a-f]{64}", digest) or name in checksums:
                    raise ValueError("Invalid snapshot checksum list")
                checksums[name] = digest

            def checked(name):
                if name not in checksums:
                    raise ValueError(f"Snapshot checksum missing: {name}")
                print(f"Downloading snapshot file: {name}", flush=True)
                path = download(base, name, stage)
                if sha256(path) != checksums[name]:
                    raise ValueError(f"Snapshot checksum mismatch: {name}")
                return path

            snapshot_path = checked("snapshot.json")
            if (repo, revision) == (DEFAULT_REPO, DEFAULT_REVISION):
                if sha256(snapshot_path) != DEFAULT_MANIFEST_SHA256:
                    raise ValueError("Pinned snapshot manifest checksum mismatch")
            snapshot = read_json(snapshot_path)
            if snapshot["schema"] != "pharo-reranker-snapshot-v1":
                raise ValueError("Unsupported snapshot schema")
            runtime = snapshot["runtime"]
            if ((runtime["os"], runtime["architecture"]) != (platform.system(), platform.machine())
                    and not can_adapt(runtime)):
                raise ValueError(f"Snapshot VM requires {runtime['os']} {runtime['architecture']}; "
                                 f"this host is {platform.system()} {platform.machine()}. "
                                 "Use a snapshot with a compatible VM, or prepare local mining data "
                                 "with pipeline-mine-training-data.sh. No automatic re-mining was performed.")
            required = {}
            for name, info in snapshot["archiveMembers"].items():
                if (name.startswith(("mining/image/", "mining/repository/")) or
                        name in ("mining/image-ready", "experiment/benchmark-selection.json", "experiment/split.json")):
                    required[name] = info
            storage = snapshot["corpusStorage"]
            needed = (storage["uncompressedBytes"] + storage["compressedBytes"] +
                      sum(info["size"] for info in required.values()))
            if shutil.disk_usage(mining).free < needed:
                raise ValueError(f"Insufficient disk space to restore snapshot: need approximately "
                                 f"{needed / 1e9:.1f} GB free, plus space for training and workers")
            unpacked = stage / "restored"
            for archive_name in (IMAGE_ARCHIVE, SOURCE_ARCHIVE):
                archive = checked(archive_name)
                extract(archive, unpacked, {n: i for n, i in required.items() if i["archive"] == archive_name})
                archive.unlink()
            for name in CORPUS_FILES:
                checked(name)
            corpus = stage / "mining/corpus"
            with gzip.open(corpus / "all.jsonl.gz", "rb") as src, (corpus / "all.jsonl").open("xb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            (corpus / "all.jsonl.gz").unlink()
            if (corpus / "all.jsonl").stat().st_size != storage["uncompressedBytes"]:
                raise ValueError("Decompressed corpus size disagrees with snapshot")
            restored = unpacked / "mining"
            manifest = verify(corpus, image=restored / "image/Pharo.image",
                              repository=restored / "repository",
                              split_path=unpacked / "experiment/split.json")
            for key in ("corpusSHA256", "imageSHA256", "repositorySHA256"):
                if manifest[key] != snapshot[key]:
                    raise ValueError(f"Corpus provenance disagrees with snapshot: {key}")
            if set(read_json(corpus / "packages.json")) != set(manifest["eligible"]):
                raise ValueError("Snapshot package list disagrees with corpus manifest")
            split = load_split(unpacked / "experiment/split.json", set(manifest["eligible"]))
            selection = read_json(unpacked / "experiment/benchmark-selection.json")
            if (set(selection["eligible"]) != set(split["eligible"]) or
                    selection["benchmark"] != split["benchmark"] or selection["seed"] != split["seed"]):
                raise ValueError("Snapshot benchmark selection disagrees with package split")
            if not os.access(restored / "image/pharo", os.X_OK):
                raise ValueError("Snapshot Pharo launcher is not executable")
            ensure_runtime(restored / "image")
            (unpacked / "experiment").rename(restored / "snapshot-selection")
            (restored / "snapshot-origin.json").write_text(json.dumps({
                "repo": repo, "revision": revision, "snapshotSHA256": sha256(snapshot_path)
            }, indent=2) + "\n")
            # The completed corpus is the publication marker; install it last.
            for name in ("image", "repository", "image-ready", "snapshot-selection", "snapshot-origin.json"):
                (restored / name).rename(mining / name)
            corpus.rename(mining / "corpus")
            print(f"Restored verified snapshot: {manifest['rows']} rows from {repo}@{revision}", flush=True)
        return 0
    finally:
        lock.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mining", type=Path)
    parser.add_argument("--repo", default=os.environ.get("RANKING_DATASET_REPO", DEFAULT_REPO))
    parser.add_argument("--revision", default=os.environ.get("RANKING_DATASET_REVISION", DEFAULT_REVISION))
    args = parser.parse_args()
    def interrupted(number, frame):
        raise SystemExit(128 + number)
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, interrupted)
    try:
        return restore(args.mining, args.repo, args.revision)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, EOFError) as error:
        print(f"Snapshot error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
