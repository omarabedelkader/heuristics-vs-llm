"""Freeze the benchmark source of one experiment and pin the image built from it.

An experiment benchmarks a frozen copy of the Smalltalk source (EXPERIMENT_DIR/repository),
loaded into its own Pharo image. When the code changes in the checkout, the next run
freezes the checkout again and loads it into the same image, so a run never measures
stale code. image-source.json records which source the image holds.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


NEEDS_UPDATE = 5


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha256(repository):
    """Hash the Smalltalk source: every file under src, by relative path and content."""
    root = repository / "src"
    if not root.is_dir():
        raise ValueError(f"Missing benchmark source directory: {root}")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == ".DS_Store":
            continue
        digest.update(path.relative_to(repository).as_posix().encode() + b"\0")
        digest.update(sha256(path).encode() + b"\n")
    return digest.hexdigest()


def origin(checkout):
    """Where the frozen source came from: the commit, and whether files differ from it."""
    def git(*arguments):
        result = subprocess.run(["git", "-C", str(checkout), *arguments], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    commit = git("rev-parse", "HEAD")
    if commit is None:
        return f"Local snapshot: {checkout}\n"
    changed = git("status", "--porcelain", "--", "src")
    return (f"Local snapshot: {checkout}\nCommit: {commit}"
            + (" with uncommitted changes\n" if changed else "\n"))


def copy_source(checkout, output, note=None):
    """Publish a complete copy of the checkout's source at output, replacing an older copy."""
    with tempfile.TemporaryDirectory(prefix="source.", dir=output.parent) as temporary:
        stage = Path(temporary) / "repository"
        shutil.copytree(checkout / "src", stage / "src", ignore=shutil.ignore_patterns(".DS_Store"))
        (stage / "origin.txt").write_text(note + "\n" if note else origin(checkout))
        if output.exists():
            # The previous copy leaves with the temporary directory.
            output.rename(Path(temporary) / "previous")
        stage.rename(output)


def sync(repository, checkout, note=None):
    """Freeze the checkout, or follow its changes; without a checkout keep the frozen copy."""
    if checkout is None or not (checkout / "src").is_dir():
        if repository.is_dir():
            return
        raise ValueError(f"No benchmark source: {checkout} has no src directory. "
                         "Set BENCHMARK_REPO_DIR to your HeuristicCompletion-Benchmarks-Multiples checkout.")
    if not repository.exists():
        copy_source(checkout, repository, note)
        print(f"Benchmark source: frozen from {checkout}")
    elif source_sha256(checkout) != source_sha256(repository):
        copy_source(checkout, repository, note)
        print(f"Benchmark source: code changed in {checkout}; this experiment now benchmarks it. "
              "Run directories made with the previous code cannot be resumed.")


def check_image(repository, image, pin):
    """0: the image holds the frozen source; NEEDS_UPDATE: load the source into it first."""
    if not pin.exists():
        return NEEDS_UPDATE
    saved = read_json(pin)
    if saved["imageSHA256"] != sha256(image):
        raise ValueError("Image does not match the experiment's frozen source; restore it or use a new EXPERIMENT_DIR")
    return 0 if saved["sourceSHA256"] == source_sha256(repository) else NEEDS_UPDATE


def pin_image(repository, image, pin):
    write_json(pin, dict(schema="coo-image-source-v2", sourceSHA256=source_sha256(repository),
                         imageSHA256=sha256(image)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("sync", "check-image", "pin-image"))
    parser.add_argument("--repository", type=Path, required=True, help="The experiment's frozen source")
    parser.add_argument("--checkout", type=Path, help="Local benchmark checkout, if any")
    parser.add_argument("--origin", help="sync: where a downloaded checkout came from")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--pin", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "sync":
            sync(args.repository, args.checkout, args.origin)
        elif args.action == "check-image":
            return check_image(args.repository, args.image, args.pin)
        else:
            pin_image(args.repository, args.image, args.pin)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Benchmark source error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
