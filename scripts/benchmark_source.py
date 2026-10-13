"""Freeze and check the benchmark source of one experiment.

The dataset pins the source and image that mined it. Mining again takes a day, and
most benchmark changes never touch the mined rows, so an experiment may freeze a
newer local checkout that differs from the dataset's source only in benchmark-only
code (the LLM and hybrid strategies, the publication chart and the tests). Its image
is then the dataset's image with that source loaded once, pinned in
image-source.json. Any other difference still needs a new dataset.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from ranking_corpus import load_manifest, read_json, repository_sha256, sha256, write_json


PACKAGE = "ExtendedHeuristicCompletion-Benchmarks"
# Classes of PACKAGE that produce no mined row and no re-ranker feature.
BENCHMARK_ONLY = ("CooLLM", "CooHybrid", "CooBenchmarkChart")
NEEDS_UPDATE = 5


def benchmark_only(path):
    parts = path.parts
    return len(parts) == 3 and parts[0] == "src" and (
        parts[1] == PACKAGE + "-Tests" or (parts[1] == PACKAGE and parts[2].startswith(BENCHMARK_ONLY)))


def dataset_sha256(repository):
    return repository_sha256(repository, skip=benchmark_only)


def origin(checkout):
    """Where the frozen source came from: the commit, and whether files differ from it."""
    def git(*arguments):
        result = subprocess.run(["git", "-C", str(checkout), *arguments], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    commit = git("rev-parse", "HEAD")
    if commit is None:
        return f"Local snapshot: {checkout}\n"
    changed = git("status", "--porcelain", "--", "src", "reranker", "scripts")
    return (f"Local snapshot: {checkout}\nCommit: {commit}"
            + (" with uncommitted changes\n" if changed else "\n"))


def freeze(mining, checkout, output):
    """Copy the experiment's source once: a newer benchmark-only checkout, else the dataset's."""
    source, note = mining, "the dataset's frozen source"
    if checkout is not None and (checkout / "src").is_dir() and (checkout / "reranker").is_dir():
        if repository_sha256(checkout) != repository_sha256(mining):
            if dataset_sha256(checkout) != dataset_sha256(mining):
                raise ValueError(
                    f"{checkout} differs from the dataset's source outside the LLM, hybrid, chart "
                    "and test code. Check out the benchmark revision you want, or mine a new "
                    "dataset with pipeline-mine-training-data.sh in a new MINING_DIR.")
            source, note = checkout, f"newer LLM/hybrid code from {checkout}"
    with tempfile.TemporaryDirectory(prefix="source.", dir=output.parent) as temporary:
        stage = Path(temporary) / "repository"
        if source is mining:
            shutil.copytree(mining, stage)
        else:
            for name in ("src", "scripts", "reranker"):
                if (source / name).is_dir():
                    shutil.copytree(source / name, stage / name,
                                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
            (stage / "origin.txt").write_text(origin(checkout))
        stage.rename(output)
    print(f"Benchmark source: {note}")


def check(mining, repository, checkout):
    """Reject a frozen source that changed, or that the dataset cannot serve."""
    if dataset_sha256(repository) != dataset_sha256(mining):
        raise ValueError("Frozen source does not match the corpus; restore its repository or mine a new dataset")
    if (checkout is not None and (checkout / "src").is_dir() and (checkout / "reranker").is_dir()
            and repository_sha256(checkout) != repository_sha256(repository)):
        print(f"WARNING: {checkout} differs from this experiment's frozen source, which is what "
              "will be benchmarked. Use a new EXPERIMENT_DIR to benchmark the checkout.", file=sys.stderr)


def check_image(corpus, mining, repository, image, pin):
    """0: image is the pinned one; NEEDS_UPDATE: the dataset's image, awaiting the newer source."""
    updated = repository_sha256(repository) != repository_sha256(mining)
    if not updated or not pin.exists():
        load_manifest(corpus, image=image)
        return NEEDS_UPDATE if updated else 0
    saved = read_json(pin)
    if saved["repositorySHA256"] != repository_sha256(repository) or saved["imageSHA256"] != sha256(image):
        raise ValueError("Image does not match the experiment's frozen source; restore it or use a new EXPERIMENT_DIR")
    return 0


def pin_image(corpus, repository, image, pin):
    manifest, _ = load_manifest(corpus)
    write_json(pin, dict(schema="coo-image-source-v1", repositorySHA256=repository_sha256(repository),
                         imageSHA256=sha256(image), datasetRepositorySHA256=manifest["repositorySHA256"],
                         datasetImageSHA256=manifest["imageSHA256"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "check", "check-image", "pin-image"))
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--mining", type=Path, required=True, help="The dataset's frozen source")
    parser.add_argument("--repository", type=Path, required=True, help="The experiment's frozen source")
    parser.add_argument("--checkout", type=Path, help="Local benchmark checkout, if any")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--pin", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "freeze":
            freeze(args.mining, args.checkout, args.repository)
        elif args.action == "check":
            check(args.mining, args.repository, args.checkout)
        elif args.action == "check-image":
            return check_image(args.corpus, args.mining, args.repository, args.image, args.pin)
        else:
            pin_image(args.corpus, args.repository, args.image, args.pin)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Benchmark source error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
