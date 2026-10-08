"""Prepare and partition a reusable Pharo ranking corpus using only the stdlib."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import random
from pathlib import Path
import sys
import tempfile


SCHEMA = "coo-ranking-corpus-v1"


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


def repository_sha256(repository):
    digest = hashlib.sha256()
    for directory in ("src", "reranker"):
        root = repository / directory
        if not root.is_dir():
            raise ValueError(f"Missing frozen source directory: {root}")
        for path in sorted(root.rglob("*")):
            if (not path.is_file() or "__pycache__" in path.parts
                    or path.suffix == ".pyc" or path.name == ".DS_Store"):
                continue
            digest.update(path.relative_to(repository).as_posix().encode() + b"\0")
            digest.update(sha256(path).encode() + b"\n")
    return digest.hexdigest()


def package_names(value, label):
    if (not isinstance(value, list) or not value
            or any(not isinstance(name, str) or not name for name in value)
            or len(value) != len(set(value))):
        raise ValueError(f"Invalid or duplicate package names in {label}")
    return set(value)


PARTITIONS = ("train", "validation", "test", "benchmark")
FILES = {"train": "training", "validation": "validation", "test": "test"}


def validate_split(split, eligible=None):
    if split.get("schema") != "coo-package-split-v2" or type(split.get("seed")) is not int:
        raise ValueError("A four-part coo-package-split-v2 split is required; use a new EXPERIMENT_DIR for older runs")
    pools = {key: package_names(split.get(key), key) for key in ("eligible", *PARTITIONS)}
    # Packages left out by TRAINING_PACKAGE_COUNT; absent or empty otherwise.
    unused = split.get("unused", [])
    pools["unused"] = package_names(unused, "unused") if unused else set()
    if eligible is not None and pools["eligible"] != eligible:
        raise ValueError("Saved split and prepared corpus have different package pools")
    seen = set()
    for key in (*PARTITIONS, "unused"):
        if seen & pools[key]:
            raise ValueError("Train, validation, test, benchmark and unused packages must be mutually disjoint")
        seen.update(pools[key])
    if seen != pools["eligible"]:
        raise ValueError("The partitions must contain every eligible package exactly once")
    return split


def load_split(path, eligible=None):
    return validate_split(read_json(path), eligible)


def create_split(corpus, selection_path, output, ratios=(80, 10, 10), training_packages=None):
    """Reserve the saved benchmark selection, then split only the remaining packages.

    training_packages keeps only that many (seeded random) training packages; the
    others are recorded as unused and enter no model-development file."""
    _, eligible = load_manifest(corpus)
    selection = read_json(selection_path)
    if package_names(selection.get("eligible"), "selection eligible") != eligible:
        raise ValueError("Saved selection and prepared corpus have different package pools")
    benchmark = package_names(selection.get("benchmark"), "selection benchmark")
    if not benchmark < eligible or type(selection.get("seed")) is not int:
        raise ValueError("Invalid benchmark selection or seed")
    if len(ratios) != 3 or any(type(n) is not int or n <= 0 for n in ratios) or sum(ratios) != 100:
        raise ValueError("RANKING_SPLIT_RATIOS must be three positive percentages summing to 100")
    if output.exists():
        split = load_split(output, eligible)
        if (set(split["benchmark"]) != benchmark or split["seed"] != selection["seed"]
                or split.get("ratios") != list(ratios)
                or split.get("trainingPackageCount") != training_packages):
            raise ValueError("Saved split differs from selection, ratios or TRAINING_PACKAGE_COUNT; "
                             "use a new EXPERIMENT_DIR")
        return split
    remaining = sorted(eligible - benchmark)
    if len(remaining) < 3:
        raise ValueError("Reserve at least three non-benchmark packages for train, validation and test")
    random.Random(selection["seed"]).shuffle(remaining)
    # Largest-remainder allocation, with at least one package in every ML partition.
    exact = [len(remaining) * n / 100 for n in ratios]
    counts = [int(n) for n in exact]
    for index in sorted(range(3), key=lambda i: (-(exact[i] - counts[i]), i))[:len(remaining) - sum(counts)]:
        counts[index] += 1
    for index in range(3):
        if counts[index] == 0:
            donor = max(range(3), key=lambda i: counts[i])
            counts[donor] -= 1
            counts[index] = 1
    train_end, validation_end = counts[0], counts[0] + counts[1]
    train = remaining[:train_end]
    split = dict(schema="coo-package-split-v2", seed=selection["seed"], ratios=list(ratios),
                 eligible=sorted(eligible), benchmark=selection["benchmark"],
                 train=sorted(train), validation=sorted(remaining[train_end:validation_end]),
                 test=sorted(remaining[validation_end:]))
    if training_packages is not None:
        if type(training_packages) is not int or not 0 < training_packages <= len(train):
            raise ValueError(f"TRAINING_PACKAGE_COUNT must be between 1 and {len(train)} "
                             "(the training packages available after the split)")
        # `remaining` is already seeded-shuffled, so a prefix is a random subset.
        split.update(train=sorted(train[:training_packages]), unused=sorted(train[training_packages:]),
                     trainingPackageCount=training_packages)
    validate_split(split, eligible)
    write_json(output, split)
    return split


def scan_rows(path, eligible, destinations=None, outputs=None):
    """Stream the corpus; retain zero-row packages and hash the exact input bytes."""
    counts = dict.fromkeys(sorted(eligible), 0)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for number, line in enumerate(source, 1):
            digest.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Row {number}: expected an object")
            group = row.get("group")
            if not isinstance(group, str) or group not in eligible:
                raise ValueError(f"Row {number}: unknown package {group!r}")
            if row.get("package", group) != group:
                raise ValueError(f"Row {number}: package and group disagree")
            if row.get("schema") != "coo-ranking-v1":
                raise ValueError(f"Row {number}: unsupported ranking schema")
            counts[group] += 1
            if outputs is not None and group in destinations:
                outputs[destinations[group]].write(line.rstrip(b"\r\n") + b"\n")
    if not sum(counts.values()):
        raise ValueError("Empty prepared corpus")
    return counts, digest.hexdigest()


def finalize(corpus, image, repository=None):
    manifest_path = corpus / "manifest.json"
    if manifest_path.exists():
        raise ValueError("Corpus already finalized; refusing to replace its manifest")
    eligible = package_names(read_json(corpus / "packages.json"), "packages.json")
    counts, digest = scan_rows(corpus / "all.jsonl", eligible)
    manifest = dict(schema=SCHEMA, eligible=sorted(eligible), rowsByPackage=counts,
                    rows=sum(counts.values()), corpusSHA256=digest,
                    imageSHA256=sha256(image))
    if repository is not None:
        manifest["repositorySHA256"] = repository_sha256(repository)
    write_json(manifest_path, manifest)
    return manifest


def load_manifest(corpus, image=None, repository=None):
    manifest = read_json(corpus / "manifest.json")
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported corpus manifest schema")
    eligible = package_names(manifest.get("eligible"), "manifest")
    counts = manifest.get("rowsByPackage")
    if (not isinstance(counts, dict) or set(counts) != eligible
            or any(type(n) is not int or n < 0 for n in counts.values())
            or not sum(counts.values()) or sum(counts.values()) != manifest.get("rows")):
        raise ValueError("Invalid corpus package counts")
    if image is not None and manifest.get("imageSHA256") != sha256(image):
        raise ValueError("Image does not match the prepared corpus; restore the original mining image snapshot or use a new EXPERIMENT_DIR with that snapshot")
    if repository is not None and manifest.get("repositorySHA256") != repository_sha256(repository):
        raise ValueError("Frozen source does not match the corpus; restore its repository or mine a new dataset")
    return manifest, eligible


def verify(corpus, image=None, split_path=None, repository=None):
    manifest, eligible = load_manifest(corpus, image, repository)
    if split_path is not None:
        load_split(split_path, eligible)
    if sha256(corpus / "all.jsonl") != manifest.get("corpusSHA256"):
        raise ValueError("Prepared corpus checksum mismatch; restore the complete corpus")
    return manifest


def partition(corpus, image, split_path, output, repository=None):
    manifest, eligible = load_manifest(corpus, image, repository)
    split = load_split(split_path, eligible)
    destinations = {name: FILES[key] for key in FILES for name in split[key]}
    output.mkdir(parents=True, exist_ok=True)
    for name in ("training.jsonl", "validation.jsonl", "test.jsonl", "corpus.json"):
        if (output / name).exists():
            raise ValueError(f"Refusing to overwrite {output / name}; use a fresh run directory")
    # No training file becomes visible until every row and the checksum pass.
    with tempfile.TemporaryDirectory(prefix="partition-", dir=output) as temporary:
        stage = Path(temporary)
        with ExitStack() as stack:
            outputs = {name: stack.enter_context((stage / f"{name}.jsonl").open("wb"))
                       for name in FILES.values()}
            counts, digest = scan_rows(corpus / "all.jsonl", eligible, destinations, outputs)
        if digest != manifest.get("corpusSHA256") or counts != manifest["rowsByPackage"]:
            raise ValueError("Prepared corpus checksum/count mismatch; restore the complete corpus")
        row_counts = {key: sum(counts[name] for name in split.get(key, [])) for key in (*PARTITIONS, "unused")}
        if any(row_counts[key] == 0 for key in FILES):
            raise ValueError("Saved split must have nonempty train, validation and test data; no packages were redrawn")
        audit = dict(manifest, packageSplit=split, trainingRows=row_counts["train"],
                     validationRows=row_counts["validation"], testRows=row_counts["test"],
                     excludedBenchmarkRows=row_counts["benchmark"], excludedUnusedRows=row_counts["unused"])
        audit["partitionSHA256"] = {name: sha256(stage / f"{name}.jsonl") for name in FILES.values()}
        write_json(stage / "corpus.json", audit)
        for name in ("training.jsonl", "validation.jsonl", "test.jsonl", "corpus.json"):
            (stage / name).replace(output / name)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("finalize", "verify", "partition", "split"))
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--image", type=Path, help="Mining image: required for finalize; optional provenance check otherwise")
    parser.add_argument("--repository", type=Path, help="Frozen source used by both independent images")
    parser.add_argument("--split", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--ratios", default="80,10,10")
    parser.add_argument("--training-packages", type=int, help="Keep only this many training packages")
    args = parser.parse_args()
    if args.action == "finalize" and args.image is None:
        parser.error("finalize requires --image")
    if args.action == "partition" and (args.split is None or args.output is None):
        parser.error("partition requires --split and --output")
    if args.action == "split" and (args.selection is None or args.output is None):
        parser.error("split requires --selection and --output")
    try:
        if args.action == "split":
            result = create_split(args.corpus, args.selection, args.output,
                                  tuple(int(n) for n in args.ratios.split(",")), args.training_packages)
            print("Package partitions: " + ", ".join(f"{key}={len(result.get(key, []))}"
                                                     for key in (*PARTITIONS, "unused")))
            return 0
        if args.action == "finalize":
            result = finalize(args.corpus, args.image, args.repository)
        elif args.action == "verify":
            result = verify(args.corpus, args.image, args.split, args.repository)
        else:
            result = partition(args.corpus, args.image, args.split, args.output, args.repository)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Corpus error: {error}", file=sys.stderr)
        return 1
    print(f"Corpus {args.action}: {result['rows']} rows across {len(result['eligible'])} packages")
    if args.action == "partition":
        print(f"Training: {result['trainingRows']}; validation: {result['validationRows']}; "
              f"test: {result['testRows']}; excluded benchmark rows: {result['excludedBenchmarkRows']}; "
              f"unused rows (TRAINING_PACKAGE_COUNT): {result['excludedUnusedRows']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
