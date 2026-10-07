"""Prepare and partition a reusable Pharo ranking corpus using only the stdlib."""
import argparse
from contextlib import ExitStack
import hashlib
import json
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


def load_split(path, eligible):
    split = read_json(path)
    if split.get("schema") != "coo-package-split-v1" or type(split.get("seed")) is not int:
        raise ValueError("Invalid package split schema or seed")
    pools = {key: package_names(split.get(key), key)
             for key in ("eligible", "train", "benchmark")}
    if pools["eligible"] != eligible:
        raise ValueError("Saved split and prepared corpus have different package pools")
    if pools["train"] & pools["benchmark"]:
        raise ValueError("Benchmark packages must never enter training")
    if pools["train"] | pools["benchmark"] != eligible:
        raise ValueError("Training must contain every remaining eligible package")
    return split


def scan_rows(path, eligible, train=None, outputs=None):
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
            if outputs is not None:
                outputs["training" if group in train else "test"].write(line.rstrip(b"\r\n") + b"\n")
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
        raise ValueError("Prepared corpus belongs to a different image; use the original experiment")
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
    train = set(split["train"])
    output.mkdir(parents=True, exist_ok=True)
    for name in ("training.jsonl", "test.jsonl", "corpus.json"):
        if (output / name).exists():
            raise ValueError(f"Refusing to overwrite {output / name}; use a fresh run directory")
    # No training file becomes visible until every row and the checksum pass.
    with tempfile.TemporaryDirectory(prefix="partition-", dir=output) as temporary:
        stage = Path(temporary)
        with ExitStack() as stack:
            outputs = {name: stack.enter_context((stage / f"{name}.jsonl").open("wb"))
                       for name in ("training", "test")}
            counts, digest = scan_rows(corpus / "all.jsonl", eligible, train, outputs)
        if digest != manifest.get("corpusSHA256") or counts != manifest["rowsByPackage"]:
            raise ValueError("Prepared corpus checksum/count mismatch; restore the complete corpus")
        training_rows = sum(counts[name] for name in train)
        test_rows = sum(counts[name] for name in split["benchmark"])
        if not training_rows or not test_rows:
            raise ValueError("Saved split must have nonempty training and test data; no packages were redrawn")
        audit = dict(manifest, packageSplit=split, trainingRows=training_rows, testRows=test_rows)
        write_json(stage / "corpus.json", audit)
        for name in ("training.jsonl", "test.jsonl", "corpus.json"):
            (stage / name).replace(output / name)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("finalize", "verify", "partition"))
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--image", type=Path, help="Mining image: required for finalize; optional provenance check otherwise")
    parser.add_argument("--repository", type=Path, help="Frozen source used by both independent images")
    parser.add_argument("--split", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "finalize" and args.image is None:
        parser.error("finalize requires --image")
    if args.action == "partition" and (args.split is None or args.output is None):
        parser.error("partition requires --split and --output")
    try:
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
        print(f"Training: {result['trainingRows']} rows; held-out test: {result['testRows']} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
