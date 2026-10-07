"""Train/validate/test a ranker with a fourth, strictly excluded benchmark group."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

from ranking_corpus import load_split, sha256, write_json


def read_partition(path, split, partition):
    allowed = set(split[partition])
    rows = []
    with path.open() as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (not isinstance(row, dict) or row.get("group") not in allowed
                    or row.get("package", row.get("group")) != row.get("group")
                    or row.get("schema") != "coo-ranking-v1"):
                raise ValueError(f"Forbidden or invalid {partition} row at {path}:{number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty {partition} partition")
    return rows


def train(args):
    # Validate membership before importing ML code, encoding or optimization.
    split = load_split(args.split)
    training = read_partition(args.training, split, "train")
    validation = read_partition(args.validation, split, "validation")
    import numpy as np
    import torch
    from features import SCHEMA, INPUTS, encode, request_from_row
    from model import Ranker
    from serve import Runtime

    torch.set_num_threads(1)
    torch.manual_seed(split["seed"])
    rng = random.Random(split["seed"])

    def tensors(row, k=None):
        return tuple(torch.from_numpy(v) for v in encode(request_from_row(row, k)).values())

    # Check every supplied row, including candidate misses, before fitting.
    for row in training + validation:
        encode(request_from_row(row))
    usable = [r for r in training if r["target"] in [c["name"] for c in r["candidates"]]]
    if not usable:
        raise ValueError("No positive candidates in training partition")
    model = Ranker(args.width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    def validation_metrics():
        reciprocal_ranks, losses = [], []
        model.eval()
        with torch.inference_mode():
            for row in validation:
                names = [c["name"] for c in row["candidates"]]
                if not names:
                    reciprocal_ranks.append(0.0)
                    continue
                logits = model(*tensors(row))
                ranked = [names[i] for i in np.argsort(-logits.numpy(), kind="stable")[:10]]
                reciprocal_ranks.append(1 / (ranked.index(row["target"]) + 1)
                                        if row["target"] in ranked else 0.0)
                if row["target"] in names:
                    losses.append(float(torch.nn.functional.cross_entropy(
                        logits[None, :], torch.tensor([names.index(row["target"])]))))
        return dict(validationMRR=sum(reciprocal_ranks) / len(validation),
                    validationLoss=sum(losses) / len(losses) if losses else None,
                    validationPositiveRows=len(losses))

    history = [dict(epoch=0, trainingLoss=None, **validation_metrics())]
    if not history[0]["validationPositiveRows"]:
        raise ValueError("Validation has no positive candidates; it cannot select a meaningful checkpoint")
    best_score, best_state, best_epoch = -1.0, None, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(usable)
        loss_sum = 0.0
        for row in usable:
            target = [c["name"] for c in row["candidates"]].index(row["target"])
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(*tensors(row))[None, :], torch.tensor([target]))
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
        metrics = dict(epoch=epoch, trainingLoss=loss_sum / len(usable), **validation_metrics())
        history.append(metrics)
        if metrics["validationMRR"] > best_score:
            best_score, best_epoch = metrics["validationMRR"], epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        print(json.dumps(metrics), flush=True)
    model.load_state_dict(best_state)
    model.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, tensors(usable[0]), str(args.output / "ranker.onnx"),
                      input_names=INPUTS, output_names=["logits"], opset_version=17,
                      dynamic_axes={"candidate_ids": {0: "candidates"}, "features": {0: "candidates"},
                                    "logits": {0: "candidates"}}, dynamo=False)
    metadata = dict(schema=SCHEMA, width=args.width, seed=split["seed"], epochs=args.epochs,
                    selection="validation-MRR", bestEpoch=best_epoch, validationMRR=best_score,
                    parameters=sum(p.numel() for p in model.parameters()),
                    trainGroups=split["train"], validationGroups=split["validation"],
                    testGroups=split["test"], benchmarkGroups=split["benchmark"], packageSplit=split,
                    trainingRows=len(training), trainingMisses=len(training) - len(usable),
                    validationRows=len(validation),
                    trainingSHA256=sha256(args.training), validationSHA256=sha256(args.validation),
                    fusionTrained=any(c.get("lmAgreement", False) for r in usable for c in r["candidates"]))
    write_json(args.output / "metadata.json", metadata)
    write_json(args.output / "learning-history.json", dict(
        epochs=history, bestEpoch=best_epoch, selection="validation-MRR",
        lossScope="Cross-entropy on rows with a positive candidate; training loss is the mean during each epoch",
        mrrScope="Validation MRR@10 includes all validation rows, including candidate misses"))
    runtime = Runtime(args.output)
    for row in usable[:5]:
        for k in (1, 10, 20, 30, 50):
            request = request_from_row(row, k)
            inputs = encode(request)
            if not request["candidates"]:
                continue
            with torch.inference_mode():
                expected = model(*(torch.from_numpy(v) for v in inputs.values())).numpy()
            np.testing.assert_allclose(runtime.session.run(None, inputs)[0], expected, rtol=1e-4, atol=1e-5)
    print(f"Selected epoch {best_epoch} using validation MRR only; ONNX parity passed", flush=True)


def rank_bucket(rank):
    if rank == 0:
        return 4
    return 0 if rank == 1 else 1 if rank <= 3 else 2 if rank <= 10 else 3


def ranking_metrics(ranks):
    return dict(mrr=sum(1 / r if 0 < r <= 10 else 0 for r in ranks) / len(ranks),
                **{f"accuracyAt{k}": sum(0 < r <= k for r in ranks) / len(ranks) for k in (1, 3, 10)})


def evaluate(args):
    split = load_split(args.split)
    rows = read_partition(args.data, split, "test")
    from features import request_from_row
    from serve import Runtime
    runtime = Runtime(args.model)
    if runtime.metadata.get("packageSplit") != split:
        raise ValueError("Model and evaluation package splits differ")
    summaries = []
    for k in (10, 20, 30, 50):
        if any(row.get("candidateLimit", 0) < k for row in rows):
            continue
        ranks, baseline_ranks, elapsed = [], [], []
        transitions = [[0] * 5 for _ in range(5)]
        for row in rows:
            request = request_from_row(row, k)
            start = time.perf_counter()
            result = runtime.rank(request)
            elapsed.append((time.perf_counter() - start) * 1000)
            ranked = [pair[0] for pair in result["scores"]]
            rank = ranked.index(row["target"]) + 1 if row["target"] in ranked else 0
            baseline = sorted((c for c in request["candidates"] if c["rank"] > 0), key=lambda c: c["rank"])
            names = [c["name"] for c in baseline]
            before = names.index(row["target"]) + 1 if row["target"] in names else 0
            ranks.append(rank)
            baseline_ranks.append(before)
            transitions[rank_bucket(before)][rank_bucket(rank)] += 1
        summaries.append(dict(k=k, count=len(rows), modelId=runtime.model_id,
                              partition="test", candidateRecall=sum(r > 0 for r in baseline_ranks) / len(rows),
                              unionCandidateRecall=sum(r > 0 for r in ranks) / len(rows),
                              **ranking_metrics(ranks), baseline=ranking_metrics(baseline_ranks),
                              meanMs=sum(elapsed) / len(elapsed),
                              timingScope="Python preprocessing + ONNX; no Pharo or HTTP",
                              rankTransitions=transitions))
    if not summaries:
        raise ValueError("Test corpus has no fully measured candidate limit >= 10")
    write_json(args.output, dict(schema="coo-ranker-evaluation-v2", partition="test",
                               testGroups=split["test"], dataSHA256=sha256(args.data), summaries=summaries))


def report(args):
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    history = json.loads((args.model / "learning-history.json").read_text())
    evaluation = json.loads(args.evaluation.read_text())
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42})

    def save(figure, name):
        for extension in ("png", "pdf"):
            figure.savefig(args.output / f"{name}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(figure)

    records = history["epochs"]
    epochs = [r["epoch"] for r in records]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    axes[0].plot(epochs[1:], [r["trainingLoss"] for r in records[1:]], label="Training (epoch mean)")
    axes[0].plot(epochs, [r["validationLoss"] for r in records], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy (positive candidates)")
    axes[0].legend()
    axes[1].plot(epochs, [r["validationMRR"] for r in records], marker="o", color="#167568")
    axes[1].axvline(history["bestEpoch"], linestyle="--", color="#666666", label="Selected checkpoint")
    axes[1].set(xlabel="Epoch (0 = untrained)", ylabel="Validation MRR@10", ylim=(0, 1))
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    save(figure, "learning-curves")

    summaries = evaluation["summaries"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for axis, metric, label in zip(axes, ("mrr", "accuracyAt1"), ("Test MRR@10", "Test accuracy@1")):
        for source, name, marker in (("baseline", "Dependency order", "s"), (None, "Neural re-ranker", "o")):
            axis.plot([r["k"] for r in summaries],
                      [(r[source] if source else r)[metric] for r in summaries], marker=marker, label=name)
        axis.set(xlabel="Candidate limit K", ylabel=label, ylim=(0, 1))
        axis.grid(alpha=0.2)
        axis.legend()
    save(figure, "test-performance")

    # Largest measured K is predetermined by corpus coverage, not chosen by test score.
    selected = summaries[-1]
    counts = selected["rankTransitions"]
    figure, axis = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    im = axis.imshow(counts, cmap="Blues")
    labels = ["1", "2–3", "4–10", ">10", "Absent"]
    axis.set(xticks=range(5), yticks=range(5), xticklabels=labels, yticklabels=labels,
             xlabel="Rank after re-ranking", ylabel="Rank before re-ranking",
             title=f"Test rank transitions (K={selected['k']}, n={selected['count']})")
    peak = max(max(row) for row in counts)
    for i, row in enumerate(counts):
        for j, value in enumerate(row):
            axis.text(j, i, str(value), ha="center", va="center", color="white" if value > peak / 2 else "black")
    figure.colorbar(im, ax=axis, label="Completion observations")
    save(figure, "test-rank-transitions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    training = sub.add_parser("train")
    testing = sub.add_parser("evaluate")
    plotting = sub.add_parser("report")
    for command in (training, testing):
        command.add_argument("--repository", required=True, type=Path)
        command.add_argument("--split", required=True, type=Path)
    training.add_argument("--training", required=True, type=Path)
    training.add_argument("--validation", required=True, type=Path)
    training.add_argument("--output", required=True, type=Path)
    training.add_argument("--epochs", type=int, default=10)
    training.add_argument("--width", type=int, choices=(16, 32, 64), default=32)
    testing.add_argument("--data", required=True, type=Path)
    testing.add_argument("--model", required=True, type=Path)
    testing.add_argument("--output", required=True, type=Path)
    plotting.add_argument("--model", required=True, type=Path)
    plotting.add_argument("--evaluation", required=True, type=Path)
    plotting.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "train" and args.epochs < 1:
        parser.error("epochs must be positive")
    if hasattr(args, "repository"):
        sys.path.insert(0, str(args.repository.resolve() / "reranker"))
    try:
        {"train": train, "evaluate": evaluate, "report": report}[args.action](args)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"Re-ranker workflow error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
