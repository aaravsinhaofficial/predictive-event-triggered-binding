from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from etb.utils import ensure_dir, read_jsonl


def _maybe_read_training_metrics(run_dir: Path) -> pd.DataFrame:
    metrics = run_dir / "metrics.jsonl"
    if not metrics.exists():
        return pd.DataFrame()
    return pd.DataFrame(read_jsonl(metrics))


def plot_compute_accuracy(run_dir: Path, fig_dir: Path) -> Path | None:
    train = _maybe_read_training_metrics(run_dir)
    eval_path = run_dir / "eval" / "metrics.json"
    if train.empty or not eval_path.exists():
        return None
    with eval_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    accuracy = None
    for task in ("blimp", "syntaxgym", "fillergap"):
        if task in metrics and "accuracy" in metrics[task]:
            accuracy = float(metrics[task]["accuracy"])
            break
    if accuracy is None:
        return None
    last = train.iloc[-1]
    out = fig_dir / "compute_accuracy_pareto.png"
    plt.figure(figsize=(5, 4))
    plt.scatter([last["activated_flops_per_token"]], [accuracy], s=80)
    plt.xlabel("Activated FLOPs / token")
    plt.ylabel("Accuracy")
    plt.title("Compute-Accuracy Pareto Point")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def plot_naturalstories_trace(run_dir: Path, fig_dir: Path) -> Path | None:
    path = run_dir / "eval" / "naturalstories_token_metrics.csv"
    if not path.exists():
        return None
    data = pd.read_csv(path)
    out = fig_dir / "naturalstories_gate_trace.png"
    plt.figure(figsize=(9, 4))
    x = data["token_index"]
    plt.plot(x, data["rt"], label="reading time", color="#4C78A8")
    ax2 = plt.gca().twinx()
    ax2.plot(x, data["gate_prob"], label="gate probability", color="#F58518")
    plt.gca().set_xlabel("Token index")
    plt.gca().set_ylabel("Reading time")
    ax2.set_ylabel("Gate probability")
    plt.title("Gate Trace Over Natural Stories Fixture")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def gate_breakdown(run_dir: Path, fig_dir: Path) -> Path | None:
    path = run_dir / "eval" / "naturalstories_token_metrics.csv"
    if not path.exists():
        return None
    data = pd.read_csv(path)
    data["event_class"] = data["token"].map(
        lambda tok: "punctuation" if str(tok) in {".", ",", "?", "!", ";", ":"} else "word"
    )
    summary = (
        data.groupby("event_class")
        .agg(gate_rate=("gate_activation", "mean"), gate_prob=("gate_prob", "mean"), n=("token", "size"))
        .reset_index()
    )
    csv_path = fig_dir / "gate_breakdown.csv"
    summary.to_csv(csv_path, index=False)
    out = fig_dir / "gate_breakdown.png"
    plt.figure(figsize=(5, 4))
    plt.bar(summary["event_class"], summary["gate_prob"], color=["#54A24B", "#E45756"])
    plt.ylabel("Mean gate probability")
    plt.title("Gate Rate by Event Class")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def make_figures(run_dir: str | Path) -> list[Path]:
    run = Path(run_dir)
    fig_dir = ensure_dir(run / "figures")
    made = [
        plot_compute_accuracy(run, fig_dir),
        plot_naturalstories_trace(run, fig_dir),
        gate_breakdown(run, fig_dir),
    ]
    return [path for path in made if path is not None]

