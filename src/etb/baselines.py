from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd

from etb.config import ExperimentConfig
from etb.eval.runner import evaluate
from etb.training import train
from etb.utils import ensure_dir, read_jsonl

DEFAULT_BASELINE_VARIANTS = [
    "etb",
    "cheap_only",
    "dense_gru",
    "always_on",
    "generic_dynamic",
    "anira_emergent",
    "punctuation_only",
    "random_matched",
]


def run_baseline_grid(
    config: ExperimentConfig,
    variants: list[str] | None = None,
    tasks: str = "fixture",
    max_steps: int | None = None,
) -> Path:
    selected = variants or DEFAULT_BASELINE_VARIANTS
    root = ensure_dir(Path("outputs") / "baselines" / config.run_name)
    summaries: list[dict] = []
    for variant in selected:
        raw = copy.deepcopy(config.raw)
        raw["run_name"] = f"{config.run_name}_{variant}"
        raw["output_dir"] = str(root / variant)
        raw.setdefault("model", {})["variant"] = variant
        if max_steps is not None:
            raw.setdefault("training", {})["max_steps"] = int(max_steps)
        variant_config = ExperimentConfig(raw=raw, path=config.path)
        checkpoint = train(variant_config)
        eval_metrics = evaluate(variant_config, checkpoint=checkpoint, tasks=tasks)
        train_metrics_path = variant_config.output_dir / "metrics.jsonl"
        last_train = read_jsonl(train_metrics_path)[-1] if train_metrics_path.exists() else {}
        summaries.append(_flatten_summary(variant, checkpoint, last_train, eval_metrics))

    summary_csv = root / "summary.csv"
    summary_json = root / "summary.json"
    pd.DataFrame(summaries).to_csv(summary_csv, index=False)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    return summary_csv


def _flatten_summary(
    variant: str,
    checkpoint: Path,
    train_metrics: dict,
    eval_metrics: dict,
) -> dict:
    row = {
        "variant": variant,
        "checkpoint": str(checkpoint),
        "loss": train_metrics.get("loss"),
        "gate_rate": train_metrics.get("gate_rate"),
        "gate_prob_mean": train_metrics.get("gate_prob_mean"),
        "information_gain_mean": train_metrics.get("information_gain_mean"),
        "activated_flops_per_token": train_metrics.get("activated_flops_per_token"),
    }
    lm = eval_metrics.get("language_modeling", {})
    row["perplexity"] = lm.get("perplexity")
    row["bits_per_byte"] = lm.get("bits_per_byte")
    for task in ("blimp", "syntaxgym", "fillergap"):
        if task in eval_metrics:
            row[f"{task}_accuracy"] = eval_metrics[task].get("accuracy")
    if "naturalstories" in eval_metrics:
        row["naturalstories_delta_r2_gate"] = eval_metrics["naturalstories"].get("delta_r2_gate")
    return row
