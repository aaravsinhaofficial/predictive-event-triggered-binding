from __future__ import annotations

import json
from pathlib import Path

from etb.config import ExperimentConfig
from etb.eval.blimp import evaluate_blimp
from etb.eval.fillergap import evaluate_fillergap
from etb.eval.naturalstories import evaluate_naturalstories
from etb.eval.scoring import load_checkpoint, text_perplexity
from etb.eval.syntaxgym import evaluate_syntaxgym
from etb.utils import ensure_dir


def evaluate(
    config: ExperimentConfig,
    checkpoint: str | Path,
    tasks: str = "fixture",
) -> dict:
    out_dir = ensure_dir(config.output_dir / "eval")
    model, tokenizer, device = load_checkpoint(checkpoint, config.device)
    summaries: dict[str, dict | float] = {}

    if config.data.get("eval_text_path"):
        summaries["language_modeling"] = text_perplexity(
            model,
            tokenizer,
            config.data["eval_text_path"],
            device,
        )

    task_cfg = dict(config.evaluation.get("tasks", {}))
    if tasks == "none":
        task_cfg = {}
    elif tasks != "fixture" and tasks:
        requested = {task.strip() for task in tasks.split(",") if task.strip()}
        task_cfg = {key: value for key, value in task_cfg.items() if key in requested}

    if task_cfg.get("blimp"):
        summaries["blimp"] = evaluate_blimp(model, tokenizer, device, task_cfg["blimp"], out_dir)
    if task_cfg.get("syntaxgym"):
        summaries["syntaxgym"] = evaluate_syntaxgym(
            model, tokenizer, device, task_cfg["syntaxgym"], out_dir
        )
    if task_cfg.get("fillergap"):
        summaries["fillergap"] = evaluate_fillergap(
            model, tokenizer, device, task_cfg["fillergap"], out_dir
        )
    if task_cfg.get("naturalstories"):
        summaries["naturalstories"] = evaluate_naturalstories(
            model, tokenizer, device, task_cfg["naturalstories"], out_dir
        )

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    return summaries

