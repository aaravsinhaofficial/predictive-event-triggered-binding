from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from etb.eval.scoring import sentence_log_likelihood


def evaluate_syntaxgym(
    model: Any,
    tokenizer: Any,
    device: Any,
    path: str | Path,
    out_dir: str | Path,
) -> dict:
    outputs: list[dict] = []
    correct = 0
    total = 0
    for suite_path in _suite_paths(path):
        with Path(suite_path).open("r", encoding="utf-8") as handle:
            suite = json.load(handle)
        for item in suite.get("items", []):
            scores: dict[str, float] = {}
            for condition in item.get("conditions", []):
                name = str(condition["condition"])
                sentence = str(condition["sentence"])
                scores[name] = sentence_log_likelihood(model, tokenizer, sentence, device)[
                    "log_likelihood"
                ]
            for prediction in item.get("predictions", []):
                better = str(prediction["better"])
                worse = str(prediction["worse"])
                result = int(scores[better] > scores[worse])
                correct += result
                total += 1
                outputs.append(
                    {
                        "suite": suite.get("meta", {}).get("name", Path(suite_path).stem),
                        "item_number": item.get("item_number"),
                        "better": better,
                        "worse": worse,
                        "better_log_likelihood": scores[better],
                        "worse_log_likelihood": scores[worse],
                        "result": result,
                    }
                )

    out_path = Path(out_dir) / "syntaxgym_predictions.csv"
    pd.DataFrame(outputs).to_csv(out_path, index=False)
    return {
        "task": "syntaxgym",
        "accuracy": correct / max(1, total),
        "n": total,
        "predictions": str(out_path),
    }


def _suite_paths(path: str | Path) -> list[Path]:
    source = Path(path)
    if source.is_dir():
        return sorted(source.glob("*.json"))
    return [source]
