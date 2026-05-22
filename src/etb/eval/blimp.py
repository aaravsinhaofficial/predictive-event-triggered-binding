from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from etb.eval.scoring import sentence_log_likelihood_batch
from etb.utils import read_jsonl, write_jsonl


def _load_blimp_rows(path: str | Path) -> list[dict]:
    source = Path(path)
    if source.is_dir():
        rows: list[dict] = []
        for jsonl in sorted(source.glob("*.jsonl")):
            rows.extend(read_jsonl(jsonl))
        return rows
    return read_jsonl(source)


def evaluate_blimp(
    model: Any,
    tokenizer: Any,
    device: Any,
    path: str | Path,
    out_dir: str | Path,
) -> dict:
    rows = _load_blimp_rows(path)
    outputs: list[dict] = []
    correct = 0
    by_uid: dict[str, list[int]] = {}
    good_scores = sentence_log_likelihood_batch(
        model,
        tokenizer,
        [str(row["sentence_good"]) for row in rows],
        device,
    )
    bad_scores = sentence_log_likelihood_batch(
        model,
        tokenizer,
        [str(row["sentence_bad"]) for row in rows],
        device,
    )
    for row, good, bad in zip(rows, good_scores, bad_scores, strict=True):
        is_correct = int(good["log_likelihood"] > bad["log_likelihood"])
        correct += is_correct
        uid = str(row.get("UID", "unknown"))
        by_uid.setdefault(uid, []).append(is_correct)
        outputs.append(
            {
                "UID": uid,
                "pairID": row.get("pairID"),
                "sentence_good": row["sentence_good"],
                "sentence_bad": row["sentence_bad"],
                "good_log_likelihood": good["log_likelihood"],
                "bad_log_likelihood": bad["log_likelihood"],
                "correct": is_correct,
            }
        )

    out_path = Path(out_dir) / "blimp_predictions.jsonl"
    write_jsonl(out_path, outputs)
    summary = {
        "task": "blimp",
        "accuracy": correct / max(1, len(rows)),
        "n": len(rows),
        "by_uid": {uid: sum(vals) / len(vals) for uid, vals in by_uid.items()},
        "predictions": str(out_path),
    }
    pd.DataFrame(outputs).to_csv(Path(out_dir) / "blimp_predictions.csv", index=False)
    return summary
