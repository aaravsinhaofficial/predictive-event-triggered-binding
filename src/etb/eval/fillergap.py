from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from etb.eval.scoring import sentence_log_likelihood_batch


def evaluate_fillergap(
    model: Any,
    tokenizer: Any,
    device: Any,
    path: str | Path,
    out_dir: str | Path,
) -> dict:
    data = pd.read_csv(path)
    scored = []
    records = data.to_dict("records")
    scores = sentence_log_likelihood_batch(
        model,
        tokenizer,
        [str(row["sentence"]) for row in records],
        device,
    )
    for row, score in zip(records, scores, strict=True):
        scored.append({**row, "log_likelihood": score["log_likelihood"]})
    scored_df = pd.DataFrame(scored)

    decisions = []
    for item_id, group in scored_df.groupby("item_id"):
        good = group[group["expected_good"].astype(int) == 1]
        bad = group[group["expected_good"].astype(int) == 0]
        if good.empty or bad.empty:
            continue
        good_score = float(good["log_likelihood"].max())
        bad_score = float(bad["log_likelihood"].max())
        decisions.append(
            {
                "item_id": item_id,
                "construction": group["construction"].iloc[0],
                "good_log_likelihood": good_score,
                "bad_log_likelihood": bad_score,
                "correct": int(good_score > bad_score),
            }
        )
    out_scored = Path(out_dir) / "fillergap_scored.csv"
    out_decisions = Path(out_dir) / "fillergap_decisions.csv"
    scored_df.to_csv(out_scored, index=False)
    pd.DataFrame(decisions).to_csv(out_decisions, index=False)
    return {
        "task": "fillergap",
        "accuracy": sum(row["correct"] for row in decisions) / max(1, len(decisions)),
        "n": len(decisions),
        "scored": str(out_scored),
        "decisions": str(out_decisions),
    }
