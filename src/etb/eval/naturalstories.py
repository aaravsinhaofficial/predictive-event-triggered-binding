from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from etb.eval.scoring import sentence_log_likelihood


def _r2(y: np.ndarray, x: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def evaluate_naturalstories(
    model: Any,
    tokenizer: Any,
    device: Any,
    path: str | Path,
    out_dir: str | Path,
) -> dict:
    data = pd.read_csv(path, sep="\t")
    rows: list[dict] = []
    for story_id, group in data.groupby("story_id", sort=True):
        prefix_tokens: list[str] = []
        for record in group.to_dict("records"):
            prefix_tokens.append(str(record["token"]))
            sentence = " ".join(prefix_tokens)
            scored = sentence_log_likelihood(model, tokenizer, sentence, device, return_trace=True)
            trace = scored["trace"]
            token_log_probs = [x for x in trace["token_log_probs"] if x is not None]
            surprisal = -float(token_log_probs[-1]) if token_log_probs else 0.0
            rows.append(
                {
                    "story_id": story_id,
                    "token_index": int(record["token_index"]),
                    "token": str(record["token"]),
                    "rt": float(record["rt"]),
                    "surprisal": surprisal,
                    "gate_prob": float(trace["gate_probs"][-1]) if trace["gate_probs"] else 0.0,
                    "gate_activation": (
                        float(trace["gate_activations"][-1]) if trace["gate_activations"] else 0.0
                    ),
                    "token_length": len(str(record["token"])),
                }
            )

    out = pd.DataFrame(rows)
    out_path = Path(out_dir) / "naturalstories_token_metrics.csv"
    parquet_path = Path(out_dir) / "naturalstories_token_metrics.parquet"
    out.to_csv(out_path, index=False)
    out.to_parquet(parquet_path, index=False)

    y = out["rt"].to_numpy(dtype=float)
    surprisal_x = out[["surprisal", "token_length"]].to_numpy(dtype=float)
    full_x = out[["surprisal", "gate_prob", "token_length"]].to_numpy(dtype=float)
    r2_surprisal = _r2(y, surprisal_x)
    r2_full = _r2(y, full_x)
    return {
        "task": "naturalstories",
        "n": int(len(out)),
        "r2_surprisal": r2_surprisal,
        "r2_surprisal_gate": r2_full,
        "delta_r2_gate": r2_full - r2_surprisal,
        "token_metrics": str(out_path),
        "token_metrics_parquet": str(parquet_path),
    }
