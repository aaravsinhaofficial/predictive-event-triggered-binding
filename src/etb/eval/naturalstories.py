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
        records = group.to_dict("records")
        sentence = " ".join(str(record["token"]) for record in records)
        scored = sentence_log_likelihood(model, tokenizer, sentence, device, return_trace=True)
        trace = scored["trace"]
        token_log_probs = trace["token_log_probs"]
        gate_probs = trace["gate_probs"]
        gate_activations = trace["gate_activations"]
        token_offset = 0
        for record in records:
            piece_count = len(tokenizer.encode(str(record["token"]), add_special_tokens=False))
            piece_count = max(1, piece_count)
            start = token_offset + 1
            stop = token_offset + piece_count + 1
            piece_log_probs = [x for x in token_log_probs[start:stop] if x is not None]
            surprisal = -float(sum(piece_log_probs)) if piece_log_probs else 0.0
            gate_index = min(stop - 1, len(gate_probs) - 1)
            rows.append(
                {
                    "story_id": story_id,
                    "token_index": int(record["token_index"]),
                    "token": str(record["token"]),
                    "rt": float(record["rt"]),
                    "surprisal": surprisal,
                    "gate_prob": float(gate_probs[gate_index]) if gate_probs else 0.0,
                    "gate_activation": (
                        float(gate_activations[gate_index]) if gate_activations else 0.0
                    ),
                    "token_length": len(str(record["token"])),
                }
            )
            token_offset += piece_count

    out = pd.DataFrame(rows)
    if "log_frequency" in data.columns:
        freq_lookup = {
            (int(record["story_id"]), int(record["token_index"])): float(record["log_frequency"])
            for record in data.to_dict("records")
        }
        out["log_frequency"] = [
            freq_lookup.get((int(row.story_id), int(row.token_index)), 0.0)
            for row in out.itertuples(index=False)
        ]
    out_path = Path(out_dir) / "naturalstories_token_metrics.csv"
    parquet_path = Path(out_dir) / "naturalstories_token_metrics.parquet"
    out.to_csv(out_path, index=False)
    out.to_parquet(parquet_path, index=False)

    y = out["rt"].to_numpy(dtype=float)
    controls = ["surprisal", "token_length"]
    if "log_frequency" in out.columns:
        controls.append("log_frequency")
    surprisal_x = out[controls].to_numpy(dtype=float)
    full_x = out[[*controls, "gate_prob"]].to_numpy(dtype=float)
    r2_surprisal = _r2(y, surprisal_x)
    r2_full = _r2(y, full_x)
    return {
        "task": "naturalstories",
        "n": int(len(out)),
        "baseline_controls": controls,
        "r2_surprisal": r2_surprisal,
        "r2_surprisal_gate": r2_full,
        "delta_r2_gate": r2_full - r2_surprisal,
        "token_metrics": str(out_path),
        "token_metrics_parquet": str(parquet_path),
    }
