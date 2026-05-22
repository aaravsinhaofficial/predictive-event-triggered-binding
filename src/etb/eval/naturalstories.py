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


def _cv_r2_by_group(y: np.ndarray, x: np.ndarray, groups: np.ndarray) -> float:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return float("nan")
    preds = np.zeros_like(y, dtype=float)
    for group in unique_groups:
        train = groups != group
        test = groups == group
        design_train = np.column_stack([np.ones(int(train.sum())), x[train]])
        beta, *_ = np.linalg.lstsq(design_train, y[train], rcond=None)
        design_test = np.column_stack([np.ones(int(test.sum())), x[test]])
        preds[test] = design_test @ beta
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _add_spillovers(frame: pd.DataFrame, column: str) -> None:
    grouped = frame.groupby("story_id", sort=False)[column]
    frame[f"{column}_lag1"] = grouped.shift(1).fillna(0.0)
    frame[f"{column}_lag2"] = grouped.shift(2).fillna(0.0)


def _standardized_columns(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].to_numpy(dtype=float)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return (values - mean) / np.maximum(std, 1e-6)


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
        cheap_token_log_probs = trace.get("cheap_token_log_probs") or []
        gate_probs = trace["gate_probs"]
        gate_activations = trace["gate_activations"]
        token_offset = 0
        for record in records:
            piece_count = len(tokenizer.encode(str(record["token"]), add_special_tokens=False))
            piece_count = max(1, piece_count)
            start = token_offset + 1
            stop = token_offset + piece_count + 1
            piece_log_probs = [x for x in token_log_probs[start:stop] if x is not None]
            cheap_piece_log_probs = [
                x for x in cheap_token_log_probs[start:stop] if x is not None
            ]
            surprisal = -float(sum(piece_log_probs)) if piece_log_probs else 0.0
            cheap_surprisal = (
                -float(sum(cheap_piece_log_probs)) if cheap_piece_log_probs else surprisal
            )
            gate_index = min(stop - 1, len(gate_probs) - 1)
            rows.append(
                {
                    "story_id": story_id,
                    "token_index": int(record["token_index"]),
                    "token": str(record["token"]),
                    "rt": float(record["rt"]),
                    "surprisal": surprisal,
                    "cheap_surprisal": cheap_surprisal,
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
    _add_spillovers(out, "surprisal")
    _add_spillovers(out, "cheap_surprisal")
    out["cheap_surprisal_sq"] = out["cheap_surprisal"].pow(2)
    out["cheap_surprisal_cube"] = out["cheap_surprisal"].pow(3)
    if "log_frequency" in out.columns:
        out["cheap_surprisal_x_log_frequency"] = (
            out["cheap_surprisal"] * out["log_frequency"]
        )
    else:
        out["log_frequency"] = 0.0
        out["cheap_surprisal_x_log_frequency"] = 0.0

    out_path = Path(out_dir) / "naturalstories_token_metrics.csv"
    parquet_path = Path(out_dir) / "naturalstories_token_metrics.parquet"
    out.to_csv(out_path, index=False)
    out.to_parquet(parquet_path, index=False)

    y = out["rt"].to_numpy(dtype=float)
    groups = out["story_id"].to_numpy()
    basic_controls = ["surprisal", "token_length", "log_frequency"]
    matched_controls = ["cheap_surprisal", "token_length", "log_frequency"]
    flexible_controls = [
        "cheap_surprisal",
        "cheap_surprisal_sq",
        "cheap_surprisal_cube",
        "cheap_surprisal_lag1",
        "cheap_surprisal_lag2",
        "token_length",
        "log_frequency",
        "cheap_surprisal_x_log_frequency",
    ]

    basic_x = _standardized_columns(out, basic_controls)
    basic_gate_x = np.column_stack([basic_x, out["gate_prob"].to_numpy(dtype=float)])
    matched_x = _standardized_columns(out, matched_controls)
    matched_gate_x = np.column_stack([matched_x, out["gate_prob"].to_numpy(dtype=float)])
    flexible_x = _standardized_columns(out, flexible_controls)
    flexible_gate_x = np.column_stack([flexible_x, out["gate_prob"].to_numpy(dtype=float)])

    r2_surprisal = _r2(y, basic_x)
    r2_full = _r2(y, basic_gate_x)
    r2_matched = _r2(y, matched_x)
    r2_matched_gate = _r2(y, matched_gate_x)
    r2_flexible = _r2(y, flexible_x)
    r2_flexible_gate = _r2(y, flexible_gate_x)
    cv_r2_basic = _cv_r2_by_group(y, basic_x, groups)
    cv_r2_basic_gate = _cv_r2_by_group(y, basic_gate_x, groups)
    cv_r2_flexible = _cv_r2_by_group(y, flexible_x, groups)
    cv_r2_flexible_gate = _cv_r2_by_group(y, flexible_gate_x, groups)
    gate_y = out["gate_prob"].to_numpy(dtype=float)
    return {
        "task": "naturalstories",
        "n": int(len(out)),
        "baseline_controls": basic_controls,
        "r2_surprisal": r2_surprisal,
        "r2_surprisal_gate": r2_full,
        "delta_r2_gate": r2_full - r2_surprisal,
        "matched_surprisal_controls": matched_controls,
        "r2_matched_surprisal": r2_matched,
        "r2_matched_surprisal_gate": r2_matched_gate,
        "delta_r2_gate_matched_surprisal": r2_matched_gate - r2_matched,
        "flexible_surprisal_controls": flexible_controls,
        "r2_flexible_surprisal": r2_flexible,
        "r2_flexible_surprisal_gate": r2_flexible_gate,
        "delta_r2_gate_flexible_surprisal": r2_flexible_gate - r2_flexible,
        "cv_by_story_r2_surprisal": cv_r2_basic,
        "cv_by_story_r2_surprisal_gate": cv_r2_basic_gate,
        "cv_by_story_delta_r2_gate": cv_r2_basic_gate - cv_r2_basic,
        "cv_by_story_r2_flexible_surprisal": cv_r2_flexible,
        "cv_by_story_r2_flexible_surprisal_gate": cv_r2_flexible_gate,
        "cv_by_story_delta_r2_gate_flexible_surprisal": (
            cv_r2_flexible_gate - cv_r2_flexible
        ),
        "gate_r2_from_basic_controls": _r2(gate_y, basic_x),
        "gate_r2_from_matched_controls": _r2(gate_y, matched_x),
        "gate_r2_from_flexible_controls": _r2(gate_y, flexible_x),
        "token_metrics": str(out_path),
        "token_metrics_parquet": str(parquet_path),
    }
