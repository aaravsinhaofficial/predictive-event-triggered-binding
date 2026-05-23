#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RUN_RE = re.compile(r"babylm_10m_v7_publishable_(?:(?P<condition>.+)_)?s(?P<seed>\d+)$")

KEY_METRICS = [
    "blimp_accuracy",
    "syntaxgym_accuracy",
    "fillergap_accuracy",
    "naturalstories_delta_r2_gate",
    "naturalstories_delta_r2_gate_matched_surprisal",
    "naturalstories_delta_r2_gate_flexible_surprisal",
    "naturalstories_cv_by_story_delta_r2_gate_flexible_surprisal",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build post-hoc tables for the v7 publishable/Insights analysis."
    )
    parser.add_argument("--root", type=Path, default=Path("outputs/baselines"))
    parser.add_argument("--pattern", default="babylm_10m_v7_publishable_*")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/insights/v7_publishable"))
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--permutation-repeats", type=int, default=500)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def _run_parts(run_name: str) -> tuple[str, int | None]:
    match = RUN_RE.fullmatch(run_name)
    if match is None:
        return "unknown", None
    return match.group("condition") or "main", int(match.group("seed"))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _latest_jsonl(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    last = ""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = line
    return json.loads(last) if last else {}


def _get(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def collect_results(root: Path, pattern: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(root.glob(f"{pattern}/*/eval/metrics.json")):
        variant_dir = metrics_path.parents[1]
        run_dir = metrics_path.parents[2]
        condition, seed = _run_parts(run_dir.name)
        metrics = _read_json(metrics_path)
        train = _latest_jsonl(variant_dir / "metrics.jsonl")
        ns = metrics.get("naturalstories", {})
        row = {
            "run_name": run_dir.name,
            "condition": condition,
            "seed": seed,
            "variant": variant_dir.name,
            "checkpoint": str(variant_dir / "checkpoint-final"),
            "loss": train.get("loss"),
            "lm_loss": train.get("lm_loss"),
            "gate_rate": train.get("gate_rate"),
            "gate_prob_mean": train.get("gate_prob_mean"),
            "memory_residual_scale": train.get("memory_residual_scale"),
            "contrastive_loss": train.get("contrastive_loss"),
            "rt_gate_loss": train.get("rt_gate_loss"),
            "blimp_accuracy": _get(metrics, "blimp", "accuracy"),
            "syntaxgym_accuracy": _get(metrics, "syntaxgym", "accuracy"),
            "fillergap_accuracy": _get(metrics, "fillergap", "accuracy"),
            "naturalstories_n": ns.get("n"),
            "naturalstories_delta_r2_gate": ns.get("delta_r2_gate"),
            "naturalstories_delta_r2_gate_matched_surprisal": ns.get(
                "delta_r2_gate_matched_surprisal"
            ),
            "naturalstories_delta_r2_gate_flexible_surprisal": ns.get(
                "delta_r2_gate_flexible_surprisal"
            ),
            "naturalstories_cv_by_story_delta_r2_gate": ns.get(
                "cv_by_story_delta_r2_gate"
            ),
            "naturalstories_cv_by_story_delta_r2_gate_flexible_surprisal": ns.get(
                "cv_by_story_delta_r2_gate_flexible_surprisal"
            ),
            "naturalstories_gate_r2_from_basic_controls": ns.get(
                "gate_r2_from_basic_controls"
            ),
            "naturalstories_gate_r2_from_matched_controls": ns.get(
                "gate_r2_from_matched_controls"
            ),
            "naturalstories_gate_r2_from_flexible_controls": ns.get(
                "gate_r2_from_flexible_controls"
            ),
            "naturalstories_token_metrics": ns.get("token_metrics"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _standardized(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return (values - mean) / np.maximum(std, 1e-6)


def _r2(y: np.ndarray, x: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _delta_r2(frame: pd.DataFrame, controls: list[str], gate: np.ndarray | None = None) -> float:
    clean = frame.dropna(subset=["rt", *controls, "gate_prob"]).copy()
    if clean.empty:
        return float("nan")
    y = clean["rt"].to_numpy(dtype=float)
    x = _standardized(clean[controls].to_numpy(dtype=float))
    gate_values = clean["gate_prob"].to_numpy(dtype=float) if gate is None else gate
    full_x = np.column_stack([x, gate_values])
    return _r2(y, full_x) - _r2(y, x)


def _ensure_flexible_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "cheap_surprisal_sq" not in out:
        out["cheap_surprisal_sq"] = out["cheap_surprisal"].pow(2)
    if "cheap_surprisal_cube" not in out:
        out["cheap_surprisal_cube"] = out["cheap_surprisal"].pow(3)
    if "cheap_surprisal_lag1" not in out:
        out["cheap_surprisal_lag1"] = out.groupby("story_id")["cheap_surprisal"].shift(1).fillna(0)
    if "cheap_surprisal_lag2" not in out:
        out["cheap_surprisal_lag2"] = out.groupby("story_id")["cheap_surprisal"].shift(2).fillna(0)
    if "log_frequency" not in out:
        out["log_frequency"] = 0.0
    if "cheap_surprisal_x_log_frequency" not in out:
        out["cheap_surprisal_x_log_frequency"] = out["cheap_surprisal"] * out["log_frequency"]
    return out


def _naturalstories_uncertainty(
    row: pd.Series,
    rng: np.random.Generator,
    bootstrap_repeats: int,
    permutation_repeats: int,
) -> dict[str, Any]:
    token_path = row.get("naturalstories_token_metrics")
    if not isinstance(token_path, str) or not Path(token_path).exists():
        return {}
    frame = _ensure_flexible_features(pd.read_csv(token_path))
    controls = [
        "cheap_surprisal",
        "cheap_surprisal_sq",
        "cheap_surprisal_cube",
        "cheap_surprisal_lag1",
        "cheap_surprisal_lag2",
        "token_length",
        "log_frequency",
        "cheap_surprisal_x_log_frequency",
    ]
    observed = _delta_r2(frame, controls)

    boot: list[float] = []
    if bootstrap_repeats > 0:
        grouped = [group for _, group in frame.groupby("story_id", sort=False)]
        for _ in range(bootstrap_repeats):
            parts = [
                group.iloc[rng.integers(0, len(group), size=len(group))]
                for group in grouped
                if len(group) > 0
            ]
            boot_frame = pd.concat(parts, ignore_index=True)
            boot.append(_delta_r2(boot_frame, controls))

    null: list[float] = []
    if permutation_repeats > 0:
        clean = frame.dropna(subset=["rt", *controls, "gate_prob"]).copy()
        story_indices = [
            np.flatnonzero(clean["story_id"].to_numpy() == story)
            for story in clean["story_id"].drop_duplicates().to_numpy()
        ]
        base_gate = clean["gate_prob"].to_numpy(dtype=float)
        for _ in range(permutation_repeats):
            shuffled = base_gate.copy()
            for indices in story_indices:
                shuffled[indices] = rng.permutation(shuffled[indices])
            null.append(_delta_r2(clean, controls, gate=shuffled))

    boot_array = np.array(boot, dtype=float)
    null_array = np.array(null, dtype=float)
    result = {
        "run_name": row["run_name"],
        "condition": row["condition"],
        "seed": row["seed"],
        "variant": row["variant"],
        "observed_delta_r2_flexible": observed,
        "bootstrap_repeats": bootstrap_repeats,
        "permutation_repeats": permutation_repeats,
    }
    if len(boot_array):
        result["bootstrap_mean"] = float(np.nanmean(boot_array))
        result["bootstrap_ci95_low"] = float(np.nanpercentile(boot_array, 2.5))
        result["bootstrap_ci95_high"] = float(np.nanpercentile(boot_array, 97.5))
    if len(null_array):
        result["permutation_null_mean"] = float(np.nanmean(null_array))
        result["permutation_null_ci95_low"] = float(np.nanpercentile(null_array, 2.5))
        result["permutation_null_ci95_high"] = float(np.nanpercentile(null_array, 97.5))
        result["permutation_p_one_sided"] = float(
            (1 + np.sum(null_array >= observed)) / (len(null_array) + 1)
        )
    return result


def build_uncertainty(
    results: pd.DataFrame,
    bootstrap_repeats: int,
    permutation_repeats: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _, row in results.iterrows():
        diagnostics = _naturalstories_uncertainty(
            row, rng, bootstrap_repeats, permutation_repeats
        )
        if diagnostics:
            rows.append(diagnostics)
    return pd.DataFrame(rows)


def build_etb_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    main = results[results["condition"] == "main"]
    rows: list[dict[str, Any]] = []
    for seed, group in main.groupby("seed", dropna=False):
        etb_rows = group[group["variant"] == "etb"]
        if etb_rows.empty:
            continue
        etb = etb_rows.iloc[0]
        for _, baseline in group[group["variant"] != "etb"].iterrows():
            row = {
                "seed": seed,
                "baseline": baseline["variant"],
            }
            for metric in KEY_METRICS:
                row[f"etb_{metric}"] = etb.get(metric)
                row[f"baseline_{metric}"] = baseline.get(metric)
                row[f"etb_minus_baseline_{metric}"] = _as_float(etb.get(metric)) - _as_float(
                    baseline.get(metric)
                )
            rows.append(row)
    return pd.DataFrame(rows)


def build_rt_supervision_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    etb = results[results["variant"] == "etb"]
    for seed, group in etb.groupby("seed", dropna=False):
        main_rows = group[group["condition"] == "main"]
        rt_rows = group[group["condition"] == "rt_supervised_upper_bound"]
        if main_rows.empty or rt_rows.empty:
            continue
        main = main_rows.iloc[0]
        rt = rt_rows.iloc[0]
        row = {"seed": seed}
        for metric in KEY_METRICS:
            row[f"main_{metric}"] = main.get(metric)
            row[f"rt_supervised_{metric}"] = rt.get(metric)
            row[f"rt_supervised_minus_main_{metric}"] = _as_float(rt.get(metric)) - _as_float(
                main.get(metric)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _format_float(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows yet._"
    subset = frame.loc[:, [col for col in columns if col in frame.columns]].head(max_rows)
    rendered = subset.copy()
    for col in rendered.columns:
        if pd.api.types.is_numeric_dtype(rendered[col]):
            rendered[col] = rendered[col].map(_format_float)
    headers = list(rendered.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in rendered.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def write_report(
    out_path: Path,
    results: pd.DataFrame,
    comparisons: pd.DataFrame,
    rt_comparisons: pd.DataFrame,
    uncertainty: pd.DataFrame,
) -> None:
    completed = len(results)
    main = results[results["condition"] == "main"].sort_values(["seed", "variant"])
    lines = [
        "# V7 Publishable Insights Report",
        "",
        f"Completed evaluated runs: {completed}",
        "",
        "## Clean Main Runs",
        "",
        _markdown_table(
            main,
            [
                "seed",
                "variant",
                "blimp_accuracy",
                "syntaxgym_accuracy",
                "fillergap_accuracy",
                "naturalstories_delta_r2_gate_flexible_surprisal",
                "naturalstories_cv_by_story_delta_r2_gate_flexible_surprisal",
            ],
        ),
        "",
        "## ETB Minus Baselines",
        "",
        _markdown_table(
            comparisons,
            [
                "seed",
                "baseline",
                "etb_minus_baseline_blimp_accuracy",
                "etb_minus_baseline_syntaxgym_accuracy",
                "etb_minus_baseline_fillergap_accuracy",
                "etb_minus_baseline_naturalstories_delta_r2_gate_flexible_surprisal",
            ],
        ),
        "",
        "## RT Supervision Upper Bound",
        "",
        _markdown_table(
            rt_comparisons,
            [
                "seed",
                "rt_supervised_minus_main_blimp_accuracy",
                "rt_supervised_minus_main_syntaxgym_accuracy",
                "rt_supervised_minus_main_fillergap_accuracy",
                "rt_supervised_minus_main_naturalstories_delta_r2_gate_flexible_surprisal",
            ],
        ),
        "",
        "## Natural Stories Uncertainty",
        "",
        _markdown_table(
            uncertainty,
            [
                "condition",
                "seed",
                "variant",
                "observed_delta_r2_flexible",
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
                "permutation_null_mean",
                "permutation_p_one_sided",
            ],
        ),
        "",
        "## Interpretation Checklist",
        "",
        "- Treat RT-supervised runs as an upper bound or leakage stress test, not the main claim.",
        "- Main ETB evidence requires no RT gate training, held-out Natural Stories, and matched/flexible surprisal controls.",
        "- If generic_dynamic or other simple baselines beat ETB, frame the result as a failure-analysis/strong-baseline finding.",
        "- Report seed-level variability and uncertainty before making claims about small Natural Stories deltas.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = collect_results(args.root, args.pattern)
    results.to_csv(args.out_dir / "all_results.csv", index=False)
    comparisons = build_etb_comparisons(results)
    comparisons.to_csv(args.out_dir / "etb_vs_baselines.csv", index=False)
    rt_comparisons = build_rt_supervision_comparisons(results)
    rt_comparisons.to_csv(args.out_dir / "rt_supervision_comparison.csv", index=False)
    uncertainty = build_uncertainty(
        results,
        bootstrap_repeats=args.bootstrap_repeats,
        permutation_repeats=args.permutation_repeats,
        seed=args.seed,
    )
    uncertainty.to_csv(args.out_dir / "naturalstories_uncertainty.csv", index=False)
    write_report(
        args.out_dir / "report.md",
        results=results,
        comparisons=comparisons,
        rt_comparisons=rt_comparisons,
        uncertainty=uncertainty,
    )
    print(args.out_dir / "report.md")


if __name__ == "__main__":
    main()
