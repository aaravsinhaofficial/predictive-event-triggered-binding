from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_BASELINES = ["generic_dynamic", "anira_emergent"]


def build_claim_report(summary_csv: str | Path, out_path: str | Path | None = None) -> dict[str, Any]:
    summary = pd.read_csv(summary_csv)
    report: dict[str, Any] = {
        "summary_csv": str(summary_csv),
        "passes_main_track_claim": False,
        "checks": [],
        "notes": [],
    }
    if "variant" not in summary.columns:
        raise ValueError("Baseline summary must contain a 'variant' column")
    variants = set(summary["variant"].astype(str))
    missing = {"etb", *REQUIRED_BASELINES} - variants
    if missing:
        report["notes"].append(f"Missing variants: {', '.join(sorted(missing))}")
        return _write_report(report, out_path)

    etb = _single(summary, "etb")
    required_results = []
    for baseline in REQUIRED_BASELINES:
        base = _single(summary, baseline)
        required_results.extend(
            [
                _higher_is_better(
                    etb,
                    base,
                    metric="fillergap_accuracy",
                    label=f"ETB beats {baseline} on filler-gap/island accuracy",
                ),
                _higher_is_better(
                    etb,
                    base,
                    metric="naturalstories_delta_r2_gate",
                    label=f"ETB beats {baseline} on RT residual variance beyond surprisal",
                ),
                _matched_flops(etb, base, tolerance=0.1, baseline=baseline),
            ]
        )

    report["checks"] = required_results
    report["passes_main_track_claim"] = all(check["pass"] for check in required_results)
    if not report["passes_main_track_claim"]:
        report["notes"].append(
            "This report is a decision aid: fixture runs should not be interpreted as paper evidence."
        )
    return _write_report(report, out_path)


def _single(summary: pd.DataFrame, variant: str) -> pd.Series:
    rows = summary[summary["variant"].astype(str) == variant]
    if rows.empty:
        raise ValueError(f"Missing variant: {variant}")
    return rows.iloc[0]


def _higher_is_better(etb: pd.Series, base: pd.Series, metric: str, label: str) -> dict[str, Any]:
    etb_value = _safe_float(etb.get(metric))
    base_value = _safe_float(base.get(metric))
    return {
        "label": label,
        "metric": metric,
        "etb": etb_value,
        "baseline": base_value,
        "margin": etb_value - base_value,
        "pass": etb_value > base_value,
    }


def _matched_flops(
    etb: pd.Series,
    base: pd.Series,
    tolerance: float,
    baseline: str,
) -> dict[str, Any]:
    etb_flops = _safe_float(etb.get("activated_flops_per_token"))
    base_flops = _safe_float(base.get("activated_flops_per_token"))
    rel_gap = abs(etb_flops - base_flops) / max(1.0, etb_flops)
    return {
        "label": f"ETB and {baseline} are matched within {tolerance:.0%} activated FLOPs/token",
        "metric": "activated_flops_per_token",
        "etb": etb_flops,
        "baseline": base_flops,
        "relative_gap": rel_gap,
        "pass": rel_gap <= tolerance,
    }


def _safe_float(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)


def _write_report(report: dict[str, Any], out_path: str | Path | None) -> dict[str, Any]:
    if out_path is not None:
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    return report

