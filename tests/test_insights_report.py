from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_metrics(root: Path, run_name: str, variant: str, delta: float) -> None:
    eval_dir = root / run_name / variant / "eval"
    eval_dir.mkdir(parents=True)
    token_metrics = eval_dir / "naturalstories_token_metrics.csv"
    pd.DataFrame(
        {
            "story_id": [1, 1, 1, 2, 2, 2],
            "rt": [300.0, 320.0, 360.0, 305.0, 330.0, 370.0],
            "cheap_surprisal": [1.0, 1.2, 1.5, 1.0, 1.3, 1.6],
            "cheap_surprisal_sq": [1.0, 1.44, 2.25, 1.0, 1.69, 2.56],
            "cheap_surprisal_cube": [1.0, 1.728, 3.375, 1.0, 2.197, 4.096],
            "cheap_surprisal_lag1": [0.0, 1.0, 1.2, 0.0, 1.0, 1.3],
            "cheap_surprisal_lag2": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            "token_length": [3, 4, 5, 3, 4, 5],
            "log_frequency": [5.0, 4.0, 3.0, 5.0, 4.0, 3.0],
            "cheap_surprisal_x_log_frequency": [5.0, 4.8, 4.5, 5.0, 5.2, 4.8],
            "gate_prob": [0.1, 0.4, 0.8, 0.2, 0.5, 0.9],
        }
    ).to_csv(token_metrics, index=False)
    metrics = {
        "blimp": {"accuracy": 0.8},
        "syntaxgym": {"accuracy": 0.7},
        "fillergap": {"accuracy": 0.9},
        "naturalstories": {
            "n": 6,
            "delta_r2_gate": delta,
            "delta_r2_gate_matched_surprisal": delta,
            "delta_r2_gate_flexible_surprisal": delta,
            "cv_by_story_delta_r2_gate_flexible_surprisal": delta,
            "gate_r2_from_flexible_controls": 0.2,
            "token_metrics": str(token_metrics),
        },
    }
    (eval_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    metrics_jsonl = root / run_name / variant / "metrics.jsonl"
    metrics_jsonl.write_text(json.dumps({"loss": 1.0, "gate_rate": 0.15}) + "\n", encoding="utf-8")


def test_build_insights_report(tmp_path: Path) -> None:
    root = tmp_path / "baselines"
    _write_metrics(root, "babylm_10m_v7_publishable_s23", "etb", 0.01)
    _write_metrics(root, "babylm_10m_v7_publishable_s23", "generic_dynamic", 0.02)
    _write_metrics(root, "babylm_10m_v7_publishable_rt_supervised_upper_bound_s23", "etb", 0.2)
    out_dir = tmp_path / "report"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_insights_report.py",
            "--root",
            str(root),
            "--out-dir",
            str(out_dir),
            "--bootstrap-repeats",
            "2",
            "--permutation-repeats",
            "2",
        ],
        check=True,
    )

    assert (out_dir / "report.md").exists()
    all_results = pd.read_csv(out_dir / "all_results.csv")
    assert set(all_results["condition"]) == {"main", "rt_supervised_upper_bound"}
    comparisons = pd.read_csv(out_dir / "etb_vs_baselines.csv")
    assert comparisons.loc[0, "baseline"] == "generic_dynamic"
    rt_comparisons = pd.read_csv(out_dir / "rt_supervision_comparison.csv")
    assert rt_comparisons.loc[0, "rt_supervised_minus_main_blimp_accuracy"] == 0.0
