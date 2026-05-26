#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = {
    "etb": "#1f77b4",
    "generic_dynamic": "#ff7f0e",
    "punctuation_only": "#2ca02c",
    "anira_emergent": "#9467bd",
    "random_matched": "#7f7f7f",
    "cheap_only": "#8c564b",
    "dense_gru": "#17becf",
    "rt_supervised_upper_bound": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build figures for the Insights report.")
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/insights/v7_publishable"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _save(fig: plt.Figure, out_dir: Path, name: str, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _format_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def clean_ns_delta(results: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    main = results[results["condition"] == "main"].copy()
    keep = ["etb", "generic_dynamic", "punctuation_only", "anira_emergent"]
    main = main[main["variant"].isin(keep)]
    seeds = sorted(main["seed"].dropna().astype(int).unique())
    variants = [variant for variant in keep if variant in set(main["variant"])]
    x = np.arange(len(seeds))
    width = 0.18
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, variant in enumerate(variants):
        values = []
        for seed in seeds:
            rows = main[(main["seed"] == seed) & (main["variant"] == variant)]
            value = rows["naturalstories_delta_r2_gate_flexible_surprisal"].iloc[0] if len(rows) else np.nan
            values.append(value)
        ax.bar(
            x + (i - (len(variants) - 1) / 2) * width,
            values,
            width=width,
            label=variant,
            color=PALETTE.get(variant),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(seed) for seed in seeds])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Delta R2 beyond flexible surprisal")
    ax.set_title("Clean Natural Stories Gate Signal")
    ax.legend(frameon=False, ncol=2)
    _format_ax(ax)
    _save(fig, out_dir, "clean_naturalstories_delta_r2", dpi)


def etb_vs_baselines(comparisons: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    focus = comparisons[comparisons["baseline"].isin(["generic_dynamic", "punctuation_only"])].copy()
    metrics = [
        ("etb_minus_baseline_blimp_accuracy", "BLiMP"),
        ("etb_minus_baseline_syntaxgym_accuracy", "SyntaxGym"),
        ("etb_minus_baseline_fillergap_accuracy", "FillerGap"),
        (
            "etb_minus_baseline_naturalstories_delta_r2_gate_flexible_surprisal",
            "NS flex dR2",
        ),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(11.8, 3.8), sharex=False)
    for ax, (column, label) in zip(axes, metrics, strict=True):
        rows = []
        labels = []
        colors = []
        for baseline in ["generic_dynamic", "punctuation_only"]:
            data = focus[focus["baseline"] == baseline]
            for _, row in data.iterrows():
                rows.append(row[column])
                labels.append(f"s{int(row['seed'])}\n{baseline.replace('_', ' ')}")
                colors.append(PALETTE.get(baseline, "#666666"))
        ax.bar(np.arange(len(rows)), rows, color=colors)
        ax.axhline(0, color="#222222", linewidth=1.0)
        ax.set_title(label)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        _format_ax(ax)
    axes[0].set_ylabel("ETB minus baseline")
    fig.suptitle("ETB Relative to Strong Baselines", y=1.03)
    _save(fig, out_dir, "etb_minus_strong_baselines", dpi)


def rt_supervision(results: pd.DataFrame, rt_comparison: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    ax = axes[0]
    for _, row in rt_comparison.iterrows():
        seed = int(row["seed"])
        y = [
            row["main_naturalstories_delta_r2_gate_flexible_surprisal"],
            row["rt_supervised_naturalstories_delta_r2_gate_flexible_surprisal"],
        ]
        ax.plot([0, 1], y, marker="o", linewidth=2, label=f"seed {seed}")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["main", "RT supervised"])
    ax.set_ylabel("Delta R2 beyond flexible surprisal")
    ax.set_title("RT Supervision Inflates RT Alignment")
    ax.legend(frameon=False)
    _format_ax(ax)

    ax = axes[1]
    columns = [
        ("rt_supervised_minus_main_blimp_accuracy", "BLiMP"),
        ("rt_supervised_minus_main_syntaxgym_accuracy", "SyntaxGym"),
        ("rt_supervised_minus_main_fillergap_accuracy", "FillerGap"),
        (
            "rt_supervised_minus_main_naturalstories_delta_r2_gate_flexible_surprisal",
            "NS flex dR2",
        ),
    ]
    means = [rt_comparison[col].mean() for col, _ in columns]
    labels = [label for _, label in columns]
    ax.bar(np.arange(len(labels)), means, color="#d62728")
    ax.axhline(0, color="#222222", linewidth=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("Mean RT-Supervised Minus Main")
    _format_ax(ax)
    _save(fig, out_dir, "rt_supervision_upper_bound", dpi)


def auxiliary_collapse(results: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    seed23 = results[(results["seed"] == 23) & (results["variant"] == "etb")].copy()
    rows = []
    for condition in ["main", "no_contrastive", "no_aux"]:
        selected = seed23[seed23["condition"] == condition]
        if len(selected):
            row = selected.iloc[0]
            rows.append(
                {
                    "condition": condition,
                    "BLiMP": row["blimp_accuracy"],
                    "SyntaxGym": row["syntaxgym_accuracy"],
                    "FillerGap": row["fillergap_accuracy"],
                    "NS flex dR2": row["naturalstories_delta_r2_gate_flexible_surprisal"],
                }
            )
    data = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ax = axes[0]
    metric_names = ["BLiMP", "SyntaxGym", "FillerGap"]
    x = np.arange(len(metric_names))
    width = 0.24
    for i, (_, row) in enumerate(data.iterrows()):
        ax.bar(x + (i - 1) * width, [row[m] for m in metric_names], width, label=row["condition"])
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Auxiliary Training Carries Syntax/Filler")
    ax.legend(frameon=False)
    _format_ax(ax)

    ax = axes[1]
    ax.bar(data["condition"], data["NS flex dR2"], color="#1f77b4")
    ax.set_ylabel("Delta R2")
    ax.set_title("Natural Stories Signal Also Collapses")
    _format_ax(ax)
    _save(fig, out_dir, "auxiliary_ablation_collapse", dpi)


def uncertainty(uncertainty_data: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    focus = uncertainty_data[
        (
            (uncertainty_data["condition"] == "main")
            & (uncertainty_data["variant"].isin(["etb", "generic_dynamic", "punctuation_only"]))
        )
        | (uncertainty_data["condition"] == "rt_supervised_upper_bound")
    ].copy()
    focus["label"] = focus.apply(
        lambda row: (
            f"s{int(row['seed'])} RT sup"
            if row["condition"] == "rt_supervised_upper_bound"
            else f"s{int(row['seed'])} {row['variant'].replace('_', ' ')}"
        ),
        axis=1,
    )
    focus = focus.sort_values(["condition", "seed", "variant"])
    y = np.arange(len(focus))
    x = focus["observed_delta_r2_flexible"].to_numpy(dtype=float)
    low = focus["bootstrap_ci95_low"].to_numpy(dtype=float)
    high = focus["bootstrap_ci95_high"].to_numpy(dtype=float)
    xerr = np.vstack([x - low, high - x])
    colors = [
        PALETTE["rt_supervised_upper_bound"]
        if row.condition == "rt_supervised_upper_bound"
        else PALETTE.get(row.variant, "#666666")
        for row in focus.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor="#555555", elinewidth=1.0, capsize=3)
    ax.scatter(x, y, s=42, color=colors, zorder=3)
    ax.axvline(0, color="#222222", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(focus["label"])
    ax.set_xlabel("Observed delta R2 beyond flexible surprisal")
    ax.set_title("Natural Stories Bootstrap Intervals")
    _format_ax(ax)
    _save(fig, out_dir, "naturalstories_uncertainty", dpi)


def write_index(out_dir: Path) -> None:
    figures = sorted(path.name for path in out_dir.glob("*.png"))
    lines = ["# Insights Figures", ""]
    for figure in figures:
        stem = Path(figure).stem
        lines.append(f"- `{figure}` / `{stem}.pdf`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    out_dir = args.out_dir or results_dir / "figures"
    results = pd.read_csv(results_dir / "all_results.csv")
    comparisons = pd.read_csv(results_dir / "etb_vs_baselines.csv")
    rt_comparison = pd.read_csv(results_dir / "rt_supervision_comparison.csv")
    uncertainty_data = pd.read_csv(results_dir / "naturalstories_uncertainty.csv")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": args.dpi,
        }
    )
    clean_ns_delta(results, out_dir, args.dpi)
    etb_vs_baselines(comparisons, out_dir, args.dpi)
    rt_supervision(results, rt_comparison, out_dir, args.dpi)
    auxiliary_collapse(results, out_dir, args.dpi)
    uncertainty(uncertainty_data, out_dir, args.dpi)
    write_index(out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
