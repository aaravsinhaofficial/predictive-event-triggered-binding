from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from etb.analysis.figures import make_figures
from etb.baselines import DEFAULT_BASELINE_VARIANTS, run_baseline_grid
from etb.claim_report import build_claim_report
from etb.config import load_experiment_config
from etb.data.babylm import fetch_babylm_track
from etb.eval.babylm_export import export_babylm_manifest, maybe_score_brainscore
from etb.eval.runner import evaluate as run_evaluate
from etb.eval.scoring import load_checkpoint, sentence_log_likelihood
from etb.training import train as run_train
from etb.utils import write_jsonl

app = typer.Typer(help="Predictive Event-Triggered Binding research CLI")
console = Console()


@app.command()
def train(config: Path = typer.Option(..., "--config", "-c", help="Experiment YAML config")) -> None:
    """Train a model from a YAML config."""

    checkpoint = run_train(load_experiment_config(config))
    console.print(f"[green]Training complete:[/green] {checkpoint}")


@app.command()
def evaluate(
    config: Path = typer.Option(..., "--config", "-c", help="Experiment YAML config"),
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Checkpoint directory"),
    tasks: str = typer.Option("fixture", "--tasks", help="'fixture', 'none', or comma list"),
) -> None:
    """Evaluate a checkpoint on configured text and task adapters."""

    metrics = run_evaluate(load_experiment_config(config), checkpoint=checkpoint, tasks=tasks)
    console.print(metrics)


@app.command("score-sentences")
def score_sentences(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Checkpoint directory"),
    sentences_file: Path = typer.Option(..., "--sentences-file", help="One sentence per line"),
    out: Path = typer.Option(..., "--out", help="Output JSONL"),
    device: str = typer.Option("auto", "--device"),
    traces: bool = typer.Option(False, "--traces", help="Include gate/memory traces"),
) -> None:
    """Write sentence log-likelihoods and optional gate traces."""

    model, tokenizer, resolved = load_checkpoint(checkpoint, device)
    rows = []
    with sentences_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            sentence = line.strip()
            if sentence:
                rows.append(sentence_log_likelihood(model, tokenizer, sentence, resolved, traces))
    write_jsonl(out, rows)
    console.print(f"[green]Wrote {len(rows)} scores to {out}[/green]")


@app.command("fetch-babylm")
def fetch_babylm(
    track: str = typer.Option("strict-small", "--track", help="'strict-small' or 'strict'"),
    out_dir: Path = typer.Option(Path("data/babylm"), "--out-dir"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens"),
) -> None:
    """Fetch a BabyLM 2026 text track into a local plain-text file."""

    out = fetch_babylm_track(track=track, out_dir=out_dir, max_tokens=max_tokens)
    console.print(f"[green]Prepared BabyLM text at {out}[/green]")


@app.command("analyze")
def analyze(run_dir: Path = typer.Option(..., "--run-dir", help="Run directory under outputs/")) -> None:
    """Generate paper-oriented figures for a completed run."""

    figures = make_figures(run_dir)
    if not figures:
        console.print("[yellow]No figures generated; run evaluation first.[/yellow]")
    for figure in figures:
        console.print(f"[green]Generated {figure}[/green]")


@app.command("run-baselines")
def run_baselines(
    config: Path = typer.Option(..., "--config", "-c", help="Base YAML config"),
    variants: str = typer.Option(
        ",".join(DEFAULT_BASELINE_VARIANTS),
        "--variants",
        help="Comma-separated model variants",
    ),
    tasks: str = typer.Option("fixture", "--tasks"),
    max_steps: Optional[int] = typer.Option(None, "--max-steps"),
) -> None:
    """Train and evaluate a small comparison grid from one base config."""

    selected = [variant.strip() for variant in variants.split(",") if variant.strip()]
    summary = run_baseline_grid(
        load_experiment_config(config),
        variants=selected,
        tasks=tasks,
        max_steps=max_steps,
    )
    console.print(f"[green]Wrote baseline grid summary to {summary}[/green]")


@app.command("claim-report")
def claim_report(
    summary_csv: Path = typer.Option(..., "--summary-csv"),
    out: Path = typer.Option(Path("outputs/claim_report.json"), "--out"),
) -> None:
    """Check the exact falsifiable claim against a baseline summary table."""

    report = build_claim_report(summary_csv, out)
    console.print(report)


@app.command("export-babylm")
def export_babylm(
    checkpoint: Path = typer.Option(..., "--checkpoint"),
    out_dir: Path = typer.Option(..., "--out-dir"),
    model_name: str = typer.Option("predictive-etb", "--model-name"),
    backend: str = typer.Option("hf", "--backend"),
) -> None:
    """Write a lightweight manifest for the official BabyLM 2026 eval pipeline."""

    manifest = export_babylm_manifest(checkpoint, out_dir, model_name=model_name, backend=backend)
    console.print(f"[green]Wrote BabyLM export manifest to {manifest}[/green]")


@app.command("brainscore")
def brainscore(
    model_identifier: str = typer.Option(..., "--model-identifier"),
    benchmark_identifier: str = typer.Option("Futrell2018-pearsonr", "--benchmark-identifier"),
) -> None:
    """Optional Brain-Score Language hook."""

    result = maybe_score_brainscore(model_identifier, benchmark_identifier)
    console.print(result)


if __name__ == "__main__":
    app()
