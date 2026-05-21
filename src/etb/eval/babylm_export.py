from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def export_babylm_manifest(
    checkpoint: str | Path,
    out_dir: str | Path,
    model_name: str = "predictive-etb",
    backend: str = "hf",
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_name": model_name,
        "checkpoint": str(checkpoint),
        "backend": backend,
        "notes": (
            "Upload the checkpoint directory to Hugging Face, then run the official BabyLM "
            "2026 evaluation pipeline against that model identifier."
        ),
    }
    manifest_path = out / "babylm_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    readme = out / "MODEL_CARD.md"
    readme.write_text(
        "# Predictive ETB BabyLM Export\n\n"
        "This checkpoint implements a GRU-based Predictive Event-Triggered Binding LM. "
        "It supports Hugging Face `from_pretrained` loading through the local `etb` package.\n",
        encoding="utf-8",
    )
    return manifest_path


def maybe_score_brainscore(model_identifier: str, benchmark_identifier: str) -> Any:
    try:
        from brainscore_language import score  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Brain-Score Language is optional. Install with `uv sync --extra brainscore` "
            "and review its dependency-isolation guidance before running."
        ) from exc
    return score(model_identifier=model_identifier, benchmark_identifier=benchmark_identifier)

