from __future__ import annotations

from pathlib import Path

from datasets import load_dataset
from rich.progress import track

from etb.utils import count_words

TRACK_DATASETS = {
    "strict-small": "BabyLM-community/BabyLM-2026-Strict-Small",
    "strict": "BabyLM-community/BabyLM-2026-Strict",
}


def prepare_hf_text(
    dataset_name: str,
    out_path: str | Path,
    split: str = "train",
    text_column: str = "text",
    max_tokens: int | None = None,
    streaming: bool = True,
) -> Path:
    """Write a plain-text training file from a Hugging Face text dataset."""

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    total_words = 0
    with target.open("w", encoding="utf-8") as handle:
        iterable = dataset if streaming else iter(dataset)
        for row in track(iterable, description=f"Preparing {dataset_name}"):
            text = str(row.get(text_column, "")).strip()
            if not text:
                continue
            words = count_words(text)
            if max_tokens is not None and total_words + words > max_tokens:
                remaining = max_tokens - total_words
                if remaining <= 0:
                    break
                text = " ".join(text.split()[:remaining])
                words = count_words(text)
            handle.write(text + "\n")
            total_words += words
            if max_tokens is not None and total_words >= max_tokens:
                break
    return target


def fetch_babylm_track(
    track: str,
    out_dir: str | Path,
    max_tokens: int | None = None,
) -> Path:
    if track not in TRACK_DATASETS:
        valid = ", ".join(sorted(TRACK_DATASETS))
        raise ValueError(f"Unknown BabyLM track '{track}'. Valid tracks: {valid}")
    suffix = f"_{max_tokens}" if max_tokens else ""
    out_path = Path(out_dir) / f"{track.replace('-', '_')}{suffix}.txt"
    return prepare_hf_text(TRACK_DATASETS[track], out_path=out_path, max_tokens=max_tokens)

