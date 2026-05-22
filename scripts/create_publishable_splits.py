from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic train/dev/test splits for publishable ETB evaluation."
    )
    parser.add_argument("--source-dir", type=Path, default=Path("data/eval"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/eval_publishable"))
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    args = parser.parse_args()

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "source_dir": str(args.source_dir),
        "out_dir": str(args.out_dir),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "dev_ratio": args.dev_ratio,
        "splits": {},
        "note": "Auxiliary training should use train_dev only; final evaluation should use test only.",
    }
    manifest["splits"]["blimp"] = split_blimp(
        args.source_dir / "blimp" / "data",
        args.out_dir / "blimp",
        seed=args.seed,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
    )
    manifest["splits"]["syntaxgym"] = split_syntaxgym(
        args.source_dir / "syntaxgym",
        args.out_dir / "syntaxgym",
        seed=args.seed,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
    )
    manifest["splits"]["fillergap"] = split_fillergap(
        args.source_dir / "fillergap" / "fillergap.csv",
        args.out_dir / "fillergap",
        seed=args.seed,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
    )
    manifest["splits"]["naturalstories"] = split_naturalstories(
        args.source_dir / "naturalstories" / "naturalstories.tsv",
        args.out_dir / "naturalstories",
        seed=args.seed,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
    )

    manifest_path = args.out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


def split_keys(
    keys: list[Any],
    seed: int,
    train_ratio: float,
    dev_ratio: float,
) -> dict[str, set[Any]]:
    shuffled = list(keys)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    if n == 0:
        return {"train": set(), "dev": set(), "test": set(), "train_dev": set()}
    if n == 1:
        train = set(shuffled)
        return {"train": train, "dev": set(), "test": set(), "train_dev": train}
    if n == 2:
        train = {shuffled[0]}
        test = {shuffled[1]}
        return {"train": train, "dev": set(), "test": test, "train_dev": train}

    n_train = max(1, int(round(n * train_ratio)))
    n_dev = max(1, int(round(n * dev_ratio)))
    while n_train + n_dev > n - 1:
        if n_train >= n_dev and n_train > 1:
            n_train -= 1
        elif n_dev > 1:
            n_dev -= 1
        else:
            break
    train = set(shuffled[:n_train])
    dev = set(shuffled[n_train : n_train + n_dev])
    test = set(shuffled[n_train + n_dev :])
    return {"train": train, "dev": dev, "test": test, "train_dev": train | dev}


def split_blimp(
    source_dir: Path,
    out_root: Path,
    seed: int,
    train_ratio: float,
    dev_ratio: float,
) -> dict[str, Any]:
    counts = {split: 0 for split in ("train", "dev", "test", "train_dev")}
    for jsonl_path in sorted(source_dir.glob("*.jsonl")):
        rows = read_jsonl(jsonl_path)
        row_keys = [str(row.get("pairID", index)) for index, row in enumerate(rows)]
        key_splits = split_keys(row_keys, seed + stable_int(jsonl_path.stem), train_ratio, dev_ratio)
        for split_name, split_keys_set in key_splits.items():
            selected = [
                row
                for index, row in enumerate(rows)
                if str(row.get("pairID", index)) in split_keys_set
            ]
            if not selected:
                continue
            out_path = out_root / split_name / "data" / jsonl_path.name
            write_jsonl(out_path, selected)
            counts[split_name] += len(selected)
    return {
        split: {
            "path": str(out_root / split / "data"),
            "pairs": count,
        }
        for split, count in counts.items()
    }


def split_syntaxgym(
    source_dir: Path,
    out_root: Path,
    seed: int,
    train_ratio: float,
    dev_ratio: float,
) -> dict[str, Any]:
    counts = {split: {"items": 0, "predictions": 0} for split in ("train", "dev", "test", "train_dev")}
    for suite_path in sorted(source_dir.glob("*.json")):
        with suite_path.open("r", encoding="utf-8") as handle:
            suite = json.load(handle)
        items = list(suite.get("items", []))
        item_keys = [str(item.get("item_number", index)) for index, item in enumerate(items)]
        key_splits = split_keys(item_keys, seed + stable_int(suite_path.stem), train_ratio, dev_ratio)
        for split_name, split_keys_set in key_splits.items():
            selected = [
                item
                for index, item in enumerate(items)
                if str(item.get("item_number", index)) in split_keys_set
            ]
            if not selected:
                continue
            split_suite = {**suite, "items": selected}
            out_path = out_root / split_name / suite_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as handle:
                json.dump(split_suite, handle, indent=2)
            counts[split_name]["items"] += len(selected)
            counts[split_name]["predictions"] += sum(
                len(item.get("predictions", [])) for item in selected
            )
    return {
        split: {
            "path": str(out_root / split),
            **count,
        }
        for split, count in counts.items()
    }


def split_fillergap(
    source_path: Path,
    out_root: Path,
    seed: int,
    train_ratio: float,
    dev_ratio: float,
) -> dict[str, Any]:
    data = pd.read_csv(source_path)
    item_ids = sorted(data["item_id"].astype(str).unique().tolist())
    key_splits = split_keys(item_ids, seed, train_ratio, dev_ratio)
    counts = {}
    for split_name, split_keys_set in key_splits.items():
        selected = data[data["item_id"].astype(str).isin(split_keys_set)]
        out_path = out_root / split_name / "fillergap.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(out_path, index=False)
        counts[split_name] = {
            "path": str(out_path),
            "minimal_pairs": int(selected["item_id"].nunique()) if not selected.empty else 0,
            "rows": int(len(selected)),
        }
    return counts


def split_naturalstories(
    source_path: Path,
    out_root: Path,
    seed: int,
    train_ratio: float,
    dev_ratio: float,
) -> dict[str, Any]:
    data = pd.read_csv(source_path, sep="\t")
    story_ids = sorted(data["story_id"].astype(int).unique().tolist())
    key_splits = split_keys(story_ids, seed, train_ratio, dev_ratio)
    counts = {}
    for split_name, split_keys_set in key_splits.items():
        selected = data[data["story_id"].astype(int).isin(split_keys_set)]
        out_path = out_root / split_name / "naturalstories.tsv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(out_path, sep="\t", index=False)
        counts[split_name] = {
            "path": str(out_path),
            "stories": int(selected["story_id"].nunique()) if not selected.empty else 0,
            "tokens": int(len(selected)),
        }
    return counts


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def stable_int(text: str) -> int:
    value = 0
    for char in text:
        value = (value * 33 + ord(char)) % 1_000_003
    return value


if __name__ == "__main__":
    main()
