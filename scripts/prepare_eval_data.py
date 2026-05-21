from __future__ import annotations

import argparse
import json
import math
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import get_dataset_config_names, load_dataset

SYNTAXGYM_BASE = (
    "https://raw.githubusercontent.com/cpllab/syntactic-generalization/"
    "nextflow/test_suites/json"
)
SYNTAXGYM_2020_SUITES = [
    "center_embed",
    "center_embed_mod",
    "cleft",
    "cleft_modifier",
    "fgd_hierarchy",
    "fgd_object",
    "fgd_pp",
    "fgd_subject",
    "mvrr",
    "mvrr_mod",
    "npi_orc_any",
    "npi_orc_ever",
    "npi_src_any",
    "npi_src_ever",
    "npz_ambig",
    "npz_ambig_mod",
    "npz_obj",
    "npz_obj_mod",
    "number_orc",
    "number_prep",
    "number_src",
    "reflexive_orc_fem",
    "reflexive_orc_masc",
    "reflexive_prep_fem",
    "reflexive_prep_masc",
    "reflexive_src_fem",
    "reflexive_src_masc",
    "subordination",
    "subordination_orc-orc",
    "subordination_pp-pp",
    "subordination_src-src",
]
FILLERGAP_BLIMP_CONFIGS = {
    "adjunct_island",
    "complex_NP_island",
    "coordinate_structure_constraint_complex_left_branch",
    "coordinate_structure_constraint_object_extraction",
    "left_branch_island_echo_question",
    "left_branch_island_simple_question",
    "sentential_subject_island",
    "wh_island",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare non-fixture BLiMP, SyntaxGym, filler-gap/island, and Natural Stories data."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/eval"))
    parser.add_argument(
        "--naturalstories-source",
        type=Path,
        default=Path("data/sources/naturalstories"),
        help="Clone https://github.com/languageMIT/naturalstories here first.",
    )
    parser.add_argument("--skip-blimp", action="store_true")
    parser.add_argument("--skip-syntaxgym", action="store_true")
    parser.add_argument("--skip-fillergap", action="store_true")
    parser.add_argument("--skip-naturalstories", action="store_true")
    parser.add_argument("--blimp-limit-per-config", type=int, default=None)
    parser.add_argument("--syntaxgym-limit-items", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"out_dir": str(args.out_dir), "artifacts": {}}

    blimp_rows_by_config: dict[str, list[dict[str, Any]]] = {}
    if not args.skip_blimp or not args.skip_fillergap:
        blimp_rows_by_config = prepare_blimp(
            args.out_dir / "blimp" / "data",
            limit_per_config=args.blimp_limit_per_config,
            write_files=not args.skip_blimp,
        )
        if not args.skip_blimp:
            manifest["artifacts"]["blimp"] = {
                "path": str(args.out_dir / "blimp" / "data"),
                "configs": len(blimp_rows_by_config),
                "pairs": sum(len(rows) for rows in blimp_rows_by_config.values()),
            }

    if not args.skip_syntaxgym:
        syntaxgym_summary = prepare_syntaxgym(
            args.out_dir / "syntaxgym",
            limit_items=args.syntaxgym_limit_items,
        )
        manifest["artifacts"]["syntaxgym"] = syntaxgym_summary

    if not args.skip_fillergap:
        fillergap_path, fillergap_n = prepare_fillergap_from_blimp(
            blimp_rows_by_config,
            args.out_dir / "fillergap" / "fillergap.csv",
        )
        manifest["artifacts"]["fillergap"] = {
            "path": str(fillergap_path),
            "minimal_pairs": fillergap_n,
            "source": "BLiMP island configs; closest available filler-gap/island CSV",
        }

    if not args.skip_naturalstories:
        ns_path, ns_n = prepare_naturalstories(
            args.naturalstories_source,
            args.out_dir / "naturalstories" / "naturalstories.tsv",
        )
        manifest["artifacts"]["naturalstories"] = {
            "path": str(ns_path),
            "tokens": ns_n,
            "source": str(args.naturalstories_source),
        }

    manifest_path = args.out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


def prepare_blimp(
    out_dir: Path,
    limit_per_config: int | None,
    write_files: bool,
) -> dict[str, list[dict[str, Any]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_config: dict[str, list[dict[str, Any]]] = {}
    for config_name in get_dataset_config_names("nyu-mll/blimp"):
        dataset = load_dataset("nyu-mll/blimp", config_name, split="train")
        rows: list[dict[str, Any]] = []
        for idx, row in enumerate(dataset):
            if limit_per_config is not None and idx >= limit_per_config:
                break
            rows.append(
                {
                    "sentence_good": row["sentence_good"],
                    "sentence_bad": row["sentence_bad"],
                    "UID": row.get("UID", config_name),
                    "pairID": row.get("pairID", row.get("pair_id", idx)),
                    "field": row.get("field"),
                    "linguistics_term": row.get("linguistics_term"),
                }
            )
        rows_by_config[config_name] = rows
        if write_files:
            with (out_dir / f"{config_name}.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
    return rows_by_config


def prepare_syntaxgym(out_dir: Path, limit_items: int | None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_items = 0
    total_predictions = 0
    for suite_name in SYNTAXGYM_2020_SUITES:
        with urllib.request.urlopen(f"{SYNTAXGYM_BASE}/{suite_name}.json") as response:
            raw_suite = json.loads(response.read().decode("utf-8"))
        suite = convert_syntaxgym_suite(raw_suite, limit_items=limit_items)
        total_items += len(suite["items"])
        total_predictions += sum(len(item["predictions"]) for item in suite["items"])
        with (out_dir / f"{suite_name}.json").open("w", encoding="utf-8") as handle:
            json.dump(suite, handle, indent=2)
    return {
        "path": str(out_dir),
        "suites": len(SYNTAXGYM_2020_SUITES),
        "items": total_items,
        "predictions": total_predictions,
        "source": SYNTAXGYM_BASE,
        "conversion_note": "Region formulas converted to full-sentence condition likelihood contrasts.",
    }


def convert_syntaxgym_suite(raw_suite: dict[str, Any], limit_items: int | None) -> dict[str, Any]:
    predictions = syntaxgym_predictions(raw_suite.get("predictions", []))
    items = []
    for raw_item in raw_suite.get("items", [])[:limit_items]:
        conditions = [
            {
                "condition": condition["condition_name"],
                "sentence": condition_sentence(condition),
            }
            for condition in raw_item.get("conditions", [])
        ]
        items.append(
            {
                "item_number": raw_item.get("item_number"),
                "conditions": conditions,
                "predictions": predictions,
            }
        )
    return {
        "meta": raw_suite.get("meta", {}),
        "items": items,
    }


def syntaxgym_predictions(raw_predictions: list[dict[str, Any]]) -> list[dict[str, str]]:
    converted: list[dict[str, str]] = []
    pattern = re.compile(
        r"\(\s*\d+\s*;\s*%([^%]+)%\s*\)\s*([<>])\s*"
        r"\(\s*\d+\s*;\s*%([^%]+)%\s*\)"
    )
    for prediction in raw_predictions:
        formula = str(prediction.get("formula", ""))
        for left, op, right in pattern.findall(formula):
            if op == ">":
                converted.append({"better": right, "worse": left})
            else:
                converted.append({"better": left, "worse": right})
    return converted


def condition_sentence(condition: dict[str, Any]) -> str:
    parts = [
        str(region.get("content", "")).lstrip()
        for region in condition.get("regions", [])
        if str(region.get("content", "")).strip()
    ]
    text = " ".join(parts)
    return re.sub(r"\s+([.,!?;:])", r"\1", text)


def prepare_fillergap_from_blimp(
    rows_by_config: dict[str, list[dict[str, Any]]],
    out_path: Path,
) -> tuple[Path, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for config_name in sorted(FILLERGAP_BLIMP_CONFIGS & set(rows_by_config)):
        for idx, row in enumerate(rows_by_config[config_name]):
            item_id = f"{config_name}:{row.get('pairID', idx)}"
            records.append(
                {
                    "item_id": item_id,
                    "construction": config_name,
                    "condition": "good",
                    "sentence": row["sentence_good"],
                    "expected_good": 1,
                }
            )
            records.append(
                {
                    "item_id": item_id,
                    "construction": config_name,
                    "condition": "bad",
                    "sentence": row["sentence_bad"],
                    "expected_good": 0,
                }
            )
    pd.DataFrame(records).to_csv(out_path, index=False)
    return out_path, len(records) // 2


def prepare_naturalstories(source_dir: Path, out_path: Path) -> tuple[Path, int]:
    wordinfo = source_dir / "naturalstories_RTS" / "processed_wordinfo.tsv"
    if not wordinfo.exists():
        raise FileNotFoundError(f"Missing Natural Stories processed_wordinfo.tsv under {source_dir}")
    data = pd.read_csv(wordinfo, sep="\t")
    data = data.sort_values(["item", "zone"])
    token_counts = Counter(str(token).lower() for token in data["word"])
    total = sum(token_counts.values())
    rows = []
    for record in data.to_dict("records"):
        token = str(record["word"])
        freq = token_counts[token.lower()] / max(1, total)
        rows.append(
            {
                "story_id": int(record["item"]),
                "token_index": int(record["zone"]),
                "token": token,
                "rt": float(record["meanItemRT"]),
                "log_frequency": math.log(freq),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    return out_path, len(rows)


if __name__ == "__main__":
    main()
