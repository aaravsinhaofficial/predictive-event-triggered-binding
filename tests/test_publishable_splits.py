import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def _fixture_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    (source / "blimp" / "data").mkdir(parents=True)
    (source / "syntaxgym").mkdir(parents=True)
    (source / "fillergap").mkdir(parents=True)
    (source / "naturalstories").mkdir(parents=True)
    shutil.copyfile(
        "data/fixtures/blimp/tiny.jsonl",
        source / "blimp" / "data" / "tiny.jsonl",
    )
    shutil.copyfile(
        "data/fixtures/syntaxgym/tiny_suite.json",
        source / "syntaxgym" / "tiny_suite.json",
    )
    shutil.copyfile(
        "data/fixtures/fillergap/tiny.csv",
        source / "fillergap" / "fillergap.csv",
    )
    ns = pd.read_csv("data/fixtures/naturalstories/tiny.tsv", sep="\t")
    ns_two_story = pd.concat([ns, ns.assign(story_id=2)], ignore_index=True)
    ns_two_story.to_csv(source / "naturalstories" / "naturalstories.tsv", sep="\t", index=False)
    return source


def test_create_publishable_splits(tmp_path):
    source = _fixture_source(tmp_path)
    out_dir = tmp_path / "splits"
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/create_publishable_splits.py",
            "--source-dir",
            str(source),
            "--out-dir",
            str(out_dir),
            "--seed",
            "5",
        ],
        check=True,
    )
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["splits"]["blimp"]["test"]["pairs"] > 0
    assert (out_dir / "blimp" / "train_dev" / "data" / "tiny.jsonl").exists()
    assert (out_dir / "syntaxgym" / "test" / "tiny_suite.json").exists()
    assert (out_dir / "fillergap" / "test" / "fillergap.csv").exists()
    assert (out_dir / "naturalstories" / "test" / "naturalstories.tsv").exists()
