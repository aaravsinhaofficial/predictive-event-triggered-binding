from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any]
    path: Path | None = None

    @property
    def run_name(self) -> str:
        return str(self.raw.get("run_name", "etb_run"))

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 13))

    @property
    def output_dir(self) -> Path:
        return Path(self.raw.get("output_dir", f"outputs/{self.run_name}"))

    @property
    def device(self) -> str:
        return str(self.raw.get("device", "auto"))

    @property
    def data(self) -> dict[str, Any]:
        return dict(self.raw.get("data", {}))

    @property
    def model(self) -> dict[str, Any]:
        return dict(self.raw.get("model", {}))

    @property
    def training(self) -> dict[str, Any]:
        return dict(self.raw.get("training", {}))

    @property
    def evaluation(self) -> dict[str, Any]:
        return dict(self.raw.get("evaluation", {}))


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return ExperimentConfig(raw=raw, path=config_path)


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)

