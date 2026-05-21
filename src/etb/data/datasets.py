from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast

from etb.data.babylm import prepare_hf_text


@dataclass
class TextDataPaths:
    train: Path
    eval: Path | None = None


def resolve_text_paths(data_cfg: dict[str, Any]) -> TextDataPaths:
    train_path = data_cfg.get("train_text_path") or data_cfg.get("prepared_text_path")
    if train_path:
        train = Path(train_path)
    else:
        raise ValueError("Config must provide data.train_text_path or data.prepared_text_path")

    if not train.exists():
        if data_cfg.get("download_if_missing") and data_cfg.get("hf_dataset_name"):
            prepare_hf_text(
                dataset_name=str(data_cfg["hf_dataset_name"]),
                out_path=train,
                split=str(data_cfg.get("hf_split", "train")),
                text_column=str(data_cfg.get("text_column", "text")),
                max_tokens=(
                    int(data_cfg["max_tokens"]) if data_cfg.get("max_tokens") is not None else None
                ),
            )
        else:
            raise FileNotFoundError(f"Training text not found: {train}")

    eval_path = Path(data_cfg["eval_text_path"]) if data_cfg.get("eval_text_path") else None
    return TextDataPaths(train=train, eval=eval_path)


class CausalTextDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        text_path: str | Path,
        tokenizer: PreTrainedTokenizerFast,
        block_size: int,
        max_tokens: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        tokens: list[int] = []
        with Path(text_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                encoded = tokenizer.encode(text, add_special_tokens=False)
                tokens.extend([int(tokenizer.bos_token_id), *encoded, int(tokenizer.eos_token_id)])
                if max_tokens is not None and len(tokens) >= max_tokens:
                    tokens = tokens[:max_tokens]
                    break
        if len(tokens) < 2:
            raise ValueError(f"Not enough tokens in {text_path}")
        self.tokens = tokens

    def __len__(self) -> int:
        return max(1, (len(self.tokens) - 1) // self.block_size)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = index * self.block_size
        chunk = self.tokens[start : start + self.block_size + 1]
        if len(chunk) < self.block_size + 1:
            pad = [int(self.tokenizer.pad_token_id)] * (self.block_size + 1 - len(chunk))
            chunk = chunk + pad
        input_ids = torch.tensor(chunk, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids.eq(int(self.tokenizer.pad_token_id))] = -100
        attention_mask = input_ids.ne(int(self.tokenizer.pad_token_id)).long()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def build_dataloader(
    dataset: CausalTextDataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[dict[str, torch.Tensor]]:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
