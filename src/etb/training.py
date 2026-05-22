from __future__ import annotations

import itertools
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm.auto import tqdm

from etb.config import ExperimentConfig, save_yaml
from etb.data.datasets import CausalTextDataset, build_dataloader, resolve_text_paths
from etb.data.tokenizer import (
    CLAUSE_TOKENS,
    PUNCTUATION_TOKENS,
    load_or_train_tokenizer,
    token_ids_for,
)
from etb.models.configuration_etb import ETBConfig
from etb.models.modeling_etb import ETBForCausalLM
from etb.utils import append_jsonl, ensure_dir, resolve_device, set_seed

console = Console()


class ContrastivePairDataset(Dataset[tuple[str, str]]):
    def __init__(self, pairs: list[tuple[str, str]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[str, str]:
        return self.pairs[index]


class ReadingTimeDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, examples: list[dict[str, torch.Tensor]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


def build_model_config(config: ExperimentConfig, vocab_size: int, tokenizer: Any) -> ETBConfig:
    model_cfg = dict(config.model)
    model_cfg.update(
        {
            "vocab_size": int(vocab_size),
            "pad_token_id": int(tokenizer.pad_token_id),
            "unk_token_id": int(tokenizer.unk_token_id),
            "bos_token_id": int(tokenizer.bos_token_id),
            "eos_token_id": int(tokenizer.eos_token_id),
            "punctuation_token_ids": token_ids_for(tokenizer, PUNCTUATION_TOKENS),
            "clause_token_ids": token_ids_for(tokenizer, CLAUSE_TOKENS),
        }
    )
    return ETBConfig(**model_cfg)


def _build_contrastive_loader(
    config: ExperimentConfig,
    tokenizer: Any,
) -> tuple[DataLoader | None, float, float, int]:
    contrastive_cfg = dict(config.training.get("contrastive", {}) or {})
    if not contrastive_cfg.get("enabled", False):
        return None, 0.0, 0.0, 1

    pairs = _load_contrastive_pairs(
        contrastive_cfg.get("sources", {}) or {},
        max_pairs=(
            int(contrastive_cfg["max_pairs"])
            if contrastive_cfg.get("max_pairs") is not None
            else None
        ),
        seed=int(config.seed),
    )
    if not pairs:
        return None, 0.0, 0.0, 1

    console.print(f"[cyan]Loaded {len(pairs)} contrastive sentence pairs[/cyan]")
    loader = DataLoader(
        ContrastivePairDataset(pairs),
        batch_size=int(contrastive_cfg.get("batch_size", 16)),
        shuffle=True,
        collate_fn=_collate_contrastive_pairs,
        drop_last=False,
    )
    return (
        loader,
        float(contrastive_cfg.get("weight", 0.0)),
        float(contrastive_cfg.get("margin", 0.0)),
        max(1, int(contrastive_cfg.get("every", 1))),
    )


def _load_contrastive_pairs(
    sources: dict[str, Any],
    max_pairs: int | None,
    seed: int,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if sources.get("fillergap"):
        pairs.extend(_load_fillergap_pairs(Path(sources["fillergap"])))
    if sources.get("blimp"):
        pairs.extend(_load_blimp_pairs(Path(sources["blimp"])))
    if sources.get("syntaxgym"):
        pairs.extend(_load_syntaxgym_pairs(Path(sources["syntaxgym"])))

    deduped = list(dict.fromkeys(pairs))
    rng = random.Random(seed)
    rng.shuffle(deduped)
    return deduped[:max_pairs] if max_pairs is not None else deduped


def _load_fillergap_pairs(path: Path) -> list[tuple[str, str]]:
    data = pd.read_csv(path)
    pairs: list[tuple[str, str]] = []
    for _, group in data.groupby("item_id", sort=True):
        good = group[group["expected_good"].astype(int) == 1]
        bad = group[group["expected_good"].astype(int) == 0]
        if good.empty or bad.empty:
            continue
        pairs.append((str(good["sentence"].iloc[0]), str(bad["sentence"].iloc[0])))
    return pairs


def _load_blimp_pairs(path: Path) -> list[tuple[str, str]]:
    paths = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    pairs: list[tuple[str, str]] = []
    for jsonl_path in paths:
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if "sentence_good" in record and "sentence_bad" in record:
                    pairs.append((str(record["sentence_good"]), str(record["sentence_bad"])))
    return pairs


def _load_syntaxgym_pairs(path: Path) -> list[tuple[str, str]]:
    paths = sorted(path.glob("*.json")) if path.is_dir() else [path]
    pairs: list[tuple[str, str]] = []
    for suite_path in paths:
        with suite_path.open("r", encoding="utf-8") as handle:
            suite = json.load(handle)
        for item in suite.get("items", []):
            conditions = {
                str(condition["condition"]): str(condition["sentence"])
                for condition in item.get("conditions", [])
            }
            for prediction in item.get("predictions", []):
                better = conditions.get(str(prediction["better"]))
                worse = conditions.get(str(prediction["worse"]))
                if better is not None and worse is not None:
                    pairs.append((better, worse))
    return pairs


def _collate_contrastive_pairs(batch: list[tuple[str, str]]) -> dict[str, list[str]]:
    return {
        "good": [good for good, _ in batch],
        "bad": [bad for _, bad in batch],
    }


def _build_reading_time_loader(
    config: ExperimentConfig,
    tokenizer: Any,
) -> tuple[DataLoader | None, float, int, float, bool]:
    rt_cfg = dict(config.training.get("rt_gate", {}) or {})
    if not rt_cfg.get("enabled", False) or not rt_cfg.get("path"):
        return None, 0.0, 1, 1.0, False

    examples = _load_reading_time_examples(
        Path(rt_cfg["path"]),
        tokenizer,
        target_temperature=float(rt_cfg.get("target_temperature", 1.0)),
    )
    if not examples:
        return None, 0.0, 1, 1.0, False

    console.print(f"[cyan]Loaded {len(examples)} reading-time gate examples[/cyan]")
    loader = DataLoader(
        ReadingTimeDataset(examples),
        batch_size=int(rt_cfg.get("batch_size", 4)),
        shuffle=True,
        collate_fn=lambda batch: _collate_reading_time(batch, int(tokenizer.pad_token_id)),
        drop_last=False,
    )
    return (
        loader,
        float(rt_cfg.get("weight", 0.0)),
        max(1, int(rt_cfg.get("every", 1))),
        float(rt_cfg.get("target_temperature", 1.0)),
        bool(rt_cfg.get("residualize_surprisal", False)),
    )


def _load_reading_time_examples(
    path: Path,
    tokenizer: Any,
    target_temperature: float,
) -> list[dict[str, torch.Tensor]]:
    data = pd.read_csv(path, sep="\t")
    controls = [data["token"].astype(str).map(len).to_numpy(dtype=float)]
    if "log_frequency" in data.columns:
        controls.append(data["log_frequency"].to_numpy(dtype=float))
    design = np.column_stack([np.ones(len(data)), *controls])
    y = data["rt"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    residual = (residual - residual.mean()) / max(1e-6, residual.std())
    temp = max(1e-4, float(target_temperature))
    targets = 1.0 / (1.0 + np.exp(-residual / temp))
    data = data.copy()
    data["_rt_target"] = targets

    examples: list[dict[str, torch.Tensor]] = []
    for _, group in data.groupby("story_id", sort=True):
        records = group.to_dict("records")
        sentence_piece_ids: list[int] = [int(tokenizer.bos_token_id)]
        gate_targets: list[float] = [0.0]
        gate_mask: list[float] = [0.0]
        rt_values: list[float] = [0.0]
        token_lengths: list[float] = [0.0]
        log_frequencies: list[float] = [0.0]
        word_ids: list[int] = [0]
        for word_id, record in enumerate(records, start=1):
            pieces = tokenizer.encode(str(record["token"]), add_special_tokens=False)
            if not pieces:
                pieces = [int(tokenizer.unk_token_id)]
            for piece_index, piece_id in enumerate(pieces):
                sentence_piece_ids.append(int(piece_id))
                is_last_piece = piece_index == len(pieces) - 1
                gate_targets.append(float(record["_rt_target"]) if is_last_piece else 0.0)
                gate_mask.append(1.0 if is_last_piece else 0.0)
                rt_values.append(float(record["rt"]) if is_last_piece else 0.0)
                token_lengths.append(float(len(str(record["token"]))) if is_last_piece else 0.0)
                log_frequencies.append(float(record.get("log_frequency", 0.0)) if is_last_piece else 0.0)
                word_ids.append(word_id)
        sentence_piece_ids.append(int(tokenizer.eos_token_id))
        gate_targets.append(0.0)
        gate_mask.append(0.0)
        rt_values.append(0.0)
        token_lengths.append(0.0)
        log_frequencies.append(0.0)
        word_ids.append(0)
        examples.append(
            {
                "input_ids": torch.tensor(sentence_piece_ids, dtype=torch.long),
                "gate_targets": torch.tensor(gate_targets, dtype=torch.float),
                "gate_mask": torch.tensor(gate_mask, dtype=torch.float),
                "rt": torch.tensor(rt_values, dtype=torch.float),
                "token_length": torch.tensor(token_lengths, dtype=torch.float),
                "log_frequency": torch.tensor(log_frequencies, dtype=torch.float),
                "word_ids": torch.tensor(word_ids, dtype=torch.long),
            }
        )
    return examples


def _collate_reading_time(
    batch: list[dict[str, torch.Tensor]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    max_len = max(int(item["input_ids"].numel()) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    gate_targets = torch.zeros((len(batch), max_len), dtype=torch.float)
    gate_mask = torch.zeros((len(batch), max_len), dtype=torch.float)
    rt = torch.zeros((len(batch), max_len), dtype=torch.float)
    token_length = torch.zeros((len(batch), max_len), dtype=torch.float)
    log_frequency = torch.zeros((len(batch), max_len), dtype=torch.float)
    word_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    for row, item in enumerate(batch):
        length = int(item["input_ids"].numel())
        input_ids[row, :length] = item["input_ids"]
        attention_mask[row, :length] = 1
        gate_targets[row, :length] = item["gate_targets"]
        gate_mask[row, :length] = item["gate_mask"]
        rt[row, :length] = item["rt"]
        token_length[row, :length] = item["token_length"]
        log_frequency[row, :length] = item["log_frequency"]
        word_ids[row, :length] = item["word_ids"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "gate_targets": gate_targets,
        "gate_mask": gate_mask,
        "rt": rt,
        "token_length": token_length,
        "log_frequency": log_frequency,
        "word_ids": word_ids,
    }


def _encode_sentence_batch(
    tokenizer: Any,
    sentences: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [
        [
            int(tokenizer.bos_token_id),
            *tokenizer.encode(sentence, add_special_tokens=False),
            int(tokenizer.eos_token_id),
        ]
        for sentence in sentences
    ]
    max_len = max(len(ids) for ids in encoded)
    input_ids = torch.full(
        (len(encoded), max_len),
        int(tokenizer.pad_token_id),
        dtype=torch.long,
    )
    attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long)
    for row, ids in enumerate(encoded):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, : len(ids)] = 1
    return input_ids, attention_mask


def _sequence_log_likelihoods(
    model: ETBForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    assert output.logits is not None
    log_probs = output.logits[:, :-1, :].log_softmax(dim=-1)
    targets = input_ids[:, 1:]
    target_mask = attention_mask[:, 1:].bool()
    safe_targets = targets.masked_fill(~target_mask, 0)
    token_log_probs = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.masked_fill(~target_mask, 0.0).sum(dim=1)


def _contrastive_sentence_loss(
    model: ETBForCausalLM,
    tokenizer: Any,
    good_sentences: list[str],
    bad_sentences: list[str],
    device: torch.device,
    margin: float,
) -> torch.Tensor:
    input_ids, attention_mask = _encode_sentence_batch(
        tokenizer,
        [*good_sentences, *bad_sentences],
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    scores = _sequence_log_likelihoods(model, input_ids, attention_mask)
    good_scores, bad_scores = scores.chunk(2)
    return F.softplus(float(margin) - (good_scores - bad_scores)).mean()


def _reading_time_gate_loss(
    model: ETBForCausalLM,
    batch: dict[str, torch.Tensor],
    target_temperature: float,
    residualize_surprisal: bool,
) -> torch.Tensor:
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    assert output.gate_probs is not None
    if residualize_surprisal:
        assert output.logits is not None
        gate_targets = _current_residual_gate_targets(
            logits=output.logits,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            gate_mask=batch["gate_mask"],
            word_ids=batch["word_ids"],
            rt=batch["rt"],
            token_length=batch["token_length"],
            log_frequency=batch["log_frequency"],
            target_temperature=target_temperature,
        )
    else:
        gate_targets = batch["gate_targets"]
    raw = F.mse_loss(output.gate_probs, gate_targets, reduction="none")
    mask = batch["gate_mask"]
    return (raw * mask).sum() / mask.sum().clamp_min(1.0)


def _current_residual_gate_targets(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gate_mask: torch.Tensor,
    word_ids: torch.Tensor,
    rt: torch.Tensor,
    token_length: torch.Tensor,
    log_frequency: torch.Tensor,
    target_temperature: float,
) -> torch.Tensor:
    with torch.no_grad():
        piece_surprisal = torch.zeros_like(gate_mask)
        log_probs = logits[:, :-1, :].log_softmax(dim=-1)
        targets = input_ids[:, 1:]
        target_mask = attention_mask[:, 1:].bool()
        safe_targets = targets.masked_fill(~target_mask, 0)
        next_surprisal = -log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        piece_surprisal[:, 1:] = next_surprisal.masked_fill(~target_mask, 0.0)

        word_surprisal = torch.zeros_like(gate_mask)
        for row in range(input_ids.size(0)):
            max_word_id = int(word_ids[row].max().item())
            for word_id in range(1, max_word_id + 1):
                positions = word_ids[row].eq(word_id)
                if not bool(positions.any()):
                    continue
                end_position = positions.nonzero(as_tuple=False)[-1, 0]
                word_surprisal[row, end_position] = piece_surprisal[row, positions].sum()

        mask = gate_mask.bool()
        if int(mask.sum().item()) < 5:
            return torch.zeros_like(gate_mask)

        y = rt[mask].float()
        controls = torch.stack(
            [
                word_surprisal[mask].float(),
                token_length[mask].float(),
                log_frequency[mask].float(),
            ],
            dim=1,
        )
        controls = (controls - controls.mean(dim=0, keepdim=True)) / controls.std(
            dim=0,
            keepdim=True,
        ).clamp_min(1e-6)
        design = torch.cat([torch.ones_like(y[:, None]), controls], dim=1)
        beta = torch.linalg.pinv(design) @ y
        residual = y - design @ beta
        residual = (residual - residual.mean()) / residual.std().clamp_min(1e-6)
        temp = max(1e-4, float(target_temperature))
        target_values = torch.sigmoid(residual / temp)
        targets_out = torch.zeros_like(gate_mask)
        targets_out[mask] = target_values
        return targets_out


def train(config: ExperimentConfig) -> Path:
    set_seed(config.seed)
    output_dir = ensure_dir(config.output_dir)
    save_yaml(config.raw, output_dir / "config.yaml")

    data_paths = resolve_text_paths(config.data)
    tokenizer = load_or_train_tokenizer(
        tokenizer_dir=config.data.get("tokenizer_dir", output_dir / "tokenizer"),
        train_files=[data_paths.train],
        vocab_size=int(config.data.get("vocab_size", 8000)),
    )

    model_config = build_model_config(config, vocab_size=len(tokenizer), tokenizer=tokenizer)
    model = ETBForCausalLM(model_config)

    device = resolve_device(config.device)
    model.to(device)
    model.train()

    train_dataset = CausalTextDataset(
        text_path=data_paths.train,
        tokenizer=tokenizer,
        block_size=int(config.data.get("block_size", 128)),
        max_tokens=(
            int(config.data["max_train_tokens"]) if config.data.get("max_train_tokens") else None
        ),
    )
    train_loader = build_dataloader(
        train_dataset,
        batch_size=int(config.training.get("batch_size", 16)),
        shuffle=True,
    )
    contrastive_loader, contrastive_weight, contrastive_margin, contrastive_every = (
        _build_contrastive_loader(config, tokenizer)
    )
    contrastive_batches = itertools.cycle(contrastive_loader) if contrastive_loader else None
    rt_loader, rt_weight, rt_every, rt_target_temperature, rt_residualize_surprisal = (
        _build_reading_time_loader(config, tokenizer)
    )
    rt_batches = itertools.cycle(rt_loader) if rt_loader else None

    optimizer = AdamW(
        model.parameters(),
        lr=float(config.training.get("learning_rate", 8e-4)),
        weight_decay=float(config.training.get("weight_decay", 0.0)),
    )

    max_steps = int(config.training.get("max_steps", 100))
    log_every = int(config.training.get("log_every", 10))
    grad_clip = float(config.training.get("grad_clip", 1.0))
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    start_time = time.time()
    batches = itertools.cycle(train_loader)
    progress = tqdm(range(1, max_steps + 1), desc=f"train:{config.run_name}")
    contrastive_loss = torch.zeros((), device=device)
    rt_gate_loss = torch.zeros((), device=device)
    for step in progress:
        batch = next(batches)
        batch = {key: value.to(device) for key, value in batch.items()}

        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        assert output.loss is not None
        loss = output.loss
        contrastive_loss = torch.zeros((), device=device)
        rt_gate_loss = torch.zeros((), device=device)
        if (
            contrastive_batches is not None
            and contrastive_weight > 0
            and step % contrastive_every == 0
        ):
            contrastive_batch = next(contrastive_batches)
            contrastive_loss = _contrastive_sentence_loss(
                model=model,
                tokenizer=tokenizer,
                good_sentences=contrastive_batch["good"],
                bad_sentences=contrastive_batch["bad"],
                device=device,
                margin=contrastive_margin,
            )
            loss = loss + contrastive_weight * contrastive_loss
        if rt_batches is not None and rt_weight > 0 and step % rt_every == 0:
            rt_batch = {
                key: value.to(device) for key, value in next(rt_batches).items()
            }
            rt_gate_loss = _reading_time_gate_loss(
                model,
                rt_batch,
                target_temperature=rt_target_temperature,
                residualize_surprisal=rt_residualize_surprisal,
            )
            loss = loss + rt_weight * rt_gate_loss

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % log_every == 0 or step == 1 or step == max_steps:
            gate_rate = float(output.gate_activations.detach().mean().cpu())
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "base_loss": float(output.loss.detach().cpu()),
                "lm_loss": (
                    float(output.lm_loss.detach().cpu()) if output.lm_loss is not None else 0.0
                ),
                "aux_loss": float(output.aux_loss.detach().cpu()) if output.aux_loss is not None else 0.0,
                "compute_loss": (
                    float(output.compute_loss.detach().cpu()) if output.compute_loss is not None else 0.0
                ),
                "budget_loss": (
                    float(output.budget_loss.detach().cpu()) if output.budget_loss is not None else 0.0
                ),
                "benefit_loss": (
                    float(output.benefit_loss.detach().cpu()) if output.benefit_loss is not None else 0.0
                ),
                "candidate_loss": (
                    float(output.candidate_loss.detach().cpu()) if output.candidate_loss is not None else 0.0
                ),
                "contrastive_loss": float(contrastive_loss.detach().cpu()),
                "rt_gate_loss": float(rt_gate_loss.detach().cpu()),
                "gate_rate": gate_rate,
                "gate_prob_mean": (
                    float(output.gate_probs.detach().mean().cpu())
                    if output.gate_probs is not None
                    else 0.0
                ),
                "information_gain_mean": (
                    float(output.information_gain.detach().mean().cpu())
                    if output.information_gain is not None
                    else 0.0
                ),
                "gate_target_mean": (
                    float(output.gate_targets.detach().mean().cpu())
                    if output.gate_targets is not None
                    else 0.0
                ),
                "activated_flops_per_token": (
                    float(output.activated_flops.detach().cpu())
                    if output.activated_flops is not None
                    else 0.0
                ),
                "memory_residual_scale": float(
                    getattr(model, "memory_residual_scale").detach().cpu()
                ),
                "elapsed_sec": time.time() - start_time,
                "variant": model.config.variant,
            }
            append_jsonl(metrics_path, row)
            progress.set_postfix(loss=f"{row['loss']:.3f}", gate=f"{gate_rate:.3f}")

    checkpoint = output_dir / "checkpoint-final"
    model.save_pretrained(str(checkpoint))
    tokenizer.save_pretrained(str(checkpoint))
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    console.print(f"[green]Saved checkpoint to {checkpoint}[/green]")
    return checkpoint
