from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
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
    for step in progress:
        batch = next(batches)
        batch = {key: value.to(device) for key, value in batch.items()}

        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        assert output.loss is not None
        output.loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % log_every == 0 or step == 1 or step == max_steps:
            gate_rate = float(output.gate_activations.detach().mean().cpu())
            row = {
                "step": step,
                "loss": float(output.loss.detach().cpu()),
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
