from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import PreTrainedTokenizerFast

from etb.models.modeling_etb import ETBForCausalLM
from etb.utils import resolve_device


def load_checkpoint(
    checkpoint: str | Path,
    device: str = "auto",
) -> tuple[ETBForCausalLM, PreTrainedTokenizerFast, torch.device]:
    resolved = resolve_device(device)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(checkpoint))
    model = ETBForCausalLM.from_pretrained(str(checkpoint))
    model.to(resolved)
    model.eval()
    return model, tokenizer, resolved


def encode_sentence(tokenizer: PreTrainedTokenizerFast, sentence: str) -> torch.Tensor:
    ids = tokenizer.encode(sentence, add_special_tokens=False)
    ids = [int(tokenizer.bos_token_id), *ids, int(tokenizer.eos_token_id)]
    return torch.tensor([ids], dtype=torch.long)


def encode_sentences(
    tokenizer: PreTrainedTokenizerFast,
    sentences: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = []
    for sentence in sentences:
        encoded.append(
            [
                int(tokenizer.bos_token_id),
                *tokenizer.encode(sentence, add_special_tokens=False),
                int(tokenizer.eos_token_id),
            ]
        )
    max_len = max(len(ids) for ids in encoded)
    pad_id = int(tokenizer.pad_token_id)
    input_ids = torch.full((len(encoded), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long)
    for row, ids in enumerate(encoded):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, : len(ids)] = 1
    return input_ids, attention_mask


@torch.no_grad()
def sentence_log_likelihood(
    model: ETBForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    sentence: str,
    device: torch.device,
    return_trace: bool = False,
) -> dict[str, Any]:
    input_ids = encode_sentence(tokenizer, sentence).to(device)
    attention_mask = input_ids.ne(int(tokenizer.pad_token_id)).long()
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    assert output.logits is not None
    log_probs = output.logits[:, :-1, :].log_softmax(dim=-1)
    targets = input_ids[:, 1:]
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    total = float(token_log_probs.sum().cpu())
    result: dict[str, Any] = {
        "sentence": sentence,
        "log_likelihood": total,
        "num_tokens": int(targets.numel()),
        "mean_log_likelihood": total / max(1, int(targets.numel())),
        "loss": None,
        "gate_rate": (
            float(output.gate_activations.detach().mean().cpu())
            if output.gate_activations is not None
            else 0.0
        ),
        "activated_flops_per_token": (
            float(output.activated_flops.detach().cpu())
            if output.activated_flops is not None
            else 0.0
        ),
    }
    if return_trace:
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist())
        gate_probs = (
            output.gate_probs[0].detach().cpu().tolist() if output.gate_probs is not None else []
        )
        gate_activations = (
            output.gate_activations[0].detach().cpu().tolist()
            if output.gate_activations is not None
            else []
        )
        memory_events = {}
        if output.memory_events:
            memory_events = {
                key: value[0].detach().cpu().tolist() for key, value in output.memory_events.items()
            }
        information_gain = (
            output.information_gain[0].detach().cpu().tolist()
            if output.information_gain is not None
            else []
        )
        gate_targets = (
            output.gate_targets[0].detach().cpu().tolist()
            if output.gate_targets is not None
            else []
        )
        result["trace"] = {
            "tokens": tokens,
            "gate_probs": gate_probs,
            "gate_activations": gate_activations,
            "gate_targets": gate_targets,
            "information_gain": information_gain,
            "memory_events": memory_events,
            "token_log_probs": [None, *token_log_probs[0].detach().cpu().tolist()],
        }
    return result


@torch.no_grad()
def sentence_log_likelihood_batch(
    model: ETBForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    sentences: Sequence[str],
    device: torch.device,
    batch_size: int = 64,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for start in range(0, len(sentences), batch_size):
        batch_sentences = list(sentences[start : start + batch_size])
        input_ids, attention_mask = encode_sentences(tokenizer, batch_sentences)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        assert output.logits is not None
        log_probs = output.logits[:, :-1, :].log_softmax(dim=-1)
        targets = input_ids[:, 1:]
        target_mask = attention_mask[:, 1:].bool()
        safe_targets = targets.masked_fill(~target_mask, 0)
        token_log_probs = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        token_log_probs = token_log_probs.masked_fill(~target_mask, 0.0)
        totals = token_log_probs.sum(dim=1).detach().cpu().tolist()
        counts = target_mask.sum(dim=1).detach().cpu().tolist()
        gate_rate = (
            float(output.gate_activations.detach().mean().cpu())
            if output.gate_activations is not None
            else 0.0
        )
        activated = (
            float(output.activated_flops.detach().cpu())
            if output.activated_flops is not None
            else 0.0
        )
        for sentence, total, count in zip(batch_sentences, totals, counts, strict=True):
            results.append(
                {
                    "sentence": sentence,
                    "log_likelihood": float(total),
                    "num_tokens": int(count),
                    "mean_log_likelihood": float(total) / max(1, int(count)),
                    "loss": None,
                    "gate_rate": gate_rate,
                    "activated_flops_per_token": activated,
                }
            )
    return results


@torch.no_grad()
def text_perplexity(
    model: ETBForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    text_path: str | Path,
    device: torch.device,
) -> dict[str, float]:
    total_log_prob = 0.0
    total_tokens = 0
    with Path(text_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            score = sentence_log_likelihood(model, tokenizer, text, device)
            total_log_prob += float(score["log_likelihood"])
            total_tokens += int(score["num_tokens"])
    nll = -total_log_prob / max(1, total_tokens)
    return {
        "nll": nll,
        "perplexity": float(torch.exp(torch.tensor(nll)).item()),
        "bits_per_byte": nll / torch.log(torch.tensor(2.0)).item(),
        "tokens": float(total_tokens),
    }
