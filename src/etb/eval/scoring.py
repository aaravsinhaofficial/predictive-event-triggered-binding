from __future__ import annotations

from pathlib import Path
from typing import Any

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
    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
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
        "loss": float(output.loss.detach().cpu()) if output.loss is not None else None,
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
