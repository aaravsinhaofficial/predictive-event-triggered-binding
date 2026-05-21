from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from etb.models.configuration_etb import ETBConfig


def _isin_many(values: torch.Tensor, candidates: list[int]) -> torch.Tensor:
    if not candidates:
        return torch.zeros_like(values, dtype=torch.bool)
    mask = torch.zeros_like(values, dtype=torch.bool)
    for candidate in candidates:
        mask = mask | values.eq(int(candidate))
    return mask


def boundary_features(input_ids: torch.Tensor, config: ETBConfig) -> torch.Tensor:
    """Causal surface event features for the current observed token."""

    is_punct = _isin_many(input_ids, config.punctuation_token_ids)
    is_clause = _isin_many(input_ids, config.clause_token_ids)
    is_eos = input_ids.eq(int(config.eos_token_id))
    return torch.stack([is_punct, is_clause, is_eos], dim=-1).float()


def cheap_predictive_stats(
    cheap_logits: torch.Tensor,
    input_ids: torch.Tensor,
    pad_token_id: int | None = None,
) -> torch.Tensor:
    """Return entropy at t and realized surprisal for x_t under distribution at t-1.

    The current-token surprisal is causal for routing the update after x_t has been read:
    it never uses the next token that logits[t] will be evaluated against.
    """

    log_probs = cheap_logits.log_softmax(dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)

    prev_log_probs = log_probs[:, :-1, :]
    current_tokens = input_ids[:, 1:].unsqueeze(-1)
    current_surprisal = -prev_log_probs.gather(-1, current_tokens).squeeze(-1)
    zero = torch.zeros(input_ids.size(0), 1, device=input_ids.device, dtype=cheap_logits.dtype)
    surprisal = torch.cat([zero, current_surprisal], dim=1)

    if pad_token_id is not None:
        mask = input_ids.ne(int(pad_token_id)).float()
        entropy = entropy * mask
        surprisal = surprisal * mask

    scale = max(1.0, math.log(max(2, cheap_logits.size(-1))))
    return torch.stack([entropy / scale, surprisal / scale], dim=-1)


def cue_interference(hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Causal cue-overlap proxy: max cosine similarity to previous hidden states."""

    batch, seq_len, _ = hidden.shape
    normed = F.normalize(hidden.detach(), dim=-1)
    similarity = torch.bmm(normed, normed.transpose(1, 2))
    causal = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden.device),
        diagonal=-1,
    )
    similarity = similarity.masked_fill(~causal.unsqueeze(0), float("-inf"))
    if attention_mask is not None:
        valid_previous = attention_mask[:, None, :].bool()
        similarity = similarity.masked_fill(~valid_previous, float("-inf"))
    max_similarity = similarity.max(dim=-1).values
    max_similarity = torch.where(torch.isfinite(max_similarity), max_similarity, torch.zeros_like(max_similarity))
    return max_similarity.clamp_min(0.0).unsqueeze(-1)


class EventGate(nn.Module):
    def __init__(self, config: ETBConfig) -> None:
        super().__init__()
        gate_hidden = max(16, config.hidden_size // 2)
        self.net = nn.Sequential(
            nn.Linear(config.hidden_size + 6, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        cheap_stats: torch.Tensor,
        boundaries: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat([hidden, cheap_stats.detach(), boundaries], dim=-1)
        return self.net(features).squeeze(-1)


def route_gate(
    variant: str,
    learned_logits: torch.Tensor,
    cheap_stats: torch.Tensor,
    boundaries: torch.Tensor,
    threshold: float,
    target_sparsity: float,
    selection: str,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert gate logits/features into probabilities and hard activations."""

    if variant in {"cheap_only", "dense_gru"}:
        probs = torch.zeros_like(learned_logits)
        hard = torch.zeros_like(learned_logits)
    elif variant == "always_on":
        probs = torch.ones_like(learned_logits)
        hard = torch.ones_like(learned_logits)
    elif variant == "punctuation_only":
        hard = boundaries[..., :2].amax(dim=-1).float()
        probs = hard
    elif variant == "random_matched":
        probs = torch.full_like(learned_logits, float(target_sparsity))
        hard = torch.bernoulli(probs) if training else probs
    elif variant == "generic_dynamic":
        entropy = cheap_stats[..., 0]
        centered = entropy - entropy.mean(dim=1, keepdim=True)
        scaled = centered / entropy.std(dim=1, keepdim=True).clamp_min(1e-4)
        probs = torch.sigmoid(scaled)
        hard = _select_gate(probs, target_sparsity, threshold, selection, training)
    else:
        probs = torch.sigmoid(learned_logits)
        hard_binary = _select_gate(probs, target_sparsity, threshold, selection, training)
        hard = hard_binary + probs - probs.detach() if training else hard_binary

    return probs, hard


def _select_gate(
    probs: torch.Tensor,
    target_sparsity: float,
    threshold: float,
    selection: str,
    training: bool,
) -> torch.Tensor:
    if selection == "sample" and training:
        return torch.bernoulli(probs)
    if selection == "topk":
        if target_sparsity <= 0:
            return torch.zeros_like(probs)
        k = max(1, int(round(probs.size(1) * min(1.0, target_sparsity))))
        cutoff = probs.topk(k, dim=1).values[:, -1:].expand_as(probs)
        return (probs >= cutoff).float()
    return (probs > threshold).float()
