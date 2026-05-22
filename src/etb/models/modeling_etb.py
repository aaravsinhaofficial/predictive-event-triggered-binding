from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from etb.flops import estimate_flops_per_token
from etb.models.configuration_etb import ETBConfig
from etb.models.gates import (
    EventGate,
    boundary_features,
    cheap_predictive_stats,
    cue_interference,
    route_gate,
)
from etb.models.memory import RoleFillerMemory


@dataclass
class ETBOutput(ModelOutput):
    loss: torch.Tensor | None = None
    lm_loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    gate_probs: torch.Tensor | None = None
    gate_activations: torch.Tensor | None = None
    memory_events: dict[str, torch.Tensor] | None = None
    activated_flops: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    cheap_logits: torch.Tensor | None = None
    candidate_logits: torch.Tensor | None = None
    information_gain: torch.Tensor | None = None
    gate_targets: torch.Tensor | None = None
    compute_loss: torch.Tensor | None = None
    budget_loss: torch.Tensor | None = None
    benefit_loss: torch.Tensor | None = None
    candidate_loss: torch.Tensor | None = None


class ETBForCausalLM(PreTrainedModel):
    config_class = ETBConfig
    base_model_prefix = "etb"
    supports_gradient_checkpointing = False

    def __init__(self, config: ETBConfig) -> None:
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        gru_hidden = config.hidden_size
        self.gru = nn.GRU(
            input_size=config.d_model,
            hidden_size=gru_hidden,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(gru_hidden, config.vocab_size, bias=False)
        self.gate = EventGate(config)
        self.memory = RoleFillerMemory(config)
        self.memory_lm_head = nn.Linear(gru_hidden, config.vocab_size, bias=False)
        self.memory_residual_scale = nn.Parameter(
            torch.tensor(float(config.memory_residual_scale_init))
        )

        if config.variant == "dense_gru":
            self.dense_residual = nn.Sequential(
                nn.Linear(gru_hidden, gru_hidden),
                nn.SiLU(),
                nn.Linear(gru_hidden, config.vocab_size),
            )
        else:
            self.dense_residual = None

        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def build_gate_features(
        self,
        input_ids: torch.Tensor,
        cheap_logits: torch.Tensor,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        boundaries = boundary_features(input_ids, self.config)
        stats = cheap_predictive_stats(
            cheap_logits=cheap_logits,
            input_ids=input_ids,
            pad_token_id=self.config.pad_token_id,
        )
        stats = torch.cat([stats, cue_interference(hidden, attention_mask=attention_mask)], dim=-1)
        return stats, boundaries

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **_: Any,
    ) -> ETBOutput | tuple[torch.Tensor, ...]:
        return_dict = True if return_dict is None else return_dict
        if attention_mask is None:
            attention_mask = input_ids.ne(int(self.config.pad_token_id)).long()

        embedded = self.dropout(self.embed(input_ids))
        hidden, _ = self.gru(embedded)
        hidden = self.dropout(hidden)
        cheap_logits = self.lm_head(hidden)

        cheap_stats, boundaries = self.build_gate_features(
            input_ids,
            cheap_logits,
            hidden,
            attention_mask=attention_mask,
        )
        gate_stats = cheap_stats
        gate_boundaries = boundaries
        if self.config.variant == "anira_emergent":
            gate_stats = torch.zeros_like(cheap_stats)
            gate_boundaries = torch.zeros_like(boundaries)
        gate_logits = self.gate(hidden, gate_stats, gate_boundaries)
        selection_mask = self._gate_selection_mask(attention_mask)
        if self.config.variant == "etb":
            gate_logits = gate_logits + self._gate_feature_prior(
                cheap_stats,
                boundaries,
                selection_mask,
            )
        gate_probs, gate_hard = route_gate(
            variant=self.config.variant,
            learned_logits=gate_logits,
            cheap_stats=cheap_stats,
            boundaries=boundaries,
            threshold=float(self.config.gate_threshold),
            target_sparsity=float(self.config.target_sparsity),
            selection=str(self.config.gate_selection),
            training=self.training,
            selection_mask=selection_mask,
        )

        if self.config.variant in {"cheap_only"}:
            logits = cheap_logits
            events = self._empty_events(input_ids)
        elif self.config.variant == "dense_gru" and self.dense_residual is not None:
            logits = cheap_logits + self.dense_residual(hidden)
            events = self._empty_events(input_ids)
        else:
            logits, events = self._memory_augmented_logits(hidden, cheap_logits, gate_hard)

        gate_rate = gate_hard.detach().mean()
        activated = torch.tensor(
            estimate_flops_per_token(self.config, float(gate_rate.cpu())),
            device=input_ids.device,
            dtype=hidden.dtype,
        )
        aux = self._gate_auxiliary_losses(
            cheap_logits=cheap_logits,
            hidden=hidden,
            labels=labels,
            gate_logits=gate_logits,
            gate_probs=gate_probs,
            gate_hard=gate_hard,
            attention_mask=attention_mask,
            selection_mask=selection_mask,
            boundaries=boundaries,
            cheap_stats=cheap_stats,
        )
        aux_loss = aux["aux_loss"]

        loss = None
        lm_loss = None
        if labels is not None:
            lm_loss = self._causal_lm_loss(logits, labels)
            loss = lm_loss + aux_loss

        if not return_dict:
            output: tuple[torch.Tensor, ...] = (logits, gate_probs, gate_hard)
            if loss is not None:
                output = (loss,) + output
            return output

        return ETBOutput(
            loss=loss,
            lm_loss=lm_loss,
            logits=logits,
            gate_probs=gate_probs,
            gate_activations=gate_hard,
            memory_events=events,
            activated_flops=activated,
            aux_loss=aux_loss,
            cheap_logits=cheap_logits,
            candidate_logits=aux["candidate_logits"],
            information_gain=aux["information_gain"],
            gate_targets=aux["gate_targets"],
            compute_loss=aux["compute_loss"],
            budget_loss=aux["budget_loss"],
            benefit_loss=aux["benefit_loss"],
            candidate_loss=aux["candidate_loss"],
        )

    def _gate_auxiliary_losses(
        self,
        cheap_logits: torch.Tensor,
        hidden: torch.Tensor,
        labels: torch.Tensor | None,
        gate_logits: torch.Tensor,
        gate_probs: torch.Tensor,
        gate_hard: torch.Tensor,
        attention_mask: torch.Tensor,
        selection_mask: torch.Tensor,
        boundaries: torch.Tensor,
        cheap_stats: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        zero = torch.zeros((), device=cheap_logits.device, dtype=cheap_logits.dtype)
        gate_mean = self._masked_mean(gate_probs, selection_mask.float())
        hard_gate_mean = self._masked_mean(gate_hard.detach(), selection_mask.float())
        compute_loss = float(self.config.sparsity_lambda) * gate_mean
        budget_loss = float(self.config.budget_lambda) * (
            hard_gate_mean - float(self.config.target_sparsity)
        ).pow(2)

        candidate_logits = None
        information_gain = torch.zeros_like(gate_probs)
        gate_targets = torch.zeros_like(gate_probs)
        benefit_loss = zero
        candidate_loss = zero
        prior_distill_loss = zero

        if self.config.variant == "etb" and labels is not None:
            cheap_lp, valid_next = self._shifted_token_log_probs(cheap_logits, labels)
            mode = str(self.config.gate_target_mode)
            uses_candidate_target = mode not in {"prior_topk", "prior_soft"}
            uses_candidate_loss = float(self.config.candidate_loss_lambda) > 0
            delta = torch.zeros_like(cheap_lp)
            if uses_candidate_target or uses_candidate_loss:
                candidate_logits, _ = self._memory_augmented_logits(
                    hidden,
                    cheap_logits,
                    torch.ones_like(gate_probs),
                )
                candidate_lp, _ = self._shifted_token_log_probs(candidate_logits, labels)
                delta = candidate_lp - cheap_lp

            if mode in {"prior_topk", "prior_soft"}:
                benefit_score = self._prior_score(
                    cheap_stats[:, :-1, :],
                    boundaries[:, :-1, :],
                    valid_next,
                )
            else:
                structural_prior = boundaries[:, :-1, :].amax(dim=-1) * float(
                    self.config.structural_prior_bonus
                )
                interference_prior = cheap_stats[:, :-1, 2] * float(
                    self.config.interference_prior_bonus
                )
                benefit_score = (
                    delta.detach()
                    + structural_prior
                    + interference_prior
                    - float(self.config.compute_penalty)
                )

            if mode in {"topk", "prior_topk"}:
                target = self._topk_targets(
                    benefit_score,
                    valid_next,
                    target_sparsity=float(self.config.target_sparsity),
                )
            else:
                target = torch.sigmoid(
                    benefit_score / max(1e-4, float(self.config.benefit_temperature))
                )
            pos_weight = torch.tensor(
                float(self.config.gate_positive_weight)
                if self.config.gate_positive_weight is not None
                else max(
                    1.0,
                    (1.0 - float(self.config.target_sparsity))
                    / max(1e-4, float(self.config.target_sparsity)),
                ),
                device=cheap_logits.device,
                dtype=cheap_logits.dtype,
            )
            benefit_raw = F.binary_cross_entropy_with_logits(
                gate_logits[:, :-1],
                target,
                pos_weight=pos_weight if str(self.config.gate_target_mode) == "topk" else None,
                reduction="none",
            )
            benefit_loss = float(self.config.benefit_lambda) * self._masked_mean(
                benefit_raw,
                valid_next.float(),
            )
            if float(self.config.prior_distill_lambda) > 0:
                prior_target = self._prior_distill_target(
                    cheap_stats[:, :-1, :],
                    boundaries[:, :-1, :],
                    valid_next,
                )
                prior_raw = F.binary_cross_entropy_with_logits(
                    gate_logits[:, :-1],
                    prior_target,
                    reduction="none",
                )
                prior_distill_loss = float(self.config.prior_distill_lambda) * self._masked_mean(
                    prior_raw,
                    valid_next.float(),
                )
            if candidate_logits is not None:
                candidate_loss = float(self.config.candidate_loss_lambda) * self._causal_lm_loss(
                    candidate_logits,
                    labels,
                )
            information_gain[:, :-1] = delta.masked_fill(~valid_next, 0.0)
            gate_targets[:, :-1] = target.masked_fill(~valid_next, 0.0)

        return {
            "aux_loss": (
                compute_loss
                + budget_loss
                + benefit_loss
                + candidate_loss
                + prior_distill_loss
            ),
            "candidate_logits": candidate_logits,
            "information_gain": information_gain,
            "gate_targets": gate_targets,
            "compute_loss": compute_loss,
            "budget_loss": budget_loss,
            "benefit_loss": benefit_loss,
            "candidate_loss": candidate_loss,
        }

    def _causal_lm_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def _shifted_token_log_probs(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        valid = shift_labels.ne(-100)
        safe_labels = shift_labels.masked_fill(~valid, 0)
        token_log_probs = shift_logits.log_softmax(dim=-1).gather(
            -1,
            safe_labels.unsqueeze(-1),
        )
        return token_log_probs.squeeze(-1).masked_fill(~valid, 0.0), valid

    def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    def _gate_selection_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(attention_mask, dtype=torch.float)
        mask[:, :-1] = attention_mask[:, 1:].float()
        return mask

    def _standardize(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float()
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (values * mask).sum(dim=1, keepdim=True) / denom
        centered = (values - mean) * mask
        var = (centered.pow(2) * mask).sum(dim=1, keepdim=True) / denom
        return centered / var.sqrt().clamp_min(1e-4)

    def _gate_feature_prior(
        self,
        cheap_stats: torch.Tensor,
        boundaries: torch.Tensor,
        selection_mask: torch.Tensor,
    ) -> torch.Tensor:
        prior = self._prior_score(cheap_stats, boundaries, selection_mask)
        return float(self.config.gate_feature_prior_scale) * prior

    def _prior_score(
        self,
        cheap_stats: torch.Tensor,
        boundaries: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        entropy = self._standardize(cheap_stats[..., 0], mask.float())
        surprisal = self._standardize(cheap_stats[..., 1], mask.float())
        interference = self._standardize(cheap_stats[..., 2], mask.float())
        boundary = boundaries[..., :2].amax(dim=-1)
        score = (
            float(self.config.entropy_prior_weight) * entropy
            + float(self.config.surprisal_prior_weight) * surprisal
            + float(self.config.interference_prior_weight) * interference
            + float(self.config.boundary_prior_weight) * boundary
        )
        return score.masked_fill(~mask.bool(), 0.0)

    def _prior_distill_target(
        self,
        cheap_stats: torch.Tensor,
        boundaries: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        score = self._prior_score(cheap_stats, boundaries, valid)
        temp = max(1e-4, float(self.config.prior_distill_temperature))
        return torch.sigmoid(score / temp).masked_fill(~valid, 0.0)

    def _topk_targets(
        self,
        scores: torch.Tensor,
        valid: torch.Tensor,
        target_sparsity: float,
    ) -> torch.Tensor:
        targets = torch.zeros_like(scores)
        masked = scores.masked_fill(~valid, float("-inf"))
        valid_counts = valid.sum(dim=1)
        for row, count in enumerate(valid_counts.tolist()):
            if count <= 0:
                continue
            k = max(1, int(round(int(count) * min(1.0, target_sparsity))))
            k = min(k, int(count))
            indices = masked[row].topk(k).indices
            targets[row, indices] = 1.0
        return targets

    def _memory_augmented_logits(
        self,
        hidden: torch.Tensor,
        cheap_logits: torch.Tensor,
        gate_hard: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, seq_len, _ = hidden.shape
        memory_state = self.memory.empty(batch, hidden.device, hidden.dtype)
        residuals: list[torch.Tensor] = []
        event_rows: dict[str, list[torch.Tensor]] = {
            "write_slot": [],
            "read_slot": [],
            "read_strength": [],
            "role": [],
        }

        for t in range(seq_len):
            memory_state, read_hidden, events_t = self.memory.step(
                hidden_t=hidden[:, t, :],
                memory=memory_state,
                gate_t=gate_hard[:, t],
            )
            residual = (
                self.memory_lm_head(read_hidden)
                * self.memory_residual_scale
                * gate_hard[:, t, None]
            )
            residuals.append(residual)
            for key, value in events_t.items():
                event_rows[key].append(value)

        stacked_residuals = torch.stack(residuals, dim=1)
        events = {key: torch.stack(values, dim=1) for key, values in event_rows.items()}
        return cheap_logits + stacked_residuals, events

    def _empty_events(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, seq_len = input_ids.shape
        zeros = torch.zeros(batch, seq_len, device=input_ids.device)
        return {
            "write_slot": zeros.long(),
            "read_slot": zeros.long(),
            "read_strength": zeros,
            "role": zeros.long(),
        }
