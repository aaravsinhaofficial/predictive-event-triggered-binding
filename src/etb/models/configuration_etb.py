from __future__ import annotations

from transformers import PretrainedConfig


class ETBConfig(PretrainedConfig):
    """Configuration for GRU-based Predictive Event-Triggered Binding models."""

    model_type = "etb"

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 256,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
        variant: str = "etb",
        memory_slots: int = 8,
        memory_roles: int = 8,
        vsa_dim: int = 512,
        gate_threshold: float = 0.5,
        gate_selection: str = "topk",
        target_sparsity: float = 0.15,
        sparsity_lambda: float = 0.01,
        budget_lambda: float = 0.1,
        benefit_lambda: float = 0.2,
        candidate_loss_lambda: float = 0.05,
        benefit_temperature: float = 0.25,
        compute_penalty: float = 0.1,
        structural_prior_bonus: float = 0.15,
        interference_prior_bonus: float = 0.1,
        entropy_prior_weight: float = 1.0,
        surprisal_prior_weight: float = 0.5,
        boundary_prior_weight: float = 1.0,
        interference_prior_weight: float = 0.5,
        gate_feature_prior_scale: float = 1.0,
        gate_target_mode: str = "topk",
        gate_positive_weight: float | None = None,
        candidate_delta_weight: float = 0.35,
        candidate_delta_clip: float = 1.5,
        prior_distill_lambda: float = 0.0,
        prior_distill_temperature: float = 1.0,
        memory_residual_scale_init: float = 0.1,
        memory_dropout: float = 0.0,
        memory_decay: float = 0.98,
        punctuation_token_ids: list[int] | None = None,
        clause_token_ids: list[int] | None = None,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        unk_token_id: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            unk_token_id=unk_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.variant = variant
        self.memory_slots = memory_slots
        self.memory_roles = memory_roles
        self.vsa_dim = vsa_dim
        self.gate_threshold = gate_threshold
        self.gate_selection = gate_selection
        self.target_sparsity = target_sparsity
        self.sparsity_lambda = sparsity_lambda
        self.budget_lambda = budget_lambda
        self.benefit_lambda = benefit_lambda
        self.candidate_loss_lambda = candidate_loss_lambda
        self.benefit_temperature = benefit_temperature
        self.compute_penalty = compute_penalty
        self.structural_prior_bonus = structural_prior_bonus
        self.interference_prior_bonus = interference_prior_bonus
        self.entropy_prior_weight = entropy_prior_weight
        self.surprisal_prior_weight = surprisal_prior_weight
        self.boundary_prior_weight = boundary_prior_weight
        self.interference_prior_weight = interference_prior_weight
        self.gate_feature_prior_scale = gate_feature_prior_scale
        self.gate_target_mode = gate_target_mode
        self.gate_positive_weight = gate_positive_weight
        self.candidate_delta_weight = candidate_delta_weight
        self.candidate_delta_clip = candidate_delta_clip
        self.prior_distill_lambda = prior_distill_lambda
        self.prior_distill_temperature = prior_distill_temperature
        self.memory_residual_scale_init = memory_residual_scale_init
        self.memory_dropout = memory_dropout
        self.memory_decay = memory_decay
        self.punctuation_token_ids = punctuation_token_ids or []
        self.clause_token_ids = clause_token_ids or []
