from __future__ import annotations

from etb.models.configuration_etb import ETBConfig


def estimate_flops_per_token(config: ETBConfig, gate_rate: float) -> float:
    """Order-of-magnitude activated FLOPs/token used consistently across variants."""

    embed = config.d_model
    gru = 6 * config.hidden_size * (config.hidden_size + config.d_model)
    lm_head = config.hidden_size * config.vocab_size
    gate = config.hidden_size * (config.hidden_size // 2 + 1)
    memory = (
        config.hidden_size * config.vsa_dim
        + config.memory_slots * config.vsa_dim
        + config.vsa_dim * config.hidden_size
        + config.hidden_size * config.vocab_size
    )
    return float(embed + gru + lm_head + gate + gate_rate * memory)

