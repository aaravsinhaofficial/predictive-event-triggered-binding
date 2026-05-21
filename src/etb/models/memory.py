from __future__ import annotations

import torch
from torch import nn

from etb.models.configuration_etb import ETBConfig
from etb.models.vsa import bind, cosine_slots


class RoleFillerMemory(nn.Module):
    """Tiny differentiable MAP-style role-filler memory."""

    def __init__(self, config: ETBConfig) -> None:
        super().__init__()
        self.config = config
        roles = torch.empty(config.memory_roles, config.vsa_dim)
        nn.init.normal_(roles, mean=0.0, std=1.0)
        roles = roles.sign().clamp(min=-1.0, max=1.0)
        self.register_buffer("role_vectors", roles, persistent=True)

        self.fill_proj = nn.Linear(config.hidden_size, config.vsa_dim)
        self.role_writer = nn.Linear(config.hidden_size, config.memory_roles)
        self.slot_writer = nn.Linear(config.hidden_size, config.memory_slots)
        self.read_proj = nn.Linear(config.vsa_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.memory_dropout)

    def empty(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.config.memory_slots,
            self.config.vsa_dim,
            device=device,
            dtype=dtype,
        )

    def step(
        self,
        hidden_t: torch.Tensor,
        memory: torch.Tensor,
        gate_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        fill = torch.tanh(self.fill_proj(hidden_t))
        role_probs = self.role_writer(hidden_t).softmax(dim=-1)
        role_vec = role_probs @ self.role_vectors.to(dtype=hidden_t.dtype)
        bound = bind(role_vec, fill)

        slot_probs = self.slot_writer(hidden_t).softmax(dim=-1)
        write = slot_probs.unsqueeze(-1) * bound.unsqueeze(1)
        gated_write = gate_t[:, None, None] * self.dropout(write)
        new_memory = self.config.memory_decay * memory + gated_write

        read_scores = cosine_slots(bound, new_memory)
        read_probs = read_scores.softmax(dim=-1)
        read_vsa = torch.einsum("bs,bsd->bd", read_probs, new_memory)
        read_hidden = torch.tanh(self.read_proj(read_vsa))

        events = {
            "write_slot": slot_probs.argmax(dim=-1),
            "read_slot": read_probs.argmax(dim=-1),
            "read_strength": read_probs.max(dim=-1).values,
            "role": role_probs.argmax(dim=-1),
        }
        return new_memory, read_hidden, events

