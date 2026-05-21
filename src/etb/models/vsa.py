from __future__ import annotations

import torch


def bind(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """MAP-style binding with TorchHD when available, elementwise fallback otherwise."""

    try:
        import torchhd  # type: ignore

        return torchhd.bind(left, right)
    except Exception:
        return left * right


def bundle(vectors: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return vectors.sum(dim=-2)
    return (vectors * weights.unsqueeze(-1)).sum(dim=-2)


def cosine_slots(query: torch.Tensor, slots: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    query_norm = query / query.norm(dim=-1, keepdim=True).clamp_min(eps)
    slots_norm = slots / slots.norm(dim=-1, keepdim=True).clamp_min(eps)
    return torch.einsum("bd,bsd->bs", query_norm, slots_norm)

