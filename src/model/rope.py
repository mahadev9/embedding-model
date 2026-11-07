"""Rotary position embeddings (half-split convention) driven by explicit position ids."""

from __future__ import annotations

import torch
from torch import nn


def positions_from_doc_ids(doc_ids: torch.Tensor) -> torch.Tensor:
    """Per-document positions for packed rows: 0, 1, 2, ... restarting at every new document.

    doc_ids: (B, T) integer document index per token (any value; changes mark boundaries).
    """
    _, seq_len = doc_ids.shape
    idx = torch.arange(seq_len, device=doc_ids.device).expand_as(doc_ids)
    is_start = torch.ones_like(doc_ids, dtype=torch.bool)
    is_start[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]
    start_idx = torch.where(is_start, idx, torch.zeros_like(idx)).cummax(dim=1).values
    return idx - start_idx


class RotaryEmbedding(nn.Module):
    """Produces cos/sin tables for one theta. Holds only a non-persistent frequency buffer."""

    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        exponent = torch.arange(0, head_dim, 2, dtype=torch.float32)
        self.register_buffer("inv_freq", 1.0 / (theta ** (exponent / head_dim)), persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """position_ids (B, T) -> cos, sin of shape (B, 1, T, head_dim) in float32."""
        freqs = position_ids.to(torch.float32)[..., None] * self.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[:, None], emb.sin()[:, None]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate q and k of shape (B, H, T, D); the rotation is computed in float32."""
    q_dtype, k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q.to(q_dtype), k.to(k_dtype)
