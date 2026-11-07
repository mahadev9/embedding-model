"""Bidirectional attention with document masking and optional sliding window (FlexAttention).

A batch row may contain several packed documents. Tokens attend only within their own document;
local layers additionally restrict attention to a window of +-(window // 2) tokens.
Padding tokens carry their own doc id (-1), so they attend only to other padding and never to
real text.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from .config import ModelConfig
from .rope import RotaryEmbedding, apply_rope

# flex_attention must be compiled to be fast and memory-efficient.
_flex_attention = torch.compile(flex_attention)

PAD_DOC_ID = -1


def build_block_masks(doc_ids: torch.Tensor, local_window: int) -> dict[str, BlockMask]:
    """Create the global and local block masks once per batch; every layer reuses them.

    doc_ids: (B, T) document index per token, PAD_DOC_ID for padding.
    """
    batch, seq_len = doc_ids.shape
    half = local_window // 2

    def global_mask(b, h, q_idx, kv_idx):
        return doc_ids[b, q_idx] == doc_ids[b, kv_idx]

    def local_mask(b, h, q_idx, kv_idx):
        same_doc = doc_ids[b, q_idx] == doc_ids[b, kv_idx]
        return same_doc & ((q_idx - kv_idx).abs() <= half)

    common = dict(B=batch, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=doc_ids.device)
    return {
        "global": create_block_mask(global_mask, **common, _compile=True),
        "local": create_block_mask(local_mask, **common, _compile=True),
    }


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_type: str) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        theta = cfg.rope_theta_global if layer_type == "global" else cfg.rope_theta_local
        self.rope = RotaryEmbedding(cfg.head_dim, theta)
        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        self.out = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        block_masks: dict[str, BlockMask],
    ) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, H, T, D)
        cos, sin = self.rope.cos_sin(position_ids)
        q, k = apply_rope(q, k, cos, sin)
        attn = _flex_attention(q, k, v, block_mask=block_masks[self.layer_type])
        return self.out(attn.transpose(1, 2).reshape(batch, seq_len, hidden))
