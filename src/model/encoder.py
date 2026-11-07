"""ModernBERT-style encoder body: pre-norm blocks, GeGLU, no biases, local/global attention."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import BlockMask

from .attention import Attention, build_block_masks
from .config import ModelConfig
from .rope import RotaryEmbedding, positions_from_doc_ids


class GeGLU(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.wi = nn.Linear(hidden, 2 * intermediate, bias=False)
        self.wo = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.wi(x).chunk(2, dim=-1)
        return self.wo(F.gelu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        layer_type = cfg.layer_types[layer_idx]
        # The embedding output is already normalized, so the first block skips its attention norm.
        self.attn_norm = (
            nn.Identity() if layer_idx == 0 else nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        )
        self.attn = Attention(cfg, layer_type)
        self.mlp_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.mlp = GeGLU(cfg.hidden_size, cfg.intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        rope: dict[str, tuple[torch.Tensor, torch.Tensor]],
        block_masks: dict[str, BlockMask],
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope, block_masks)
        return x + self.mlp(self.mlp_norm(x))


class Encoder(nn.Module):
    """Token ids -> contextual hidden states. Accepts padded or packed rows via doc_ids."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.embed_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        # One set of cos/sin tables per layer type, computed once per forward and shared by layers.
        self.rope = nn.ModuleDict(
            {
                "local": RotaryEmbedding(cfg.head_dim, cfg.rope_theta_local),
                "global": RotaryEmbedding(cfg.head_dim, cfg.rope_theta_global),
            }
        )
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.num_layers))
        self.final_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.reset_parameters()

    def reset_parameters(self, std: float = 0.02) -> None:
        """Truncated-normal init; residual output projections are scaled by 1/sqrt(2 * layers)."""
        residual_std = std / math.sqrt(2 * self.cfg.num_layers)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear | nn.Embedding):
                is_residual_out = name.endswith(("attn.out", "mlp.wo"))
                nn.init.trunc_normal_(
                    module.weight,
                    std=residual_std if is_residual_out else std,
                    a=-3 * std,
                    b=3 * std,
                )

    def forward(
        self,
        input_ids: torch.Tensor,
        doc_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        block_masks: dict[str, BlockMask] | None = None,
    ) -> torch.Tensor:
        """
        input_ids: (B, T) token ids; T should be a multiple of 128 (the FlexAttention block size).
        doc_ids: (B, T) document index per token (-1 for padding), defining who attends to whom.
        position_ids: (B, T) RoPE positions; defaults to positions restarting at each document.
        block_masks: precomputed masks from build_block_masks. Build them OUTSIDE any
            torch.compile region (mask construction graph-breaks the compiler) and pass them in.
        """
        if position_ids is None:
            position_ids = positions_from_doc_ids(doc_ids)
        if block_masks is None:
            block_masks = build_block_masks(doc_ids, self.cfg.local_window)
        rope = {name: module(position_ids) for name, module in self.rope.items()}
        x = self.embed_norm(self.embed(input_ids))
        for block in self.blocks:
            x = block(x, rope, block_masks)
        return self.final_norm(x)
