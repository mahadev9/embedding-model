"""Task heads: masked-language modelling (pretraining) and pooled Matryoshka embeddings."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .attention import PAD_DOC_ID
from .config import ModelConfig
from .encoder import Encoder


class MLMModel(nn.Module):
    """Encoder + ModernBERT-style MLM head (dense, GELU, norm, tied decoder with bias)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.head_dense = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.head_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.decoder_bias = nn.Parameter(torch.zeros(cfg.vocab_size))
        nn.init.trunc_normal_(self.head_dense.weight, std=0.02, a=-0.06, b=0.06)

    def forward(
        self,
        input_ids: torch.Tensor,
        doc_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the MLM loss if labels are given (ignore index -100), else full logits.

        Only masked positions pass through the vocabulary projection, which saves most of the
        head's memory when ~30% of tokens are masked.
        """
        hidden = self.encoder(input_ids, doc_ids, position_ids)
        if labels is None:
            return self._logits(hidden)
        masked = labels != -100
        logits = self._logits(hidden[masked])
        return F.cross_entropy(logits.float(), labels[masked])

    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.head_norm(F.gelu(self.head_dense(hidden)))
        return F.linear(hidden, self.encoder.embed.weight, self.decoder_bias)


class EmbeddingModel(nn.Module):
    """Encoder + pooling + optional projection -> L2-normalized, Matryoshka-truncatable vectors."""

    def __init__(self, cfg: ModelConfig, encoder: Encoder | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder if encoder is not None else Encoder(cfg)
        self.projection = (
            nn.Linear(cfg.hidden_size, cfg.output_dim, bias=False)
            if cfg.has_output_projection
            else nn.Identity()
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        dim: int | None = None,
    ) -> torch.Tensor:
        """input_ids, attention_mask: (B, T) padded batch, one text per row (mask 1 = real token).

        dim: truncate to a Matryoshka size (must be one of cfg.matryoshka_dims) and re-normalize.
        """
        doc_ids = torch.where(attention_mask.bool(), 0, PAD_DOC_ID).to(torch.int32)
        hidden = self.encoder(input_ids, doc_ids)
        if self.cfg.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.embed_from_pooled(pooled, dim)

    def embed_from_pooled(self, pooled: torch.Tensor, dim: int | None = None) -> torch.Tensor:
        emb = self.projection(pooled)
        if dim is not None:
            if dim not in self.cfg.matryoshka_dims:
                raise ValueError(f"dim must be one of {self.cfg.matryoshka_dims}, got {dim}")
            emb = emb[..., :dim]
        return F.normalize(emb.float(), dim=-1)
