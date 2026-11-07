"""Task heads: masked-language modelling (pretraining) and pooled Matryoshka embeddings."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import BlockMask

from .attention import PAD_DOC_ID
from .config import ModelConfig
from .encoder import Encoder
from .rope import positions_from_doc_ids


def select_masked(labels: torch.Tensor, num_masked: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick exactly `num_masked` flat positions from a (B, T) label tensor (-100 = not masked).

    Masked positions come first. If fewer than `num_masked` are masked, the remaining slots point
    at unmasked tokens with label -100, which the loss ignores. If more are masked, the surplus is
    dropped, so choose num_masked at or above the expected count. A fixed count keeps tensor
    shapes static, which `torch.compile` needs. The data loader can instead emit exactly
    `num_masked` positions per batch and skip this function.
    """
    flat = labels.reshape(-1)
    positions = torch.topk((flat != -100).to(torch.float32), num_masked, sorted=False).indices
    return positions, flat[positions]


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
        mlm_positions: torch.Tensor | None = None,
        mlm_labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        block_masks: dict[str, BlockMask] | None = None,
    ) -> torch.Tensor:
        """Return the MLM loss if mlm_positions/mlm_labels are given, else full logits.

        mlm_positions: (K,) flat indices into the (B*T) token grid, with K fixed across steps.
        mlm_labels: (K,) target token ids, -100 for unused slots (see `select_masked`).
        Only those K positions pass through the vocabulary projection, which saves most of the
        head's memory and keeps tensor shapes static for `torch.compile`.
        """
        hidden = self.encoder(input_ids, doc_ids, position_ids, block_masks)
        if mlm_positions is None:
            return self._logits(hidden)
        selected = hidden.reshape(-1, hidden.size(-1))[mlm_positions]
        return F.cross_entropy(self._logits(selected).float(), mlm_labels, ignore_index=-100)

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
        block_masks: dict[str, BlockMask] | None = None,
    ) -> torch.Tensor:
        """Padded batch, one text per row (mask 1 = real token). Simple, but short texts waste
        compute on padding (FlexAttention works in 128-token blocks); prefer `encode_packed`.

        dim: truncate to a Matryoshka size (must be one of cfg.matryoshka_dims) and re-normalize.
        """
        doc_ids = torch.where(attention_mask.bool(), 0, PAD_DOC_ID).to(torch.int32)
        hidden = self.encoder(input_ids, doc_ids, block_masks=block_masks)
        if self.cfg.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.embed_from_pooled(pooled, dim)

    def encode_packed(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        num_segments: int,
        dim: int | None = None,
        block_masks: dict[str, BlockMask] | None = None,
    ) -> torch.Tensor:
        """Many texts per row, no padding between them. Returns (num_segments, dim) embeddings.

        segment_ids: (B, T) int32 text index per token, numbered 0..num_segments-1 in row-major
            order (the order the texts appear when reading rows top to bottom, left to right),
            and -1 for trailing padding. Each text must be contiguous.
        num_segments: total number of texts in the batch. Fixed per batch for static shapes.
        """
        hidden = self.encoder(input_ids, segment_ids, block_masks=block_masks)
        flat_hidden = hidden.reshape(-1, hidden.size(-1))
        flat_segments = segment_ids.reshape(-1)
        valid = flat_segments >= 0

        if self.cfg.pooling == "cls":
            is_first = (positions_from_doc_ids(segment_ids).reshape(-1) == 0) & valid
            starts = torch.nonzero_static(is_first, size=num_segments).squeeze(1)
            pooled = flat_hidden[starts]
        else:
            weights = valid.to(torch.float32)
            sums = torch.zeros(
                num_segments, flat_hidden.size(-1), device=hidden.device, dtype=torch.float32
            )
            sums.index_add_(0, flat_segments.clamp(min=0), flat_hidden.float() * weights[:, None])
            counts = torch.zeros(num_segments, device=hidden.device, dtype=torch.float32)
            counts.index_add_(0, flat_segments.clamp(min=0), weights)
            pooled = sums / counts.clamp(min=1)[:, None]
        return self.embed_from_pooled(pooled, dim)

    def embed_from_pooled(self, pooled: torch.Tensor, dim: int | None = None) -> torch.Tensor:
        emb = self.projection(pooled)
        if dim is not None:
            if dim not in self.cfg.matryoshka_dims:
                raise ValueError(f"dim must be one of {self.cfg.matryoshka_dims}, got {dim}")
            emb = emb[..., :dim]
        return F.normalize(emb.float(), dim=-1)
