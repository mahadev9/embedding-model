"""Model configuration: loaded from YAML, validated, with analytic parameter counts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

import yaml

Pooling = Literal["cls", "mean"]


@dataclass(frozen=True)
class ModelConfig:
    """ModernBERT-style bidirectional encoder with a Matryoshka embedding head."""

    name: str

    # Transformer body
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int  # GeGLU intermediate width (the up/gate projection is 2x this)
    vocab_size: int = 96000
    norm_eps: float = 1e-5

    # Attention layout: layer i is global when i % global_every == 0, otherwise sliding-window.
    # The last layer is always global, so the pooled CLS token sees the whole document.
    global_every: int = 3
    local_window: int = 128
    max_position: int = 8192
    rope_theta_local: float = 10_000.0
    # 10k during pretraining; raised to 160k when the context is extended to 8192.
    rope_theta_global: float = 10_000.0

    # Embedding head
    pooling: Pooling = "cls"
    output_dim: int = 768
    matryoshka_dims: tuple[int, ...] = (768, 512, 256, 128)

    def __post_init__(self) -> None:
        errors: list[str] = []
        if self.hidden_size % self.num_heads:
            errors.append("hidden_size must be divisible by num_heads")
        elif self.head_dim != 64:
            errors.append(f"head_dim must be 64, got {self.head_dim}")
        if self.vocab_size % 64:
            errors.append("vocab_size must be a multiple of 64")
        if self.intermediate_size % 64 or self.hidden_size % 64:
            errors.append("hidden_size and intermediate_size must be multiples of 64")
        if self.global_every < 1:
            errors.append("global_every must be >= 1")
        if self.local_window < 2 or self.local_window % 2:
            errors.append("local_window must be a positive even number")
        if self.max_position < self.local_window:
            errors.append("max_position must be >= local_window")
        if self.pooling not in ("cls", "mean"):
            errors.append("pooling must be 'cls' or 'mean'")
        if not self.matryoshka_dims or max(self.matryoshka_dims) != self.output_dim:
            errors.append("matryoshka_dims must be non-empty and its max must equal output_dim")
        if list(self.matryoshka_dims) != sorted(self.matryoshka_dims, reverse=True):
            errors.append("matryoshka_dims must be sorted in descending order")
        if errors:
            raise ValueError(f"Invalid model config '{self.name}': " + "; ".join(errors))

    # ------------------------------------------------------------------ derived

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def layer_types(self) -> tuple[Literal["global", "local"], ...]:
        return tuple(
            "global" if i % self.global_every == 0 or i == self.num_layers - 1 else "local"
            for i in range(self.num_layers)
        )

    @property
    def has_output_projection(self) -> bool:
        return self.hidden_size != self.output_dim

    # ------------------------------------------------------------- parameter counts

    @property
    def params_per_layer(self) -> int:
        h, i = self.hidden_size, self.intermediate_size
        attention = 4 * h * h  # fused QKV (3h*h) + output (h*h), no biases
        feed_forward = 3 * h * i  # GeGLU: in (h x 2i) + out (i x h), no biases
        norms = 2 * h  # two RMSNorm weight vectors
        return attention + feed_forward + norms

    @property
    def non_embedding_params(self) -> int:
        """Transformer body: all layers, plus the embedding and final norms.

        The first block has no attention norm (the embedding norm precedes it), hence -1 vector.
        """
        return self.num_layers * self.params_per_layer + (2 - 1) * self.hidden_size

    @property
    def embedding_params(self) -> int:
        return self.vocab_size * self.hidden_size

    @property
    def projection_params(self) -> int:
        return self.hidden_size * self.output_dim if self.has_output_projection else 0

    @property
    def total_params(self) -> int:
        """Body + embedding table + output projection (the tied MLM head adds only a vocab bias)."""
        return self.non_embedding_params + self.embedding_params + self.projection_params

    # ------------------------------------------------------------------------- io

    @classmethod
    def from_dict(cls, data: dict) -> ModelConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown model config keys: {sorted(unknown)}")
        data = dict(data)
        if "matryoshka_dims" in data:
            data["matryoshka_dims"] = tuple(data["matryoshka_dims"])
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        n_global = self.layer_types.count("global")
        return (
            f"{self.name}: {self.num_layers}L x {self.hidden_size}d, {self.num_heads} heads, "
            f"ffn {self.intermediate_size} | non-embedding {self.non_embedding_params / 1e6:.1f}M, "
            f"embedding {self.embedding_params / 1e6:.1f}M, total {self.total_params / 1e6:.1f}M | "
            f"{n_global} global / {self.num_layers - n_global} local layers"
        )


def load_model_config(name_or_path: str | Path) -> ModelConfig:
    """Load by size name ('50m', '100m', '200m') from configs/model/, or by explicit path."""
    path = Path(name_or_path)
    if not path.suffix:
        path = Path(__file__).resolve().parents[2] / "configs" / "model" / f"{path.name}.yaml"
    return ModelConfig.from_yaml(path)


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:] or ["50m", "100m", "200m"]:
        print(load_model_config(arg).summary())
