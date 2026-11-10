"""MLM data: deterministic multilingual row sampling from the token files, and GPU masking.

Rows are contiguous seq_len-token windows of each language's token stream, so documents are packed
with no padding; document boundaries come from the [CLS] ids. Which rows a step reads depends only
on (seed, step), so training resumes exactly with no saved loader state.
"""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.data.paths import TOKENIZED_DIR

CLS_ID, SEP_ID, MASK_ID = 2, 3, 4  # see SPECIAL_TOKENS in src/tokenizer/common.py


@dataclass
class LanguageStream:
    code: str
    train: np.memmap
    val: np.memmap
    windows: int  # number of full seq_len windows in the pinned train data
    runs: int  # windows grouped into runs of run_rows consecutive rows


class CorpusReader:
    """Reads train rows (pinned to a snapshot) and fixed validation rows."""

    def __init__(
        self,
        snapshot: str,
        seq_len: int,
        run_rows: int = 16,
        seed: int = 0,
        root: Path = TOKENIZED_DIR,
    ) -> None:
        self.seq_len, self.run_rows, self.seed = seq_len, run_rows, seed
        counts = json.loads((root / "snapshots" / f"{snapshot}.json").read_text())
        counts.pop("total_train_tokens", None)
        self.streams: dict[str, LanguageStream] = {}
        for code, c in counts.items():
            train = np.memmap(root / code / "train.bin", dtype="<i4", mode="r")[: c["train_tokens"]]
            val = np.memmap(root / code / "val.bin", dtype="<i4", mode="r")[: c["val_tokens"]]
            windows = len(train) // seq_len
            self.streams[code] = LanguageStream(code, train, val, windows, windows // run_rows)
        self.codes = list(self.streams)
        tokens = np.array([counts[c]["train_tokens"] for c in self.codes], dtype=np.float64)
        self.probs = tokens / tokens.sum()  # the corpus's alpha=0.5 mix
        self._perm_cache: dict[tuple[str, int], np.ndarray] = {}

    # ------------------------------------------------------------------ training rows

    def _run_permutation(self, code: str, epoch: int) -> np.ndarray:
        key = (code, epoch)
        if key not in self._perm_cache:
            if len(self._perm_cache) > 64:
                self._perm_cache.clear()
            rng = np.random.default_rng([self.seed, self.codes.index(code), epoch, 7])
            self._perm_cache[key] = rng.permutation(self.streams[code].runs)
        return self._perm_cache[key]

    def counts_for_step(self, step: int, rows: int) -> np.ndarray:
        """Rows drawn from each language at a step (a multinomial from the mix, seeded by step)."""
        rng = np.random.default_rng([self.seed, step, 11])
        return rng.multinomial(rows, self.probs)

    def cursors_at(self, step: int, rows: int) -> np.ndarray:
        """Total rows consumed per language before `step`. Replays the draws (cheap) on resume."""
        cursors = np.zeros(len(self.codes), dtype=np.int64)
        for s in range(step):
            cursors += self.counts_for_step(s, rows)
        return cursors

    def _take_rows(self, code: str, start: int, n: int) -> np.ndarray:
        """n rows from a language's shuffled stream starting at row index `start`."""
        stream = self.streams[code]
        per_epoch = stream.runs * self.run_rows
        out = np.empty((n, self.seq_len), dtype=np.int32)
        i = 0
        while i < n:
            g = start + i
            epoch, within = divmod(g, per_epoch)
            run = self._run_permutation(code, epoch)[within // self.run_rows]
            offset = within % self.run_rows
            take = min(self.run_rows - offset, n - i)  # rest of this run: one contiguous read
            first = run * self.run_rows + offset
            block = stream.train[first * self.seq_len : (first + take) * self.seq_len]
            out[i : i + take] = block.reshape(take, self.seq_len)
            i += take
        return out

    def step_rows(self, step: int, rows: int, cursors: np.ndarray) -> np.ndarray:
        """The (rows, seq_len) int32 array for a step. Advances `cursors` in place."""
        counts = self.counts_for_step(step, rows)
        parts = []
        for k, code in enumerate(self.codes):
            if counts[k]:
                parts.append(self._take_rows(code, int(cursors[k]), int(counts[k])))
        cursors += counts
        batch = np.concatenate(parts)
        order = np.random.default_rng([self.seed, step, 13]).permutation(rows)
        return batch[order]

    def iter_batches(self, start_step: int, total_steps: int, rows: int) -> Iterator[torch.Tensor]:
        cursors = self.cursors_at(start_step, rows)
        for step in range(start_step, total_steps):
            yield torch.from_numpy(self.step_rows(step, rows, cursors))

    # ------------------------------------------------------------------ validation rows

    def val_rows(self, code: str, rows: int) -> torch.Tensor:
        """The first `rows` validation windows of a language, as a fixed (rows, seq_len) array."""
        val = self.streams[code].val
        n = min(rows, len(val) // self.seq_len)
        return torch.from_numpy(
            np.ascontiguousarray(val[: n * self.seq_len]).reshape(n, self.seq_len)
        )


class Prefetcher:
    """Runs a batch generator on a background thread so disk reads overlap with the GPU.

    Tensors are pinned for fast transfer; any other batch object is passed through unchanged.
    """

    def __init__(self, iterator: Iterator[torch.Tensor], depth: int = 6) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._error: BaseException | None = None
        threading.Thread(target=self._run, args=(iterator,), daemon=True).start()

    def _run(self, iterator: Iterator[torch.Tensor]) -> None:
        try:
            for batch in iterator:
                self._queue.put(batch.pin_memory() if hasattr(batch, "pin_memory") else batch)
        except BaseException as err:  # surfaced to the consumer
            self._error = err
        self._queue.put(None)

    def __iter__(self) -> Iterator[torch.Tensor]:
        while (batch := self._queue.get()) is not None:
            yield batch
        if self._error is not None:
            raise self._error


# ---------------------------------------------------------------------------------- masking


def make_mlm_batch(
    rows: torch.Tensor, mask_rate: float, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Turn a (B, T) int32 row tensor on the GPU into model inputs.

    Returns (input_ids, doc_ids, positions, targets). Exactly K = mask_rate * B * T positions are
    masked (static shape for torch.compile), chosen uniformly among non-special tokens and always
    replaced by [MASK]. doc_ids increase at every [CLS], so attention never crosses documents.
    """
    ids = rows.long()
    batch, seq_len = ids.shape
    doc_ids = (ids == CLS_ID).cumsum(dim=1).to(torch.int32)
    scores = torch.rand(batch, seq_len, device=ids.device, generator=generator)
    scores = scores.masked_fill((ids == CLS_ID) | (ids == SEP_ID), 2.0)  # never masked
    num_masked = int(mask_rate * batch * seq_len)
    positions = torch.topk(scores.view(-1), num_masked, largest=False, sorted=False).indices
    flat = ids.view(-1)
    targets = flat[positions]
    inputs = flat.clone()
    inputs[positions] = MASK_ID
    return inputs.view(batch, seq_len), doc_ids, positions, targets
