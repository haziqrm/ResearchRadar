"""Text embedders.

:class:`Specter2Embedder` produces 768-dim SPECTER2 (proximity adapter) vectors.
:class:`FakeEmbedder` is a deterministic, dependency-free stand-in for tests and
smoke runs.  All embedders return L2-normalised ``float32`` arrays of shape
``(n_texts, dim)``.  ``torch`` and ``transformers`` are imported lazily so the
rest of the package works without them.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np

SPECTER2_BASE = "allenai/specter2_base"
SPECTER2_ADAPTER = "allenai/specter2"
SPECTER2_DIM = 768


class Embedder(Protocol):
    """Anything that turns texts into normalised vectors."""

    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale rows to unit length; all-zero rows stay zero."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.where(norms == 0, 1.0, norms)).astype(np.float32)


class FakeEmbedder:
    """Hashed bag-of-words embedder: deterministic, so texts sharing words are close."""

    name = "fake-hash"

    def __init__(self, dim: int = 64) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                sign = 1.0 if (value >> 32) & 1 else -1.0
                out[row, value % self.dim] += sign
        return l2_normalize(out)


class Specter2Embedder:
    """SPECTER2 base model with the proximity adapter (CLS-token embedding)."""

    dim = SPECTER2_DIM

    def __init__(
        self,
        *,
        batch_size: int = 16,
        max_length: int = 512,
        device: str = "cpu",
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 1 <= max_length <= 512:
            raise ValueError("max_length must be between 1 and 512")
        self.name = f"{SPECTER2_BASE}+{SPECTER2_ADAPTER}"
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from adapters import AutoAdapterModel
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "Specter2Embedder needs torch, transformers and adapters "
                "(see the B section of requirements.txt)"
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(SPECTER2_BASE)
        model = AutoAdapterModel.from_pretrained(SPECTER2_BASE)
        model.load_adapter(SPECTER2_ADAPTER, source="hf", load_as="specter2", set_active=True)
        model.to(self.device)
        model.eval()
        self._model = model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        import torch

        self._load()
        # The text builder writes "[SEP]" literally; the tokenizer's own
        # separator token is what SPECTER2 was trained with.
        sep = self._tokenizer.sep_token
        prepared = [text.replace("[SEP]", sep) for text in texts]

        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(prepared), self.batch_size):
                batch = prepared[start:start + self.batch_size]
                inputs = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                    return_token_type_ids=False,
                ).to(self.device)
                hidden = self._model(**inputs).last_hidden_state[:, 0, :]
                chunks.append(hidden.cpu().numpy())
        return l2_normalize(np.concatenate(chunks, axis=0))
