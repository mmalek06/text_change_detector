import gc
from typing import Protocol

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    def encode(self, sentences: list[str], normalize_embeddings: bool = True) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        device: str | None = None,
        dtype: torch.dtype | None = torch.float16,
        batch_size: int | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_kwargs = {"torch_dtype": dtype} if dtype is not None else {}
        self._model: SentenceTransformer | None = SentenceTransformer(
            model_name, device=self.device, model_kwargs=model_kwargs
        )
        self._batch_size = batch_size

    def encode(self, sentences: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        kwargs = {"normalize_embeddings": normalize_embeddings}

        if self._batch_size is not None:
            kwargs["batch_size"] = self._batch_size

        return self._model.encode(sentences, **kwargs)

    def close(self) -> None:
        self._model = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
