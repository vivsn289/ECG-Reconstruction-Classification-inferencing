# src_ecg_1d/retrieval/embedder.py
#
# Thin wrapper around sentence-transformers, used by both the textbook and
# case indices so that model loading and normalization stay consistent.

from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from configs.config_1d import EMBEDDING_MODEL


class Embedder:
    """Wraps a sentence-transformers model for retrieval embeddings.

    - Embedding dimension: 384 (all-MiniLM-L6-v2)
    - Loaded once, reused across textbook and case index building/querying
    - All embeddings are L2-normalized so that FAISS inner product
      (IndexFlatIP) computes cosine similarity.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text(s) into L2-normalized 384-d vectors.

        Args:
            texts: single string or list of strings.

        Returns:
            np.ndarray of shape (N, 384), float32, L2-normalized rows.
            If a single string is passed, N == 1.
        """
        single = isinstance(texts, str)
        text_list = [texts] if single else list(texts)

        embeddings = self.model.encode(text_list, convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        return embeddings
