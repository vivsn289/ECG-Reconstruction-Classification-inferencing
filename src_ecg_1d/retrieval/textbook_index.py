# src_ecg_1d/retrieval/textbook_index.py
#
# Index A: FAISS index over curated cardiology textbook passages describing
# ECG morphological patterns. Built once from knowledge_base/ecg_passages.json
# and queried at inference time using the evidence-grounded query string
# produced by Layer 1 (contrastive/evidence.py).

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import List

import faiss
import numpy as np

from src_ecg_1d.retrieval.embedder import Embedder

_INDEX_FILENAME = "textbook_index.faiss"
_METADATA_FILENAME = "textbook_metadata.json"


@dataclass
class TextbookPassage:
    """One curated cardiology passage from the knowledge base."""
    passage_id: str
    text: str
    condition: str        # superclass this passage relates to: "MI", "STTC", etc.
    morphology: str        # ECG feature described: "ST-elevation", "LBBB", etc.
    leads: List[str]       # relevant leads, e.g. ["II", "III", "aVF"]


@dataclass
class TextbookResult:
    """One retrieved passage with its similarity score."""
    passage: TextbookPassage
    similarity_score: float


class TextbookIndex:
    """FAISS index over cardiology textbook passages.

    Build once from ecg_passages.json, save to disk, load at inference.
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.index: faiss.IndexFlatIP = None
        self.passages: List[TextbookPassage] = []

    def build(self, passages_json_path: str) -> None:
        """Load passages from JSON, embed, build FAISS index.

        Args:
            passages_json_path: path to a JSON file containing a list of
                passage dicts with keys "id", "text", "condition",
                "morphology", "leads" (see knowledge_base/ecg_passages.json).
        """
        with open(passages_json_path, "r") as f:
            raw_passages = json.load(f)

        self.passages = [
            TextbookPassage(
                passage_id=p["id"],
                text=p["text"],
                condition=p["condition"],
                morphology=p["morphology"],
                leads=p["leads"],
            )
            for p in raw_passages
        ]

        texts = [p.text for p in self.passages]
        vectors = self.embedder.encode(texts)

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def query(self, query_text: str, top_k: int = 5) -> List[TextbookResult]:
        """Embed the query, search the index, return top-k passages with scores."""
        query_vec = self.embedder.encode(query_text)
        k = min(top_k, len(self.passages))
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(TextbookResult(
                passage=self.passages[idx],
                similarity_score=float(score),
            ))
        return results

    def save(self, directory: str) -> None:
        """Save FAISS index + metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, _INDEX_FILENAME))

        metadata = [asdict(p) for p in self.passages]
        with open(os.path.join(directory, _METADATA_FILENAME), "w") as f:
            json.dump(metadata, f, indent=2)

    def load(self, directory: str) -> None:
        """Load a previously saved index from disk."""
        self.index = faiss.read_index(os.path.join(directory, _INDEX_FILENAME))

        with open(os.path.join(directory, _METADATA_FILENAME), "r") as f:
            metadata = json.load(f)
        self.passages = [TextbookPassage(**p) for p in metadata]
