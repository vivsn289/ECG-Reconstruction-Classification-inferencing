# src_ecg_1d/retrieval/case_index.py
#
# Index B: FAISS index over training-set ΔIG evidence summaries.
#
# Each entry pairs one training ECG's natural-language evidence summary
# (from contrastive/evidence.py::evidence_to_query) with its record id,
# ground-truth labels, and model predictions. The expensive part — running
# ΔIG over every training record — happens in build_case_index.py; this
# module only embeds pre-computed entries and manages the FAISS index.

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import List

import faiss
import numpy as np

from src_ecg_1d.retrieval.embedder import Embedder

_INDEX_FILENAME = "case_index.faiss"
_METADATA_FILENAME = "case_metadata.json"


@dataclass
class CaseEntry:
    """One training record's ΔIG evidence summary and predictions."""
    record_id: int               # PTB-XL ecg_id
    evidence_summary: str        # natural-language summary from evidence_to_query()
    true_labels: List[str]       # ground truth superclasses (multi-hot decoded)
    pred_class: str              # model's highest-confidence prediction
    runner_up_class: str         # second-highest
    pred_probs: List[float]      # sigmoid probabilities for all 5 classes


@dataclass
class CaseResult:
    """One retrieved training case with its similarity score."""
    entry: CaseEntry
    similarity_score: float


class CaseIndex:
    """FAISS index over training set ΔIG evidence summaries."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.index: faiss.IndexFlatIP = None
        self.entries: List[CaseEntry] = []

    def build(self, entries: List[CaseEntry]) -> None:
        """Build index from pre-computed case entries.

        The ΔIG computation and evidence extraction happen in
        build_case_index.py; this just embeds the summaries and builds
        the FAISS index.
        """
        self.entries = list(entries)

        summaries = [e.evidence_summary for e in self.entries]
        vectors = self.embedder.encode(summaries)

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def query(self, query_text: str, top_k: int = 5) -> List[CaseResult]:
        """Embed the query, search the index, return top-k cases with scores."""
        query_vec = self.embedder.encode(query_text)
        k = min(top_k, len(self.entries))
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(CaseResult(
                entry=self.entries[idx],
                similarity_score=float(score),
            ))
        return results

    def save(self, directory: str) -> None:
        """Save FAISS index + metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, _INDEX_FILENAME))

        metadata = [asdict(e) for e in self.entries]
        with open(os.path.join(directory, _METADATA_FILENAME), "w") as f:
            json.dump(metadata, f, indent=2)

    def load(self, directory: str) -> None:
        """Load a previously saved index from disk."""
        self.index = faiss.read_index(os.path.join(directory, _INDEX_FILENAME))

        with open(os.path.join(directory, _METADATA_FILENAME), "r") as f:
            metadata = json.load(f)
        self.entries = [CaseEntry(**e) for e in metadata]
