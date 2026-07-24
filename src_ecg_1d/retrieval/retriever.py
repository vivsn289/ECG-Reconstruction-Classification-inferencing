# src_ecg_1d/retrieval/retriever.py
#
# Layer 3: unified retrieval across the textbook (Index A) and case-based
# (Index B) FAISS indices. This is what Layer 4 (prompt construction) calls
# at inference time, using the evidence-grounded query string produced by
# Layer 1 (contrastive/evidence.py::evidence_to_query).

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from configs.config_1d import CASE_TOP_K, INDEX_DIR, TEXTBOOK_TOP_K
from src_ecg_1d.retrieval.case_index import CaseIndex, CaseResult
from src_ecg_1d.retrieval.embedder import Embedder
from src_ecg_1d.retrieval.textbook_index import TextbookIndex, TextbookResult


@dataclass
class RetrievalResult:
    """Everything retrieved for one inference-time ECG."""
    query_text: str
    textbook_results: List[TextbookResult]
    case_results: List[CaseResult]


class Retriever:
    """Unified retrieval across textbook and case indices.

    Usage at inference time:
        retriever = Retriever(index_dir="data/indices/")

        # From Layer 1 output:
        evidence_items = extract_evidence(delta_ig, ...)
        query = evidence_to_query(evidence_items, pred_class, runner_up)

        # Retrieve:
        result = retriever.retrieve(query, textbook_k=5, case_k=5)
    """

    def __init__(self, index_dir: str = INDEX_DIR, embedder: Optional[Embedder] = None):
        """Load both indices from disk.

        Args:
            index_dir: directory containing "textbook/" and "case/" subdirectories.
            embedder: optional pre-loaded Embedder (creates one if not provided).
        """
        self.embedder = embedder or Embedder()

        self.textbook_index = TextbookIndex(self.embedder)
        self.textbook_index.load(os.path.join(index_dir, "textbook"))

        self.case_index = CaseIndex(self.embedder)
        self.case_index.load(os.path.join(index_dir, "case"))

    def retrieve(
        self,
        query_text: str,
        textbook_k: int = TEXTBOOK_TOP_K,
        case_k: int = CASE_TOP_K,
    ) -> RetrievalResult:
        """Query both indices and return unified results.

        The query_text should come from evidence_to_query() in Layer 1.
        """
        textbook_results = self.textbook_index.query(query_text, top_k=textbook_k)
        case_results = self.case_index.query(query_text, top_k=case_k)

        return RetrievalResult(
            query_text=query_text,
            textbook_results=textbook_results,
            case_results=case_results,
        )
