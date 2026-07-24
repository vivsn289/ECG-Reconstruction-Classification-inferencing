# src_ecg_1d/verification/verifier.py
#
# Layer 5: orchestrates all 4 checks over an LLM generation.
#
# Each parsed claim from Layer 4 passes through all 4 independent checks;
# results are aggregated into a VerificationReport with per-claim and
# summary-level metrics.

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import torch

from src_ecg_1d.generation.generator import GenerationResult
from src_ecg_1d.retrieval.embedder import Embedder
from src_ecg_1d.verification.check_attribution_crosscheck import check_attribution_crosscheck
from src_ecg_1d.verification.check_semantic_consistency import check_semantic_consistency
from src_ecg_1d.verification.check_tag_completeness import check_tag_completeness
from src_ecg_1d.verification.check_tag_validity import check_tag_validity
from src_ecg_1d.verification.report import CheckStatus, ClaimVerification, VerificationReport

_CHECK_NAMES = [
    "tag_completeness",
    "tag_validity",
    "semantic_consistency",
    "attribution_crosscheck",
]


class Verifier:
    """Runs all 4 checks on an LLM generation and produces a report."""

    def __init__(self, embedder: Optional[Embedder] = None):
        """Embedder is required for Check 3 (semantic consistency).

        If None, one will be created (uses the same all-MiniLM-L6-v2 model
        as the retrieval layer — pass in retriever.embedder to reuse the
        already-loaded model instead of loading it twice).
        """
        self.embedder = embedder or Embedder()

    def verify(
        self,
        generation_result: GenerationResult,
        delta_ig: Union["torch.Tensor", np.ndarray],
        lead_names: List[str],
    ) -> VerificationReport:
        """Run all 4 checks on every parsed claim.

        Args:
            generation_result: output of Layer 4 (ExplanationGenerator.generate).
            delta_ig: (12, T) ΔIG map from Layer 1.
            lead_names: LEAD_NAMES from config.

        Returns:
            VerificationReport with per-claim results and aggregate metrics.
        """
        tag_registry = generation_result.structured_prompt.tag_registry
        parsed = generation_result.parsed_response

        claim_verifications = []
        for claim in parsed.claims:
            check_results = [
                check_tag_completeness(claim, tag_registry),
                check_tag_validity(claim, tag_registry),
                check_semantic_consistency(claim, tag_registry, self.embedder),
                check_attribution_crosscheck(claim, tag_registry, delta_ig, lead_names),
            ]
            claim_verifications.append(ClaimVerification(claim=claim, check_results=check_results))

        aggregate = self._compute_aggregate_metrics(claim_verifications, parsed)

        return VerificationReport(
            claim_verifications=claim_verifications,
            untagged_sentences=parsed.untagged_sentences,
            aggregate_metrics=aggregate,
        )

    def _compute_aggregate_metrics(self, verifications: List[ClaimVerification], parsed) -> dict:
        """Compute summary metrics for the whole response."""
        total = len(verifications)
        passed = sum(1 for v in verifications if v.overall_status == CheckStatus.PASS)
        failed = sum(1 for v in verifications if v.overall_status == CheckStatus.FAIL)
        warned = sum(1 for v in verifications if v.overall_status == CheckStatus.WARN)

        check_failures = {name: 0 for name in _CHECK_NAMES}
        for v in verifications:
            for r in v.check_results:
                if r.status == CheckStatus.FAIL:
                    check_failures[r.check_name] += 1

        return {
            "total_claims": total,
            "passed": passed,
            "failed": failed,
            "warned": warned,
            "pass_rate": passed / total if total > 0 else 0.0,
            "check_failure_counts": check_failures,
            "untagged_sentence_count": len(parsed.untagged_sentences),
        }
