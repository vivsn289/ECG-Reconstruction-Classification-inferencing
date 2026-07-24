# src_ecg_1d/verification/check_tag_completeness.py
#
# Layer 5, Check 1: tag completeness.
#
# Catches: the LLM makes a clinical claim without citing any evidence tag.

from __future__ import annotations

from typing import Dict

from src_ecg_1d.generation.parser import ParsedClaim
from src_ecg_1d.generation.prompt_builder import TaggedEvidence
from src_ecg_1d.verification.report import CheckResult, CheckStatus


def check_tag_completeness(
    claim: ParsedClaim,
    tag_registry: Dict[str, TaggedEvidence],
) -> CheckResult:
    """Check 1: every clinical claim must cite at least one evidence tag.

    A claim without any tags means the LLM made an assertion it couldn't
    ground. Exception: [UNSUPPORTED] claims are allowed — they're an
    honest admission of uncertainty, not a hallucination.

    Returns:
        PASS if the claim has >=1 tag OR is marked [UNSUPPORTED].
        FAIL otherwise.
    """
    if claim.is_unsupported:
        return CheckResult(
            check_name="tag_completeness",
            status=CheckStatus.PASS,
            message="Explicitly marked as unsupported",
            details={"tag_count": 0},
        )

    if len(claim.tag_ids) == 0:
        return CheckResult(
            check_name="tag_completeness",
            status=CheckStatus.FAIL,
            message="Clinical claim made without citing any evidence",
            details={"tag_count": 0, "claim_text": claim.claim_text[:100]},
        )

    return CheckResult(
        check_name="tag_completeness",
        status=CheckStatus.PASS,
        message=f"Claim has {len(claim.tag_ids)} evidence tag(s)",
        details={"tag_count": len(claim.tag_ids)},
    )
