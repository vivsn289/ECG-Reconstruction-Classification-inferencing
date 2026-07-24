# src_ecg_1d/verification/check_tag_validity.py
#
# Layer 5, Check 2: tag validity.
#
# Catches: the LLM cites a tag that doesn't exist in the registry
# (e.g. cites [E7] when only E1-E3 were ever provided) — a fabricated
# reference.

from __future__ import annotations

from typing import Dict

from src_ecg_1d.generation.parser import ParsedClaim
from src_ecg_1d.generation.prompt_builder import TaggedEvidence
from src_ecg_1d.verification.report import CheckResult, CheckStatus


def check_tag_validity(
    claim: ParsedClaim,
    tag_registry: Dict[str, TaggedEvidence],
) -> CheckResult:
    """Check 2: every cited tag must exist in the tag registry.

    A fabricated tag means the LLM invented a reference that was never
    provided — a clear hallucination signal.

    Returns:
        PASS if all cited tags exist in the registry (or there are none to check).
        FAIL if any cited tag is not in the registry.
    """
    if claim.is_unsupported or len(claim.tag_ids) == 0:
        return CheckResult(
            check_name="tag_validity",
            status=CheckStatus.PASS,
            message="No tags to validate",
            details={},
        )

    invalid_tags = [tag for tag in claim.tag_ids if tag not in tag_registry]

    if invalid_tags:
        return CheckResult(
            check_name="tag_validity",
            status=CheckStatus.FAIL,
            message=f"Fabricated tag(s): {invalid_tags}",
            details={
                "invalid_tags": invalid_tags,
                "valid_tag_range": sorted(tag_registry.keys()),
            },
        )

    return CheckResult(
        check_name="tag_validity",
        status=CheckStatus.PASS,
        message="All cited tags exist",
        details={"tags": claim.tag_ids},
    )
