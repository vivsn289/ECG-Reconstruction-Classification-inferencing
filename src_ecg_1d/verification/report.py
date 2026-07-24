# src_ecg_1d/verification/report.py
#
# Layer 5: structured verification report dataclasses.
#
# Every check (check_tag_completeness.py, check_tag_validity.py,
# check_semantic_consistency.py, check_attribution_crosscheck.py) returns
# a CheckResult. Verifier.verify() aggregates the 4 results per claim into
# a ClaimVerification, and all claims for one generation into a
# VerificationReport.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

from src_ecg_1d.generation.parser import ParsedClaim


class CheckStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"       # borderline case, worth noting


@dataclass
class CheckResult:
    """Result of one check on one claim."""
    check_name: str              # "tag_completeness", "tag_validity", etc.
    status: CheckStatus
    message: str                 # human-readable explanation
    details: dict                # check-specific data (scores, indices, etc.)


@dataclass
class ClaimVerification:
    """All 4 check results for one claim."""
    claim: ParsedClaim
    check_results: List[CheckResult]

    @property
    def overall_status(self) -> CheckStatus:
        """FAIL if any check FAILs, WARN if any WARN, else PASS."""
        statuses = [r.status for r in self.check_results]
        if CheckStatus.FAIL in statuses:
            return CheckStatus.FAIL
        if CheckStatus.WARN in statuses:
            return CheckStatus.WARN
        return CheckStatus.PASS


@dataclass
class VerificationReport:
    """Full verification of one LLM generation."""
    claim_verifications: List[ClaimVerification]
    untagged_sentences: List[str]        # from ParsedResponse
    aggregate_metrics: dict               # summary stats

    def summary(self) -> str:
        """Return a printable summary."""
        m = self.aggregate_metrics
        lines = [
            f"Total claims: {m['total_claims']}  "
            f"(pass={m['passed']}, fail={m['failed']}, warn={m['warned']})",
            f"Pass rate: {m['pass_rate']:.1%}",
            f"Untagged clinical sentences: {m['untagged_sentence_count']}",
            "",
            "Per-check failure counts:",
        ]
        for check_name, count in m["check_failure_counts"].items():
            lines.append(f"  {check_name}: {count}")

        failing = [cv for cv in self.claim_verifications if cv.overall_status == CheckStatus.FAIL]
        if failing:
            lines.append("")
            lines.append(f"Top failing claims (showing up to 5 of {len(failing)}):")
            for cv in failing[:5]:
                failed_checks = [r.check_name for r in cv.check_results if r.status == CheckStatus.FAIL]
                lines.append(f"  - [{', '.join(failed_checks)}] {cv.claim.claim_text[:100]}")

        return "\n".join(lines)
