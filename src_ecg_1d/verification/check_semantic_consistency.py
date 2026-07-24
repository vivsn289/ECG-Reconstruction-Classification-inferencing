# src_ecg_1d/verification/check_semantic_consistency.py
#
# Layer 5, Check 3: semantic consistency.
#
# Catches: the LLM cites a tag whose content doesn't actually support the
# claim (tag exists and is valid, but the claim diverges from it) —
# e.g. "ST elevation in V1 [E1]" when E1's content is actually about lead II.
#
# Combines rule-based checks (lead names, morphology terms extracted via
# regex/keyword match) with an embedding-similarity check, so a claim has
# to diverge on both surface-level facts and semantic content to fail.

from __future__ import annotations

import re
from typing import Dict, Set

import numpy as np

from configs.config_1d import SEMANTIC_SIMILARITY_FAIL_THRESHOLD, SEMANTIC_SIMILARITY_WARN_THRESHOLD
from src_ecg_1d.generation.parser import ParsedClaim
from src_ecg_1d.generation.prompt_builder import TaggedEvidence
from src_ecg_1d.retrieval.embedder import Embedder
from src_ecg_1d.verification.report import CheckResult, CheckStatus

# Matches: I, II, III, aVR, aVL, aVF, V1-V6 — the 12 standard ECG leads.
_LEAD_PATTERN = re.compile(r"\b(I{1,3}|aVR|aVL|aVF|V[1-6])\b")

_MORPHOLOGY_TERMS = {
    "elevation", "depression", "inversion", "flattening",
    "prolongation", "shortening", "widening",
    "P-wave", "QRS", "ST", "T-wave", "U-wave",
    "bundle branch", "LBBB", "RBBB",
    "hypertrophy", "LVH", "RVH",
    "ischemia", "infarction",
}


def extract_lead_names(text: str) -> Set[str]:
    """Extract lead references (e.g. 'II', 'aVF', 'V2') from text via regex."""
    return set(_LEAD_PATTERN.findall(text))


def extract_morphology_terms(text: str) -> Set[str]:
    """Extract morphological terms (elevation, depression, etc.) from text."""
    text_lower = text.lower()
    return {term for term in _MORPHOLOGY_TERMS if term.lower() in text_lower}


def check_semantic_consistency(
    claim: ParsedClaim,
    tag_registry: Dict[str, TaggedEvidence],
    embedder: Embedder,
    similarity_warn_threshold: float = SEMANTIC_SIMILARITY_WARN_THRESHOLD,
    similarity_fail_threshold: float = SEMANTIC_SIMILARITY_FAIL_THRESHOLD,
) -> CheckResult:
    """Check 3: claim content must match the content of its cited evidence.

    Returns:
        PASS if leads/morphology terms in the claim are backed by the
            cited evidence and embedding similarity is above the warn
            threshold.
        WARN if similarity is between the fail and warn thresholds.
        FAIL if the claim mentions leads/morphology absent from every
            cited tag, or similarity falls below the fail threshold.
    """
    if claim.is_unsupported or len(claim.tag_ids) == 0:
        return CheckResult(
            check_name="semantic_consistency",
            status=CheckStatus.PASS,
            message="No tags to verify against",
            details={},
        )

    valid_tags = [t for t in claim.tag_ids if t in tag_registry]
    if not valid_tags:
        return CheckResult(
            check_name="semantic_consistency",
            status=CheckStatus.FAIL,
            message="No valid tags to verify against",
            details={},
        )

    # --- Rule-based checks ---
    claim_leads = extract_lead_names(claim.claim_text)
    claim_morphologies = extract_morphology_terms(claim.claim_text)

    evidence_leads: Set[str] = set()
    evidence_morphologies: Set[str] = set()
    for tag in valid_tags:
        content = tag_registry[tag].content
        evidence_leads.update(extract_lead_names(content))
        evidence_morphologies.update(extract_morphology_terms(content))

    lead_mismatches = claim_leads - evidence_leads if claim_leads else set()
    morphology_mismatches = claim_morphologies - evidence_morphologies if claim_morphologies else set()

    # --- Embedding-based check ---
    claim_vec = embedder.encode(claim.claim_text)
    combined_evidence = " ".join(tag_registry[t].content for t in valid_tags)
    evidence_vec = embedder.encode(combined_evidence)
    similarity = float(np.dot(claim_vec[0], evidence_vec[0]))  # cosine (already L2-normalized)

    # --- Aggregate ---
    if lead_mismatches or morphology_mismatches or similarity < similarity_fail_threshold:
        return CheckResult(
            check_name="semantic_consistency",
            status=CheckStatus.FAIL,
            message="Claim content diverges from cited evidence",
            details={
                "similarity_score": similarity,
                "unsupported_leads": sorted(lead_mismatches),
                "unsupported_morphologies": sorted(morphology_mismatches),
            },
        )

    if similarity < similarity_warn_threshold:
        return CheckResult(
            check_name="semantic_consistency",
            status=CheckStatus.WARN,
            message=f"Similarity to cited evidence is borderline ({similarity:.2f})",
            details={"similarity_score": similarity},
        )

    return CheckResult(
        check_name="semantic_consistency",
        status=CheckStatus.PASS,
        message=f"Claim is consistent with cited evidence (sim={similarity:.2f})",
        details={"similarity_score": similarity},
    )
