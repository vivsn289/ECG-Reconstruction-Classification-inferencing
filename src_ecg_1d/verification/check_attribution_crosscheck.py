# src_ecg_1d/verification/check_attribution_crosscheck.py
#
# Layer 5, Check 4: attribution cross-check.
#
# Catches: the LLM emphasizes a lead that doesn't actually have significant
# ΔIG values. Even if tags are valid AND semantically consistent (Checks
# 2-3), the underlying attribution may be weak — meaning the LLM latched
# onto a minor evidence item and inflated it into a primary claim. This is
# the check that ties the explanation back to the classifier's actual
# decision behavior, not just to prompt-following.

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
import torch

from configs.config_1d import ATTRIBUTION_SIGNIFICANCE_PERCENTILE
from src_ecg_1d.generation.parser import ParsedClaim
from src_ecg_1d.generation.prompt_builder import TaggedEvidence
from src_ecg_1d.verification.check_semantic_consistency import extract_lead_names
from src_ecg_1d.verification.report import CheckResult, CheckStatus


def check_attribution_crosscheck(
    claim: ParsedClaim,
    tag_registry: Dict[str, TaggedEvidence],
    delta_ig: Union["torch.Tensor", np.ndarray],
    lead_names: List[str],
    significance_threshold_percentile: float = ATTRIBUTION_SIGNIFICANCE_PERCENTILE,
) -> CheckResult:
    """Check 4: emphasized regions must have significant ΔIG values.

    Extracts lead names from the claim text, then checks whether |ΔIG| in
    those leads exceeds a significance threshold derived from this
    record's own attribution distribution (a percentile, since absolute
    ΔIG magnitude varies record to record).

    Returns:
        PASS if all emphasized leads have significant ΔIG, or the claim
            names no specific leads.
        WARN if some (but not all) emphasized leads are significant.
        FAIL if any emphasized lead's ΔIG is well below the threshold
            (potential confabulation).
    """
    if claim.is_unsupported or len(claim.tag_ids) == 0:
        return CheckResult(
            check_name="attribution_crosscheck",
            status=CheckStatus.PASS,
            message="No tags — no attribution to check",
            details={},
        )

    claim_leads = extract_lead_names(claim.claim_text)
    if not claim_leads:
        return CheckResult(
            check_name="attribution_crosscheck",
            status=CheckStatus.PASS,
            message="Claim does not emphasize specific leads",
            details={},
        )

    delta_ig_np = delta_ig.detach().cpu().numpy() if isinstance(delta_ig, torch.Tensor) else np.asarray(delta_ig)
    all_magnitudes = np.abs(delta_ig_np)
    threshold = float(np.percentile(all_magnitudes, significance_threshold_percentile))

    lead_significance = {}
    weak_leads = []
    strong_leads = []

    for lead_name in claim_leads:
        if lead_name not in lead_names:
            continue  # not a valid lead — nothing to cross-check
        lead_idx = lead_names.index(lead_name)
        lead_ig = np.abs(delta_ig_np[lead_idx])
        lead_max = float(lead_ig.max())
        lead_mean = float(lead_ig.mean())
        lead_significance[lead_name] = {
            "max": lead_max,
            "mean": lead_mean,
            "significant": lead_max >= threshold,
        }
        if lead_max < threshold * 0.5:
            weak_leads.append(lead_name)
        elif lead_max >= threshold:
            strong_leads.append(lead_name)

    if weak_leads:
        return CheckResult(
            check_name="attribution_crosscheck",
            status=CheckStatus.FAIL,
            message=f"Claim emphasizes leads with weak ΔIG: {weak_leads}",
            details={
                "weak_leads": weak_leads,
                "strong_leads": strong_leads,
                "threshold": threshold,
                "lead_significance": lead_significance,
            },
        )

    if len(strong_leads) < len(lead_significance):
        return CheckResult(
            check_name="attribution_crosscheck",
            status=CheckStatus.WARN,
            message="Some emphasized leads have marginal ΔIG",
            details={
                "strong_leads": strong_leads,
                "lead_significance": lead_significance,
            },
        )

    return CheckResult(
        check_name="attribution_crosscheck",
        status=CheckStatus.PASS,
        message="All emphasized leads have significant ΔIG",
        details={"lead_significance": lead_significance},
    )
