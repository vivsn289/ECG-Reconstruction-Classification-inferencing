# src_ecg_1d/contrastive/evidence.py
#
# Structured evidence extraction from contrastive IG attribution maps.
#
# Converts a (12, T) ΔIG tensor into a list of EvidenceItem dataclasses,
# each describing a significant attribution window in one ECG lead with
# polarity, magnitude, morphological region, and optional attention agreement.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from configs.config_1d import LEAD_NAMES, LABEL_DECODER, SAMPLING_RATE


# One cardiac cycle is approximated as 1 second (60 BPM) = SAMPLING_RATE samples.
# Morphological boundaries are expressed as fractions of that cycle duration.
_CARDIAC_CYCLE_SAMPLES = SAMPLING_RATE

_MORPHOLOGY_REGIONS = [
    ("P-wave",          0,                           int(0.08 * SAMPLING_RATE)),
    ("QRS",             int(0.08 * SAMPLING_RATE),   int(0.20 * SAMPLING_RATE)),
    ("ST-segment",      int(0.20 * SAMPLING_RATE),   int(0.40 * SAMPLING_RATE)),
    ("T-wave",          int(0.40 * SAMPLING_RATE),   int(0.60 * SAMPLING_RATE)),
    ("U-wave/baseline", int(0.60 * SAMPLING_RATE),   _CARDIAC_CYCLE_SAMPLES),
]

# Clinical descriptor for each (region, polarity) pair, used in the query string.
_REGION_POLARITY_DESCRIPTOR = {
    ("P-wave",          "positive"): "P-wave activation",
    ("P-wave",          "negative"): "P-wave suppression",
    ("QRS",             "positive"): "QRS activation",
    ("QRS",             "negative"): "QRS suppression",
    ("ST-segment",      "positive"): "ST elevation",
    ("ST-segment",      "negative"): "ST depression",
    ("T-wave",          "positive"): "tall T-wave",
    ("T-wave",          "negative"): "T-wave inversion",
    ("U-wave/baseline", "positive"): "U-wave/baseline changes",
    ("U-wave/baseline", "negative"): "U-wave/baseline changes",
}


def get_clinical_descriptor(morphology_region: str, polarity: str) -> str:
    """Map an (morphology_region, polarity) pair to its clinical descriptor,
    e.g. ("ST-segment", "positive") -> "ST elevation".

    Public wrapper around _REGION_POLARITY_DESCRIPTOR so downstream layers
    (Layer 4 prompt construction, Layer 5 verification) can render the
    same clinical term used internally by evidence_to_query(), instead of
    duplicating this mapping.
    """
    return _REGION_POLARITY_DESCRIPTOR.get((morphology_region, polarity), f"{morphology_region} changes")


@dataclass
class EvidenceItem:
    """A single significant attribution window in one ECG lead.

    Produced by extract_evidence() and consumed by evidence_to_query()
    and by downstream RAG layers (retrieval, generation, verification).

    Attributes:
        lead_index: 0-based lead index (0 = "I", ..., 11 = "V6").
        lead_name: human-readable lead label (e.g. "II", "V1").
        start_timestep: inclusive start of the significant window.
        end_timestep: exclusive end of the significant window.
        polarity: "positive" — window favors predicted class;
                  "negative" — window favors runner-up class.
        magnitude: mean absolute ΔIG within this window.
        morphology_region: estimated ECG region based on position within
            the cardiac cycle: "P-wave", "QRS", "ST-segment", "T-wave",
            or "U-wave/baseline".
        attention_agreement: True when the attention rollout also highlights
            this window (only meaningful when attention was passed to
            extract_evidence; otherwise always False).
    """
    lead_index: int
    lead_name: str
    start_timestep: int
    end_timestep: int
    polarity: str
    magnitude: float
    morphology_region: str
    attention_agreement: bool = False


def estimate_morphology_region(start: int, end: int) -> str:
    """Estimate the ECG morphological region for a timestep window.

    Maps the window midpoint into one cardiac cycle using modular arithmetic,
    then matches it against standard ECG interval boundaries at SAMPLING_RATE Hz.
    This handles multi-cycle recordings (e.g. 10-second signals at 100 Hz).

    Args:
        start: inclusive start timestep.
        end: exclusive end timestep.

    Returns:
        One of: "P-wave", "QRS", "ST-segment", "T-wave", "U-wave/baseline".
    """
    mid_in_cycle = ((start + end) // 2) % _CARDIAC_CYCLE_SAMPLES

    for name, lo, hi in _MORPHOLOGY_REGIONS:
        if lo <= mid_in_cycle < hi:
            return name

    return "U-wave/baseline"


def _merge_mask_to_windows(mask: np.ndarray, max_gap: int) -> List[tuple]:
    """Convert a boolean mask to (start, end) windows, bridging small gaps.

    Args:
        mask: 1D boolean array of shape (T,).
        max_gap: gaps of this many or fewer False samples between two True
            regions are filled, merging the regions into one window.

    Returns:
        List of (start, end) tuples where end is exclusive.
    """
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return []

    windows = []
    seg_start = int(indices[0])
    seg_end = int(indices[0])

    for idx in indices[1:]:
        idx = int(idx)
        if idx - seg_end <= max_gap:
            seg_end = idx
        else:
            windows.append((seg_start, seg_end + 1))
            seg_start = idx
            seg_end = idx

    windows.append((seg_start, seg_end + 1))
    return windows


def extract_evidence(
    delta_ig: np.ndarray,
    lead_names: List[str] = LEAD_NAMES,
    attention_upsampled: Optional[np.ndarray] = None,
    threshold_percentile: float = 85.0,
    max_gap: int = 20,
    attention_threshold: float = 0.5,
) -> List[EvidenceItem]:
    """Extract structured evidence items from a contrastive IG map.

    For each lead, thresholds the absolute ΔIG to keep the top
    (100 - threshold_percentile)% of timesteps, groups contiguous
    above-threshold regions into windows (bridging small gaps), and
    annotates each window with polarity, magnitude, and morphological region.

    Args:
        delta_ig: numpy array of shape (12, T) — contrastive IG per lead.
        lead_names: list of 12 lead name strings.
        attention_upsampled: optional 1D array of shape (T,) containing
            attention importance scores already upsampled to match T.
            When provided, sets EvidenceItem.attention_agreement.
        threshold_percentile: percentile of |ΔIG| used as the significance
            threshold per lead. Default 85 → top 15% of timesteps.
        max_gap: maximum gap in timesteps between significant regions to
            merge into a single window. Prevents fragmentation from noise.
        attention_threshold: mean attention value in a window above which
            attention_agreement is set to True.

    Returns:
        List of EvidenceItem objects sorted by magnitude, highest first.
    """
    num_leads, T = delta_ig.shape
    evidence: List[EvidenceItem] = []

    for lead_idx in range(num_leads):
        lead_delta = delta_ig[lead_idx]
        abs_delta = np.abs(lead_delta)

        threshold = float(np.percentile(abs_delta, threshold_percentile))
        if threshold < 1e-10:
            continue

        mask = abs_delta > threshold
        windows = _merge_mask_to_windows(mask, max_gap)

        for start, end in windows:
            window_delta = lead_delta[start:end]
            magnitude = float(np.abs(window_delta).mean())
            polarity = "positive" if float(window_delta.mean()) >= 0 else "negative"
            region = estimate_morphology_region(start, end)

            attn_agree = False
            if attention_upsampled is not None:
                attn_agree = bool(attention_upsampled[start:end].mean() > attention_threshold)

            evidence.append(EvidenceItem(
                lead_index=lead_idx,
                lead_name=lead_names[lead_idx],
                start_timestep=start,
                end_timestep=end,
                polarity=polarity,
                magnitude=magnitude,
                morphology_region=region,
                attention_agreement=attn_agree,
            ))

    evidence.sort(key=lambda e: e.magnitude, reverse=True)
    return evidence


def _magnitude_label(magnitude: float, all_magnitudes: List[float]) -> str:
    """Classify a magnitude value as 'strong', 'moderate', or 'weak'."""
    if not all_magnitudes:
        return "moderate"
    p33 = float(np.percentile(all_magnitudes, 33))
    p66 = float(np.percentile(all_magnitudes, 66))
    if magnitude >= p66:
        return "strong"
    if magnitude >= p33:
        return "moderate"
    return "weak"


def evidence_to_query(
    evidence_list: List[EvidenceItem],
    pred_class_name: str,
    runner_up_class_name: str,
) -> str:
    """Convert evidence items into a natural-language retrieval query string.

    Groups evidence by (morphology_region, polarity), then emits one sentence
    per group describing the affected leads, magnitude label, and class
    association. Designed to be used as a semantic search query in
    RAG Layers 2–3.

    Args:
        evidence_list: list of EvidenceItem objects from extract_evidence().
        pred_class_name: name of the predicted class (e.g. "MI").
        runner_up_class_name: name of the runner-up class (e.g. "STTC").

    Returns:
        Natural-language string, or empty string if evidence_list is empty.

    Example output:
        "ST elevation in leads II, aVF with strong positive contribution
        toward MI over STTC. T-wave inversion in V2 with moderate negative
        contribution favoring STTC."
    """
    if not evidence_list:
        return ""

    all_magnitudes = [e.magnitude for e in evidence_list]

    # Group by (morphology_region, polarity)
    groups: dict = {}
    for item in evidence_list:
        key = (item.morphology_region, item.polarity)
        groups.setdefault(key, []).append(item)

    # Order groups by descending mean magnitude
    sorted_groups = sorted(
        groups.items(),
        key=lambda kv: float(np.mean([e.magnitude for e in kv[1]])),
        reverse=True,
    )

    sentences = []
    for (region, polarity), items in sorted_groups:
        # Collect unique lead names ordered by magnitude (highest first)
        seen: set = set()
        leads = []
        for item in sorted(items, key=lambda e: e.magnitude, reverse=True):
            if item.lead_name not in seen:
                leads.append(item.lead_name)
                seen.add(item.lead_name)

        group_mean_mag = float(np.mean([e.magnitude for e in items]))
        mag_label = _magnitude_label(group_mean_mag, all_magnitudes)
        descriptor = get_clinical_descriptor(region, polarity)
        leads_str = ", ".join(leads)

        if polarity == "positive":
            direction = f"positive contribution toward {pred_class_name} over {runner_up_class_name}"
        else:
            direction = f"negative contribution favoring {runner_up_class_name}"

        sentences.append(
            f"{descriptor} in leads {leads_str} with {mag_label} {direction}"
        )

    return ". ".join(sentences) + "."
