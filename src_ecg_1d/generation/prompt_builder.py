# src_ecg_1d/generation/prompt_builder.py
#
# Layer 4: structured prompt construction.
#
# Builds the 4-block prompt sent to the LLM from Layer 1 (contrastive
# evidence) and Layer 2-3 (retrieval) outputs. Every evidence item and
# retrieved passage/case gets a unique tag ID (E1, T1, C1, ...) so the LLM
# can cite it inline and Layer 5 can mechanically verify each claim.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from configs.config_1d import CLASS_NAMES, LLM_MAX_EVIDENCE_ITEMS, LLM_SYSTEM_PROMPT_PATH
from src_ecg_1d.contrastive.evidence import EvidenceItem, get_clinical_descriptor
from src_ecg_1d.retrieval.retriever import RetrievalResult

# Full clinical names for the PTB-XL diagnostic superclasses, used in Block 1.
_CLASS_FULL_NAMES = {
    "NORM": "Normal ECG (NORM)",
    "MI":   "Myocardial Infarction (MI)",
    "STTC": "ST/T Change (STTC)",
    "CD":   "Conduction Disturbance (CD)",
    "HYP":  "Hypertrophy (HYP)",
}


@dataclass
class TaggedEvidence:
    """An evidence item with a unique tag ID for the LLM to cite."""
    tag_id: str              # "E1", "T1", "C1", etc.
    source_type: str         # "evidence", "textbook", "case"
    content: str             # human-readable text
    metadata: dict = field(default_factory=dict)  # original source object


@dataclass
class StructuredPrompt:
    """The final prompt sent to the LLM, plus the tag registry for verification."""
    system_prompt: str
    user_prompt: str
    tag_registry: Dict[str, TaggedEvidence]


def _magnitude_label(magnitude: float, all_magnitudes: List[float]) -> str:
    """Classify a magnitude value as 'strong', 'moderate', or 'weak' relative
    to the other evidence items shown in this prompt."""
    if not all_magnitudes:
        return "moderate"
    p33 = float(np.percentile(all_magnitudes, 33))
    p66 = float(np.percentile(all_magnitudes, 66))
    if magnitude >= p66:
        return "strong"
    if magnitude >= p33:
        return "moderate"
    return "weak"


class PromptBuilder:
    """Builds structured 4-block prompts from Layer 1-3 outputs.

    Block 1: Classification result (prediction + runner-up + probabilities)
    Block 2: Contrastive evidence items from ΔIG (tagged E1, E2, ...)
    Block 3: Retrieved textbook (tagged T1, T2, ...) and cases (tagged C1, C2, ...)
    Block 4: Instructions and constraints
    """

    def __init__(self, system_prompt_path: str = LLM_SYSTEM_PROMPT_PATH):
        with open(system_prompt_path, "r") as f:
            self.system_prompt = f.read().strip()

    def build(
        self,
        pred_class: str,
        runner_up_class: str,
        pred_probs: List[float],
        evidence_items: List[EvidenceItem],
        retrieval_result: RetrievalResult,
        max_evidence_items: int = LLM_MAX_EVIDENCE_ITEMS,
    ) -> StructuredPrompt:
        """Construct the full prompt.

        Args:
            pred_class: predicted class short name, e.g. "MI".
            runner_up_class: runner-up class short name, e.g. "STTC".
            pred_probs: sigmoid probabilities for all classes, ordered per
                CLASS_NAMES (multi-label — do not sum to 1).
            evidence_items: EvidenceItem list from extract_evidence(),
                already sorted by magnitude (highest first).
            retrieval_result: RetrievalResult from Retriever.retrieve().
            max_evidence_items: cap on how many evidence items to include
                in Block 2, to keep the prompt compact.

        Returns:
            StructuredPrompt with the system prompt, the 4-block user
            prompt, and a tag registry mapping every tag ID to its
            underlying content for Layer 5 verification.
        """
        tag_registry: Dict[str, TaggedEvidence] = {}
        shown_evidence = evidence_items[:max_evidence_items]

        block1 = self._build_block1(pred_class, runner_up_class, pred_probs)
        block2 = self._build_block2(shown_evidence, pred_class, runner_up_class, tag_registry)
        block3 = self._build_block3(retrieval_result, tag_registry)
        block4 = self._build_block4(pred_class, runner_up_class)

        user_prompt = "\n\n".join([block1, block2, block3, block4])

        return StructuredPrompt(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            tag_registry=tag_registry,
        )

    def _build_block1(self, pred_class: str, runner_up_class: str, pred_probs: List[float]) -> str:
        pred_prob = pred_probs[CLASS_NAMES.index(pred_class)]
        runner_up_prob = pred_probs[CLASS_NAMES.index(runner_up_class)]

        lines = [
            "## CLASSIFICATION RESULT",
            f"The classifier predicted: {_CLASS_FULL_NAMES.get(pred_class, pred_class)}",
            f"Probability: {pred_prob:.2f}",
            f"Runner-up class: {_CLASS_FULL_NAMES.get(runner_up_class, runner_up_class)}",
            f"Runner-up probability: {runner_up_prob:.2f}",
            "",
            "All class probabilities (sigmoid, multi-label):",
        ]
        for name, prob in zip(CLASS_NAMES, pred_probs):
            lines.append(f"  {name}: {prob:.2f}")

        return "\n".join(lines)

    def _build_block2(
        self,
        evidence_items: List[EvidenceItem],
        pred_class: str,
        runner_up_class: str,
        tag_registry: Dict[str, TaggedEvidence],
    ) -> str:
        lines = [
            "## MODEL DECISION EVIDENCE (Contrastive ΔIG)",
            "",
            f"The following signal regions drove the model's decision to prefer "
            f"{pred_class} over {runner_up_class}:",
            "",
        ]

        if not evidence_items:
            lines.append("(No significant evidence regions were found.)")
            return "\n".join(lines)

        all_magnitudes = [e.magnitude for e in evidence_items]

        for i, item in enumerate(evidence_items, start=1):
            tag_id = f"E{i}"
            mag_label = _magnitude_label(item.magnitude, all_magnitudes)
            favors = pred_class if item.polarity == "positive" else runner_up_class
            direction = (
                f"favors {pred_class} over {runner_up_class}"
                if item.polarity == "positive"
                else f"would have favored {runner_up_class}"
            )
            attn_str = "yes" if item.attention_agreement else "no"
            descriptor = get_clinical_descriptor(item.morphology_region, item.polarity)

            content = (
                f"Lead {item.lead_name}, timesteps {item.start_timestep}-{item.end_timestep} "
                f"({item.morphology_region} region)\n"
                f"     Clinical correlate: {descriptor}\n"
                f"     Polarity: {item.polarity} ({direction})\n"
                f"     Magnitude: {mag_label} ({item.magnitude:.2f})\n"
                f"     Attention agreement: {attn_str}"
            )
            lines.append(f"[{tag_id}] {content}")
            lines.append("")

            tag_registry[tag_id] = TaggedEvidence(
                tag_id=tag_id,
                source_type="evidence",
                content=content,
                metadata={"evidence_item": item, "favors": favors},
            )

        return "\n".join(lines).rstrip()

    def _build_block3(
        self,
        retrieval_result: RetrievalResult,
        tag_registry: Dict[str, TaggedEvidence],
    ) -> str:
        lines = ["## RETRIEVED CLINICAL KNOWLEDGE (Textbook)", ""]

        if not retrieval_result.textbook_results:
            lines.append("(No textbook passages retrieved.)")
        for i, result in enumerate(retrieval_result.textbook_results, start=1):
            tag_id = f"T{i}"
            passage = result.passage
            leads_str = ", ".join(passage.leads)
            content = (
                f"(similarity: {result.similarity_score:.2f}) "
                f"[{passage.condition} / {passage.morphology} / leads {leads_str}]\n"
                f"{passage.text}"
            )
            lines.append(f"[{tag_id}] {content}")
            lines.append("")

            tag_registry[tag_id] = TaggedEvidence(
                tag_id=tag_id,
                source_type="textbook",
                content=content,
                metadata={"textbook_result": result},
            )

        lines.append("## RETRIEVED SIMILAR CASES (from training set)")
        lines.append("")

        if not retrieval_result.case_results:
            lines.append("(No similar cases retrieved.)")
        for i, result in enumerate(retrieval_result.case_results, start=1):
            tag_id = f"C{i}"
            entry = result.entry
            pred_prob = entry.pred_probs[CLASS_NAMES.index(entry.pred_class)]
            runner_up_prob = entry.pred_probs[CLASS_NAMES.index(entry.runner_up_class)]
            true_labels_str = ", ".join(entry.true_labels)
            content = (
                f"(similarity: {result.similarity_score:.2f}) Record ID {entry.record_id}\n"
                f"     True labels: {true_labels_str}\n"
                f"     Model prediction: {entry.pred_class} ({pred_prob:.2f}), "
                f"runner-up {entry.runner_up_class} ({runner_up_prob:.2f})\n"
                f"     Evidence summary: \"{entry.evidence_summary}\""
            )
            lines.append(f"[{tag_id}] {content}")
            lines.append("")

            tag_registry[tag_id] = TaggedEvidence(
                tag_id=tag_id,
                source_type="case",
                content=content,
                metadata={"case_result": result},
            )

        return "\n".join(lines).rstrip()

    def _build_block4(self, pred_class: str, runner_up_class: str) -> str:
        return (
            "## YOUR TASK\n\n"
            "Write a clinical explanation of the model's prediction. You must follow these rules:\n\n"
            "1. Every clinical claim you make MUST cite at least one evidence tag.\n"
            "   Use the format [E1], [T2], [C3], or combinations like [E1, T2].\n\n"
            "2. Do NOT make claims that are not supported by the tagged evidence above.\n"
            "   If evidence is insufficient, say so explicitly.\n\n"
            f"3. Explain WHY the model chose {pred_class} over {runner_up_class} specifically — "
            "what evidence tipped the decision boundary.\n\n"
            "4. Structure your response as:\n"
            "   - Summary sentence (1-2 lines)\n"
            "   - Key findings (numbered, each tagged)\n"
            "   - Discriminating evidence (what distinguished the predicted class from the runner-up)\n"
            "   - Confidence assessment (based on evidence strength)\n\n"
            "5. If a claim has no supporting evidence, use [UNSUPPORTED] — do not fabricate.\n\n"
            "Begin your explanation now."
        )
