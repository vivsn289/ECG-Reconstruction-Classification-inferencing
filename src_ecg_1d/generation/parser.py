# src_ecg_1d/generation/parser.py
#
# Layer 4: parse tagged LLM output into structured claims.
#
# The LLM returns text with inline tags like [E1], [T2, C1]. Layer 5 needs
# a structured representation to verify each claim's citations independently
# of the surrounding prose.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src_ecg_1d.generation.prompt_builder import TaggedEvidence

_TAG_GROUP_PATTERN = re.compile(r"\[([ETC]\d+(?:\s*,\s*[ETC]\d+)*)\]")
_UNSUPPORTED_PATTERN = re.compile(r"\[UNSUPPORTED\]")
_SENTENCE_PATTERN = re.compile(r"[^.!?\n]+[.!?]*")

# Simple keyword list used to flag sentences with clinical content that
# lack an evidence tag. Not exhaustive — a rough quality signal, not a
# guarantee (see the module docstring in run_generation.py's test plan).
_CLINICAL_KEYWORDS = [
    "elevation", "depression", "inversion", "qrs", "st segment", "st-segment",
    "p-wave", "t-wave", "u-wave", "lead", "leads", "morphology", "arrhythmia",
    "hypertrophy", "conduction", "infarction", "ischemia", "ischemic", "axis",
    "bundle branch", "block", "tachycardia", "bradycardia", "wave", "segment",
]


@dataclass
class ParsedClaim:
    """A single claim extracted from the LLM's response."""
    claim_text: str
    tag_ids: List[str] = field(default_factory=list)
    is_unsupported: bool = False
    position: Tuple[int, int] = (0, 0)


@dataclass
class ParsedResponse:
    """The full parsed LLM response."""
    raw_text: str
    claims: List[ParsedClaim]
    untagged_sentences: List[str]


def _iter_lines_with_offset(text: str):
    offset = 0
    for line in text.split("\n"):
        yield line, offset
        offset += len(line) + 1  # +1 for the split newline


def _split_sentences(text: str) -> List[Tuple[str, int, int]]:
    """Split text into (sentence, start_offset, end_offset), preserving
    character offsets into the original raw text.

    Splits on paragraph/line boundaries first (numbered list items often
    have no terminating punctuation), then on sentence-ending punctuation
    within each line.
    """
    sentences = []
    for line, line_offset in _iter_lines_with_offset(text):
        if not line.strip():
            continue
        for match in _SENTENCE_PATTERN.finditer(line):
            raw = match.group()
            stripped = raw.strip()
            if not stripped:
                continue
            # Skip pure numbering/punctuation fragments (e.g. "1.", "2)")
            # left behind when a numbered list item's digit gets split from
            # its text by the sentence-boundary regex — these aren't claims.
            if not any(ch.isalpha() for ch in stripped):
                continue
            lead_ws = len(raw) - len(raw.lstrip())
            start = line_offset + match.start() + lead_ws
            end = start + len(stripped)
            sentences.append((stripped, start, end))
    return sentences


def _extract_tag_ids(sentence: str) -> List[str]:
    tag_ids: List[str] = []
    for group in _TAG_GROUP_PATTERN.findall(sentence):
        for tag_id in group.split(","):
            tag_id = tag_id.strip()
            if tag_id not in tag_ids:
                tag_ids.append(tag_id)
    return tag_ids


def _has_clinical_content(sentence: str) -> bool:
    lower = sentence.lower()
    return any(keyword in lower for keyword in _CLINICAL_KEYWORDS)


def parse_llm_response(
    response_text: str,
    tag_registry: Dict[str, TaggedEvidence],
) -> ParsedResponse:
    """Extract claims and their tags from the LLM's response.

    Args:
        response_text: raw text returned by the LLM.
        tag_registry: the tag registry from StructuredPrompt, used only to
            validate that cited tags actually exist (unknown tags are kept
            in the claim as-is — Layer 5 decides how to treat them).

    Returns:
        ParsedResponse with one ParsedClaim per sentence, plus a list of
        sentences that read as clinical claims but cite no evidence tag.
    """
    claims: List[ParsedClaim] = []
    untagged_sentences: List[str] = []

    for sentence, start, end in _split_sentences(response_text):
        tag_ids = _extract_tag_ids(sentence)
        is_unsupported = bool(_UNSUPPORTED_PATTERN.search(sentence))

        claims.append(ParsedClaim(
            claim_text=sentence,
            tag_ids=tag_ids,
            is_unsupported=is_unsupported,
            position=(start, end),
        ))

        if not tag_ids and not is_unsupported and _has_clinical_content(sentence):
            untagged_sentences.append(sentence)

    return ParsedResponse(
        raw_text=response_text,
        claims=claims,
        untagged_sentences=untagged_sentences,
    )
