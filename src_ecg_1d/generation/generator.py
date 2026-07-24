# src_ecg_1d/generation/generator.py
#
# Layer 4: end-to-end orchestration.
#
# Ties prompt construction, the LLM call, and response parsing into one
# entry point that Layer 1-3 outputs feed into directly.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from configs.config_1d import LLM_MAX_TOKENS, LLM_TEMPERATURE
from src_ecg_1d.contrastive.evidence import EvidenceItem
from src_ecg_1d.generation.llm_client import LLMClient
from src_ecg_1d.generation.parser import ParsedResponse, parse_llm_response
from src_ecg_1d.generation.prompt_builder import PromptBuilder, StructuredPrompt
from src_ecg_1d.retrieval.retriever import RetrievalResult


@dataclass
class GenerationResult:
    """Complete output of Layer 4."""
    structured_prompt: StructuredPrompt   # what was sent to the LLM (includes tag_registry)
    raw_response: str                     # what the LLM returned
    parsed_response: ParsedResponse       # structured claim list


class ExplanationGenerator:
    """Orchestrates prompt construction → LLM call → response parsing."""

    def __init__(self, llm_client: LLMClient, prompt_builder: Optional[PromptBuilder] = None):
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder or PromptBuilder()

    def generate(
        self,
        pred_class: str,
        runner_up_class: str,
        pred_probs: List[float],
        evidence_items: List[EvidenceItem],
        retrieval_result: RetrievalResult,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
    ) -> GenerationResult:
        """Full pipeline: build prompt, call LLM, parse response."""
        structured_prompt = self.prompt_builder.build(
            pred_class=pred_class,
            runner_up_class=runner_up_class,
            pred_probs=pred_probs,
            evidence_items=evidence_items,
            retrieval_result=retrieval_result,
        )

        raw_response = self.llm_client.generate(
            system_prompt=structured_prompt.system_prompt,
            user_prompt=structured_prompt.user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        parsed_response = parse_llm_response(raw_response, structured_prompt.tag_registry)

        return GenerationResult(
            structured_prompt=structured_prompt,
            raw_response=raw_response,
            parsed_response=parsed_response,
        )
