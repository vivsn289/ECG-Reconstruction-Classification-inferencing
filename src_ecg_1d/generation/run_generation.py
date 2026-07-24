# src_ecg_1d/generation/run_generation.py
#
# Layer 4 end-to-end test: Prompt Construction + LLM Generation.
#
# For each record in EXPLAIN_RECORD_INDICES:
#   1. Runs Layer 1 (contrastive IG + evidence extraction)
#   2. Runs Layer 2-3 (retrieval)
#   3. Runs Layer 4 (prompt build -> LLM call -> parse)
#   4. Prints the structured prompt, raw LLM response, and parsed claims
#   5. Saves the full trace to results_1d/generation_record_<idx>.json
#
# Usage (from repository root):
#   python -m src_ecg_1d.generation.run_generation
#
# Requires ANTHROPIC_API_KEY (or OPENAI_API_KEY, with LLM_PROVIDER="openai"
# in configs/config_1d.py) set in the environment.

import dataclasses
import json
import os

import numpy as np
import torch

from configs.config_1d import (
    CHECKPOINT_DIR,
    DATA_ROOT,
    EXPLAIN_RECORD_INDICES,
    LABEL_DECODER,
    LEAD_NAMES,
    LLM_PROVIDER,
    NUM_CLASSES,
    RESULTS_DIR,
    SAMPLING_RATE,
)
from src_ecg_1d.contrastive.delta_ig import compute_record_contrastive_ig
from src_ecg_1d.contrastive.evidence import evidence_to_query, extract_evidence
from src_ecg_1d.data.loaders import PTBXLECGLoader
from src_ecg_1d.generation.generator import ExplanationGenerator
from src_ecg_1d.generation.llm_client import build_llm_client
from src_ecg_1d.models.ecg_model import ECGClassifier1D
from src_ecg_1d.retrieval.retriever import Retriever

CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def explain_and_generate(model, loader, records, retriever, generator, record_idx: int) -> None:
    """Run the full Layer 1-4 pipeline for one record."""
    record = records[record_idx]
    ecg = loader.load_ecg(record["record_path"])  # (12, T)
    true_labels = sorted(record["labels"])

    print(f"\n{'=' * 70}")
    print(f"[INFO] Record {record_idx} | True labels: {true_labels} | Shape: {ecg.shape}")

    # ── Layer 1: Contrastive IG + evidence ──────────────────────────────────
    print("[LAYER 1] Computing contrastive IG + evidence ...")
    delta_ig_full, attention_full, window_metadata = compute_record_contrastive_ig(
        model, ecg, DEVICE, include_attention=True
    )

    mean_probs = np.mean([m["probs"] for m in window_metadata], axis=0)
    pred_probs = [float(p) for p in mean_probs]
    sorted_idx = np.argsort(mean_probs)[::-1]
    pred_class = LABEL_DECODER[int(sorted_idx[0])]
    runner_up_class = LABEL_DECODER[int(sorted_idx[1])]

    evidence = extract_evidence(delta_ig_full, lead_names=LEAD_NAMES, attention_upsampled=attention_full)
    print(f"  Predicted: {pred_class}  |  Runner-up: {runner_up_class}  |  {len(evidence)} evidence items")

    # ── Layer 2-3: Retrieval ─────────────────────────────────────────────────
    print("[LAYER 2-3] Retrieving textbook passages + similar cases ...")
    query = evidence_to_query(evidence, pred_class, runner_up_class)
    retrieval_result = retriever.retrieve(query)
    print(f"  Query: {query}")
    print(
        f"  Retrieved {len(retrieval_result.textbook_results)} textbook passages, "
        f"{len(retrieval_result.case_results)} similar cases"
    )

    # ── Layer 4: Prompt build -> LLM call -> parse ──────────────────────────
    print("[LAYER 4] Building prompt and calling the LLM ...")
    result = generator.generate(
        pred_class=pred_class,
        runner_up_class=runner_up_class,
        pred_probs=pred_probs,
        evidence_items=evidence,
        retrieval_result=retrieval_result,
    )

    print("\n[STRUCTURED PROMPT — SYSTEM]")
    print(result.structured_prompt.system_prompt)
    print("\n[STRUCTURED PROMPT — USER]")
    print(result.structured_prompt.user_prompt)
    print(f"\n[TAG REGISTRY] {len(result.structured_prompt.tag_registry)} tags: "
          f"{list(result.structured_prompt.tag_registry.keys())}")

    print("\n[RAW LLM RESPONSE]")
    print(result.raw_response)

    print("\n[PARSED CLAIMS]")
    for claim in result.parsed_response.claims:
        tag_str = ", ".join(claim.tag_ids) if claim.tag_ids else ("UNSUPPORTED" if claim.is_unsupported else "-")
        print(f"  [{tag_str:>20}] {claim.claim_text}")

    if result.parsed_response.untagged_sentences:
        print(f"\n[WARNING] {len(result.parsed_response.untagged_sentences)} clinical claim(s) with no tag:")
        for sentence in result.parsed_response.untagged_sentences:
            print(f"  - {sentence}")

    # ── Save trace to disk ──────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"generation_record_{record_idx}.json")
    with open(out_path, "w") as f:
        json.dump(dataclasses.asdict(result), f, indent=2, default=str)
    print(f"\n[SAVED] {out_path}")


def main() -> None:
    print(f"[INFO] Device: {DEVICE}")

    model = ECGClassifier1D(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    loader = PTBXLECGLoader(DATA_ROOT, sampling_rate=SAMPLING_RATE)
    records = loader.get_records()

    retriever = Retriever()
    llm_client = build_llm_client(provider=LLM_PROVIDER)
    generator = ExplanationGenerator(llm_client=llm_client)

    for record_idx in EXPLAIN_RECORD_INDICES:
        explain_and_generate(model, loader, records, retriever, generator, record_idx)


if __name__ == "__main__":
    main()
