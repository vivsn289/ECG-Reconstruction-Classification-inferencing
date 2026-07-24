# src_ecg_1d/verification/run_verification.py
#
# Full pipeline end-to-end test: Layers 1 -> 2/3 -> 4 -> 5.
#
# For each record in EXPLAIN_RECORD_INDICES:
#   1. Layer 1: contrastive IG + evidence extraction
#   2. Layer 2-3: retrieval
#   3. Layer 4: prompt build -> LLM call -> parse
#   4. Layer 5: verify every parsed claim against the evidence
#   5. Print the verification report summary
#   6. Save the full trace to results_1d/verification/pipeline_record_<idx>.json
#
# Usage (from repository root):
#   python -m src_ecg_1d.verification.run_verification
#
# Requires ANTHROPIC_API_KEY (or OPENAI_API_KEY, with LLM_PROVIDER="openai"
# in configs/config_1d.py) set in the environment.

import dataclasses
import json
import os
from enum import Enum

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
    SAMPLING_RATE,
    VERIFICATION_OUTPUT_DIR,
)
from src_ecg_1d.contrastive.delta_ig import compute_record_contrastive_ig
from src_ecg_1d.contrastive.evidence import evidence_to_query, extract_evidence
from src_ecg_1d.data.loaders import PTBXLECGLoader
from src_ecg_1d.generation.generator import ExplanationGenerator
from src_ecg_1d.generation.llm_client import build_llm_client
from src_ecg_1d.models.ecg_model import ECGClassifier1D
from src_ecg_1d.retrieval.retriever import Retriever
from src_ecg_1d.verification.verifier import Verifier

CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _json_default(obj):
    """json.dump fallback: unwrap Enums to their value, dataclass-ify the rest."""
    if isinstance(obj, Enum):
        return obj.value
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return str(obj)


def run_pipeline(model, loader, records, retriever, generator, verifier, record_idx: int) -> None:
    """Run the full Layer 1-5 pipeline for one record."""
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
    print(
        f"  Retrieved {len(retrieval_result.textbook_results)} textbook passages, "
        f"{len(retrieval_result.case_results)} similar cases"
    )

    # ── Layer 4: Prompt build -> LLM call -> parse ──────────────────────────
    print("[LAYER 4] Building prompt and calling the LLM ...")
    generation_result = generator.generate(
        pred_class=pred_class,
        runner_up_class=runner_up_class,
        pred_probs=pred_probs,
        evidence_items=evidence,
        retrieval_result=retrieval_result,
    )
    print(f"  {len(generation_result.parsed_response.claims)} claims parsed from the LLM response")

    # ── Layer 5: Verification ───────────────────────────────────────────────
    print("[LAYER 5] Verifying claims against evidence ...")
    report = verifier.verify(generation_result, delta_ig=delta_ig_full, lead_names=LEAD_NAMES)

    print("\n[VERIFICATION REPORT]")
    print(report.summary())

    # ── Save full pipeline trace to disk ────────────────────────────────────
    os.makedirs(VERIFICATION_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(VERIFICATION_OUTPUT_DIR, f"pipeline_record_{record_idx}.json")

    trace = {
        "record_idx": record_idx,
        "true_labels": true_labels,
        "prediction": {
            "pred_class": pred_class,
            "runner_up_class": runner_up_class,
            "pred_probs": pred_probs,
        },
        "evidence_items": [dataclasses.asdict(e) for e in evidence],
        "retrieval": {
            "query": query,
            "textbook_results": [dataclasses.asdict(r) for r in retrieval_result.textbook_results],
            "case_results": [dataclasses.asdict(r) for r in retrieval_result.case_results],
        },
        "structured_prompt": {
            "system_prompt": generation_result.structured_prompt.system_prompt,
            "user_prompt": generation_result.structured_prompt.user_prompt,
            "tag_registry": {
                tag_id: dataclasses.asdict(tagged)
                for tag_id, tagged in generation_result.structured_prompt.tag_registry.items()
            },
        },
        "raw_response": generation_result.raw_response,
        "parsed_claims": [dataclasses.asdict(c) for c in generation_result.parsed_response.claims],
        "untagged_sentences": generation_result.parsed_response.untagged_sentences,
        "verification": {
            "claim_verifications": [
                {
                    "claim": dataclasses.asdict(cv.claim),
                    "overall_status": cv.overall_status.value,
                    "check_results": [dataclasses.asdict(r) for r in cv.check_results],
                }
                for cv in report.claim_verifications
            ],
            "aggregate_metrics": report.aggregate_metrics,
        },
    }

    with open(out_path, "w") as f:
        json.dump(trace, f, indent=2, default=_json_default)
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
    # Reuse the retriever's already-loaded embedder instead of loading a
    # second copy of the sentence-transformers model for Check 3.
    verifier = Verifier(embedder=retriever.embedder)

    for record_idx in EXPLAIN_RECORD_INDICES:
        run_pipeline(model, loader, records, retriever, generator, verifier, record_idx)


if __name__ == "__main__":
    main()
