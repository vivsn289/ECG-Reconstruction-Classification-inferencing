# src_ecg_1d/retrieval/build_case_index.py
#
# One-time (GPU-heavy) build script for Index B (case-based training index).
#
# Iterates over all training records (strat_fold 1-8), runs contrastive IG
# on each, extracts evidence, converts it to a natural-language summary, and
# builds a FAISS index over the summaries. Checkpoints progress to disk so
# a crashed or interrupted run can be resumed with --resume.
#
# Usage (from repository root):
#   python -m src_ecg_1d.retrieval.build_case_index
#   python -m src_ecg_1d.retrieval.build_case_index --resume
#   python -m src_ecg_1d.retrieval.build_case_index --limit 50   # smoke test

import argparse
import json
import os

import torch
from tqdm import tqdm

from configs.config_1d import (
    CASE_INDEX_CHECKPOINT_INTERVAL,
    CHECKPOINT_DIR,
    DATA_ROOT,
    INDEX_DIR,
    LABEL_DECODER,
    LEAD_NAMES,
    NUM_CLASSES,
    SAMPLING_RATE,
)
from src_ecg_1d.contrastive.delta_ig import compute_record_contrastive_ig
from src_ecg_1d.contrastive.evidence import evidence_to_query, extract_evidence
from src_ecg_1d.data.loaders import PTBXLECGLoader
from src_ecg_1d.models.ecg_model import ECGClassifier1D
from src_ecg_1d.retrieval.case_index import CaseEntry, CaseIndex
from src_ecg_1d.retrieval.embedder import Embedder

CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = os.path.join(INDEX_DIR, "case")
BUILD_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "build_checkpoint.json")

# strat_fold protocol: folds 1-8 train, 9 val, 10 test.
TRAIN_FOLDS = set(range(1, 9))


def load_checkpoint(path: str):
    """Load a build checkpoint, returning (last_processed_index, entries)."""
    with open(path, "r") as f:
        data = json.load(f)
    entries = [CaseEntry(**e) for e in data["entries"]]
    return data["last_processed_index"], entries


def save_checkpoint(path: str, last_processed_index: int, entries) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "last_processed_index": last_processed_index,
        "entries": [
            {
                "record_id": e.record_id,
                "evidence_summary": e.evidence_summary,
                "true_labels": e.true_labels,
                "pred_class": e.pred_class,
                "runner_up_class": e.runner_up_class,
                "pred_probs": e.pred_probs,
            }
            for e in entries
        ],
    }
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    os.replace(tmp_path, path)


def process_record(model, loader, record) -> CaseEntry:
    """Run the ΔIG -> evidence -> query pipeline for one training record."""
    ecg = loader.load_ecg(record["record_path"])  # (12, T)

    delta_ig_full, attention_full, window_metadata = compute_record_contrastive_ig(
        model, ecg, DEVICE, include_attention=True
    )

    mean_probs = torch.tensor([m["probs"] for m in window_metadata]).mean(dim=0).numpy()
    sorted_idx = mean_probs.argsort()[::-1]
    pred_class = int(sorted_idx[0])
    runner_up_class = int(sorted_idx[1])
    pred_name = LABEL_DECODER[pred_class]
    runner_up_name = LABEL_DECODER[runner_up_class]

    evidence = extract_evidence(
        delta_ig_full,
        lead_names=LEAD_NAMES,
        attention_upsampled=attention_full,
    )
    query = evidence_to_query(evidence, pred_name, runner_up_name)

    return CaseEntry(
        record_id=int(record["ecg_id"]),
        evidence_summary=query,
        true_labels=sorted(record["labels"]),
        pred_class=pred_name,
        runner_up_class=runner_up_name,
        pred_probs=[float(p) for p in mean_probs],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                         help="Resume from build_checkpoint.json if present.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Process only the first N training records (for smoke testing).")
    parser.add_argument("--checkpoint-interval", type=int, default=CASE_INDEX_CHECKPOINT_INTERVAL,
                         help="Checkpoint to disk every N processed records.")
    args = parser.parse_args()

    print(f"[INFO] Device: {DEVICE}")

    model = ECGClassifier1D(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    loader = PTBXLECGLoader(DATA_ROOT, sampling_rate=SAMPLING_RATE)
    all_records = loader.get_records()
    train_records = [r for r in all_records if r["strat_fold"] in TRAIN_FOLDS]

    if args.limit is not None:
        train_records = train_records[: args.limit]

    print(f"[INFO] {len(train_records)} training records to process "
          f"(strat_fold in {sorted(TRAIN_FOLDS)})")

    entries = []
    start_index = 0

    if args.resume and os.path.exists(BUILD_CHECKPOINT_PATH):
        start_index_prev, entries = load_checkpoint(BUILD_CHECKPOINT_PATH)
        start_index = start_index_prev + 1
        print(f"[INFO] Resuming from index {start_index} "
              f"({len(entries)} entries already collected)")

    pbar = tqdm(
        range(start_index, len(train_records)),
        initial=start_index,
        total=len(train_records),
        desc="Building case index",
    )

    for i in pbar:
        record = train_records[i]
        try:
            entry = process_record(model, loader, record)
            entries.append(entry)
        except Exception as exc:
            print(f"\n[WARN] Skipping record ecg_id={record.get('ecg_id')} "
                  f"(index {i}): {exc}")

        if (i + 1) % args.checkpoint_interval == 0:
            save_checkpoint(BUILD_CHECKPOINT_PATH, i, entries)
            pbar.set_postfix(checkpointed=i + 1)

    save_checkpoint(BUILD_CHECKPOINT_PATH, len(train_records) - 1, entries)
    print(f"[INFO] Collected {len(entries)} case entries "
          f"({len(train_records) - len(entries)} skipped)")

    print("[INFO] Building FAISS case index ...")
    embedder = Embedder()
    case_index = CaseIndex(embedder)
    case_index.build(entries)
    case_index.save(OUTPUT_DIR)

    print(f"[DONE] Indexed {len(case_index.entries)} cases")
    print(f"       Index size: {case_index.index.ntotal} vectors, dim {case_index.index.d}")
    print(f"       Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
