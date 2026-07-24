# src_ecg_1d/retrieval/build_textbook_index.py
#
# One-time build script for Index A (textbook passages).
#
# Usage (from repository root):
#   python -m src_ecg_1d.retrieval.build_textbook_index

import os

from configs.config_1d import INDEX_DIR, TEXTBOOK_PASSAGES_PATH
from src_ecg_1d.retrieval.embedder import Embedder
from src_ecg_1d.retrieval.textbook_index import TextbookIndex

OUTPUT_DIR = os.path.join(INDEX_DIR, "textbook")


def main() -> None:
    print(f"[INFO] Building textbook index from {TEXTBOOK_PASSAGES_PATH}")

    embedder = Embedder()
    index = TextbookIndex(embedder)
    index.build(TEXTBOOK_PASSAGES_PATH)

    index.save(OUTPUT_DIR)

    print(f"[DONE] Indexed {len(index.passages)} passages")
    print(f"       Index size: {index.index.ntotal} vectors, dim {index.index.d}")
    print(f"       Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
