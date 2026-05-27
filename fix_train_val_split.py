# This file has been moved to scripts/fix_train_val_split.py.
# It is kept here temporarily to avoid breaking any references.
# Please import from scripts/fix_train_val_split.py going forward.

# This is a one-off utility that splits pre-generated ECG images (PNG format)
# from the legacy image pipeline (src/) into train/val folders.
# It is NOT needed for the 1D pipeline (src_ecg_1d/), which does not use images.

import scripts.fix_train_val_split  # noqa: F401
