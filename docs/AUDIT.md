# Repository Audit — ECG Research Framework

**Date:** 2026-05-17
**Status:** Completed — see REFACTOR_NOTES.md for applied changes

---

## Issues Found & Resolved

### Critical Bugs

| ID  | File                                     | Issue                                         | Status    |
|-----|------------------------------------------|-----------------------------------------------|-----------|
| B1  | `src_ecg_1d/train_1d.py`               | Val dataset received training augmentations    | **Fixed** |
| B2  | `scripts/generate_lead_attribution.py`  | Same IG score assigned to all 12 leads         | **Fixed** |

**B1 detail:** `ECGWindowDataset` was created once with `train_transform`, then split
via `random_split`. Both subsets shared the same underlying dataset object, so val
samples were augmented. Fix: split at the record level before creating datasets,
create separate train/val datasets with different transforms.

**B2 detail:** `ig_score = np.abs(ig).mean()` computed a scalar over all channels
and time, then added it equally to all 12 `lead_scores[lead]`. This produces a
flat bar chart. Fix: `per_lead = np.abs(ig).mean(axis=1)` computes mean absolute
IG per lead correctly.

---

### Duplicate Code

| ID  | Files                                                  | Issue                         | Resolution             |
|-----|--------------------------------------------------------|-------------------------------|------------------------|
| D1  | `explainability/attention_maps.py` + `attention_utils.py` | Two versions of `aggregate_attention()` | `attention_maps.py` → deprecated stub forwarding to `attention_utils.py` |
| D2  | `explainability/integrated_gradients.py` + `run_record_explainability.py` | Two IG implementations | Module version improved; inline version kept in script |

---

### Configuration Problems

| ID  | Issue                                                  | Resolution             |
|-----|--------------------------------------------------------|------------------------|
| C1  | All hyperparameters hardcoded across 4+ files         | Created `configs/config_1d.py` |
| C2  | Label ordering inconsistency between `src/` and `src_ecg_1d/` pipelines | Canonical `LABEL_ENCODER` in `configs/config_1d.py` |
| C3  | `src/config.py` prints on import                      | Print removed          |
| C4  | `src/config_pytorch.py` prints on import              | Print removed          |

---

### Missing Infrastructure

| ID  | Issue                                  | Resolution                   |
|-----|----------------------------------------|------------------------------|
| I1  | No `README.md`                        | Created `README.md`          |
| I2  | `requirements.txt` missing PyTorch deps | Updated to include both pipelines |
| I3  | `.gitignore` too minimal               | Updated with standard Python patterns |
| I4  | No `__init__.py` in `training/`       | Added                        |
| I5  | No `__init__.py` in `explainability/` | Added                        |
| I6  | No `__init__.py` in `scripts/`        | Added                        |
| I7  | No `__init__.py` in `configs/`        | Added                        |

---

### Code Organization

| ID  | Issue                                       | Resolution                                       |
|-----|---------------------------------------------|--------------------------------------------------|
| O1  | `fix_train_val_split.py` at root           | Copied to `scripts/`; root file is a redirect stub |
| O2  | `training_metrics.csv` output at root      | Training now saves to `results_1d/`              |
| O3  | `src_ecg_1d.zip` at root                  | Added to `.gitignore`                            |

---

## Preserved (Intentionally Not Changed)

- **`src/` pipeline** — complete legacy experiment; kept intact for reference
- **All model architectures** — CNN, ECA, Transformer weights/structure unchanged
- **`dataset_maker.py`** — uses dominant SCP code selection (different from `loaders.py`
  which uses first-match); both strategies are reasonable; not consolidated
- **`results/` and `results_pytorch/`** — historical outputs; kept as-is
- **`src/ecg_to_image.py`** — 705-line image generation utility; not modified

---

## Remaining Technical Debt (Future Work)

| ID  | Issue                                            | Priority  |
|-----|--------------------------------------------------|-----------|
| T1  | No experiment tracking (W&B, MLflow, or CSV+git) | High      |
| T2  | Checkpoint saves only `state_dict`; no config snapshot | Medium |
| T3  | `ECGWindowDataset` loads all windows into RAM    | Low       |
| T4  | No test suite for data pipeline and metrics      | Medium    |
| T5  | `run_record_explainability.py` uses `plt.show()` — not scriptable | Low |
| T6  | `dataset_maker.py` (src/) uses dominant SCP code; `loaders.py` uses first-match — inconsistency between pipelines | Low |
| T7  | `trainer_1d.py` uses deprecated `torch.cuda.amp.GradScaler` API | Low |
| T8  | No `strat_fold`-based splitting in 1D pipeline (uses random record split) | Medium |
