# src_ecg_1d/contrastive/delta_ig.py
#
# Contrastive Integrated Gradients (ΔIG) for ECG explainability.
#
# ΔIG = IG(predicted_class) − IG(runner_up_class)
#
# Positive ΔIG at a (lead, timestep) means that region pushed the model
# toward the predicted class over the runner-up.
# Negative ΔIG means that region would have supported the runner-up instead.

import numpy as np
import torch

from configs.config_1d import (
    IG_STEPS,
    WINDOW_SIZE,
    STRIDE,
    LABEL_DECODER,
)
from src_ecg_1d.explainability.integrated_gradients import integrated_gradients
from src_ecg_1d.explainability.attention_utils import aggregate_attention
from src_ecg_1d.explainability.visualization import upsample_attention
from src_ecg_1d.data.windowing import sliding_window


def compute_contrastive_ig(
    model,
    x: torch.Tensor,
    baseline: torch.Tensor = None,
    steps: int = IG_STEPS,
) -> tuple:
    """Compute contrastive IG for a single ECG window.

    Runs IG twice — once for the predicted class and once for the runner-up —
    then subtracts to get ΔIG, which highlights what the model uses to
    distinguish its top two hypotheses.

    Args:
        model: ECGClassifier1D in eval mode, on the target device.
        x: input tensor of shape (1, 12, 1000), on the same device as model.
        baseline: baseline tensor of the same shape as x.
            If None, uses per-channel mean of x broadcast across time.
        steps: number of Riemann approximation steps for IG.

    Returns:
        delta_ig: tensor of shape (12, 1000).
            Positive values support the predicted class; negative support runner-up.
        metadata: dict with keys:
            "pred_class"         — int, highest sigmoid-probability class index
            "runner_up_class"    — int, second-highest probability class index
            "pred_class_name"    — str, e.g. "MI"
            "runner_up_class_name" — str, e.g. "STTC"
            "probs"              — np.ndarray (num_classes,), sigmoid probabilities
                                    (independent per class, do not sum to 1 —
                                    the model is trained multi-label)
    """
    model.eval()

    if baseline is None:
        channel_means = x.mean(dim=-1, keepdim=True)  # (1, 12, 1)
        baseline = channel_means.expand_as(x)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0]  # (num_classes,) — independent per-class probs

    probs_np = probs.cpu().numpy()
    sorted_indices = np.argsort(probs_np)[::-1]
    pred_class = int(sorted_indices[0])
    runner_up_class = int(sorted_indices[1])

    ig_pred = integrated_gradients(model, x, pred_class, baseline, steps)      # (12, 1000)
    ig_runner = integrated_gradients(model, x, runner_up_class, baseline, steps)  # (12, 1000)

    delta_ig = ig_pred - ig_runner

    metadata = {
        "pred_class": pred_class,
        "runner_up_class": runner_up_class,
        "pred_class_name": LABEL_DECODER[pred_class],
        "runner_up_class_name": LABEL_DECODER[runner_up_class],
        "probs": probs_np,
    }

    return delta_ig, metadata


def compute_record_contrastive_ig(
    model,
    ecg: np.ndarray,
    device: torch.device,
    include_attention: bool = True,
) -> tuple:
    """Compute contrastive IG over all sliding windows of a full ECG recording.

    Stitches per-window ΔIG tensors into a full-length per-lead attribution
    map using overlap averaging. Per-lead structure is preserved so downstream
    evidence extraction can reason about individual leads.

    Args:
        model: ECGClassifier1D in eval mode.
        ecg: numpy array of shape (12, total_length) — the full ECG signal.
        device: torch device on which to run inference.
        include_attention: if True, also compute and return a full-length
            attention importance array for use in evidence cross-checking.

    Returns:
        delta_ig_full: float32 numpy array of shape (12, total_length).
            Per-lead ΔIG averaged over overlapping windows.
        attention_full: float32 numpy array of shape (total_length,), or None
            when include_attention is False. Upsampled attention rollout
            stitched across windows with overlap averaging.
        window_metadata: list of per-window dicts (from compute_contrastive_ig),
            each augmented with "window_index", "start", "end".
    """
    _, total_length = ecg.shape
    windows = sliding_window(ecg, WINDOW_SIZE, STRIDE)

    delta_ig_accum = np.zeros((12, total_length), dtype=np.float64)
    overlap_count = np.zeros(total_length, dtype=np.float64)

    attn_accum = np.zeros(total_length, dtype=np.float64) if include_attention else None
    attn_overlap = np.zeros(total_length, dtype=np.float64) if include_attention else None

    window_metadata = []

    for i, window in enumerate(windows):
        start = i * STRIDE
        end = start + WINDOW_SIZE

        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)

        # Per-channel mean baseline: shape (1, 12, 1) → expanded to (1, 12, 1000)
        channel_means = x.mean(dim=-1, keepdim=True)
        baseline = channel_means.expand_as(x)

        delta_ig_window, meta = compute_contrastive_ig(model, x, baseline)

        delta_ig_accum[:, start:end] += delta_ig_window.cpu().numpy()
        overlap_count[start:end] += 1.0

        meta["window_index"] = i
        meta["start"] = start
        meta["end"] = end
        window_metadata.append(meta)

        if include_attention:
            with torch.no_grad():
                _, attentions = model(x, return_attention=True)
            importance = aggregate_attention(attentions)         # (1, T=63)
            attn_1d = importance[0].cpu().numpy()               # (T=63,)
            attn_up = upsample_attention(attn_1d, WINDOW_SIZE)  # (1000,)
            attn_accum[start:end] += attn_up
            attn_overlap[start:end] += 1.0

    overlap_count[overlap_count == 0] = 1.0
    delta_ig_full = (delta_ig_accum / overlap_count[np.newaxis, :]).astype(np.float32)

    attention_full = None
    if include_attention:
        attn_overlap[attn_overlap == 0] = 1.0
        attention_full = (attn_accum / attn_overlap).astype(np.float32)

    return delta_ig_full, attention_full, window_metadata
