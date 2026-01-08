# src_ecg_1d/explainability/attention_maps.py

import torch


def aggregate_attention(attentions):
    """
    Aggregate transformer attention maps into a single importance score per time step.

    Args:
        attentions (list of tensors):
            Each tensor has shape (N, heads, T, T)

    Returns:
        importance (tensor):
            Shape (N, T)
    """
    # Stack layers: (L, N, heads, T, T)
    attn = torch.stack(attentions)

    # Average over layers and heads
    attn = attn.mean(dim=0)      # (N, heads, T, T)
    attn = attn.mean(dim=1)      # (N, T, T)

    # Sum attention received by each time step
    importance = attn.sum(dim=1) # (N, T)

    # Normalize for visualization
    importance = importance / (importance.max(dim=1, keepdim=True)[0] + 1e-8)

    return importance
