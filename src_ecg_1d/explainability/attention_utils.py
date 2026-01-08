# src_ecg_1d/explainability/attention_utils.py

import torch


def aggregate_attention(attentions):
    """
    Aggregate Transformer attention across layers and heads.

    Expected input:
        attentions: list of tensors
            Each tensor shape: (B, num_heads, T, T)

    Returns:
        Tensor of shape (B, T)
        representing time-wise importance.
    """

    # Stack layers → (L, B, H, T, T)
    attn = torch.stack(attentions, dim=0)

    # Mean over layers and heads → (B, T, T)
    attn = attn.mean(dim=(0, 2))

    # Aggregate "attention received" per time step
    # Sum over source positions
    importance = attn.sum(dim=-1)

    # Normalize
    importance = importance / (importance.max(dim=1, keepdim=True).values + 1e-8)

    return importance
