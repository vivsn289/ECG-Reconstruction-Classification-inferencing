# src_ecg_1d/explainability/attention_utils.py
#
# Utilities for aggregating transformer attention weights into
# per-timestep importance scores.
#
# This is the canonical attention aggregation module.
# (attention_maps.py is a deprecated alias — do not add new code there.)

import torch


def aggregate_attention(attentions):
    """Aggregate multi-layer, multi-head attention into a per-timestep score.

    Args:
        attentions: list of attention tensors, one per transformer layer.
            Each tensor has shape (N, num_heads, T, T) where:
              N = batch size
              T = number of time patches (after CNN downsampling)

    Returns:
        importance: tensor of shape (N, T) with normalized per-timestep
            importance scores (range [0, 1] per sample).

    How it works:
        1. Stack all layers → (num_layers, N, num_heads, T, T)
        2. Average across layers and heads → (N, T, T)
        3. Sum over the "attention received" axis (dim=-1 / key axis)
           to get a scalar importance per query timestep → (N, T)
        4. Normalize each sample independently to [0, 1]
    """
    # Stack layers: (num_layers, N, num_heads, T, T)
    attn = torch.stack(attentions, dim=0)

    # Average across layers and heads: (N, T, T)
    attn = attn.mean(dim=(0, 2))

    # Sum attention received per timestep (aggregate query axis): (N, T)
    importance = attn.sum(dim=-1)

    # Normalize per sample to [0, 1]
    importance = importance / (importance.max(dim=1, keepdim=True).values + 1e-8)

    return importance
