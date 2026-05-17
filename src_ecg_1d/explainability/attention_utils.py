# src_ecg_1d/explainability/attention_utils.py

import torch


def aggregate_attention(attentions):

    attn = torch.stack(attentions, dim=0)


    attn = attn.mean(dim=(0, 2))

    # Aggregate "attention received" per time step

    importance = attn.sum(dim=-1)


    importance = importance / (importance.max(dim=1, keepdim=True).values + 1e-8)

    return importance
