# src_ecg_1d/explainability/attention_maps.py

import torch


def aggregate_attention(attentions):


    attn = torch.stack(attentions)

    attn = attn.mean(dim=0)      
    attn = attn.mean(dim=1)     

    importance = attn.sum(dim=1)

    importance = importance / (importance.max(dim=1, keepdim=True)[0] + 1e-8)

    return importance
