# src_ecg_1d/models/transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention for temporal ECG features.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (N, T, C)
            return_attention: whether to return attention weights

        Returns:
            out: (N, T, C)
            attn (optional): (N, heads, T, T)
        """
        N, T, C = x.shape

        qkv = self.qkv(x)  # (N, T, 3C)
        qkv = qkv.reshape(N, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_scores = attn_scores / (self.head_dim ** 0.5)

        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(N, T, C)
        out = self.out_proj(out)

        if return_attention:
            return out, attn
        return out


class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer encoder block (pre-norm).
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (N, T, C)

        Returns:
            x: (N, T, C)
            attn (optional)
        """
        if return_attention:
            attn_out, attn = self.attn(self.norm1(x), return_attention=True)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, attn
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x


class ECGTransformer(nn.Module):
    """
    Stack of Transformer encoder blocks for temporal reasoning.
    """

    def __init__(
        self,
        embed_dim,
        num_heads=4,
        depth=2,
        dropout=0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (N, T, C)

        Returns:
            x: (N, T, C)
            attentions (optional): list of attention maps
        """
        attentions = []

        for layer in self.layers:
            if return_attention:
                x, attn = layer(x, return_attention=True)
                attentions.append(attn)
            else:
                x = layer(x)

        if return_attention:
            return x, attentions
        return x
