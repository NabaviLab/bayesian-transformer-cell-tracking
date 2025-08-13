from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .bayesian_layers import BayesianLinear

class BayesianMultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=2, prior_mu=0.0, prior_sigma=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = BayesianLinear(embed_dim, embed_dim, prior_mu, prior_sigma)
        self.k_proj = BayesianLinear(embed_dim, embed_dim, prior_mu, prior_sigma)
        self.v_proj = BayesianLinear(embed_dim, embed_dim, prior_mu, prior_sigma)
        self.out_proj = BayesianLinear(embed_dim, embed_dim, prior_mu, prior_sigma)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x, sample=True):
        # x: (B, L, D)
        B, L, D = x.shape
        Q = self.q_proj(x, sample=sample)
        K = self.k_proj(x, sample=sample)
        V = self.v_proj(x, sample=sample)
        # reshape to heads: (B, H, L, Dh)
        def split_heads(t):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        Q = split_heads(Q); K = split_heads(K); V = split_heads(V)
        scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) * self.scale
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bhkd->bhqd", attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        return self.out_proj(out, sample=sample)
