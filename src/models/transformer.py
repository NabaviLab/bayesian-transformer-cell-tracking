from __future__ import annotations
import torch
import torch.nn as nn
from .attention import BayesianMultiheadSelfAttention
from .bayesian_layers import BayesianLinear

class BayesianTransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=2, ff_hidden_dim=256,
                 prior_mu=0.0, prior_sigma=0.1, use_layernorm=True, dropout=0.1):
        super().__init__()
        self.attn = BayesianMultiheadSelfAttention(embed_dim, num_heads, prior_mu, prior_sigma)
        self.use_layernorm = use_layernorm
        self.dropout = nn.Dropout(dropout)
        if use_layernorm:
            self.ln1 = nn.LayerNorm(embed_dim)
            self.ln2 = nn.LayerNorm(embed_dim)
        self.ff1 = BayesianLinear(embed_dim, ff_hidden_dim, prior_mu, prior_sigma)
        self.ff2 = BayesianLinear(ff_hidden_dim, embed_dim, prior_mu, prior_sigma)
        self.act = nn.ReLU()

    def forward(self, x, sample=True):
        a = self.attn(x, sample=sample)
        a = self.dropout(a)
        x = self.ln1(x + a) if self.use_layernorm else (x + a)
        f = self.act(self.ff1(x, sample=sample))
        f = self.dropout(self.ff2(f, sample=sample))
        x = self.ln2(x + f) if self.use_layernorm else (x + f)
        return x

class BayesianTransformerEncoder(nn.Module):
    def __init__(self, embed_dim=64, num_heads=2, ff_hidden_dim=256,
                 num_layers=2, prior_mu=0.0, prior_sigma=0.1,
                 dropout=0.1, use_layernorm=True):
        super().__init__()
        self.layers = nn.ModuleList([
            BayesianTransformerEncoderBlock(
                embed_dim, num_heads, ff_hidden_dim,
                prior_mu, prior_sigma, use_layernorm, dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, x, sample=True):
        for blk in self.layers:
            x = blk(x, sample=sample)
        return x
