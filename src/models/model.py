from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .bayesian_layers import BayesianLinear
from .transformer import BayesianTransformerEncoder

class BayesianTransformerForCellTracking(nn.Module):
    """
    Projects tabular features -> uncertainty-aware embedding (mu_e, logvar_e).
    """
    def __init__(self,
                 input_dim: int,
                 embed_dim: int = 64,
                 num_heads: int = 2,
                 ff_hidden_dim: int = 256,
                 num_layers: int = 2,
                 output_dim: int = 32,  # embedding dimension d'
                 prior_mu: float = 0.0,
                 prior_sigma: float = 0.1,
                 dropout: float = 0.1,
                 use_layernorm: bool = True,
                 save_feature_attention_every: int = 0,
                 out_fig_dir: str | None = None,
                 alpha: float = 0.9):
        super().__init__()
        self.feature_attention = BayesianLinear(input_dim, input_dim, prior_mu, prior_sigma)
        self.input_proj = BayesianLinear(input_dim, embed_dim, prior_mu, prior_sigma)
        self.encoder = BayesianTransformerEncoder(
            embed_dim=embed_dim, num_heads=num_heads, ff_hidden_dim=ff_hidden_dim,
            num_layers=num_layers, prior_mu=prior_mu, prior_sigma=prior_sigma,
            dropout=dropout, use_layernorm=use_layernorm
        )
        self.out_mu = BayesianLinear(embed_dim, output_dim, prior_mu, prior_sigma)
        self.out_logvar = BayesianLinear(embed_dim, output_dim, prior_mu, prior_sigma)

        self.alpha = alpha
        self.register_buffer('prev_attention', torch.zeros(input_dim))
        self.save_every = save_feature_attention_every
        self.out_fig_dir = out_fig_dir

    def forward(self, x, sample=True, frame_idx: int | None = None):
        # x: (B, 1, F)
        B, L, F = x.shape
        scores = self.feature_attention(x, sample=sample)          # (B,1,F)
        scores = torch.softmax(scores, dim=-1)
        if self.training:
            mean_attn = scores.mean(dim=0).squeeze(0)              # (F,)
            smoothed = self.alpha * self.prev_attention + (1 - self.alpha) * mean_attn
            self.prev_attention = smoothed.detach()
        else:
            smoothed = scores.mean(dim=0).squeeze(0)

        # broadcast and apply
        x = x * smoothed.view(1, 1, -1)                            # (B,1,F)

        # (B,1,F) -> (B,1,D)
        h = self.input_proj(x, sample=sample)
        h = self.encoder(h, sample=sample)

        mu_e = self.out_mu(h, sample=sample)                       # (B,1,d')
        logvar_e = self.out_logvar(h, sample=sample)               # (B,1,d')

        # Optional: save feature attention bar every N frames
        if (self.out_fig_dir is not None and frame_idx is not None and
            self.save_every and frame_idx % self.save_every == 0 and not self.training):
            import matplotlib.pyplot as plt
            import numpy as np
            plt.figure(figsize=(8, 3))
            plt.bar(np.arange(F), smoothed.detach().cpu().numpy())
            plt.title(f"Feature Attention @ frame={frame_idx}")
            from pathlib import Path
            Path(self.out_fig_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(f"{self.out_fig_dir}/feature_attention_{frame_idx}.png", bbox_inches='tight', dpi=180)
            plt.close()

        return mu_e, logvar_e, smoothed
