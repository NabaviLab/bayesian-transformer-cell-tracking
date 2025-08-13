from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianLinear(nn.Module):
    """
    W ~ N(mu, diag(sigma^2)); reparameterized sampling
    """
    def __init__(self, in_features, out_features, prior_mu=0.0, prior_sigma=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.ones(out_features, in_features) * -5.0)
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.ones(out_features) * -5.0)
        nn.init.xavier_normal_(self.weight_mu)

        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

    def forward(self, x, sample: bool = True):
        if sample:
            w_std = torch.exp(0.5 * self.weight_logvar)
            b_std = torch.exp(0.5 * self.bias_logvar)
            w = self.weight_mu + torch.randn_like(w_std) * w_std
            b = self.bias_mu + torch.randn_like(b_std) * b_std
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)
