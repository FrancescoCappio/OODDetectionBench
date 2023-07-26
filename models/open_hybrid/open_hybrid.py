import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .resflow import (
    InducedNormConv2d,
    InducedNormLinear,
    LogitTransform,
    ResidualFlow,
    SpectralNormConv2d,
    SpectralNormLinear,
)


class OpenHybrid(nn.Module):
    def __init__(self, encoder, latent_dim, cls_hidden_dim, num_classes):
        super().__init__()

        self.latent_dim = latent_dim
        self.flow_input_size = (latent_dim, 1, 1)

        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

        # Network
        self.encoder = encoder
        self.classifier = (
            nn.Linear(latent_dim, num_classes)
            if cls_hidden_dim is None
            else nn.Sequential(
                nn.Linear(latent_dim, cls_hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(cls_hidden_dim, num_classes),
            )
        )
        self.flow_module = ResidualFlow(
            self.flow_input_size,
            init_layer=LogitTransform(0.05),
            actnorm=True,
        )

    def _compute_ll(self, z, delta_logp):
        nvals = 256
        std_normal_logprob = -(math.log(2 * math.pi) + z.pow(2)) / 2
        logpz = std_normal_logprob.view(z.size(0), -1).sum(1, keepdim=True)
        logpx = logpz - delta_logp - np.log(nvals) * np.prod(self.flow_input_size)
        return logpx

    def _compute_ll_v2(z, jac):
        logpx = -0.5 * torch.sum(z**2, dim=(1,)) + jac
        return logpx

    def _compute_ll_v3(self, z, ldj):
        logpz = self.prior.log_prob(z).sum(dim=1, keepdim=True)
        logpx = ldj + logpz
        return logpx

    def forward(self, x, classify=True, flow=True):
        feats = self.encoder(x)
        out = ()
        if classify:
            cls_logits = self.classifier(feats)
            out += (cls_logits,)
        if flow:
            # perform min-max normalization
            rescaled_feats = (feats - feats.min()) / (feats.max() - feats.min())
            # add two extra trailing dimensions
            for _ in range(2):
                rescaled_feats = rescaled_feats.unsqueeze(-1)
            z, delta_logp = self.flow_module(rescaled_feats, 0)
            logpx = self._compute_ll(z, delta_logp)
            out += (logpx,)
        return out if len(out) > 1 else out[0]

    @torch.no_grad()
    def update_lipschitz(self):
        for m in self.flow_module.modules():
            if isinstance(m, SpectralNormConv2d) or isinstance(m, SpectralNormLinear):
                m.compute_weight(update=True)
            if isinstance(m, InducedNormConv2d) or isinstance(m, InducedNormLinear):
                m.compute_weight(update=True)
