import math
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
from torch import nn

from .differnet import *
from .resflow import *


class OpenHybrid(nn.Module):
    def __init__(self, encoder, latent_dim, cls_hidden_dim, num_classes, flow_module="resflow"):
        super().__init__()

        self.latent_dim = latent_dim

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

        if flow_module == "resflow":
            self.flow_module = ResidualFlow(
                (latent_dim, 1, 1),
                init_layer=LogitTransform(0.05),
                actnorm=True,
            )
        elif flow_module == "differnet":
            self.flow_module = build_nf_head(latent_dim)
        else:
            raise RuntimeError(f"Unknown flow module {flow_module}")

        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

    def _compute_ll(self, z, delta_logp):
        nvals = 256
        std_normal_logprob = -(math.log(2 * math.pi) + z.pow(2)) / 2
        logpz = std_normal_logprob.view(z.size(0), -1).sum(1, keepdim=True)
        logpx = logpz - delta_logp - np.log(nvals) * self.latent_dim
        return logpx

    def _compute_ll_v2(self, z, jac):
        logpx = -0.5 * torch.sum(z**2, dim=(1,)) + jac
        return logpx

    def _compute_ll_v3(self, z, ldj):
        logpz = self.prior.log_prob(z).sum(dim=1, keepdim=True)
        logpx = ldj + logpz
        return logpx

    def forward(self, x, classify=True, flow=True, enc_grad=True):
        with nullcontext() if enc_grad else torch.no_grad():
            feats = self.encoder(x)
        out = ()
        if classify:
            cls_logits = self.classifier(feats)
            out += (cls_logits,)
        if flow:
            if isinstance(self.flow_module, ResidualFlow):
                # perform min-max normalization
                rescaled_feats = (feats - feats.min()) / (feats.max() - feats.min())
                # add two extra trailing dimensions
                for _ in range(2):
                    rescaled_feats = rescaled_feats.unsqueeze(-1)
                z, delta_logp = self.flow_module(rescaled_feats, 0)
                logpx = self._compute_ll(z, delta_logp)
            else:  # DifferNet
                z = self.flow_module(feats)
                jac = self.flow_module.jacobian(run_forward=False)
                logpx = self._compute_ll_v2(z, jac)
            out += (logpx,)
        return out if len(out) > 1 else out[0]


@contextmanager
def flow_update_context(flow_module, grad_rescale):
    if isinstance(flow_module, ResidualFlow):
        # rescale and clip grads
        if grad_rescale > 1:
            with torch.no_grad():
                for p in flow_module.parameters():
                    if p.grad is not None:
                        p.grad /= grad_rescale
        nn.utils.clip_grad.clip_grad_norm_(flow_module.parameters(), 1.0)
        yield
        # update_lipschitz
        with torch.no_grad():
            for m in flow_module.modules():
                if isinstance(m, SpectralNormConv2d) or isinstance(m, SpectralNormLinear):
                    m.compute_weight(update=True)
                if isinstance(m, InducedNormConv2d) or isinstance(m, InducedNormLinear):
                    m.compute_weight(update=True)
    else:
        yield
