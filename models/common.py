from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn

from utils.utils import trunc_normal_

from .nf import build_nf_head

BATCH_NORM_EPSILON = 1e-5

def normalize_feats(feats):
    if isinstance(feats, torch.Tensor):
        return feats/feats.norm(dim=1,keepdim=True).expand(-1,feats.shape[1])
    elif isinstance(feats, np.ndarray):
        return feats/np.linalg.norm(feats, axis=1, keepdims=True)
    else:
        raise NotImplementedError("Unknown type")

class SimCLRContrastiveHead(nn.Module):
    def __init__(self, channels_in, out_dim=128, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i != num_layers - 1:
                dim, relu = channels_in, True
            else:
                dim, relu = out_dim, False
            self.layers.append(nn.Linear(channels_in, dim, bias=False))
            bn = nn.BatchNorm1d(dim, eps=BATCH_NORM_EPSILON, affine=True)
            if i == num_layers - 1:
                nn.init.zeros_(bn.bias)
            self.layers.append(bn)
            if relu:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for b in self.layers:
            x = b(x)
        return normalize_feats(x)

class CSIContrastiveHead(nn.Module):
    def __init__(self, channels_in, out_dim=128):
        super().__init__()
        self.simclr_layer = nn.Sequential(
            nn.Linear(channels_in, channels_in),
            nn.ReLU(),
            nn.Linear(channels_in, out_dim),
        )

    def forward(self, x):
        return normalize_feats(self.simclr_layer(x))

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class WrapperWithContrastiveHead(nn.Module): 
    """
    Wrap a base model composed of a Backbone + fc (cls head) adding a parallel contrastive head 
    """
    def __init__(self, base_model, out_dim, contrastive_type="simclr", add_cls_head=False, n_classes=100):
        """
        Arguments:
            base_model (nn.Module)
            out_dim (int): output size of the base model (feats)
            contrastive_type (str): type of contrastive head: simclr (for simclr/supclr), CSI (for CSI/supCSI), DINO
            add_cls_head (bool): add also a classification head?
            n_classes (int): output_num for classification
        """
        super().__init__()
        assert contrastive_type in ["simclr", "CSI", "DINO"], f"Unknown contrastive head type {contrastive_type}"
        self.base_model = base_model
        self.add_cls_head = add_cls_head
        self.contrastive_out_dim = 128

        if self.add_cls_head:
            self.fc = nn.Linear(in_features=out_dim, out_features=n_classes)
        if contrastive_type == "simclr":
            self.contrastive_head = SimCLRContrastiveHead(channels_in=out_dim, out_dim=self.contrastive_out_dim)
        elif contrastive_type == "CSI":
            self.contrastive_head = CSIContrastiveHead(channels_in=out_dim, out_dim=self.contrastive_out_dim)
        elif contrastive_type  == "DINO":
            self.contrastive_out_dim = 65536
            self.contrastive_head = DINOHead(in_dim=out_dim, out_dim=self.contrastive_out_dim)

    def forward(self, x, contrastive=False): 
        if self.add_cls_head:
            feats = self.base_model(x)
            logits = self.fc(feats)
        else:
            logits, feats = self.base_model(x)
        if contrastive:
            return logits, self.contrastive_head(feats)
        return logits, feats


class WrapperWithFC(nn.Module):
    """
    Wrap a base model adding a final fc on top of it
    """
    def __init__(self, base_model, out_dim, n_classes, half_precision=False, base_output_map=None):
        """
        Arguments: 
            base_model (nn.Module)
            out_dim (int): output size of the base model 
            n_classes (int): output_size of the wrapper
            half_precision (bool): use half precision for fc
            base_output_map (fn): extract feats from base model output
        """
        super().__init__()
        self.base_model = base_model 
        self.fc = nn.Linear(in_features=out_dim, out_features=n_classes)
        self.half_precision = half_precision
        self.base_output_map = base_output_map
        if half_precision:
            self.fc = self.fc.half()

    def forward(self, x):
        if self.half_precision:
            x = x.half()
        feats = self.base_model(x)
        if self.base_output_map:
            feats = self.base_output_map(feats)
        return self.fc(feats), feats


class WrapperWithNF(nn.Module):
    def __init__(self, base_model, out_dim, n_classes):
        super().__init__()
        self.base_model = base_model
        cls_hidden_dim = out_dim // 2
        self.cls_head = (
            nn.Linear(out_dim, n_classes)
            if cls_hidden_dim is None
            else nn.Sequential(
                nn.Linear(out_dim, cls_hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(cls_hidden_dim, n_classes),
            )
        )
        self.nf_head = build_nf_head(out_dim)
    
    def _compute_ll(self, z, jac):
        logpx = -0.5 * torch.sum(z**2, dim=(1,)) + jac
        return logpx

    def forward(self, x, classify=True, flow=False, enc_grad=True):
        with nullcontext() if enc_grad else torch.no_grad():
            feats = self.base_model(x)
        out = ()
        if classify:
            cls_logits = self.cls_head(feats)
            out += (cls_logits,)
        if flow:
            z = self.nf_head(feats)
            jac = self.nf_head.jacobian(run_forward=False)
            logpx = self._compute_ll(z, jac)
            out += (logpx,)
        out += (feats,)
        return out if len(out) > 1 else out[0]
    