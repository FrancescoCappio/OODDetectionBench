import torch.nn as nn
from utils.utils import trunc_normal_

BATCH_NORM_EPSILON = 1e-5

def normalize_feats(feats):
    return feats/feats.norm(dim=1,keepdim=True).expand(-1,feats.shape[1])

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
        #self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        #self.last_layer.weight_g.data.fill_(1)
        #if norm_last_layer:
        #    self.last_layer.weight_g.requires_grad = False

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
    def __init__(self, base_model, out_dim, contrastive_type="simclr"):
        """
        Arguments:
            base_model (nn.Module)
            out_dim (int): output size of the base model (feats)
            contrastive_type (str): type of contrastive head: simclr (for simclr/supclr), CSI (for CSI/supCSI), DINO
        """
        super().__init__()
        self.base_model = base_model
        assert contrastive_type in ["simclr", "CSI", "DINO"], f"Unknown contrastive head type {contrastive_type}"

        if contrastive_type == "simclr":
            self.contrastive_head = SimCLRContrastiveHead(channels_in=out_dim)
        elif contrastive_type == "CSI":
            self.contrastive_head = CSIContrastiveHead(channels_in=out_dim)
        elif contrastive_type  == "DINO":
            self.contrastive_head = DINOHead(in_dim=out_dim, out_dim=65536)

    def forward(self, x, contrastive=False): 
        logits, feats = self.base_model(x)
        if contrastive:
            return logits, self.contrastive_head(feats)
        return logits, feats


class WrapperWithFC(nn.Module):
    """
    Wrap a base model adding a final fc on top of it
    """
    def __init__(self, base_model, out_dim, n_classes, half_precision=False):
        """
        Arguments: 
            base_model (nn.Module)
            out_dim (int): output size of the base model 
            n_classes (int): output_size of the wrapper
            half_precision (bool): use half precision for fc
        """
        super().__init__()
        self.base_model = base_model 
        self.fc = nn.Linear(in_features=out_dim, out_features=n_classes)
        self.half_precision = half_precision
        if half_precision:
            self.fc = self.fc.half()

    def forward(self, x):
        if self.half_precision:
            x = x.half()
        feats = self.base_model(x)
        return self.fc(feats), feats
