import torch.nn as nn

BATCH_NORM_EPSILON = 1e-5

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
        return x

class CSIContrastiveHead(nn.Module):
    def __init__(self, channels_in, out_dim=128):
        super().__init__()
        self.simclr_layer = nn.Sequential(
            nn.Linear(channels_in, channels_in),
            nn.ReLU(),
            nn.Linear(channels_in, out_dim),
        )

    def forward(self, x):
        return self.simclr_layer(x)

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
