import torch
from torch import nn
from torchvision import models
import types


def get_resnet(network, ckpt=None, n_known_classes=1000, random_init=False):

    if network != "resnet101":
        raise NotImplementedError(f"Unknown network {network}")

    if random_init: 
        model = models.resnet101()
    else:
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    
    if ckpt is not None:
        model.load_state_dict(ckpt, strict=True)

    output_num = model.fc.in_features

    # we need default fc params to not change across runs
    torch.manual_seed(42)
    if n_known_classes is None:
        model.fc = None
    elif n_known_classes != 1000:
        model.fc = nn.Linear(in_features=output_num, out_features=n_known_classes)

    def feats_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        if self.fc is None:
            return x
        return self.fc(x), x

    model.forward = types.MethodType(feats_forward, model)

    return model, output_num
