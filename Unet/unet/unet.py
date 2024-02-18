# 백본 이랑 유넷을 연결시키는 다연결시켜서 full network
import torch
import torch.nn as nn

from . import network
from .backbone import get_backbone
from .deocoder import UNetDecoder

class UNet(nn.Module):
    def __init__(self, in_channels=3, classes=5, backbone_name="unet"):
        self.encoder = get_backbone(backbone_name)
        self.decoder = UNetDecoder()

    def forward(self, x):
        features = {None: None} if None in self.shortcut_features else dict()
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                features[name] = x
            if name == self.bb_out_name:
                break

        return x, features
    
    @property
    def layer():
        layer_concat = {
            "unet": ["module1", "module2", "module3", "module4", "module5"],
            "resnet50": ["relu", "layer1", "layer2", "layer3", "layer4"],
            "efficientnetb0": [],
            "vgg19": [],
        }