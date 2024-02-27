import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (ResNet50_Weights,
                                EfficientNet_B0_Weights,
                                VGG19_BN_Weights)

from . import network

def get_backbone(backbone_name):
    model_loaders = {
        'resnet50': [models.resnet50, ResNet50_Weights],
        'efficientnetb0': [models.efficientnet_b0, EfficientNet_B0_Weights],
        'vgg19': [models.vgg19_bn, VGG19_BN_Weights],
        'unet': [UNetEncoder, None]
    }
    
    if backbone_name in model_loaders:
        backbone, pretrained_weight = model_loaders[backbone_name]
        if pretrained_weight:
            backbone = backbone(weights=pretrained_weight.DEFAULT)
        else:
            backbone = backbone()
        if "vgg" in backbone_name:
            backbone = backbone.features
    else:
        raise NotImplementedError(f"{backbone_name} is not available")
    
    return backbone

class UNetEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(UNetEncoder, self).__init__()
        self.module1 = network.DoubleConv(in_channels, out_channels)
        self.module2 = network.Down(out_channels, out_channels*2)
        self.module3 = network.Down(out_channels*2, out_channels*4)
        self.module4 = network.Down(out_channels*4, out_channels*8)
        # link block
        self.module5 = network.Down(out_channels*8, out_channels*16)

    def forward(self, x):
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.module4(x)
        out = self.module5(x)

        return out
    
if __name__ == "__main__":
    from torchvision.models.feature_extraction import create_feature_extractor
    from torchsummary import summary
    backbone_name = "vgg19"
    layer_concat = {
            "unet": ["module1", "module2", "module3", "module4", "module5"],
            "resnet50": ["relu", "layer1", "layer2", "layer3", "layer4"],
            "efficientnetb0": ["features.1", "features.2", "features.3", "features.5", "features.7"],
            "vgg19": ["12", "25", "38", "51", "52"],
        }
    print(get_backbone(backbone_name))
    encoder = create_feature_extractor(get_backbone(backbone_name).to("cuda"), layer_concat[backbone_name])

    print(summary(encoder, input_size=(3, 640, 640)))