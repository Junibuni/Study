# 백본 이랑 유넷을 연결시키는 다연결시켜서 full network
import torch
import torch.nn as nn

from . import network
from .backbone import get_backbone
from .deocoder import UNetDecoder

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=5, backbone_name="unet"):
        super(UNet, self).__init__()
        self.backbone_name = backbone_name
        self.encoder = get_backbone(backbone_name)

        self.module1 = network.Up(self.size[0], 512, self.size[1])
        self.module2 = network.Up(512, 256, self.size[2])
        self.module3 = network.Up(256, 128, self.size[3])
        self.module4 = network.Up(128, 64, self.size[4])
        self.final_module = network.FinalConv(64, num_classes)

    def forward(self, x):
      
        return x, features
    
    @property
    def layer(self):
        # skip1, skip2, skip3, skip4, bottleneck
        layer_concat = {
            "unet": ["module1", "module2", "module3", "module4", "module5"],
            "resnet50": ["relu", "layer1", "layer2", "layer3", "layer4"],
            "efficientnetb0": [],
            "vgg19": [],
        }
    
    @property
    def size(self):
        size_dict = {
            "unet": [],
            "resnet50": [],
            "efficientnetb0": [320, 112, 40, 24, 16],
            "vgg19": [],
        }
        return size_dict[self.backbone_name]
    
    def extract_features(model, input_tensor, layer_names):
        extracted_features = {}
        cnt = 1
        
        def hook(module, input, output):
            nonlocal extracted_features, cnt
            extracted_features[f"skip_{cnt}"] = output
            cnt += 1

        hook_handles = []
        for name, module in model.named_modules():
            if name in layer_names:
                hook_handles.append(module.register_forward_hook(hook))

        model(input_tensor)

        for handle in hook_handles:
            handle.remove()

        return extracted_features