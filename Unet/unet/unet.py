# 백본 이랑 유넷을 연결시키는 다연결시켜서 full network
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models.feature_extraction import create_feature_extractor

from . import network
from .backbone import get_backbone
from .losses import * #TODO

class UNet(pl.LightningModule):
    def __init__(self, in_channels=3, num_classes=5, backbone_name="unet", loss_fn="crossentropy"):
        super(UNet, self).__init__()
        self.loss_fn = loss_fn

        self.backbone_name = backbone_name
        self.encoder = create_feature_extractor(get_backbone(backbone_name), self.layer)

        self.module1 = network.Up(self.size[0], 512, self.size[1])
        self.module2 = network.Up(512, 256, self.size[2])
        self.module3 = network.Up(256, 128, self.size[3])
        self.module4 = network.Up(128, 64, self.size[4])
        self.final_module = network.FinalConv(64, num_classes)

    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        layer = self.layer

        backbone_feature = features[layer.pop()]

        # Decoder
        x = self.module1(backbone_feature, features[layer.pop()])
        x = self.module2(x, features[layer.pop()])
        x = self.module3(x, features[layer.pop()])
        x = self.module4(x, features[layer.pop()])
        out = self.final_module(x)
        return out
    
    @property
    def layer(self):
        # skip1, skip2, skip3, skip4, bottleneck
        layer_concat = {
            "unet": ["module1", "module2", "module3", "module4", "module5"],
            "resnet50": ["relu", "layer1", "layer2", "layer3", "layer4"],
            "efficientnetb0": ["features.1", "features.2", "features.3", "features.5", "features.7"],
            "vgg19": ["12", "25", "38", "51", "52"],
        }
        return layer_concat[self.backbone_name].copy()
    
    @property
    def size(self):
        # channel size of: bottleneck, skip4, 3, 2, 1
        size_dict = {
            "unet": [1024, 512, 256, 128, 64],
            "resnet50": [2048, 1024, 512, 256, 64],
            "efficientnetb0": [320, 112, 40, 24, 16],
            "vgg19": [512, 512, 512, 256, 128],
        }
        return size_dict[self.backbone_name].copy()
    
    """def extract_features(model, input_tensor, layer_names):
        extracted_features = {}

        def register_hook(name):
            def hook(module, input, output):
                extracted_features[name] = output
            return hook
        
        hook_handles = []
        for name, module in model.named_modules():
            if name in layer_names:
                hook_handles.append(module.register_forward_hook(register_hook(name)))

        model(input_tensor)

        for handle in hook_handles:
            handle.remove()

        return extracted_features"""
    
    def loss_function(self):
        match self.loss_fn:
            case "cross_entropy":
                criterion = nn.CrossEntropyLoss()
            case "miou":
                #TODO: implement miou
                pass
            case "focal":
                #TODO: implement focal
                pass
            case _:
                raise NotImplementedError(f"{self.loss_fn} is not valid loss")
        
        return criterion

    def training_step(self, train_batch, batch_idx):
        data, target = train_batch

        output = self.forward(data)
        loss = self.loss_fn(output, target)
        logs = {'train_loss': loss}

        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        data, target = val_batch

        output = self.forward(data)
        loss = self.loss_fn(output, target)

        #TODO
        pass

    def validation_epoch_end(self, outputs):
        #TODO
        pass

    def test_step(self, batch, batch_idx):
        #TODO
        pass

    def test_epoch_end(self, outputs):
        #TODO
        pass