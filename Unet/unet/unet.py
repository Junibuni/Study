import torch.nn as nn
from torchmetrics.classification import (Dice,
                                         MulticlassF1Score,
                                         MulticlassAccuracy,
                                         MulticlassPrecision,
                                         MulticlassRecall,
                                         MulticlassJaccardIndex,)
import lightning.pytorch as pl
from torchvision.models.feature_extraction import create_feature_extractor

from . import network
from .backbone import get_backbone
from .losses import FocalLoss, MeanIoU

class UNet(pl.LightningModule):
    def __init__(self, *, optimizer, optim_params, criterion_params, in_channels=3, num_classes=5, backbone_name="unet", loss_fn="crossentropy"):
        super(UNet, self).__init__()
        self.save_hyperparameters()

        self.validation_step_outputs = []
        self.criterion = self.loss_function(loss_fn)
        self.metrics = ["dice", "f1", "acc", "precision", "recall", "jaccard"]
        self._init_metrics({"num_classes": num_classes})

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
    
    def loss_function(self, loss_fn):
        match loss_fn:
            case "crossentropy":
                criterion = nn.CrossEntropyLoss(**self.hparams.criterion_params)
            case "miou":
                criterion = MeanIoU(**self.hparams.criterion_params)
            case "focal":
                criterion = FocalLoss(**self.hparams.criterion_params)
            case _:
                raise NotImplementedError(f"{loss_fn} is not valid loss")
        
        return criterion

    def configure_optimizers(self):
        assert self.hparams.optimizer and self.hparams.optim_params, "Optimizer not passed!"
        optimizer = self.hparams.optimizer(self.parameters(),**self.hparams.optim_params)

        return optimizer
    
    def shared_step(self, data, target):
        output = self.forward(data)
        loss = self.criterion(output, target)

        return loss, output, target
    
    def training_step(self, train_batch, batch_idx):
        loss, _, _ = self.shared_step(*train_batch)
        self.log_dict({'step_train_loss': loss})

        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, y_hat, y = self.shared_step(*val_batch)
        self.log_dict({'step_val_loss': loss})

        self._accumulate_metrics(y, y_hat)
         
        return loss

    def on_validation_epoch_end(self):
        metrics = self._metrics()
        for name, val in metrics.items():
            self.log(f"valid-{name}", val)

        self._reset_metrics()
    
    def _init_metrics(self, metric_kwargs={}):
        self.dice = Dice(**metric_kwargs)
        self.f1 = MulticlassF1Score(**metric_kwargs)
        self.acc = MulticlassAccuracy(**metric_kwargs)
        self.precision = MulticlassPrecision(**metric_kwargs)
        self.recall = MulticlassRecall(**metric_kwargs)
        self.jaccard = MulticlassJaccardIndex(**metric_kwargs)

    def _accumulate_metrics(self, y, y_hat):
        for metric in self.metrics:
            getattr(self, metric)(y_hat, y)
    
    def _reset_metrics(self):
        for metric in self.metrics:
            getattr(self, metric).reset()
        
    def _metrics(self):
        return {
            "dice": self.dice.compute(),
            "f1": self.f1.compute(),
            "acc": self.acc.compute(),
            "precision": self.precision.compute(),
            "recall": self.recall.compute(),
            "jaccard": self.jaccard.compute(),
        }