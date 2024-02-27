import torch
import torch.nn as nn
import torchvision
from torchmetrics.classification import (Dice,
                                         MulticlassF1Score,
                                         MulticlassAccuracy,
                                         MulticlassPrecision,
                                         MulticlassRecall,
                                         MulticlassJaccardIndex,)
import lightning.pytorch as pl
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np

from . import network
from .utils import add_mask
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
        self.encoder = create_feature_extractor(get_backbone(self.backbone_name), self.layer)

        last_channel = 64
        self.module1 = network.Up(self.size[0], 512, self.size[1])
        self.module2 = network.Up(512, 256, self.size[2])
        self.module3 = network.Up(256, 128, self.size[3])
        self.module4 = network.Up(128, last_channel, self.size[4])
        self.additional_module = nn.Identity()

        
        if "unet" not in self.backbone_name:
            last_channel = 32
            self.additional_module = nn.Sequential([
                nn.ConvTranspose2d(64, last_channel, kernel_size=2, stride=2), # double up channel
                network.DoubleConv(last_channel, 16)
            ])
            last_channel = 16
            
        self.final_module = network.FinalConv(last_channel, num_classes)

        self.img_stack_original = []
        self.img_stack_predicted = []

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
        #skip1, skip2, skip3, skip4, bottleneck
        layer_concat = {
            "unet": ["module1", "module2", "module3", "module4", "module5"],
            "resnet50": ["relu", "layer1", "layer2", "layer3", "layer4"],
            "efficientnetb0": ["features.1", "features.2", "features.3", "features.5", "features.7"],
            "vgg19": ["12", "25", "38", "51", "52"],
        }
        return layer_concat[self.backbone_name].copy()
    
    @property
    def size(self):
        #channel size of: bottleneck, skip4, 3, 2, 1
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
        self.batch_size = len(val_batch[0])
        self.log_dict({'step_val_loss': loss})

        self._accumulate_metrics(y, y_hat)

        y_hat = torch.argmax(y_hat, dim=1)
        
        if batch_idx % 20 == 0:
            orig_images = self.unnormalize(val_batch[0])
            class_colors = torch.tensor([(0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)], device=orig_images.device)
            class_colors = class_colors.unsqueeze(-1)

            for image, gt_mask, dt_mask in zip(orig_images, y, y_hat):
                gt_image = add_mask(image.clone(), gt_mask, class_colors)
                dt_image = add_mask(image.clone(), dt_mask, class_colors)

                self.img_stack_original.append(gt_image)
                self.img_stack_predicted.append(dt_image)
   
        return loss

    def unnormalize(self, tensor):
        mean = [0.5735, 0.5618, 0.5681]
        std = [0.2341, 0.2348, 0.2343]

        copy_tensor = tensor.clone()
        for i in range(copy_tensor.shape[1]):
            copy_tensor[:, i] = copy_tensor[:, i] * std[i] + mean[i]

        return copy_tensor

    def on_validation_epoch_end(self):
        metrics = self._metrics()
        for name, val in metrics.items():
            self.log(f"valid-{name}", val)

        self._reset_metrics()

        grid_original = torchvision.utils.make_grid(self.img_stack_original, nrow=self.batch_size)
        grid_predicted = torchvision.utils.make_grid(self.img_stack_predicted, nrow=self.batch_size)
        for logger in self.trainer.loggers:
            if isinstance(logger, pl.loggers.tensorboard.TensorBoardLogger):
                logger.experiment.add_image('Original mask', grid_original, self.global_step)
                logger.experiment.add_image('Predicted mask', grid_predicted, self.global_step)

        self.img_stack_original = []
        self.img_stack_predicted = []

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
        return {metric: getattr(self, metric).compute() for metric in self.metrics}