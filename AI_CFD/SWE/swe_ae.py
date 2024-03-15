import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
from torchmetrics import Accuracy, Precision, Recall, F1Score
import lightning.pytorch as pl
import numpy as np

from models import Encoder, Decoder
from utils import depth_gradient_loss

class SWE_AE(pl.LightningModule):
    def __init__(self, *, optim_params, scheduler_params, input_size, mode="ae", znum=16):
        # mode: ae, comp, sim
        super(SWE_AE, self).__init__()
        self.save_hyperparameters()

        self.encoder = Encoder(in_c=1, znum=self.hparams.znum, in_shape=self.hparams.input_size)
        self.decoder = Decoder(out_c=1, znum=self.hparams.znum, out_shape=self.hparams.input_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)

        loss = self._get_loss(x, y_hat)
        self.log_dict({'step_train_loss': loss})

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.hparams.optim_params)
        scheduler = CosineAnnealingLR(optimizer=optimizer, **self.hparams.scheduler_params)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "step_val_loss"}

    def _get_loss(self, y, y_hat):
        mseloss = nn.MSELoss(y, y_hat)
        grad_loss = depth_gradient_loss(y, y_hat)

        loss = 1.0*mseloss + 0.8*grad_loss
        return loss
    
class CombinedModel(pl.LightningModule):
    def __init__(self, autoencoder, linear_fcn):
        super(CombinedModel, self).__init__()
        self.autoencoder = autoencoder
        self.integration_net = nn.Sequential(
            nn.Linear(),
            nn.ReLU(),
            nn.BatchNorm1d(),
            nn.Dropout(0.1),
            
            nn.Linear(),
            nn.ReLU(),
            nn.BatchNorm1d(),
            nn.Dropout(0.1),

            nn.Linear(),
            nn.ReLU(),
            nn.BatchNorm1d(),
            nn.Dropout(0.1)
        )

        for param in self.autoencoder.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.autoencoder.encoder(x)
        x = self.integration_net(x)
        return x
    
    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass
