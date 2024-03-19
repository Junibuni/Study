import os
from collections import deque

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import torchvision
from torchmetrics import Accuracy, Precision, Recall, F1Score
import lightning.pytorch as pl
import numpy as np

from models import Encoder, Decoder
from utils import depth_gradient_loss

class SWE_AE(pl.LightningModule):
    def __init__(self, *, optim_params, scheduler_params, input_size, mode="ae", znum=16, pnum=2):
        # mode: ae, comp, sim
        super(SWE_AE, self).__init__()
        self.save_hyperparameters()

        self.encoder = Encoder(in_c=1, znum=self.hparams.znum, in_shape=self.hparams.input_size)
        self.decoder = Decoder(out_c=1, znum=self.hparams.znum, out_shape=self.hparams.input_size)

        mask = np.zeros((znum,))
        mask[-pnum:] = 1
        self.mask = torch.tensor(mask, dtype=torch.float32)

    def forward(self, x):
        x = self.encoder(x)
        latent_vec = x.clone().detach()
        x = self.decoder(x)
        return x, latent_vec

    def training_step(self, batch, batch_idx):
        x, p = batch
        y_hat, latent_vec = self.forward(x)

        loss = self._get_loss(x, y_hat, latent_vec, p)
        self.log_dict({'step_train_loss': loss})

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.hparams.optim_params)
        scheduler = CosineAnnealingLR(optimizer=optimizer, **self.hparams.scheduler_params)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "step_val_loss"}

    def _get_loss(self, y, y_hat, latent_vec, p):
        mse_loss = F.mse_loss(y, y_hat, reduction="sum") # mass conservation
        grad_loss = depth_gradient_loss(y, y_hat)

        p = p * self.mask
        latent_vec = latent_vec * self.mask
        latent_vec_loss = torch.mean((p - latent_vec) ** 2)
        
        loss = 1.0*mse_loss + 0.8*grad_loss + 0.5*latent_vec_loss
        return loss
    
    def on_train_start(self):
        self.mask = self.mask.to(self.device)
    
class LinearNet(pl.LightningModule):
    def __init__(self, *, autoencoder, znum=16, pnum=2, batch_size=4):
        super(LinearNet, self).__init__()
        self.save_hyperparameters()

        # Inputs [c_t, Δp_t] = [z_t, p_t, Δp_t]
        # Outputs [Δz_t]
        out_shape = self.hparams.znum - self.hparams.pnum
        self.integration_net = nn.Sequential(
            nn.Linear(self.hparams.znum, 1024, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 512, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),

            nn.Linear(512, out_shape), # Δz_t (znum-pnum): pnum is supervised
            nn.ReLU(),
            nn.BatchNorm1d(out_shape),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.integration_net(x)
        return x
    
    def training_step(self, batch, batch_idx):
        # loss =  1/30 sum L2(Δz_t - T(x))
        # get 30 data as batch, yhat for 31st data
        
        # TODO
        self.log_dict({'step_train_loss': loss})

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.hparams.optim_params)
        return optimizer

if __name__ == "__main__":
    input_size = (1, 1, 384, 256)
    input = torch.randn(input_size)

    model_input = dict(
        optim_params = dict(lr=1e-4),
        scheduler_params = dict(T_max=100),
        input_size = (1, 1, 384, 256)
    )

    torch.set_grad_enabled(False)
    ae = SWE_AE(**model_input).eval()
    out = ae(input)
    print(out)
    out = ae.encoder(input)
    print(out)