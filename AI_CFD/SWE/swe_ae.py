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
    def __init__(self, *, optim_params, scheduler_params, input_size, mode="ae", cnum=16, pnum=2, in_c=1, out_c=1, loss_ratio=[1., 1., 0.5]):
        # mode: ae, comp, sim
        super(SWE_AE, self).__init__()
        self.save_hyperparameters()

        self.encoder = Encoder(in_c=self.hparams.in_c, cnum=self.hparams.cnum, in_shape=self.hparams.input_size)
        self.decoder = Decoder(out_c=self.hparams.out_c, cnum=self.hparams.cnum, out_shape=self.hparams.input_size)

        mask = np.zeros((self.hparams.cnum,))
        mask[-self.hparams.pnum:] = 1
        mask = torch.tensor(mask, dtype=torch.float32)
        self.register_buffer("mask", mask)

    def forward(self, x):
        latent_vec = self.encoder(x)
        out = self.decoder(latent_vec)
        return out, latent_vec

    def training_step(self, batch, batch_idx):
        x, p = batch
        y_hat, latent_vec = self.forward(x)

        depth_map = x.c[:, 0:1, :, :] # extract first channel
        loss = self._get_loss(depth_map, y_hat, latent_vec, p)
        self.log_dict({'step_train_loss': loss})

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.hparams.optim_params)
        scheduler = CosineAnnealingLR(optimizer=optimizer, **self.hparams.scheduler_params) # no need to monitor

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _get_loss(self, y, y_hat, latent_vec, p):
        mse_loss = F.mse_loss(y, y_hat, reduction="sum") # mass conservation
        grad_loss = depth_gradient_loss(y, y_hat)

        p = p * self.mask
        latent_vec = latent_vec * self.mask
        latent_vec_loss = torch.mean((p - latent_vec) ** 2)
        
        ratio = self.hparams.loss_ratio
        loss = ratio[0]*mse_loss + ratio[1]*grad_loss + ratio[2]*latent_vec_loss

        # logging multitask learning
        self.log_dict({'mse_loss': mse_loss, 'grad_loss': grad_loss, 'latent_vec_loss': latent_vec_loss})

        return loss
    
class LinearNet(pl.LightningModule):
    def __init__(self, *, cnum=32, pnum=6, batch_size=4):
        super(LinearNet, self).__init__()
        self.save_hyperparameters()

        # Inputs [c_t, Δp_t] = [z_t, p_t, Δp_t]
        # Outputs [Δz_t]
        in_shape = self.hparams.cnum + self.hparams.pnum
        out_shape = self.hparams.cnum - self.hparams.pnum
        self.integration_net = nn.Sequential(
            nn.Linear(in_shape, 1024, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 512, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),

            nn.Linear(512, out_shape, bias=False), # Δz_t (cnum-pnum): pnum is supervised
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
    # input_size = (1, 1, 384, 256)
    # input = torch.randn(input_size)

    # model_input = dict(
    #     optim_params = dict(lr=1e-4),
    #     scheduler_params = dict(T_max=100),
    #     input_size = (1, 1, 384, 256)
    # )

    # torch.set_grad_enabled(False)
    # ae = SWE_AE(**model_input).eval()
    # out = ae(input)
    # print(out)
    # out = ae.encoder(input)
    # print(out)

    input_size = (1, 32)
    ct = torch.randn(input_size)
    torch.set_grad_enabled(False)
    
    model = LinearNet(cnum=32, pnum=6).eval()

    ref_value = torch.tensor([1, 0, 0, 0, 0, 1])
    ref_value = ref_value.unsqueeze(0)
    
    for i in range(3):
        xt = torch.cat([ct, ref_value], dim=1)
        print("xt", xt, xt.shape)
        # in = [c_t, Δp_t] = [z_t, p_t, p_{t+1}-p_t]
        dz = model(xt)
        print("dz", dz, dz.shape)
        # out = Δz_t
        # z_{t+1} = z_t + Δz_t
        z_t1 = dz + ct[:, :26]
        #ref_val here is actually Δp_t = p_{t+1}-p_t
        c_t1 = torch.cat((z_t1, ref_value), dim=1)
        print("c_t1", c_t1, c_t1.shape)

        ct = c_t1
        
    # out = model(input)
    # print(input, input.shape)
    # print(out, out.shape)
