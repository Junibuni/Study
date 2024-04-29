import os
from collections import deque

from lightning.pytorch.core.optimizer import LightningOptimizer
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, StepLR
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.autograd import Variable
import torchvision
from torchmetrics import Accuracy, Precision, Recall, F1Score

import lightning.pytorch as pl
import numpy as np

from models import Encoder, Decoder
from utils import depth_gradient_loss, CosineAnnealingWarmupRestarts

class SWE_AE(pl.LightningModule):
    def __init__(self, *, optim_params, scheduler_params, input_size, mode="ae", cnum=32, pnum=6, in_c=3, out_c=1, loss_ratio=[1., 1., 1.0]):
        # mode: ae, comp, sim
        super(SWE_AE, self).__init__()
        self.save_hyperparameters()

        self.encoder = Encoder(in_c=self.hparams.in_c, cnum=self.hparams.cnum-self.hparams.pnum, in_shape=self.hparams.input_size)
        self.decoder = Decoder(out_c=self.hparams.out_c, cnum=self.hparams.cnum, out_shape=self.hparams.input_size)

        mask = np.zeros((self.hparams.cnum,))
        mask[-self.hparams.pnum:] = 1
        mask = torch.tensor(mask, dtype=torch.float32)
        self.register_buffer("mask", mask)

    def forward(self, x, p):
        latent_vec = self.encoder(x)
        # lv_copy = latent_vec.clone().detach()
        # lv_copy[:, -self.hparams.pnum:] = p[:, -self.hparams.pnum:]
        out = self.decoder(torch.cat((latent_vec, p[:, -self.hparams.pnum:]), dim=1))
        return out, latent_vec

    def shared_step(self, x, p, mode):
        y_hat, latent_vec = self.forward(x, p)
        depth_map = x[:, 0:1, :, :] # extract first channel
        loss = self._get_loss(depth_map, y_hat, latent_vec, p, mode)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(*batch, mode="train")
        self.log_dict({'step_train_loss': loss})

        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self.shared_step(*val_batch, mode="val")
        self.log_dict({'step_val_loss': loss})

        return loss
    
    def test_step(self, test_batch, batch_idx, dataloader_idx=0):
        loss = self.shared_step(*test_batch, mode="test")
        self.log_dict({'step_test_loss': loss})

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.hparams.optim_params)
        #scheduler = CosineAnnealingLR(optimizer=optimizer, **self.hparams.scheduler_params) # no need to monitor
        scheduler = CosineAnnealingWarmupRestarts(optimizer, **self.hparams.scheduler_params)

        return [optimizer], [scheduler]

    def _get_loss(self, y, y_hat, latent_vec, p, mode):
        #mse_loss = F.mse_loss(y, y_hat, reduction="sum") # mass conservation
        mse_loss = F.l1_loss(y, y_hat, reduction="mean")
        #grad_loss = depth_gradient_loss(y, y_hat)

        # p = p * self.mask
        # latent_vec = latent_vec * self.mask
        # latent_vec_loss = F.mse_loss(latent_vec, p, reduction="sum")
        
        ratio = self.hparams.loss_ratio
        # loss = ratio[0]*mse_loss + ratio[1]*grad_loss + ratio[2]*latent_vec_loss

        # loss = ratio[0]*mse_loss + ratio[2]*latent_vec_loss
        loss = mse_loss
        # logging multitask learning
        if mode in ["train", "test"]:
            # self.log_dict({'mse_loss': mse_loss, 'grad_loss': grad_loss, 'latent_vec_loss': latent_vec_loss})
            # self.log_dict({'mse_loss': mse_loss, 'latent_vec_loss': latent_vec_loss})
            self.log_dict({'mse_loss': mse_loss})
        return loss
    
class ManifoldNavigator(pl.LightningModule):
    def __init__(self, *, cnum=32, pnum=6, model_type="linear", batch_size, optim_params, scheduler_params, hidden_shape=128, dropout=0, weight_decay=0.001):
        super(ManifoldNavigator, self).__init__()
        assert (model_type in ["linear", "lstm"]), f"{model_type} is not valid"
        self.save_hyperparameters()

        in_shape = self.hparams.cnum
        hidden_shape = self.hparams.hidden_shape
        out_shape = self.hparams.cnum - self.hparams.pnum
        self.weight_decay= self.hparams.weight_decay

        match self.hparams.model_type:
            case "linear":
                self.integration_net = LinearNavigator(in_shape, out_shape)
            case "lstm":
                self.integration_net = LSTMNavigator(in_shape, hidden_shape, out_shape, batch_size, dropout)
            case _:
                raise NotImplementedError(f"model not implementd")
        
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self.integration_net(x)
        return x
    
    def shared_step(self, input_data, target_data):
        output = self.forward(input_data)
        loss = self.loss(output, target_data)

        #loss += self.l2_regularization()

        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(*batch)

        self.log_dict({'step_train_loss': loss})

        return loss
    
    def validation_step(self, val_batch, batch_idx):
        loss = self.shared_step(*val_batch)
        self.log_dict({'step_val_loss': loss})

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.hparams.optim_params)
        scheduler = StepLR(optimizer, **self.hparams.scheduler_params)
        return [optimizer], [scheduler]
    
    def l2_regularization(self):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
        return self.weight_decay * l2_loss
    
class LSTMNavigator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, dropout, num_layers=1):
        super(LSTMNavigator, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            batch_first=True,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
        h_0 = Variable(torch.zeros(num_layers, batch_size, hidden_size))
        c_0 = Variable(torch.zeros(num_layers, batch_size, hidden_size))
        self.register_buffer("h_0", h_0)
        self.register_buffer("c_0", c_0)

    def forward(self, input):
        # lstm_out = (batch_size, seq_len, hidden_size)
        h_0 = self.h_0.clone().detach()
        c_0 = self.c_0.clone().detach()
        lstm_out, _ = self.lstm(input, (h_0, c_0))
        output = self.fc(lstm_out[:, -1, :])
        return output
    
class LinearNavigator(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 512, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),

            nn.Linear(512, output_size, bias=False), # Δz_t (cnum-pnum): pnum is supervised
            nn.ReLU(),
            nn.BatchNorm1d(output_size),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        batch_size, seq_len, feature_len = x.size()
        x = x.view(batch_size * seq_len, feature_len)
        x = self.layers(x)
        x = x.view(batch_size, seq_len, -1)
        return x[:, -1, :]

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
    
    model = ManifoldNavigator(cnum=32, pnum=6, model_type="lstm", batch_size=4).eval()

    ref_value = torch.tensor([1, 0, 0, 0, 0, 1])
    ref_value = ref_value.unsqueeze(0)

    input = torch.randn(4, 360, 32)
    print("input shape:", input.shape)
    output = model(input)
    print("output shape:", output.shape)

    # for i in range(3):
    #     xt = torch.cat([ct, ref_value], dim=1)
    #     print("xt", xt, xt.shape)
    #     # in = [c_t, Δp_t] = [z_t, p_t, p_{t+1}-p_t]
    #     dz = model(xt)
    #     print("dz", dz, dz.shape)
    #     # out = Δz_t
    #     # z_{t+1} = z_t + Δz_t
    #     z_t1 = dz + ct[:, :26]
    #     #ref_val here is actually Δp_t = p_{t+1}-p_t
    #     c_t1 = torch.cat((z_t1, ref_value), dim=1)
    #     print("c_t1", c_t1, c_t1.shape, end="\n")

    #     ct = c_t1
        
    # out = model(input)
    # print(input, input.shape)
    # print(out, out.shape)
