import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchinfo import summary
from lightning.pytorch import Trainer
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.loggers import CSVLogger

from .utils.dataloader import DataModule
from .unet.unet import UNet

dataset_root = "Unet\datasets"
batchsize = 32

csv_logger = CSVLogger("Unet/logs/", name="unet", version=0.0)
tb_logger = pl_loggers.TensorBoardLogger(save_dir="Unet/logs/")

trainer = Trainer(logger=[csv_logger, tb_logger], accelerator="gpu", precision="16-mixed")
model = UNet()

data_module = DataModule(dataset_root=dataset_root, batch_size=batchsize)

trainer.fit()