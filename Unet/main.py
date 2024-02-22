import argparse

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

def argument_parser():
    parser = argparse.ArgumentParser(description="Unet")

    parser.add_argument("--mode", default="train", choices=["train", "test"], type=str)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_epoch", default=100, type=int)
    parser.add_argument("--dataset_pth", default="Unet\datasets", type=str)
    parser.add_argument("--device", default="gpu", type=str)
    parser.add_argument("--precision", default="16-mixed", type=str)

    parser.add_argument("--seed", default=999, type=int, dest="seed")

    return parser.parse_args()

def main(args):
    dataset_root = "Unet\datasets"
    batchsize = 32

    csv_logger = CSVLogger("Unet/logs/", name="unet", version=0.0)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="Unet/logs/")

    trainer = Trainer(logger=[csv_logger, tb_logger], accelerator="gpu", precision="16-mixed")
    model = UNet()

    data_module = DataModule(dataset_root=dataset_root, batch_size=batchsize)

    trainer.fit()

if __name__ == "__main":
    args = argument_parser()
    main(args)