import os
import argparse

import torch
from lightning.pytorch import seed_everything
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from dataloader import DataModule
from swe_ae import SWE_AE

def argument_parser():
    parser = argparse.ArgumentParser(description="AI-SWE")

    parser.add_argument("--log_pth", default="AI_CFD\SWE\logs", type=str)
    parser.add_argument("--mode", default="train", choices=["train", "test"], type=str)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_epoch", default=150, type=int)
    parser.add_argument("--device", default="gpu", type=str)
    parser.add_argument("--precision", default="16-mixed", type=str)
    parser.add_argument("--max_train_batch", default=1.0, type=float) #for test (ratio)
    parser.add_argument("--log_step", default=10, type=int)
    parser.add_argument("--dataset_pth", default=r"AI_CFD\SWE\datasets", type=str)
    parser.add_argument("--seed", default=42, type=int, dest="seed")

    return parser.parse_args()

def main(args):
    torch.set_float32_matmul_precision("medium")
    seed_everything(args.seed)

    version_name = "ae3"
    csv_logger = CSVLogger(args.log_pth, name="CSVLogger", version=version_name)
    tb_logger = TensorBoardLogger(save_dir=args.log_pth, name="TBLogger", version=version_name)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(logger=[csv_logger, tb_logger], 
                      accelerator=args.device, 
                      devices=[0],
                      precision=args.precision, 
                      limit_train_batches=args.max_train_batch,
                      num_sanity_val_steps=0, #num validation batches to check 
                      log_every_n_steps=args.log_step,
                      max_epochs=args.num_epoch,
                      callbacks=[lr_monitor]
                      )
    
    data_module = DataModule(dataset_root=args.dataset_pth, batch_size=args.batch_size)

    model_input = dict(
        optim_params = dict(lr=args.lr, betas=(0.9, 0.999)),
        scheduler_params = dict(T_max=100),
        input_size = (args.batch_size, 1, 384, 256)
    )

    model = SWE_AE(**model_input)
    print("Train")
    trainer.fit(model=model, datamodule=data_module)
    
if __name__ == "__main__":
    args = argument_parser()
    main(args)