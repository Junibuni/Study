import os
import argparse

import torch
from lightning.pytorch import seed_everything
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from dataloader import LinearDataModule
from swe_ae import ManifoldNavigator

def argument_parser():
    parser = argparse.ArgumentParser(description="AI-SWE")

    parser.add_argument("--log_pth", default="AI_CFD\SWE\logs", type=str)
    parser.add_argument("--mode", default="train", choices=["train", "test"], type=str)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_epoch", default=100000, type=int)
    parser.add_argument("--device", default="gpu", type=str)
    parser.add_argument("--precision", default="16-mixed", type=str)
    parser.add_argument("--max_train_batch", default=1.0, type=float) #for test (ratio)
    parser.add_argument("--log_step", default=1, type=int)
    parser.add_argument("--dataset_pth", default=r"AI_CFD\SWE\datasets", type=str)
    parser.add_argument("--seed", default=42, type=int, dest="seed")
    parser.add_argument("--cnum", default=32, type=int, dest="cnum")
    parser.add_argument("--pnum", default=6, type=int, dest="pnum")


    return parser.parse_args()

def main(args):
    torch.set_float32_matmul_precision("medium")
    seed_everything(args.seed)

    version_name = "drop03"
    csv_logger = CSVLogger(args.log_pth, name="latnet\CSVLogger", version=version_name)
    tb_logger = TensorBoardLogger(save_dir=args.log_pth, name="latnet\TBLogger", version=version_name)

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
    
    #perhaps sorting?
    data_module = LinearDataModule(data_dir=args.dataset_pth, batch_size=args.batch_size, seqlen=350)
    
    model_input = dict(
        optim_params = dict(lr=args.lr),
        scheduler_params = dict(gamma = 0.95,
                                step_size = 2000),
        cnum = args.cnum,
        pnum = args.pnum,
        model_type = "lstm",
        batch_size=args.batch_size,
        hidden_shape = 28,
        dropout = 0.3
    )

    model = ManifoldNavigator(**model_input)
    print("Train")
    trainer.fit(model=model, datamodule=data_module)
    
if __name__ == "__main__":
    args = argument_parser()
    main(args)