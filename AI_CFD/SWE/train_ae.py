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
    parser.add_argument("--num_epoch", default=10, type=int)
    parser.add_argument("--device", default="gpu", type=str)
    parser.add_argument("--precision", default="16-mixed", type=str)
    parser.add_argument("--max_train_batch", default=1.0, type=float) #for test (ratio)
    parser.add_argument("--log_step", default=10, type=int)
    parser.add_argument("--dataset_pth", default=r"AI_CFD\SWE\datasets", type=str)
    parser.add_argument("--ckpt_pth", default=r"", type=str)
    parser.add_argument("--seed", default=42, type=int, dest="seed")

    parser.add_argument("--cnum", default=32, type=int)
    parser.add_argument("--pnum", default=6, type=int)

    return parser.parse_args()

def main(args):
    torch.set_float32_matmul_precision("medium")
    seed_everything(args.seed)

    version_name = "detach_normp"
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
    
    data_module = DataModule(dataset_root=args.dataset_pth, batch_size=args.batch_size, cnum=args.cnum, pnum=args.pnum, normp=True)

    model_input = dict(
        optim_params = dict(lr=args.lr, betas=(0.9, 0.999)),
        scheduler_params = dict(first_cycle_steps = 20,
                                cycle_mult = 1,
                                max_lr = args.lr,
                                min_lr = 1e-6,
                                warmup_steps = 2,
                                gamma = 0.5,
                                last_epoch = -1),
        input_size = (args.batch_size, 4, 384, 256),
        cnum = args.cnum,
        pnum = args.pnum,
        in_c = 3,
        out_c = 1,
        loss_ratio = [0.2, 0.1, 10.] #mse, grad, latentvec
    )

    model = SWE_AE(**model_input)
    print("Train")
    if args.mode == "train":
        trainer.fit(model=model, datamodule=data_module)
    elif args.mode == "test":
        trainer.test(model=model, datamodule=data_module, ckpt_path=args.ckpt_pth)
    
if __name__ == "__main__":
    args = argument_parser()
    main(args)