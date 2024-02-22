import argparse

import torch
from torchinfo import summary
from lightning.pytorch import Trainer
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.loggers import CSVLogger

from utils.dataloader import DataModule
from unet.unet import UNet

def argument_parser():
    parser = argparse.ArgumentParser(description="Unet")

    parser.add_argument("--mode", default="train", choices=["train", "test"], type=str)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_epoch", default=100, type=int)
    parser.add_argument("--dataset_pth", default="Unet\datasets", type=str)
    parser.add_argument("--log_pth", default="Unet\logs", type=str)
    parser.add_argument("--device", default="gpu", type=str)
    parser.add_argument("--precision", default="16-mixed", type=str)
    parser.add_argument("--loss_fn", default="crossentropy", type=str)
    parser.add_argument("--max_train_batch", default=0.1, type=float)
    parser.add_argument("--version_name", default="test", type=str)
    parser.add_argument("--max_epochs", default=-1, type=int)

    parser.add_argument("--seed", default=999, type=int, dest="seed")

    return parser.parse_args()

def main(args):
    print(f"Loading Loggers")
    csv_logger = CSVLogger(args.log_pth, name="unet", version=args.version_name)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.log_pth)

    trainer = Trainer(logger=[csv_logger, tb_logger], 
                      accelerator=args.device, 
                      devices=[0],
                      precision=args.precision, 
                      limit_train_batches=args.max_train_batch,
                      num_sanity_val_steps=0, 
                      log_every_n_steps=10,
                      max_epochs=1000)
    
    print("Initialize DataModule")
    data_module = DataModule(dataset_root=args.dataset_pth, batch_size=args.batch_size)

    model_input = dict(
        optimizer = torch.optim.Adam,
        optim_params = dict(lr=0.001, betas=(0.9, 0.999), weight_decay=0.01),
        loss_fn = args.loss_fn,
        criterion_params = {},
    )

    print("Load Model")
    model = UNet(**model_input)

    print("Train")
    trainer.fit(model=model, datamodule=data_module)

if __name__ == "__main__":
    args = argument_parser()
    main(args)