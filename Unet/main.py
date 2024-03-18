import os
import argparse

import torch
from lightning.pytorch import seed_everything
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from unet.dataloader import DataModule
from unet.unet import UNet

def argument_parser():
    parser = argparse.ArgumentParser(description="Unet")

    parser.add_argument("--mode", default="train", choices=["train", "test"], type=str)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_epoch", default=100, type=int)
    parser.add_argument("--dataset_pth", default="Unet\datasets", type=str)
    parser.add_argument("--log_pth", default="Unet\logs", type=str)
    parser.add_argument("--device", default="gpu", type=str)
    parser.add_argument("--precision", default="16-mixed", type=str)
    parser.add_argument("--loss_fn", default="crossentropy", type=str)
    parser.add_argument("--max_train_batch", default=1.0, type=float)
    parser.add_argument("--backbone", default="efficientnetb0", choices=["unet", "resnet50", "efficientnetb0", "vgg19"], type=str)
    parser.add_argument("--log_step", default=10, type=int)
    parser.add_argument("--continue_train", default=False, type=bool)
    parser.add_argument("--continue_pth", default=r"", type=str)

    parser.add_argument("--seed", default=42, type=int, dest="seed")

    return parser.parse_args()

def main(args):
    torch.set_float32_matmul_precision("medium")
    seed_everything(args.seed)

    print(f"Loading Loggers")
    version_name = f"lr_{args.lr:.0e}_{args.loss_fn}_lrscheduled_freeze4"
    csv_logger = CSVLogger(args.log_pth, name=os.path.join(args.backbone, "CSVLogger"), version=version_name)
    tb_logger = TensorBoardLogger(save_dir=args.log_pth, name=os.path.join(args.backbone, "TBLogger"), version=version_name)

    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(logger=[csv_logger, tb_logger], 
                      accelerator=args.device, 
                      devices=[0],
                      precision=args.precision, 
                      limit_train_batches=args.max_train_batch,
                      num_sanity_val_steps=0, 
                      log_every_n_steps=args.log_step,
                      max_epochs=args.num_epoch,
                      callbacks=[lr_monitor]
                      )
    
    print("Initialize DataModule")
    data_module = DataModule(dataset_root=args.dataset_pth, batch_size=args.batch_size)
  
    model_input = dict(
        optimizer = torch.optim.Adam,
        optim_params = dict(lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4),
        loss_fn = args.loss_fn,
        criterion_params = {}, 
        backbone_name = args.backbone,
        freeze_until = 4
    )

    print("Load Model")
    if args.continue_train:
        print("Train_Continue")
        model = UNet.load_from_checkpoint(args.continue_pth) 
        trainer.fit(model=model, datamodule=data_module, ckpt_path=args.continue_pth)
    else:
        model = UNet(**model_input)
        print("Train")
        trainer.fit(model=model, datamodule=data_module)
    
    

if __name__ == "__main__":
    args = argument_parser()
    main(args)