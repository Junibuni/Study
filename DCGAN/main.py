import os
import argparse
import random
import pickle
import torch
from torch import optim
from torch import nn

from models import Generator, Discriminator
from utils import (train_one_epoch, 
                   evaluate_one_epoch, 
                   init_weight, 
                   load_ckpt,
                   mkdir)
from dataloader import get_dataloader
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device set to {device}")
    
    netG = Generator(nc=args.nc, nz=args.nz, ngf=args.ngf)
    netD = Discriminator(nc=args.nc, ndf=args.ndf)
    netG.apply(init_weight)
    netD.apply(init_weight)

    
    criterion = nn.BCELoss()
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta, 0.999))

    datapath = os.path.join(args.data_dir, args.dataset)
    ckpt_path = os.path.join(args.ckpt_path, args.dataset)
    log_path = os.path.join(args.log_dir, args.dataset)
    mkdir(datapath, ckpt_path, log_path)

    if args.mode == "train":
        loss_G_train = []
        loss_D_real_train = []
        loss_D_fake_train = []
        loss_D_train = []

        dataloader = get_dataloader(datapath, args.dataset, args.batch_size, train=True)

        if args.train_continue:
            epoch, netG, optimizerG, netD, optimizerD = load_ckpt(ckpt_path, epoch, netG, optimizerG, netD, optimizerD, train=True)
            
        for epoch in range(args.num_epoch):
            loss_G_train_per_epoch, loss_D_real_train_per_epoch, loss_D_fake_train_per_epoch, loss_D_train_per_epoch = \
            train_one_epoch(
               netG,
               optimizerG,
               netD,
               optimizerD,
               device,
               criterion,
               dataloader,
               epoch,
               args.num_epoch,
               ckpt_path
            )
            loss_G_train.extend(loss_G_train_per_epoch)
            loss_D_real_train.extend(loss_D_real_train_per_epoch)
            loss_D_fake_train.extend(loss_D_fake_train_per_epoch)
            loss_D_train.extend(loss_D_train_per_epoch)
        
        #save losses
        losses_header = ["loss_G_train", "loss_D_real_train", "loss_D_fake_train", "loss_D_train"]
        losses_list = [loss_G_train, loss_D_real_train, loss_D_fake_train, loss_D_train]
        loss_dict = dict(zip(losses_header, losses_list))
        loss_save_path = os.path.join(os.getcwd(), "losses", args.dataset, ".pkl")
        with open(loss_save_path, "wb") as file:
            pickle.dump(loss_dict, file)
            print(f"losses saved in {loss_save_path}")

    elif args.mode == "eval":
        epoch, netG = load_ckpt(ckpt_path, epoch, netG, optimizerG, netD, optimizerD, train=False)
        evaluate_one_epoch()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCGAN")

    parser.add_argument("--mode", default="train", choices=["train", "test"], type=str, dest="mode")
    parser.add_argument("--lr", default=2e-4, type=float, dest="lr")
    parser.add_argument("--batch_size", default=128, type=int, dest="batch_size")
    parser.add_argument("--num_epoch", default=5, type=int, dest="num_epoch")
    parser.add_argument("--train_continue", default=False, type=bool, dest="train_continue")
    parser.add_argument("--ckpt_path", default="./DCGAN/checkpoint", type=str, dest="ckpt_path")
    parser.add_argument("--log_dir", default="./DCGAN/log", type=str, dest="log_dir")
    parser.add_argument("--data_dir", default="./DCGAN/datasets", type=str, dest="data_dir")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "celeba"], type=str, dest="dataset")

    parser.add_argument("--nc", default=3, type=int, dest="nc")
    parser.add_argument("--nz", default=100, type=int, dest="nz")
    parser.add_argument("--ngf", default=64, type=int, dest="ngf")
    parser.add_argument("--ndf", default=64, type=int, dest="ndf")
    parser.add_argument("--beta", default=0.5, type=float, dest="beta")

    parser.add_argument("--seed", default=999, type=int, dest="seed")
    
    args = parser.parse_args()

    main(args)