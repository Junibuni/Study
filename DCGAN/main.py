import os
import argparse
import random
import torch
from torch import optim
from torch import nn

from models import Generator, Discriminator
from utils import train_one_epoch, evaluate_one_epoch, init_weight




def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    """
    DATA LOADER HERE
    """

    ckpt = args.ckpt_path
    netG = Generator(nc=args.nc, nz=args.nz, ngf=args.ngf)
    netD = Discriminator(nc=args.nc, ndf=args.ndf)
    init_weight(netG,)
    
    criterion = nn.BCELoss()
    
    if args.mode == "train":
        for epoch in args.num_epoch:
            train_one_epoch(
               generator = netG,
               discriminator = netD,
            )
        pass

    elif args.mode == "eval":
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCGAN")

    parser.add_argument("--mode", default="train", choices=["train", "test"], type=str, dest="mode")
    parser.add_argument("--lr", default=2e-4, type=float, dest="lr")
    parser.add_argument("--batch_size", default=128, type=int, dest="batch_size")
    parser.add_argument("--num_epoch", default=5, type=int, dest="num_epoch")
    parser.add_argument("--ckpt_path", default="./checkpoint", type=str, dest="ckpt_path")
    parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")

    parser.add_argument("--nc", default=3, type=int, dest="nc")
    parser.add_argument("--nz", default=100, type=int, dest="nz")
    parser.add_argument("--ngf", default=128, type=int, dest="ngf")
    parser.add_argument("--ndf", default=128, type=int, dest="ndf")
    parser.add_argument("--beta", default=0.5, type=float, dest="beta")

    parser.add_argument("--seed", default=999, type=int, dest="seed")
    
    args = parser.parse_args()

    main(args)