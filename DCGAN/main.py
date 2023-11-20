import os
import torch

CWD = os.getcwd()

batch_size = 128
image_size = 64
nc = 3 #RGB channels
nz = 100 #input dim for Generator
ngf = 64
ndf = 64
num_epoch = 5
# optimizer
lr = 0.0002
beta = 0.5