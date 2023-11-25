import torch
from torch import nn

from tqdm import tqdm 

def train_one_epoch(generator, discriminator, device, dataloader):
    generator.to(device).train()
    discriminator.to(device).train()
    
    loss_G_train = []
    loss_D_real_train = []
    loss_D_fake_tr
    for batch_idx, (images, _) in tqdm(enumerate(dataloader), "Train"):
        

def evaluate_one_epoch():
    pass

@torch.no_grad()
def init_weight(m: nn.Module):
    if type(m) in (nn.Conv2d, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    elif type(m) in (nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        
    if hasattr(m, "bias") and m.bias is not None:
        nn.init.constant_(m.bias.data, 0.0)