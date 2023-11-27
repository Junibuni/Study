import os
import torch
from torch import nn
from tqdm import tqdm 

def mkdir(*path):
    for p in path:
        if not os.path.exists(p):
            os.makedirs(p)

def save_ckpt(ckpt_dir, epoch, netG, optimG, netD, optimD):
    mkdir(ckpt_dir)
    save_dict = {
        "epoch": epoch,
        "netG": netG,
        "netD": netD,
        "optimG": optimG,
        "optimD": optimD,
    }
    torch.save(save_dict, f"{ckpt_dir}/epoch{epoch}.pth")

def load_ckpt(ckpt_dir, netG, optimG, netD, optimD):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return epoch, netG, optimG, netD, optimD
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_list = os.listdir(ckpt_dir)
    ckpt_list.sort(key=lambda x:int(''.join(filter(str.isdigit, x))))

    dict_model = torch.load(f"{ckpt_dir}/{ckpt_list[-1]}", map_location=device)

    netG.load_state_dict(dict_model['netG'])
    netD.load_state_dict(dict_model['netD'])
    optimG.load_state_dict(dict_model['optimG'])
    optimD.load_state_dict(dict_model['optimD'])
    epoch = int(ckpt_list[-1].split('epoch')[1].split('.pth')[0])

    return epoch, netG, optimG, netD, optimD

def train_one_epoch(
    netG,
    optimG, 
    netD, 
    optimD,
    device, 
    criterion, 
    dataloader,
    epoch,
    num_epochs,
    ckpt_dir,
):
    netG.to(device).train()
    netD.to(device).train()
    
    loss_G_train = []
    loss_D_real_train = []
    loss_D_fake_train = [] #D(G(z))
    
    for batch_idx, (images, _) in tqdm(enumerate(dataloader)):
        batch_size = len(images)
        images = images.to(device)
        """
        Update D
        """
        optimD.zero_grad()
        # Real data
        real_labels = torch.ones((batch_size, 1, 1, 1), device=device)
        real_output_D = netD(images)
        real_loss_D = criterion(real_output_D, real_labels)
        real_loss_D.backward()
        loss_D_real_train.append(real_loss_D.item())

        # Random data (only train discriminator)
        # Generator input shape torch.Size([B, 100, 1, 1])
        fake_labels = torch.zeros((batch_size, 1, 1, 1), device=device)
        input_z = torch.randn(batch_size, 100, 1, 1, device=device)
        fake_output = netG(input_z)
        fake_output_D = netD(fake_output.detach())
        fake_loss_D = criterion(fake_output_D, fake_labels)
        fake_loss_D.backward()
        loss_D_fake_train.append(fake_loss_D.item())

        loss_D = real_loss_D + fake_loss_D
        optimD.step()

        """
        update G
        """
        optimG.zero_grad()
        fake_output = netD(fake_output)
        loss_G = criterion(fake_output, real_labels)
        loss_G.backward()
        loss_G_train.append(loss_G.item())
        optimG.step()

        if batch_idx % 10 == 0:
            print(f"[{epoch}/{num_epochs}][{batch_idx}/{batch_size}]\tLoss_D: {loss_D.item()}\tLoss_G: {loss_G.item()}")

        if epoch % 2 == 0:
            save_ckpt(ckpt_dir, epoch, netG, optimG, netD, optimD)

@torch.no_grad()
def evaluate_one_epoch(
    generator,
    discriminator, 
    device, 
    criterion, 
    dataloader
    ):
    
    generator.eval()
    discriminator.eval()
    input_z = torch.randn(batchsize, 100, 1, 1, device=device)
    output = generator(input_z)

    pass

@torch.no_grad()
def init_weight(m: nn.Module):
    if type(m) in (nn.Conv2d, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    elif type(m) in (nn.BatchNorm2d, ):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        
    if hasattr(m, "bias") and m.bias is not None:
        nn.init.constant_(m.bias.data, 0.0)