import os
import torch
from torch import nn
from tqdm import tqdm 
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

def mkdir(*path):
    for p in path:
        if not os.path.exists(p):
            os.makedirs(p)
            print(f"creating: {p}")

def save_ckpt(ckpt_dir, epoch, netG, optimG, netD, optimD):
    mkdir(ckpt_dir)
    save_dict = {
        "epoch": epoch,
        "netG": netG.state_dict(),
        "netD": netD.state_dict(),
        "optimG": optimG.state_dict(),
        "optimD": optimD.state_dict(),
    }
    torch.save(save_dict, f"{ckpt_dir}/epoch{epoch}.pth")

def load_ckpt(ckpt_dir, netG, optimG, netD, optimD, train=True):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return epoch, netG, optimG, netD, optimD
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_list = os.listdir(ckpt_dir)
    ckpt_list.sort(key=lambda x:int(''.join(filter(str.isdigit, x))))
    epoch = int(ckpt_list[-1].split('epoch')[1].split('.pth')[0])
    dict_model = torch.load(f"{ckpt_dir}/{ckpt_list[-1]}", map_location=device)
    print(f"load with epoch {epoch}")
    if train:
        netG.load_state_dict(dict_model['netG'])
        netD.load_state_dict(dict_model['netD'])
        optimG.load_state_dict(dict_model['optimG'])
        optimD.load_state_dict(dict_model['optimD'])
        return epoch, netG, optimG, netD, optimD
    else:
        netG.load_state_dict(dict_model['netG'])
        return epoch, netG

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
    result_dir,
    dataset,
):
    netG.to(device).train()
    netD.to(device).train()
    
    loss_G_train = []
    loss_D_real_train = []
    loss_D_fake_train = [] #D(G(z))
    loss_D_train = []
    
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
        loss_D_train.append(loss_D.item())
        optimD.step()

        """
        update G
        """
        optimG.zero_grad()
        fake_logits = netD(fake_output)
        loss_G = criterion(fake_logits, real_labels)
        loss_G.backward()
        loss_G_train.append(loss_G.item())
        optimG.step()

        if batch_idx % 10 == 0:
            print(f"[{epoch}/{num_epochs}][{batch_idx}/{len(dataloader)}]\tLoss_D: {loss_D.item()}\tLoss_G: {loss_G.item()}")

    save_ckpt(ckpt_dir, epoch, netG, optimG, netD, optimD)
    save_img(fake_output, result_dir, epoch)

    return loss_G_train, loss_D_real_train, loss_D_fake_train, loss_D_train

@torch.no_grad()
def evaluate_one_epoch(
    generator,
    args
    ):

    generator.eval()
    input_z = torch.randn(args.batch_size, args.nz, 1, 1) + 0.1
    output = generator(input_z)
    for i in range(output.shape[0]):
        img_name = f"{i:04d}-output.png"
        img = Denormalize_Tanh()(output[i, ...].permute(1, 2, 0).numpy())
        if args.nc == 3:
            plt.imsave(os.path.join(args.result_dir, args.dataset, img_name), img)
        elif args.nc == 1:
            plt.imsave(os.path.join(args.result_dir, args.dataset, img_name), img, cmap=cm.gray)

@torch.no_grad()
def init_weight(m: nn.Module):
    if type(m) in (nn.Conv2d, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    elif type(m) in (nn.BatchNorm2d, ):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        
    if hasattr(m, "bias") and m.bias is not None:
        nn.init.constant_(m.bias.data, 0.0)

class Denormalize_Tanh:
    def __call__(self, data):
        return (data+1)/2
    
def save_img(img, result_dir, epoch):
    img = img.detach()
    batch_size = img.size(0)
    random_idx = torch.randperm(batch_size)[:16]
    selected_imgs = img[random_idx]
    #[1, 3, 64*nrow, 64*nrow]
    result_grid_output = make_grid(selected_imgs, nrow=4, normalize=True, value_range=(-1, 1))
    
    result_grid_output = result_grid_output.squeeze().permute(1, 2, 0).cpu().numpy()
    #assert result_grid_output.any()

    log_dir = os.path.join(result_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    save_path = os.path.join(log_dir, f"epoch_{epoch}.png")
    print(f"img log saved {save_path}")
    plt.imsave(save_path, result_grid_output)
