import torch
from torchvision import transforms
import numpy as np
from dataloader import CustomDataset
from swe_ae import SWE_AE
from tqdm import tqdm

from utils import unnormalize

ckpt_pth = r"AI_CFD\SWE\logs\CSVLogger\norm_p\checkpoints\epoch=4-step=19440.ckpt"
checkpoint = torch.load(ckpt_pth)

model_input = dict(
    optim_params = dict(lr=1e-4),
    scheduler_params = dict(T_max=100),
    input_size = (1, 3, 384, 256)
)
model = SWE_AE(**model_input)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

data = CustomDataset(r"AI_CFD\SWE\datasets")
# result = []
# model.to("cuda")
# with torch.no_grad():
#     for i in tqdm(range(len(data))):
#         input = data[i][0].unsqueeze(0).to("cuda")
#         _, lv = model(input)
#         result.append(lv)

#     tensor_list = torch.cat(result, dim=0)

#     for i, tensor in enumerate(tqdm(tensor_list)):
#         torch.save(tensor, fr'AI_CFD\SWE\datasets\linear_net\tensor_{i}.pt')

img = data[264][0]


import matplotlib.pyplot as plt
with torch.no_grad():
    print(img.shape)
    imgcrop = img.clone()[0:1, :, :].permute(1, 2, 0)[23:, 47:-2]
    print(imgcrop.shape)
    plt.imshow(unnormalize(imgcrop))
    plt.clim(0, 0.2)
    plt.show()

    out, lv = model(img.unsqueeze(0))
    print(lv, lv.shape)
    print(out, out.shape)
    out_numpy = unnormalize(out.clone().detach()).numpy().squeeze()
    plt.imshow(out_numpy[23:, 47:-2])
    plt.clim(0, 0.2)
    plt.show()



    rmse_error = np.sqrt((unnormalize(out).detach().numpy().squeeze() - unnormalize(img[0:1, :, :]).detach().numpy().squeeze())**2)
    print(rmse_error.shape)
    plt.imshow(rmse_error[23:, 47:-2])
    plt.colorbar()
    plt.show()

    rsmpe_error = np.sqrt((unnormalize(out).detach().numpy().squeeze() - unnormalize(img[0:1, :, :]).detach().numpy().squeeze())**2/unnormalize(img[0:1, :, :]).squeeze())*100
    plt.imshow(rsmpe_error[23:, 47:-2])
    plt.clim(0, 1)
    plt.colorbar()
    plt.show()
