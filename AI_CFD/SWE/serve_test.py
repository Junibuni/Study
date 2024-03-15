import torch
from torchvision import transforms
import numpy as np
from dataloader import CustomDataset
from swe_ae import SWE_AE

from utils import unnormalize

ckpt_pth = r"AI_CFD\SWE\logs\CSVLogger\test\checkpoints\epoch=62-step=5355.ckpt"
checkpoint = torch.load(ckpt_pth)

model_input = dict(
    optim_params = dict(lr=1e-4),
    criterion_params = dict(), 
    scheduler_params = dict(T_max=100),
    input_size = (1, 1, 384, 256)
)
model = SWE_AE(**model_input)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

data = CustomDataset(r"C:\Users\20230008\Documents\Study\AI_CFD\SWE\datasets")

img = data[264]


import matplotlib.pyplot as plt
print(img.shape)
plt.imshow(img.permute(1, 2, 0))
plt.show()

out = model(img.unsqueeze(0))
print(out, out.shape)
out_numpy = out.detach().numpy().squeeze()
plt.imshow(out_numpy)
plt.show()



mse_error = np.sqrt((unnormalize(out).detach().numpy().squeeze() - unnormalize(img).detach().numpy().squeeze())**2)
plt.imshow(mse_error)
plt.colorbar()
plt.show()
