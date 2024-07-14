import re
import os

import torch
from torchvision import transforms
import numpy as np
from dataloader import CustomDataset
from swe_ae import SWE_AE
from tqdm import tqdm

from utils import unnormalize
import matplotlib.pyplot as plt

ckpt_pth = r"D:\Study\AI_CFD\SWE\logs\epoch=917-step=223074.ckpt"
checkpoint = torch.load(ckpt_pth)

model_input = dict(
    optim_params = dict(lr=2e-4),
    scheduler_params = dict(),
    input_size = (1, 3, 384, 256),
    cnum = 64
)
model = SWE_AE(**model_input)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.to("cuda")

data = CustomDataset(r"AI_CFD\SWE\datasets", normp=True)

def extract_last_number(filename):
    return int(re.findall(r'\d+', filename.split('\\')[-1])[-1])

def group_and_sort_by_depth_last_number(data):
    grouped_data = {}
    for item in data:
        grouped_data.setdefault(item['case_num'], []).append(item)
    for group in grouped_data.values():
        group.sort(key=lambda x: extract_last_number(x['depth']))
    return grouped_data

sorted_data = group_and_sort_by_depth_last_number(data.grouped_files)
save_data_pth = r"AI_CFD\SWE\datasets\linear"
for i in range(1, 7):
    for j in range(1, 4):
        for k in range(1, 4):
            case_num = int(str(i)+str(j)+str(k))
            savepth = os.path.join(save_data_pth, f"{case_num}.npy")
            data.grouped_files = sorted_data[case_num]

            lvec_stack = []
            for idx in tqdm(range(len(data)), desc=f"{case_num}"):
                with torch.no_grad():
                    _img = data[idx][0].to("cuda")
                    _lv = data[idx][1].to("cuda")
                    lv = model.encoder(_img.unsqueeze(0))

                    lv = torch.concat((lv, _lv[-6:].unsqueeze(0)), dim=1)

                    a = np.array(lv.squeeze().cpu())
                    lvec_stack.append(a)
            np.save(savepth, lvec_stack)