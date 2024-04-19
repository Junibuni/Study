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

ckpt_pth = r"AI_CFD\SWE\logs\CSVLogger\detach\checkpoints\epoch=9-step=12960.ckpt"
checkpoint = torch.load(ckpt_pth)

model_input = dict(
    optim_params = dict(lr=2e-4),
    scheduler_params = dict(),
    input_size = (1, 3, 384, 256)
)
model = SWE_AE(**model_input)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.to("cuda")

data = CustomDataset(r"AI_CFD\SWE\datasets")

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
for i in range(6):
    for j in range(3):
        for k in range(3):
            case_num = int(str(i)+str(j)+str(k))
            savepth = os.path.join(save_data_pth, f"{case_num}.npy")
            data.grouped_files = sorted_data[case_num]

            lvec_stack = []
            for i in tqdm(range(len(data))):
                with torch.no_grad():
                    _img = data[i][0].to("cuda")
                    # _lv = data[i][1].to("cuda")
                    #depend on the model , whether concat or modify the last 6
                    lv = model.encoder(_img.unsqueeze(0))
                    a = np.array(lv.squeeze().cpu())
                    lvec_stack.append(a)
            np.save(savepth, lvec_stack)