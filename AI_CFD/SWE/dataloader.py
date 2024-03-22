#%%
import os
import glob
import re

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import lightning.pytorch as pl
import numpy as np
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, root, split="train", znum=32, pnum=6):
        super(CustomDataset, self).__init__()
        self.znum = znum
        self.pnum = pnum
        self.file_path = os.path.join(root, split)
        cases = os.listdir(self.file_path)

        csv_data = pd.read_csv(os.path.join(root, "case_data.csv"))
        
        self.grouped_files = []
        for c in cases:
            file_groups = {}
            file_list = glob.glob(os.path.join(self.file_path, c) + "/*.asc")

            for filename in file_list:
                dirname = os.path.dirname(filename)
                basename = os.path.basename(filename) 
                n = os.path.splitext(basename)[0]
                file_type, file_number = n.split('_')
                case_name = dirname.split('/')[1]

                manhole_duration = (int(case_name)%10)*120
                if int(file_number) < manhole_duration:
                    row_data = csv_data[csv_data["Case #"] == int(case_name)]
                    assert row_data
                    manhole_data = row_data.values.tolist()[0][1:-1]
                else:
                    manhole_data = [0] * 6

                file_groups.setdefault(file_number, {}).setdefault(file_type, filename)
                file_groups[file_number]["manhole_data"] = manhole_data

            for file_number, file_dict in file_groups.items():
                self.grouped_files.append(file_dict)
        
        self.desired_order = ['depth', 'xvel', 'yvel']

    def __getitem__(self, index):
        data = self.grouped_files[index]
        img = []
        for key in self.desired_order:
            file_path = data[key]
            part_data = np.genfromtxt(file_path, skip_header=6)
            # replace nan
            part_data[np.isnan(part_data)] = 0.0
            avg_pool_data = np.pad(self._avg_pool(part_data), ((23,0), (47, 0)), mode="constant")
            img.append(avg_pool_data)

        # TODO: Normalize by channel   
        preprocess = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.0,), (0.2,))
                                        ])
        
        img = preprocess(img)
        p = torch.zeros(self.znum)
        p[-self.pnum:] = torch.tensor(data["manhole_data"])

        return img.float(), p.float()
    
    def __len__(self):
        return len(self.grouped_files)
    
    def _avg_pool(self, arr):
        #if 722, 417
        arr = np.pad(arr, ((0, 0), (0, 1)), mode='constant')
        x, y = arr.shape
        new_x, new_y = x//2, y//2
        arr = np.mean(arr.reshape(new_x, 2, new_y, 2), axis=(1, 3))
        return arr
#%%

class DataModule(pl.LightningDataModule):
    def __init__(self, *, dataset_root, batch_size, shuffle=True):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_root = dataset_root
        self.shuffle = shuffle

    def prepare_data(self):
        CustomDataset(self.dataset_root, "train")
        CustomDataset(self.dataset_root, "test")

    def setup(self, stage):
        self.train_set = CustomDataset(self.dataset_root, "train")
        self.test_set = CustomDataset(self.dataset_root, "test")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
# %%

# TODO    
class LinearDataSet(Dataset):
    pass
