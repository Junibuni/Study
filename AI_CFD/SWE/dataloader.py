#%%
import os
import glob
import re

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torchvision.transforms import transforms
import lightning.pytorch as pl
import numpy as np
import pandas as pd

from utils import norm_p

class CustomDataset(Dataset):
    def __init__(self, root, split="train", cnum=32, pnum=6):
        super(CustomDataset, self).__init__()
        self.cnum = cnum
        self.pnum = pnum
        self.file_path = os.path.join(root, split)
        cases = os.listdir(self.file_path)

        csv_file_path = os.path.join(root, "case_data.csv")
        assert os.path.isfile(csv_file_path), f"The csv file({csv_file_path}) does not exist."
        csv_data = pd.read_csv(csv_file_path)
        
        self.grouped_files = []
        for c in cases:
            file_groups = {}
            file_list = glob.glob(os.path.join(self.file_path, c) + "/*.asc")

            for filename in file_list:
                dirname = os.path.dirname(filename)
                basename = os.path.basename(filename) 
                n = os.path.splitext(basename)[0]
                file_type, file_number = n.split('_')
                case_name = os.path.split(dirname)[1]
                if int(case_name) % 10 != 2:
                    continue

                manhole_duration = (int(case_name)%10)*120
                if int(file_number) < manhole_duration:
                    row_data = csv_data[csv_data["Case #"] == int(case_name)]
                    assert not row_data.empty, f"data not found on csv file for {case_name}"
                    manhole_data = row_data.values.tolist()[0][1:-1]
                else:
                    manhole_data = [0] * pnum
                assert (len(manhole_data) == pnum), f"length of pnum({pnum}) does not match csv data({len(manhole_data)})"

                file_groups.setdefault(file_number, {}).setdefault(file_type, filename)
                file_groups[file_number]["manhole_data"] = manhole_data

            for file_number, file_dict in file_groups.items():
                self.grouped_files.append(file_dict)
        
        self.desired_order = ['depth', 'xvel', 'yvel']
        #self.grouped_files = np.random.choice(self.grouped_files, size=int(len(self.grouped_files)*0.6), replace=False)
    def __getitem__(self, index):
        data = self.grouped_files[index]
        img = []
        for key in self.desired_order:
            file_path = data[key]
            part_data = np.genfromtxt(file_path, skip_header=6)
            # replace nan
            part_data[np.isnan(part_data)] = 0.0
            avg_pool_data = self._avg_pool(part_data)
            img.append(avg_pool_data)
        img = np.pad(img, ((0,0), (23,0), (47, 0)), mode="constant")
        # TODO: Normalize by channel   
        norm_img = transforms.Normalize((0.0059, 0.0004, -0.0045),(0.0198, 0.0300, 0.0297))

        img = torch.from_numpy(img)
        
        p = torch.zeros(self.cnum)
        p[-self.pnum:] = torch.tensor(data["manhole_data"])
        
        img = norm_img(img)
        #TODO: norm_p ?
        #p = norm_p(p)
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
    def __init__(self, *, dataset_root, batch_size, shuffle=True, train_val_test_split=(0.8, 0.1, 0.1), cnum=32, pnum=6):
        super().__init__()
        self.batch_size = batch_size
        self.data = CustomDataset(dataset_root, cnum=cnum, pnum=pnum)

        self.shuffle = shuffle
        self.train_val_test_split = train_val_test_split

    def setup(self, stage):
        train_size = int(self.train_val_test_split[0] * len(self.data))
        val_size = int(self.train_val_test_split[1] * len(self.data))
        test_size = len(self.data) - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.data, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
# %%

# TODO    
class LinearDataSet(Dataset):
    pass
