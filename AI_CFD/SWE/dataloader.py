#%%
import os
import glob
import re

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import lightning.pytorch as pl
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, root, split="train", znum=16):
        super(CustomDataset, self).__init__()
        self.znum = znum
        self.file_path = os.path.join(root, split)
        files = glob.glob(self.file_path + "/*.asc")
        self.files = sorted(files, key=self._numerical_sort_key)

    def __getitem__(self, index):
        data = np.genfromtxt(self.files[index], skip_header=6)
        # if 716, 412
        # img = np.pad(data, ((52,0), (100, 0)), mode="constant")

        #if reduce
        img = np.pad(self.avg_pool(data), ((26,0), (50, 0)), mode="constant")
        
        preprocess = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.0,), (0.2,))
                                        ])
        img = preprocess(img)
        p = torch.zeros(self.znum)
        p[-2:] = torch.tensor([1, 1])
        return img.float(), p.float()
    
    def __len__(self):
        return len(self.files)
    
    def avg_pool(self, arr):
        x, y = arr.shape
        new_x, new_y = x//2, y//2
        arr = np.mean(arr.reshape(new_x, 2, new_y, 2), axis=(1, 3))
        return arr

    def _get_filename_from_path(self, filepath):
        return os.path.basename(filepath)
    
    def _numerical_sort_key(self, filepath):
        filename = self._get_filename_from_path(filepath)
        
        match = re.search(r'\d+', filename)
        if match:
            return int(match.group())
        else:
            return filename 
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
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
# %%
    
class LinearDataSet(Dataset):
    def __init__(self, root, split="train"):
        super(LinearDataSet, self).__init__()
        self.file_path = os.path.join(root, "linear_net")
        files = glob.glob(self.file_path + "/*.pt")
        self.files = sorted(files, key=self._numerical_sort_key)

    def __getitem__(self, index):
        data = torch.load(self.files[index])
        return data.float()
    
    def __len__(self):
        return len(self.files)
    
    def _get_filename_from_path(self, filepath):
        return os.path.basename(filepath)
    
    def _numerical_sort_key(self, filepath):
        filename = self._get_filename_from_path(filepath)
        
        match = re.search(r'\d+', filename)
        if match:
            return int(match.group())
        else:
            return filename 
# %%

class LinearDataModule(pl.LightningDataModule):
    def __init__(self, *, dataset_root, batch_size=30, shuffle=False):
        super(LinearDataModule, self).__init__()
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
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)