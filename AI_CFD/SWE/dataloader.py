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
from sklearn.model_selection import train_test_split

from utils import norm_p

class CustomDataset(Dataset):
    def __init__(self, root, split="train", cnum=32, pnum=6, *, norm=True, normp=False):
        super(CustomDataset, self).__init__()
        self.cnum = cnum
        self.pnum = pnum
        self.normp = normp
        self.norm = norm
        self.file_path = os.path.join(root, "train")
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
                # if int(case_name) % 10 != 2:
                #     continue

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
                file_groups[file_number]["case_num"] = int(case_name)

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

        img = torch.from_numpy(img)
        
        p = torch.zeros(self.cnum)
        p[-self.pnum:] = torch.tensor(data["manhole_data"])
        
        if self.norm:
            norm_img = transforms.Normalize((0.0059, 0.0004, -0.0045),(0.0198, 0.0300, 0.0297))
            img = norm_img(img)
        if self.normp:
            p = norm_p(p)
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
    def __init__(self, *, dataset_root, batch_size, shuffle=True, train_val_test_split=(0.8, 0.1, 0.1), cnum=32, pnum=6, normp=False):
        super().__init__()
        self.batch_size = batch_size
        self.data = CustomDataset(dataset_root, cnum=cnum, pnum=pnum, normp=normp)

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
 
# class LinearDataSet(Dataset):
#     def __init__(self, file_list, root_dir, seqlen=30, pnum=6):
#         super(LinearDataSet, self).__init__()
#         self.root_dir = os.path.join(root_dir, "linear")
#         self.sequence_length = seqlen
#         self.pnum = pnum
#         self.file_list = file_list

#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, idx):
#         file_name = os.path.join(self.root_dir, self.file_list[idx])
#         data = np.load(file_name)  
#         num_frames = data.shape[0]

#         #start_index = np.random.randint(0, num_frames - self.sequence_length)
#         start_index = 0
#         sequence = data[start_index:start_index+self.sequence_length]
#         target = data[start_index+self.sequence_length][:-self.pnum]

#         sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
#         target_tensor = torch.tensor(target, dtype=torch.float32)

#         return sequence_tensor, target_tensor
#%%

class LinearDataSet(Dataset):
    def __init__(self, root_dir, seqlen=30, pnum=6):
        super(LinearDataSet, self).__init__()
        self.root_dir = os.path.join(root_dir, "linear")
        self.sequence_length = seqlen
        self.pnum = pnum
        
        self.data_files = os.listdir(self.root_dir)
        self.data = self.load_data()

    def load_data(self):
        data = []
        for file in self.data_files:
            file_path = os.path.join(self.root_dir, file)
            numpy_data = np.load(file_path)
            for i, _ in enumerate(numpy_data):
                idx = i + self.sequence_length
                idx_target = idx + 1
                if idx_target > len(numpy_data):
                    break
                seq_x = numpy_data[i:idx]
                seq_y = numpy_data[idx_target-1, :-self.pnum]
                data.append((seq_x, seq_y))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence, target = self.data[idx]
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)

        return sequence_tensor, target_tensor
    
class LinearDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=32, validation_split=0.2, seqlen=30, pnum=6, seed=42):
        super(LinearDataModule, self).__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.seqlen = seqlen
        self.pnum = pnum
        self.seed = seed

    def prepare_data(self):
        # Download, prepare, or preprocess data here.
        pass

    def setup(self, stage=None):
        dataset = LinearDataSet(self.root_dir, seqlen=self.seqlen, pnum=self.pnum)
        train_size = int((1 - self.validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(self.seed))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
#%%

# class LinearDataModule(pl.LightningDataModule):
#     def __init__(self, data_dir, batch_size=32, seqlen=30, pnum=6, val_size=0.3, random_state=42):
#         super().__init__()
#         self.data_dir = data_dir
#         self.batch_size = batch_size
#         self.seqlen = seqlen
#         self.pnum = pnum
#         self.val_size = val_size
#         self.random_state = random_state

#     def setup(self, stage=None):
#         file_list = os.listdir(os.path.join(self.data_dir, "linear"))
#         train_files, val_files = train_test_split(file_list, test_size=self.val_size, random_state=self.random_state)
        
#         self.train_dataset = LinearDataSet(train_files, self.data_dir, seqlen=self.seqlen, pnum=self.pnum)
#         self.val_dataset = LinearDataSet(val_files, self.data_dir, seqlen=self.seqlen, pnum=self.pnum)

#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True, shuffle=True)

#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
#%%