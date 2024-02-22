#%%
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.io import read_image
import torchvision.transforms.functional as TF
from pycocotools.coco import COCO
import lightning.pytorch as pl
import numpy as np

class MyCocoDataset(Dataset):
    def __init__(self, root, split="train"):
        super(MyCocoDataset, self).__init__()
        annoFile = os.path.join(root, split, "_annotations.coco.json")

        self.root = root
        self.split = split
        self.coco = COCO(annoFile)
        self.catIds = self.coco.getCatIds()
        self.imgIds = self.coco.getImgIds()

    def __getitem__(self, index):
        imageId = self.imgIds[index]
        annIds = self.coco.getAnnIds(imgIds = imageId)
        anns = self.coco.loadAnns(annIds)
        
        imageInfo = self.coco.loadImgs(imageId)[0]
        mask = self.poly_to_mask(imageInfo, anns)

        # CHW, RGB, in tensor form
        img = read_image(os.path.join(self.root, self.split, imageInfo['file_name'])).float()
        img /= 255.0

        preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.5735, 0.5618, 0.5681],
                                     std=[0.2341, 0.2348, 0.2343]),
        ])

        img = preprocess(img)
        return img, mask
    
    def __len__(self):
        return len(self.imgIds)
    
    def poly_to_mask(self, img_info, anns):
        anns_img = torch.zeros((img_info['height'],img_info['width']))
        for ann in anns:
            anns_img = torch.maximum(anns_img, self.coco.annToMask(ann)*ann['category_id'])
        
        return anns_img.to(torch.int64)

#%%
def torch_divmod(dividend:torch.Tensor, divisor:int):
    quotient = dividend.div(divisor, rounding_mode="floor")
    remainder = torch.remainder(dividend, divisor)
    return quotient, remainder

def collate_fn(batch):
    images, masks = zip(*batch)

    max_height = max(img.size()[1] for img in images)
    max_width = max(img.size()[0] for img in images)

    pad_height = max_height - torch.tensor([img.size()[1] for img in images])
    pad_width = max_width - torch.tensor([img.size()[0] for img in images])

    h_quotient, h_remainder = torch_divmod(pad_width, 2)
    v_quotient, v_remainder = torch_divmod(pad_height, 2)

    pad_left = h_quotient
    pad_right = h_quotient + h_remainder
    pad_top = v_quotient
    pad_bottom = v_quotient + v_remainder

    padded_images = []
    padded_masks = []

    for i, image_mask in enumerate(zip(images, masks)):
        img, mask = image_mask
        padding = (pad_left[i], pad_top[i], pad_right[i], pad_bottom[i])
        padded_img = TF.pad(img, padding, fill=0)
        padded_mask = TF.pad(mask, padding, fill=0)

        padded_images.append(padded_img)
        padded_masks.append(padded_mask)

    return torch.stack(padded_images), torch.stack(padded_masks)

class DataModule(pl.LightningDataModule):
    def __init__(self, *, dataset_root, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_root = dataset_root

    def prepare_data(self):
        MyCocoDataset(self.dataset_root, "train")
        MyCocoDataset(self.dataset_root, "valid")
        MyCocoDataset(self.dataset_root, "test")

    def setup(self, stage):
        self.train_set = MyCocoDataset(self.dataset_root, "train")
        self.valid_set = MyCocoDataset(self.dataset_root, "valid")
        self.test_set = MyCocoDataset(self.dataset_root, "test")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True)
# %%