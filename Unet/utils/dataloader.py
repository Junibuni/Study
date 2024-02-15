#%%
import os

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.io import read_image
from PIL import Image
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
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

        masks = list()
        categories = list()
        for an in anns:
            masks.append(torch.tensor(self.coco.annToMask(an), dtype=torch.int64))
            categories.append(an["category_id"])
        
        imageInfo = self.coco.loadImgs(imageId)[0]

        # CHW, RGB, in tensor form
        img = read_image(os.path.join(self.root, self.split, imageInfo['file_name'])).float()
        img /= 255.0

        preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.5735, 0.5618, 0.5681],
                                     std=[0.2341, 0.2348, 0.2343]),
        ])

        img = preprocess(img)
        return img, dict(mask=masks, category_id=categories)
    
    def __len__(self):
        return len(self.imgIds)

#%%
