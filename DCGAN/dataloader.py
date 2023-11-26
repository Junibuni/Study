import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

def get_dataloader(data_dir, batch_size=128, train=True):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(root=data_dir, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)

    return dataloader