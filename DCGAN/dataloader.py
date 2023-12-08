import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

def get_dataloader(data_dir, dataset, batch_size=128, train=True):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if dataset == "cifar10":
        dataset = datasets.CIFAR10(root=data_dir, download=True, transform=transform)
    elif dataset == "celeba":
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    else:
        raise ValueError("Use Valid Dataset")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)

    return dataloader