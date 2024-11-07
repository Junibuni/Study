import os

from PIL import Image, ImageSequence
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# https://www.kaggle.com/datasets/sayehkargari/disney-characters-dataset/data
class CharacterDataset(Dataset):
    def __init__(self, data_dir, image_size=(128, 128), transform=None):
        self.data_dir = data_dir
        self.image_size = image_size
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp'))]

    def __len__(self):
        return len(self.image_files)

    def _load_image(self, image_path):
        with Image.open(image_path) as img:
            if img.format == 'GIF':
                img = next(ImageSequence.Iterator(img))
            if img.mode == "P" or (img.mode == "RGBA" and 'transparency' in img.info):
                img = img.convert("RGBA")
            elif img.mode != "RGB":
                img = img.convert("RGB")
            return self.transform(img)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = self._load_image(img_path)
        return image

def get_dataloader(data_dir, batch_size=32, image_size=(128, 128), shuffle=True, num_workers=4):
    dataset = CharacterDataset(data_dir=data_dir, image_size=image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_dataloader(data_dir, batch_size=32, image_size=(128, 128), shuffle=True, num_workers=4):
    dataset = CharacterDataset(data_dir=data_dir, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
