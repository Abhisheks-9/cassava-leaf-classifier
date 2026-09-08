import os
import numpy as np
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
import torch
from torch.utils.data import Dataset

from src import config


def get_transforms(split="val"):
    if split == "train":
        return albumentations.Compose([
            albumentations.Resize(config.WIDTH, config.HEIGHT),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=(-90, 90)),
            albumentations.Normalize(config.MEAN, config.STD, max_pixel_value=255.0),
            ToTensorV2(),
        ])
    return albumentations.Compose([
        albumentations.Resize(config.WIDTH, config.HEIGHT),
        albumentations.Normalize(config.MEAN, config.STD, max_pixel_value=255.0),
        ToTensorV2(),
    ])


class CassavaDataset(Dataset):
    def __init__(self, image_ids, labels, augmentations=None, folder="train_images"):
        self.image_ids = image_ids
        self.labels = labels
        self.augmentations = augmentations
        self.folder = folder

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_path = os.path.join(config.BASE_DIR, self.folder, self.image_ids[idx])
        img = np.array(Image.open(img_path).convert("RGB"))

        if self.augmentations:
            img = self.augmentations(image=img)["image"]

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label
