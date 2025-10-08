from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2






train_transform = A.Compose([
    A.Resize(256, 256),

    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.4),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.6),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),

    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


val_transform = A.Compose([
    A.Resize(256, 256),

    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


class AlbumentationsFlowers102(Flowers102):
    def __init__(self, root, transform=None, **kwargs):
        super().__init__(root, **kwargs)
        self.albu_transform = transform

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)

        # Convert PIL image to numpy array
        image = np.array(image)

        # Apply Albumentations
        if self.albu_transform:
            transformed = self.albu_transform(image=image)
            image = transformed['image']

        return image, label







import torch



train_dataset = AlbumentationsFlowers102(root="./data",split="test",transform=train_transform,download=False)


test_dataset = AlbumentationsFlowers102(root="./data",split="val",transform=val_transform,download=False)

val_dataset = AlbumentationsFlowers102(root="./data",split="train",transform=val_transform,download=False)




train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)



