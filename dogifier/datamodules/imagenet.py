from torch.utils.data import DataLoader
from torchvision import datasets
import os
from torchvision import transforms as T
import pytorch_lightning as pl

from typing import Any, Optional


class ImagenetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        num_workers: int = 12,
        batch_size: int = 512,
        shuffle: bool = True,
        *args: Any,
        **kwargs: Any
    ):
        super().__init__(*args, **kwargs)

        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle

    def train_dataloader(self) -> DataLoader:
        transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        dataset = datasets.ImageFolder(os.path.join(self.data_dir, "train"), transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader:
        transform = T.Compose([
            T.Resize(256, interpolation=3),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        dataset = datasets.ImageFolder(os.path.join(self.data_dir, "val"), transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)
