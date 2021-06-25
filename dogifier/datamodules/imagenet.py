from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
from pytorch_lightning import LightningDataModule

from typing import Any


class ImagenetDataModule(LightningDataModule):
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
        dataset = datasets.ImageNet(self.data_dir, split="train", transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader:
        transform = T.Compose([
            T.Resize(256, interpolation=3),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        dataset = datasets.ImageNet(self.data_dir, split="val", transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        transform = T.Compose([
            T.Resize(256, interpolation=3),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        dataset = datasets.ImageNet(self.data_dir, split="test", transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)
