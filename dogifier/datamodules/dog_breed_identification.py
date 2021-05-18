import torchvision.transforms as T
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import pytorch_lightning as pl

from PIL import Image
import pandas as pd
import os
from typing import List

class DogBreedIdentification(VisionDataset):
    def __init__(self, data_dir: str, train: bool = True, transform = None):
        super(DogBreedIdentification, self).__init__(data_dir, None, transform)
        self.train = train

        df_submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
        CATEGORY_NAMES = df_submission.columns.values[1:].tolist()
        CATEGORY_NAME_TO_ID = {}
        for i, name in enumerate( CATEGORY_NAMES ):
            CATEGORY_NAME_TO_ID[name] = i

        self.image_dir = os.path.join(data_dir, "train" if train else "test")
        image_mask = [ image_name.split('.')[0] for image_name in os.listdir(self.image_dir) ]
        df_labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))
        df_labels['breed_id'] = df_labels['breed'].map( CATEGORY_NAME_TO_ID )
        
        df = df_labels[['id', 'breed_id']]
        self.df = df[df.id.isin(image_mask)]

    def __getitem__(self, index):
        item = self.df.loc[index]
        image_name = item["id"]  + ".jpg"
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path)
        target = item["breed_id"]

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.df)

class DogBreedIdentificationDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './', batch_size: int = 64, num_workers: int = 8, image_size: List[int] = [28, 28]):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.dims = (1, image_size[0], image_size[1])
        self.num_classes = 10

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            dbi_full = DogBreedIdentification(self.data_dir, train=True, transform=self.transform)
            self.dbi_train, self.dbi_val = random_split(dbi_full, [int(len(dbi_full) * 0.8), len(dbi_full) - int(len(dbi_full) * 0.8)])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.dbi_test = DogBreedIdentification(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.dbi_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dbi_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dbi_test, batch_size=self.batch_size, num_workers=self.num_workers)
