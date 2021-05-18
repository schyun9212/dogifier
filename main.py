from omegaconf import DictConfig, OmegaConf
import hydra
import pytorch_lightning as pl
from torchvision.transforms.transforms import Resize
import os
from PIL import Image

import torch
import torchvision.transforms as T
import json

from dogifier.datamodules.dog_breed_identification import DogBreedIdentificationDataModule
from dogifier.classifier import Classifier

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # dm = MNISTDataModule(**cfg.datamodule)
    # model = Classifier("vit_base_patch16_224")
    # model = Classifier("resnext101_32x8d")
    dm = DogBreedIdentificationDataModule("/home/appuser/datasets/dog-breed-identification")
    model = Classifier("efficientnet_l2")
    model.eval()

    trainer = pl.Trainer(**cfg.trainer)
    trainer.fit(model, datamodule=dm)
    image_dir = "/home/appuser/datasets/dogNet"
    transforms = T.Compose(
        [
            T.Resize((224, 224), 3),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    with open(os.path.join(hydra.utils.get_original_cwd(), "imagenet_class_index.json"), 'r') as f:
        imagenet_class_idx = json.load(f)
        imagenet_idx_to_class = [ item[1] for item in imagenet_class_idx.values() ]

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path)
        image = transforms(image)
        image = image.unsqueeze(0)
        result = model(image)
        result = torch.argmax(result)
        print(imagenet_idx_to_class[int(result)])

if __name__ == "__main__":
    main()
