import os

from .imagenet import ImagenetDataModule


CATALOG = {
    "ILSVRC2012": ImagenetDataModule, # ImageNet-1k Dataset
}


def build_datamodule(datamodule_cfg):
    data_dir = datamodule_cfg.data_dir
    dataset_name = os.path.basename(data_dir)
    return CATALOG[dataset_name](**datamodule_cfg)
