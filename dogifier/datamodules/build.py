from .imagenet import ImagenetDataModule


CATALOG = {
    "ImageNet": ImagenetDataModule
}


def build_datamodule(datamodule_cfg):
    return CATALOG[datamodule_cfg.name](**datamodule_cfg)
