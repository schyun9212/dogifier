import argparse
import pytorch_lightning as pl

from dogifier.datamodules import ImagenetDataModule
from dogifier.config import get_cfg
from dogifier.model import Dogifier


def main(args) -> None:
    cfg = get_cfg()
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)

    model = Dogifier(cfg.MODEL)
    model.eval()

    dm = ImagenetDataModule("/home/appuser/datasets/ImageNet")
    trainer = pl.Trainer(gpus=1)

    if args.mode == "fit":
        trainer.fit(model, dm)
    elif args.mode == "test":
        trainer.test(model, dm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default=None)
    parser.add_argument("--mode", choices=("fit", "test"), required=True)
    args = parser.parse_args()
    
    main(args)
