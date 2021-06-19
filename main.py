import argparse
import os

from pytorch_lightning import Trainer, callbacks

from dogifier.datamodules import ImagenetDataModule
from dogifier.config import get_cfg
from dogifier.model import Dogifier
from dogifier.checkpoint import build_checkpoint_callback


def main(args) -> None:
    cfg = get_cfg()
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)

    model = Dogifier(cfg.MODEL)
    model.eval()

    dm = ImagenetDataModule("/home/appuser/datasets/ImageNet")
    model_dir = os.path.join(args.model_root, args.expr)
    checkpoint_callbacks = build_checkpoint_callback(model_dir, ["val_loss", "val_top1_acc", "val_top5_acc"])
    trainer = Trainer(
        gpus=1,
        auto_lr_find=True,
        callbacks = checkpoint_callbacks
    )
        
    if args.mode == "fit":
        trainer.fit(model, dm)
    elif args.mode == "test":
        trainer.test(model, dm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default=None)
    parser.add_argument("--model-root", type=str, default="/home/appuser/models")
    parser.add_argument("--expr", type=str, required=True)
    parser.add_argument("--mode", choices=("fit", "test"), required=True)
    args = parser.parse_args()
    
    main(args)
