import os

import hydra
from pytorch_lightning import Trainer
from omegaconf import DictConfig

from dogifier.datamodules.build import build_datamodule
from dogifier.model import Dogifier
from dogifier.callback import build_checkpoint_callback, build_early_stop_callback


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    model = Dogifier(**cfg.model)
    model.eval()

    model_dir = os.getcwd()
    dm = build_datamodule(cfg.datamodule)
    checkpoint_callbacks = build_checkpoint_callback(model_dir, ["val_top1_acc", "val_top5_acc"])
    early_stop_callbacks = build_early_stop_callback("val_top1_acc")
    callbacks = checkpoint_callbacks + early_stop_callbacks
    trainer = Trainer(
        **cfg.trainer,
        callbacks=callbacks
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
