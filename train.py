import os

import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from dogifier.datamodules.build import build_datamodule
from dogifier.model import Dogifier
from dogifier.callback import build_checkpoint_callback, build_early_stop_callback


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    pl_logger = TensorBoardLogger(os.getcwd(), "")
    os.makedirs(pl_logger.log_dir, exist_ok=True)
    with open(os.path.join(pl_logger.log_dir, "config.yaml"), 'w') as f:
        OmegaConf.save(cfg, f)

    dm = build_datamodule(cfg.datamodule)
    checkpoint_callbacks = build_checkpoint_callback(pl_logger.log_dir, ["val_top1_acc", "val_top5_acc"])
    early_stop_callbacks = build_early_stop_callback("val_top1_acc")
    callbacks = checkpoint_callbacks + early_stop_callbacks

    trainer = Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        logger=pl_logger
    )

    if os.path.isfile(cfg.ckpt_path):
        model = Dogifier.load_from_checkpoint(cfg.ckpt_path)
    else:
        model = Dogifier(**cfg.model)
    model.eval()

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
