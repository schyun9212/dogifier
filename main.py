import os

import hydra
from pytorch_lightning import Trainer

from omegaconf import DictConfig
from pytorch_lightning.core import datamodule

from dogifier.datamodules.build import build_datamodule
from dogifier.model import Dogifier
from dogifier.checkpoint import build_checkpoint_callback
from dogifier.utils import get_num_params, save_ckpt_from_result


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    model = Dogifier(cfg.model)
    model.eval()

    dm = build_datamodule(cfg.datamodule)
    model_dir = os.path.join(cfg.model_root, cfg.experiment)
    checkpoint_callbacks = build_checkpoint_callback(model_dir, ["val_loss", "val_top1_acc", "val_top5_acc"])
    trainer = Trainer(
        gpus=1,
        auto_lr_find=True,
        callbacks=checkpoint_callbacks
    )

    if cfg.mode == "fit":
        if get_num_params(model, requires_grad=True) > 0:
            trainer.fit(model, datamodule=dm)
        else:
            val_result = trainer.validate(model, datamodule=dm)[0]
            save_ckpt_from_result(trainer, val_result, model_dir)
    elif cfg.model == "val":
        trainer.validate(model, datamodule=dm)
    elif cfg.mode == "test":
        trainer.test(model, dataomodule=dm)


if __name__ == "__main__":
    main()
