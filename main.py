import os

import hydra
from pytorch_lightning import Trainer
from omegaconf import DictConfig

from dogifier.datamodules.build import build_datamodule
from dogifier.model import Dogifier
from dogifier.callback import build_checkpoint_callback, build_early_stop_callback
from dogifier.utils import get_num_params, save_ckpt_from_result, get_expr_name


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    model = Dogifier(cfg.model)
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
