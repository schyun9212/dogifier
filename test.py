from pytorch_lightning import Trainer
from omegaconf import DictConfig, OmegaConf
from typing import Optional

from dogifier.datamodules.build import build_datamodule
from dogifier.model import Dogifier


def main(cfg: DictConfig, weight_file: Optional[str] = None) -> None:
    if weight_file:
        model = Dogifier.load_from_checkpoint(weight_file)
    else:
        model = Dogifier(cfg.model)
    model.eval()

    dm = build_datamodule(cfg.datamodule)
    trainer = Trainer(**cfg.trainer)
    trainer.test(model, dataomodule=dm)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--weight-file", type=str, required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_file)
    main(cfg, args.weight_file)
