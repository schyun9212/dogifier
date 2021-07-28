from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .utils.common import parse_ckpt_template


def build_checkpoint_callback(model_dir, targets=[], precision=".2f", save_top_k=2):
    callbacks = []

    for target in targets:
        mode = "min" if "loss" in target else "max"

        callback = ModelCheckpoint(
            dirpath=model_dir,
            monitor=target,
            filename=parse_ckpt_template(target, precision),
            save_top_k=save_top_k,
            mode=mode
        )
        callbacks.append(callback)
    
    return callbacks


def build_early_stop_callback(monitor, min_delta=0.0, patience=10, mode="max"):
    callback = EarlyStopping(
        monitor=monitor,
        min_delta=min_delta,
        patience=patience,
        mode=mode
    )
    return [callback]
