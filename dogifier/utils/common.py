import os
from datetime import datetime
from typing import Optional
from pytz import timezone
import subprocess
import torch.nn as nn


def get_num_params(model: nn.Module, requires_grad: Optional[bool] = True) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad == requires_grad))


def parse_ckpt_template(target: str, precision: Optional[str] = ".2f") -> str:
    return "{epoch}-{" + target + ":" + precision + "}"


def get_datetime(
    tz: Optional[str] = "Asia/Seoul",
    dt_format: Optional[str] = "%Y%m%d%H%M%S"
) -> str:
    dt = datetime.now(timezone(tz))
    dt = dt.strftime(dt_format)
    return dt


def get_git_revision_hash(dst_dir: Optional[str] = None) -> str:
    working_dir = os.getcwd()
    if dst_dir:
        os.chdir(dst_dir)
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode()
    os.chdir(working_dir)
    return git_hash


def get_git_revision_short_hash(dst_dir: Optional[str] = None) -> str:
    working_dir = os.getcwd()
    if dst_dir:
        os.chdir(dst_dir)
    git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode()
    os.chdir(working_dir)
    return git_hash
