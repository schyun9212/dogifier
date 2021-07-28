import os
from datetime import datetime
from typing import Optional
from pytz import timezone
import subprocess
import json
import torch.nn as nn


def get_num_params(model: nn.Module, requires_grad: Optional[bool] = True):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad == requires_grad))


def parse_ckpt_template(target: str, precision: Optional[str] = ".2f"):
    return "{epoch}-{" + target + ":" + precision + "}"


def get_datetime(
    tz: Optional[str] = "Asia/Seoul",
    dt_format: Optional[str] = "%Y%m%d%H%M%S"
):
    dt = datetime.now(timezone(tz))
    dt = dt.strftime(dt_format)
    return dt


def get_git_revision_hash(dst_dir: Optional[str] = None):
    working_dir = os.getcwd()
    if dst_dir:
        os.chdir(dst_dir)
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode()
    os.chdir(working_dir)
    return git_hash


def get_git_revision_short_hash(dst_dir: Optional[str] = None):
    working_dir = os.getcwd()
    if dst_dir:
        os.chdir(dst_dir)
    git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode()
    os.chdir(working_dir)
    return git_hash


def build_imagenet_class_map():
    with open("data/imagenet_class_index.json", 'r') as f:
        imagenet_class_idx = json.load(f)
        imagenet_idx_to_class = [ item[1] for item in imagenet_class_idx.values() ]
    return imagenet_idx_to_class
