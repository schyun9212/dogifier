import os
from datetime import datetime
from typing import Optional
from pytz import timezone
import subprocess


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def get_num_params(model, requires_grad=True):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad == requires_grad))


def parse_ckpt_template(target, precision=".2f"):
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
