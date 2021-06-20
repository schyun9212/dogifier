import os


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


def save_ckpt_from_result(trainer, result, model_dir="outputs"):
    for metric, scalar in result.items():
        ckpt_template = parse_ckpt_template(metric)
        metric_pos = ckpt_template.find(metric)
        ckpt_template = f"epoch={ckpt_template[:metric_pos - 1]}{metric}={ckpt_template[metric_pos - 1:]}.ckpt"
        ckpt_name = ckpt_template.format(**{"epoch": 0, metric: scalar})
        ckpt_path = os.path.join(model_dir, ckpt_name)
        trainer.save_checkpoint(ckpt_path)
