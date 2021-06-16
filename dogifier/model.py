import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm


def build_timm_model(timm_cfg):
    model = timm.create_model(timm_cfg.NAME, pretrained=True)
    return model


def build_dino_model(dino_cfg):
    model = torch.hub.load("facebookresearch/dino:main", dino_cfg.NAME)
    return model


class Dogifier(pl.LightningModule):
    def __init__(self, model_cfg):
        super(Dogifier, self).__init__()
        
        if model_cfg.TYPE == "timm":
            self.backbone = build_timm_model(model_cfg.TIMM)
        elif model_cfg.TYPE == "dino":
            self.backbone = build_dino_model(model_cfg.DINO)
        else:
            raise NotImplementedError

        if self.backbone.num_features != model_cfg.NUM_LABELS:
            self.linear = nn.Linear(self.backbone.num_features, model_cfg.NUM_LABELS)
            self.linear.weight.data.normal_(mean=0.0, std=0.01)
            self.linear.bias.data.zero_()
        else:
            self.linear = None

    def forward(self, image):
        x = self.backbone(image)
        if self.linear is not None:
            x = self.linear(x)
        x = F.softmax(x, dim=1)
        return x
