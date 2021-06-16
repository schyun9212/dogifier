import torch
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

    def forward(self, image):
        x = self.backbone(image)
        x = F.softmax(x, dim=1)
        return x
