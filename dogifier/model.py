import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm

import dogifier.utils as utils


def build_timm_model(timm_cfg):
    model = timm.create_model(timm_cfg.NAME, pretrained=True)
    return model


def build_dino_model(dino_cfg):
    model = torch.hub.load("facebookresearch/dino:main", dino_cfg.NAME)
    return model


class Dogifier(pl.LightningModule):
    def __init__(self, model_cfg):
        super(Dogifier, self).__init__()
        self.cfg = model_cfg
        
        if self.cfg.TYPE == "timm":
            self.backbone = build_timm_model(self.cfg.TIMM)
        elif self.cfg.TYPE == "dino":
            self.backbone = build_dino_model(self.cfg.DINO)
        else:
            raise NotImplementedError

        if self.backbone.num_features != self.cfg.NUM_LABELS:
            self.head = nn.Linear(self.backbone.num_features, model_cfg.NUM_LABELS)
            self.head.weight.data.normal_(mean=0.0, std=0.01)
            self.head.bias.data.zero_()
        else:
            self.head = nn.Identity()
        
        self._freeze()
    
    def _freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _forward_features(self, image):
        x = self.backbone(image)
        x = self.head(x)
        return x

    def forward(self, image):
        x = self._forward_features(image)
        x = F.softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        _, y = batch
        loss, logits = self.shared_step(batch)
        acc1, acc5 = utils.accuracy(logits, y, (1, 5))

        self.log("val_loss", loss)
        self.log("val_top1_acc", acc1)
        self.log("val_top5_acc", acc5)

    def shared_step(self, batch):
        x, y = batch
        logits = self._forward_features(x)
        loss = F.cross_entropy(logits, y)
        return loss, logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.LR)
        return optimizer
