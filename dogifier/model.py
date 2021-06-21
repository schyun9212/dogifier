import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm

import dogifier.utils as utils


def build_backbone(backbone_cfg):
    if backbone_cfg.type == "timm":
        model = timm.create_model(backbone_cfg.name, pretrained=True)
    elif backbone_cfg.type == "dino":
        model = torch.hub.load("facebookresearch/dino:main", backbone_cfg.name)
    else:
        raise NotImplementedError
    return model


class Dogifier(pl.LightningModule):
    def __init__(self, model_cfg):
        super(Dogifier, self).__init__()
        self.cfg = model_cfg
        
        self.backbone = build_backbone(self.cfg.backbone)

        if isinstance(self.backbone.head, nn.Identity):
            backbone_out_features = self.backbone.num_features
        else:
            backbone_out_features = self.backbone.head.out_features

        if backbone_out_features != self.cfg.num_labels:
            self.head = nn.Linear(self.backbone.num_features, self.cfg.num_labels)
            self.head.weight.data.normal_(mean=0.0, std=0.01)
            self.head.bias.data.zero_()
        else:
            self.head = nn.Identity()
        
        if self.cfg.freeze_backbone:
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        return optimizer
