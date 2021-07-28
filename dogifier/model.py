import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
import torchvision.transforms as T
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union, List

from .metric import accuracy
from .utils.wordtree import WordTree


TIMM_MODEL_CATALOG = dict.fromkeys(timm.list_models(pretrained=True), None)
DINO_MODEL_CATALOG = {
    "dino_vits16", "dino_vits8",
    "dino_vitb16", "dino_vitb8",
    "dino_xcit_small_12_p16", "dino_xcit_small_12_p8",
    "dino_xcit_medium_24_p16", "dino_xcit_medium_24_p8",
    "dino_resnet50"
}


def build_backbone(name: str):
    if name in TIMM_MODEL_CATALOG:
        model = timm.create_model(name, pretrained=True)
    elif name in DINO_MODEL_CATALOG:
        model = torch.hub.load("facebookresearch/dino:main", name)
    else:
        raise NotImplementedError
    return model


class Dogifier(pl.LightningModule):
    def __init__(
        self,
        backbone_name: str, *,
        num_classes: Optional[int] = 1000,
        freeze: Optional[bool] = False,
        lr: Optional[float] = 0.001

    ):
        super(Dogifier, self).__init__()
        self.wordtree = WordTree()
        self.lr = lr

        # Setup network
        self.backbone = build_backbone(backbone_name)

        if isinstance(self.backbone.head, nn.Identity):
            backbone_out_features = self.backbone.num_features
        else:
            backbone_out_features = self.backbone.head.out_features

        if backbone_out_features != num_classes:
            self.head = nn.Linear(self.backbone.num_features, num_classes)
            self.head.weight.data.normal_(mean=0.0, std=0.01)
            self.head.bias.data.zero_()
        else:
            self.head = nn.Identity()

        if freeze:
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
        acc1, acc5 = accuracy(logits, y, (1, 5))

        self.log("val_loss", loss)
        self.log("val_top1_acc", acc1, prog_bar=True)
        self.log("val_top5_acc", acc5)

        metrics = {
            "loss": loss,
            "top1_acc": acc1,
            "top5_acc": acc5
        }

        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        
        self.log("test_loss", metrics["loss"])
        self.log("test_top1_acc", metrics["top1_acc"], prog_bar=True)
        self.log("test_top5_acc", metrics["top5_acc"])

    def shared_step(self, batch):
        x, y = batch
        logits = self._forward_features(x)
        loss = F.cross_entropy(logits, y)
        return loss, logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def classify(
        self,
        images: Union[Image.Image, List[Image.Image]],
        wordtree_target: Optional[str] = None
    ):
        if not isinstance(images, list):
            images = [ images ]

        transforms = T.Compose(
            [
                T.Resize((224, 224), 3),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        image_batch = [ transforms(image) for image in images ]
        image_batch = torch.stack(image_batch)
        logits = self.forward(image_batch)
        classes = torch.argmax(logits, dim=1)

        if wordtree_target:
            task = lambda x: self.wordtree.search_ancestor(x, wordtree_target)
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                classes = list(executor.map(task, classes), total=len(classes))
        return classes
