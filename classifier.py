import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from collections import OrderedDict

class Classifier(pl.LightningModule):
    def __init__(self, model_name='resnext101_32x8d', num_classes=1):
        super(Classifier, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)

    def forward(self, image):
        x = self.backbone(image)
        x = F.softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch

        logits = self(x)
        loss = F.nll_loss(logits, y)
        tqdm_dict = { "loss": loss }
        output = OrderedDict({
            "loss": loss,
            "progress_bar": tqdm_dict,
            "log": tqdm_dict
        })
        return output
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002)
        return [optimizer], []
