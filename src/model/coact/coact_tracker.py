import os
from typing import Dict, Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from monai.utils import set_determinism

from src.utils.geometry import IHPSphere


class COACTTracker(pl.LightningModule):

    def __init__(
        self,
        level: int = 4,
        lr: float = 0.001,
        seed: int = 0,
        **kwargs
    ):

        super().__init__()
        self.sphere = IHPSphere(subdivision=level)

        set_determinism(seed)
        self.save_hyperparameters()

        self.lr = lr
        self.optimizer, self.lr_scheduler = kwargs["optimizer"], kwargs["lr_scheduler"]
        self.lr_scheduler_params = kwargs["lr_scheduler_params"]
        self.model = kwargs["model"](**kwargs["model_params"])

        self.bce_loss = nn.BCELoss(reduction="none")
        self.cosine_similarity = nn.CosineSimilarity()

    def get_direction(self, direction_heatmap: torch.Tensor):
        """ Gets the direction with maximum response among all vertices. """

        ind = torch.argmax(direction_heatmap, dim=1).detach().cpu().numpy()
        direction = torch.tensor(self.sphere.cartverts[ind, :]).reshape(-1, 3)

        return direction

    def forward(self, data: Dict | torch.Tensor, **kwds) -> torch.Tensor:

        if isinstance(data, Dict):
            bs = len(data["sample"]["index"])
            pred: torch.Tensor = self.model(data["sample"]["sphere"])
        else:
            bs = kwds["bs"]
            pred: torch.Tensor = self.model(data)
        pred = pred.view(bs, -1)

        return pred

    def shared_step(self, data, stage):

        index, graph, label = data

        pred: torch.Tensor = self(graph, bs=len(index)).float()
        label: torch.Tensor = label.float()
        bs = len(index)

        # Compute loss
        # NOTE: softmax is applied according to original paper. NOT working!
        # pred = torch.nn.functional.softmax(pred, dim=-1)
        prob = torch.nn.functional.sigmoid(pred)
        regularization = torch.norm(prob, p=1)
        bce_loss = self.bce_loss(prob, label).view(bs, -1)
        with torch.no_grad():
            weights = torch.zeros_like(bce_loss)
            pos_w = 1.0 / torch.sum(label)
            neg_w = 1.0 / torch.sum(1 - label)
            weights[label == 1] = pos_w
            weights[label == 0] = neg_w
            weights = weights.to(bce_loss.device)
        bce_loss = torch.mean(bce_loss * weights)
        loss = regularization + bce_loss

        # Compute matching and direction similarities
        with torch.no_grad():
            cosine_similarity = torch.abs(self.cosine_similarity(
                self.get_direction(pred).view(-1, 3),
                self.get_direction(label).view(-1, 3)
            )).mean()

            self.log(f"{stage}/bce_loss", bce_loss,
                     on_epoch=False, on_step=True, logger=True, batch_size=1)
            self.log(f"{stage}/L1_norm_avg", regularization,
                     on_epoch=False, on_step=True, logger=True, batch_size=1)
            self.log(f"{stage}/cosine_similarity", cosine_similarity,
                     on_epoch=False, on_step=True, logger=True, batch_size=1)

        return loss

    def training_step(self, data, batch_idx):

        return self.shared_step(data, stage="train")

    def validation_step(self, data, batch_idx):

        return self.shared_step(data, stage="val")

    def configure_optimizers(self):

        optimizer = self.optimizer(self.model.parameters(), lr=self.lr)
        lr_scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_params)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
