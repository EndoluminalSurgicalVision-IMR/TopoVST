import os
from typing import Dict, Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import CosineSimilarity, MSELoss, BCELoss
from monai.utils import set_determinism

from src.utils.geometry import IcoSphere
from src.utils.spherical_graph_utils import LocalMaxEvaluator


class AdaSIREMultiTaskMultiLossTracker(pl.LightningModule):
    """
    AdaSIRE pytorch lightning wrapper for multi-task (direction & radius) multi
    loss network.
    """

    def __init__(
        self,
        **kwargs
    ):

        super().__init__()
        self.sphere = IcoSphere(subdivisions=kwargs["subdivisions"])
        set_determinism(seed=kwargs["seed"])
        self.save_hyperparameters()

        self.lr = kwargs["lr"]
        self.optimizer, self.lr_scheduler = kwargs["optimizer"], kwargs["lr_scheduler"]
        self.lr_scheduler_params = kwargs["lr_scheduler_params"]
        self.model = kwargs["model"](**kwargs["model_params"])

        self.d_loss = MSELoss(reduction="mean")
        self.r_loss = MSELoss(reduction="mean")
        self.cosine_similarity = CosineSimilarity()

    def get_direction(self, direction_heatmap: torch.tensor):
        """ Gets the direction with maximum response among all vertices. """

        ind = torch.argmax(direction_heatmap, dim=1).detach().cpu().numpy()
        direction = torch.tensor(self.sphere.cartverts[ind, :]).reshape(-1, 3)

        return direction

    def forward(self, data):
        """
        Return multi-scale predictions.
        """

        nverts = data["global"]["nverts"][0].item()
        dir_spheres, radii = self.model(
            data["sample"]["spheres"], nverts=nverts)
        dirs_vertex_max, _ = torch.max(dir_spheres, dim=1)

        return dir_spheres, radii.squeeze(-1), dirs_vertex_max

    def shared_step(self, data, stage):

        d_scales, r, d_vertmax = self(data)
        d_scale_label, r_labels = data["sample"]["dir_target_perscale"], \
            data["sample"]["radii_target"]
        d_vertmax_label = data["sample"]["dir_target"]
        radius = data["sample"]["radius"]
        s_scales = data["global"]["scales"]

        # Loss computation
        dir_loss_perscale = self.d_loss(d_scales, d_scale_label)
        radii_loss_perscale = self.r_loss(r, r_labels)
        dir_loss_vertmax = self.d_loss(d_vertmax, d_vertmax_label)

        alpha = 10.0
        beta = 1.0
        gamma = 10.0
        loss = alpha * dir_loss_perscale + beta * dir_loss_vertmax + \
            gamma * radii_loss_perscale

        with torch.no_grad():
            cosine_similarity = torch.abs(self.cosine_similarity(
                self.get_direction(d_vertmax.squeeze()).view(-1, 3),
                self.get_direction(d_vertmax_label.squeeze()).view(-1, 3)
            )).mean()
            valid_r_inds = torch.argwhere((r > 0.2) & (r < 0.8))
            valid_r_mask = torch.zeros_like(r).float()
            valid_r_mask[valid_r_inds] = 1.0
            r_estimate = torch.sum(s_scales * r * valid_r_mask, dim=1) / \
                torch.sum(valid_r_mask, dim=1)
            phy_r_loss = self.r_loss(r_estimate, radius)
            self.log(f"{stage}/scaledir_mse_loss", dir_loss_perscale,
                     on_epoch=False, on_step=True, logger=True, batch_size=1)
            self.log(f"{stage}/vertexdir_mse_loss", dir_loss_vertmax,
                     on_epoch=False, on_step=True, logger=True, batch_size=1)
            self.log(f"{stage}/radii_mse_loss", radii_loss_perscale,
                     on_epoch=False, on_step=True, logger=True, batch_size=1)
            self.log(f"{stage}/phy_radii_mse_loss", phy_r_loss,
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


class AdaSIREMultiTaskScaleFusionTracker(pl.LightningModule):
    """
    AdaSIRE pytorch lightning wrapper for multi-task (direction & radius) scale
    fusion network.
    """

    def __init__(
        self,
        **kwargs
    ):

        super().__init__()
        self.sphere = IcoSphere(subdivisions=kwargs["subdivisions"])
        set_determinism(seed=kwargs["seed"])
        self.save_hyperparameters()

        self.optimizer = kwargs["optimizer"]
        self.lr_scheduler = kwargs["lr_scheduler"]
        self.opt_params = kwargs["optimizer_params"]
        self.lr_params = kwargs["lr_scheduler_params"]
        self.model: nn.Module = kwargs["model"](**kwargs["model_params"])

        self.d_loss = kwargs["loss"](**kwargs["loss_params"])
        self.d_alpha = kwargs["direction_loss_alpha"]
        self.r_loss = MSELoss(reduction="mean")
        self.evaluator = LocalMaxEvaluator(subdivisions=kwargs["subdivisions"],
                                           device="cpu")

    def forward(self, graph, **kwds):
        """
        Return direction and radius predictions.
        """

        nverts = len(self.sphere.cartverts)
        save_features = kwds.get("save_features", False)
        directions, radius, weights, features = self.model(
            graph, nverts=nverts, save_features=save_features)

        if not save_features:
            return directions, radius, weights
        else:
            return directions, radius, weights, features

    def shared_step(self, data, stage):

        graph, d_target, aug, index, r_target = data
        bs = len(index)
        d, r, w = self(graph)
        r = r * ~aug
        r_target = r_target * ~aug

        # Loss computation: Standard BCE loss and MSE loss
        d_loss = self.d_loss(d, d_target, step=self.global_step)
        r_loss = self.r_loss(r, r_target)
        loss = self.d_alpha * d_loss + r_loss

        with torch.no_grad():
            eval_res = self.evaluator(d, d_target)
            self.log(f"{stage}/cosine_similarity", eval_res["cosine_similarity"],
                     on_epoch=False, on_step=True, logger=True, batch_size=bs)
            self.log(f"{stage}/num_response_err", eval_res["num_response_err"],
                     on_epoch=False, on_step=True, logger=True, batch_size=bs)
            self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"],
                     on_epoch=False, on_step=True, logger=True, batch_size=bs)
            self.log(f"{stage}/direction_loss", d_loss,
                     on_epoch=False, on_step=True, logger=True, batch_size=bs)
            self.log(f"{stage}/radius_loss", r_loss,
                     on_epoch=False, on_step=True, logger=True, batch_size=bs)
            self.log(f"{stage}/loss", loss,
                     on_epoch=False, on_step=True, logger=True, batch_size=bs)

        return loss

    def training_step(self, data, batch_idx):

        return self.shared_step(data, stage="train")

    def validation_step(self, data, batch_idx):

        return self.shared_step(data, stage="val")

    def configure_optimizers(self):

        optimizer = self.optimizer(self.model.parameters(), **self.opt_params)
        lr_scheduler = self.lr_scheduler(optimizer, **self.lr_params)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
            },
        }
