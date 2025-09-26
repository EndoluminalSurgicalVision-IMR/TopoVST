import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

# Needs to be built from source
from gem_cnn.nn.gem_res_net_block import GemResNetBlock
from torch.nn import Sequential

from src.model.networks.special_layers import (
    MLP,
    ScaleAttnFusionLayer,
    ScaleMLPFusionLayer,
    ScaleFusionTransformer,
    ScaleSimpleMeanLayer,
)


class GEMGCNSIRE(nn.Module):
    def __init__(self, channels=16, out_channels=1, convs=3, n_rings=2, max_order=2):
        """
        OnionNet implementation of the GEMGCN used in the Faust experiment in the ICLR paper.
        nverts: number of vertices in the onion
        """
        super(GEMGCNSIRE, self).__init__()

        # onion structure
        self.out_channels = out_channels

        kwargs = dict(
            n_rings=n_rings, band_limit=max_order, num_samples=7, checkpoint=False, batch=100000, batch_norm=False
        )

        model = [GemResNetBlock(32, channels, 0, max_order, **kwargs)]
        for i in range(convs - 2):
            model += [GemResNetBlock(channels, channels,
                                     max_order, max_order, **kwargs)]
        model += [GemResNetBlock(channels, channels, max_order, 0, **kwargs)]

        self.model = Sequential(*model)

        # Dense final layer
        self.lin1 = nn.Linear(channels, out_channels)

    def forward(self, data, nverts: int = 642, num_scales: int = None, **kwds):
        attr0 = (data.edge_index, data.precomp, data.connection)
        bs = len(data.ptr) - 1

        x: torch.Tensor = data.features.reshape(-1, 32, 1).float()

        for layer in self.model:
            x = layer(x, *attr0)

        # Take trivial feature
        x = x[:, :, 0]
        features = x.clone().detach().view(bs, num_scales, nverts, -1)
        x = self.lin1(x).squeeze()  # Disable ReLU activation if returning prob

        if num_scales is not None:
            all_nverts = nverts * num_scales * self.out_channels

            x = torch.cat(
                [
                    x[i * all_nverts: (i + 1) *
                      all_nverts].view(num_scales, -1, self.out_channels)
                    for i in range(len(x) // all_nverts)
                ],
                dim=1,
            ).transpose(0, 1)

        if not kwds.get("save_features", False):
            return x, None
        else:
            return x, features


class GEMGCNSIRERadius(nn.Module):

    def __init__(self, channels=16, out_channels=1, convs=3, n_rings=2, max_order=2):
        """
        OnionNet implementation of the GEMGCN SIRE backbone, with additional radius
        regression head.
        """

        super(GEMGCNSIRERadius, self).__init__()

        # onion structure
        self.out_channels = out_channels

        kwargs = dict(
            n_rings=n_rings, band_limit=max_order, num_samples=7, checkpoint=False, batch=100000, batch_norm=False
        )

        model = [GemResNetBlock(32, channels, 0, max_order, **kwargs)]
        for i in range(convs - 2):
            model += [GemResNetBlock(channels, channels,
                                     max_order, max_order, **kwargs)]
        model += [GemResNetBlock(channels, channels, max_order, 0, **kwargs)]

        self.model = Sequential(*model)

        # Dense final layer
        self.direction_head = nn.Linear(channels, out_channels)
        self.radius_head = nn.Sequential(
            nn.Linear(in_features=channels, out_features=channels),
            nn.Linear(in_features=channels, out_features=out_channels),
        )
        self.radius_out = nn.LazyLinear(out_features=out_channels)

    def forward(self, data, nverts: int = 642, num_scales: int = None, **kwds):

        attr0 = (data.edge_index, data.precomp, data.connection)
        bs = len(data.ptr) - 1

        x: torch.Tensor = data.features.reshape(-1, 32, 1).float()

        for layer in self.model:
            x = layer(x, *attr0)

        # Take trivial feature
        x = x[:, :, 0]  # (bs * num_scales * nverts, 32)
        features = x.clone().detach().view(bs, num_scales, nverts, -1)
        d = self.direction_head(x).squeeze()  # Returning logits
        r = x.view(bs, num_scales, nverts, -1)
        r = F.relu(self.radius_head(r))
        r = r.squeeze(-1)  # (bs, num_scales, nverts)
        r = self.radius_out(r)  # (bs, num_scales, 1)

        # Reorganizing predicted logits into (bs, nverts, num_scales, 1)
        if num_scales is not None:
            all_nverts = nverts * num_scales * self.out_channels

            d = torch.cat(
                [
                    d[i * all_nverts: (i + 1) *
                      all_nverts].view(num_scales, -1, self.out_channels)
                    for i in range(len(d) // all_nverts)
                ],
                dim=1,
            ).transpose(0, 1)

        if not kwds.get("save_features", False):
            return d, r, None
        else:
            return d, r, features


class GEMGCNMultiScaleFeatureEncoder(nn.Module):
    """
    Multi-scale node features extractor with GEM-GCN backbone.
    """

    def __init__(
        self,
        in_features: int = 32,
        out_channels: int = 1,
        embed_dim: int = 64,
        convs=3,
        n_rings: int = 2,
        max_order: int = 2,
    ):

        super(GEMGCNMultiScaleFeatureEncoder, self).__init__()

        self.in_features = in_features
        self.out_channels = out_channels

        kwargs = dict(
            n_rings=n_rings,
            band_limit=max_order,
            num_samples=7,
            checkpoint=False,
            batch=100000,
            batch_norm=False,
        )

        model = [GemResNetBlock(in_features, embed_dim,
                                0, max_order, **kwargs)]
        for i in range(convs - 2):
            model += [GemResNetBlock(embed_dim, embed_dim,
                                     max_order, max_order, **kwargs)]
        model += [GemResNetBlock(embed_dim, embed_dim, max_order, 0, **kwargs)]

        self.gnns = Sequential(*model)
        self.scale_fusion = ScaleAttnFusionLayer(embed_dim, num_heads=4)
        self.proj_layer = MLP(
            input_dim=embed_dim, embed_dim=embed_dim, out_dim=embed_dim)

    def forward(self, data: Data, **kwds):

        attr0 = (data.edge_index, data.precomp, data.connection)
        x: torch.Tensor = data.features.view(-1, self.in_features, 1)

        for layer in self.gnns:
            x = layer(x, *attr0)

        x = x[:, :, 0]  # Take trivial feature

        # Scale fusion and feature extraction
        features = self.scale_fusion(
            x, batch=data.batch, nverts=kwds["nverts"])
        proj = self.proj_layer(features)

        return features, proj


class GEMGCNAdaSIREMultiTaskScaleAverage(nn.Module):
    """
    GEM-GCN implementation of AdaSIRE, with features on different scales fused
    near the end of prediction using direct averaging. Served as a baseline to
    Other feature fusion methods.
    """

    def __init__(
        self,
        in_features: int = 32,
        out_channels: int = 1,
        embed_dim: int = 64,
        convs=3,
        n_rings: int = 2,
        max_order: int = 2,
    ):

        super(GEMGCNAdaSIREMultiTaskScaleAverage, self).__init__()

        self.in_features = in_features
        self.out_channels = out_channels

        kwargs = dict(
            n_rings=n_rings,
            band_limit=max_order,
            num_samples=7,
            checkpoint=False,
            batch=100000,
            batch_norm=False,
        )

        model = [GemResNetBlock(in_features, embed_dim,
                                0, max_order, **kwargs)]
        for i in range(convs - 2):
            model += [GemResNetBlock(embed_dim, embed_dim,
                                     max_order, max_order, **kwargs)]
        model += [GemResNetBlock(embed_dim, embed_dim, max_order, 0, **kwargs)]

        self.gnns = Sequential(*model)
        self.scale_fusion = ScaleSimpleMeanLayer()
        self.direction_head = MLP(input_dim=embed_dim, out_dim=1, actv="elu")
        self.radius_proj = MLP(input_dim=embed_dim, out_dim=1, actv="elu")
        self.radius_head = nn.LazyLinear(out_features=1)

    def forward(self, data: Data, **kwds):

        bs = len(data.ptr) - 1
        batch: torch.Tensor = data.batch
        attr0 = (data.edge_index, data.precomp, data.connection)

        x: torch.Tensor = data.features.view(-1, self.in_features, 1)

        for layer in self.gnns:
            x = layer(x, *attr0)

        x = x[:, :, 0]  # Take trivial feature from the orders

        # Perform scale feature fusion
        x = self.scale_fusion(x, batch=batch, nverts=kwds["nverts"])
        features = x.clone().detach()  # For feature checking
        d = self.direction_head(x)
        d = d.squeeze(-1)  # (bs, nverts)
        r = self.radius_head(self.radius_proj(x).view(bs, -1))  # (bs, 1)
        r = r.squeeze(-1)  # (bs, )

        return d, r, None, features if kwds.get("save_features", False) else None


class GEMGCNAdaSIREMultiTaskScaleFusion(nn.Module):
    """
    GEM-GCN implementation of AdaSIRE, with features on different scales fused
    near the end of prediction using MLP.
    """

    def __init__(
        self,
        in_features: int = 32,
        out_channels: int = 1,
        embed_dim: int = 64,
        convs=3,
        n_rings: int = 2,
        max_order: int = 2,
    ):

        super(GEMGCNAdaSIREMultiTaskScaleFusion, self).__init__()

        self.in_features = in_features
        self.out_channels = out_channels

        kwargs = dict(
            n_rings=n_rings,
            band_limit=max_order,
            num_samples=7,
            checkpoint=False,
            batch=100000,
            batch_norm=False,
        )

        model = [GemResNetBlock(in_features, embed_dim,
                                0, max_order, **kwargs)]
        for i in range(convs - 2):
            model += [GemResNetBlock(embed_dim, embed_dim,
                                     max_order, max_order, **kwargs)]
        model += [GemResNetBlock(embed_dim, embed_dim, max_order, 0, **kwargs)]

        self.gnns = Sequential(*model)
        self.scale_fusion = ScaleMLPFusionLayer(embed_dim=embed_dim)
        self.direction_head = MLP(input_dim=embed_dim, out_dim=1, actv="elu")
        self.radius_proj = MLP(input_dim=embed_dim, out_dim=1, actv="elu")
        self.radius_head = nn.LazyLinear(out_features=1)

    def forward(self, data: Data, **kwds):

        bs = len(data.ptr) - 1
        batch: torch.Tensor = data.batch
        attr0 = (data.edge_index, data.precomp, data.connection)

        x: torch.Tensor = data.features.view(-1, self.in_features, 1)

        for layer in self.gnns:
            x = layer(x, *attr0)

        x = x[:, :, 0]  # Take trivial feature from the orders

        # Perform scale feature fusion
        x, w_l = self.scale_fusion(x, batch=batch, nverts=kwds["nverts"])
        features = x.clone().detach()  # For feature checking
        d = self.direction_head(x)
        d = d.squeeze(-1)  # (bs, nverts)
        r = self.radius_head(self.radius_proj(x).view(bs, -1))  # (bs, 1)
        r = r.squeeze(-1)  # (bs, )

        return d, r, w_l, features if kwds.get("save_features", False) else None


class GEMGCNAdaSIREMultiTaskTransformer(nn.Module):
    """
    GEM-GCN implementation of AdaSIRE, with features on different scales fused
    near the end of prediction using MLP.
    """

    def __init__(
        self,
        in_features: int = 32,
        out_channels: int = 1,
        num_scales: int = 10,
        embed_dim: int = 64,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        convs=3,
        n_rings: int = 2,
        max_order: int = 2,
    ):

        super(GEMGCNAdaSIREMultiTaskTransformer, self).__init__()

        self.in_features = in_features
        self.out_channels = out_channels

        kwargs = dict(
            n_rings=n_rings,
            band_limit=max_order,
            num_samples=7,
            checkpoint=False,
            batch=100000,
            batch_norm=False,
        )

        model = [GemResNetBlock(in_features, embed_dim,
                                0, max_order, **kwargs)]
        for i in range(convs - 2):
            model += [GemResNetBlock(embed_dim, embed_dim,
                                     max_order, max_order, **kwargs)]
        model += [GemResNetBlock(embed_dim, embed_dim, max_order, 0, **kwargs)]

        self.gnns = Sequential(*model)

        self.scale_fusion = ScaleFusionTransformer(
            num_scales=num_scales,
            feature_dim=embed_dim,
            dim=2 * embed_dim,
            depth=num_transformer_layers,
            heads=num_heads,
            mlp_dim=4 * embed_dim,
            dim_head=embed_dim,
            dropout=0.2,
            emb_dropout=0.2,
        )

        self.direction_head = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1)
        )
        self.radius_proj = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1)
        )
        self.radius_head = nn.LazyLinear(out_features=1)

    def forward(self, data: Data, **kwds):

        bs = len(data.ptr) - 1
        batch: torch.Tensor = data.batch
        attr0 = (data.edge_index, data.precomp, data.connection)

        x: torch.Tensor = data.features.view(-1, self.in_features, 1)

        for layer in self.gnns:
            x = layer(x, *attr0)

        x = x[:, :, 0]  # Take trivial feature from the orders

        # Perform scale feature fusion
        d_tok, r_tok = self.scale_fusion(x, batch=batch, nverts=kwds["nverts"])
        d = self.direction_head(d_tok)
        d = d.squeeze(-1)  # (bs, nverts)
        r = self.radius_head(self.radius_proj(r_tok).view(bs, -1))  # (bs, 1)
        r = r.squeeze(-1)  # (bs, )

        return d, r, None, None


class GEMGCNAdaSIREMultiTaskMaxScale(nn.Module):
    """
    GEM-GCN implementation of AdaSIRE, with features on different scales maxed
    near the end of prediction using MLP.
    """

    def __init__(
        self,
        in_features: int = 32,
        out_channels: int = 1,
        embed_dim: int = 64,
        convs=3,
        n_rings: int = 2,
        max_order: int = 2,
    ):

        super(GEMGCNAdaSIREMultiTaskMaxScale, self).__init__()

        self.in_features = in_features
        self.out_channels = out_channels

        kwargs = dict(
            n_rings=n_rings,
            band_limit=max_order,
            num_samples=7,
            checkpoint=False,
            batch=100000,
            batch_norm=False,
        )

        model = [GemResNetBlock(in_features, embed_dim,
                                0, max_order, **kwargs)]
        for i in range(convs - 2):
            model += [GemResNetBlock(embed_dim, embed_dim,
                                     max_order, max_order, **kwargs)]
        model += [GemResNetBlock(embed_dim, embed_dim, max_order, 0, **kwargs)]

        self.gnns = Sequential(*model)
        self.scale_fusion = ScaleMLPFusionLayer(embed_dim=embed_dim)
        self.direction_head = MLP(input_dim=embed_dim, out_dim=1)
        self.radius_proj = MLP(input_dim=embed_dim, out_dim=1)
        self.radius_head = nn.LazyLinear(out_features=1)

    def forward(self, data: Data, **kwds):

        batch: torch.Tensor = data.batch
        attr0 = (data.edge_index, data.precomp, data.connection)

        V = kwds["nverts"]
        B = batch.max().item() + 1

        x: torch.Tensor = data.features.view(-1, self.in_features, 1)

        for layer in self.gnns:
            x = layer(x, *attr0)

        x = x[:, :, 0]  # Take trivial feature from the orders

        features = x.clone().detach()  # For feature checking
        E = x.shape[-1]
        d = self.direction_head(x).squeeze()  # Returning logits
        r = x.view(B, -1, V, E)
        r = self.radius_proj(r)
        r = r.squeeze(-1)  # (bs, num_scales, nverts)
        r = self.radius_head(r)  # (bs, num_scales, 1)
        r = r.squeeze(-1)

        # Perform argmax operation
        d = d.view(B, -1, V)
        d, _ = torch.max(d, dim=1, keepdim=False)
        r, _ = torch.max(r, dim=1, keepdim=False)

        if kwds.get("save_features", False):
            return d, r, None, features
        else:
            return d, r, None, None
