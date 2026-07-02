# GCN based on Icosahedral Hexagonal-pentagonal Grids geometry.
from ast import literal_eval
from typing import List

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

from src.model.networks.gcn_blocks import GCNUnetEncBlock, GCNUnetDecBlock
from src.model.networks.ihp_pooling import getIHP_pool_unpool
from src.utils.geometry import IHPSphere


class IHPGCNUnet(nn.Module):
    """ Implementation of IHP geometry based GCN network following COACT
    original paper(https://doi.org/10.1002/mp.16873). """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        levels: List[int] = [0, 1, 2, 3, 4],
    ) -> None:
        """ Construct the GNN UNet in COACT.

        Args:
            in_channels(int): Length of input features.
            out_channels(int): Length of output channels.
            heads(int): Number of heads in GATConv. Default 4.
            levels(List[int]): Different Levels of IHP Grids along the encoder.
            Default to [0, 1, 2, 3, 4].
        """

        super(IHPGCNUnet, self).__init__()
        if isinstance(levels, str):
            levels = literal_eval(levels)
        self.levels = sorted(levels)  # IHP at different levels
        if len(self.levels) < 2:
            raise ValueError("Need >= 2 levels. Got %d" % len(self.levels))

        # Construct pooling and unpooling layers based on different levels
        self.poolings, self.unpoolings = getIHP_pool_unpool(self.levels)

        # Construct in/out feature length according to COACT original paper
        enc_configs = []
        for idx in range(len(self.levels)):
            in_pow = 0 if idx == 0 else self.levels[idx - 1]
            out_pow = self.levels[idx] if idx < len(
                self.levels) - 1 else self.levels[-2]
            enc_configs.append({"in_channels": 2 ** in_pow * in_channels,
                                "out_channels": 2 ** out_pow * in_channels})
        encoders = [GCNUnetEncBlock(**config) for config in enc_configs]
        self.encoders = nn.ModuleList(encoders)
        dec_configs = []
        for idx in range(len(self.levels) - 1, 0, -1):
            in_feats = 2 ** self.levels[idx] * in_channels
            out_feats = 2 ** self.levels[idx - 2] * \
                in_channels if idx > 1 else in_channels
            dec_configs.append({"in_channels": in_feats,
                                "out_channels": out_feats})
        decoders = [GCNUnetDecBlock(**config) for config in dec_configs]
        self.decoders = nn.ModuleList(decoders)

        self.final_conv = GCNConv(
            dec_configs[-1]["out_channels"], out_channels)

    def forward(self, data) -> torch.Tensor:

        x = data.features
        edge_index = data.edge_index

        # Encoder part
        enc_results = []
        conv = self.encoders[0]  # The first encoder block
        x = conv(x, edge_index)
        enc_results.append(x)
        for conv, out_level in zip(self.encoders[1:], self.levels[::-1][1:]):
            # Pooling
            pooling = self.poolings["PoolingLevel%dTo%d" %
                                    (out_level + 1, out_level)]
            x, edge_index = pooling(x, edge_index)
            # Convolution
            x = conv(x, edge_index)
            # Stack convolution results
            enc_results.append(x)
        enc_results.pop(-1)  # Last encoder result is not needed

        # Decoder part
        for idx, (conv, in_level) in enumerate(zip(self.decoders, self.levels[:-1])):
            # Unpooling
            unpool = self.unpoolings["UnpoolingLevel%dTo%d" %
                                     (in_level, in_level + 1)]
            x, edge_index = unpool(x, edge_index)
            # Concat with encoder output
            enc_x = enc_results[::-1][idx]
            x = torch.concat((enc_x, x), dim=-1)
            # Convolution
            x = conv(x, edge_index)

        # Final conv
        x = self.final_conv(x, edge_index)
        return x


if __name__ == "__main__":

    # Test input
    x = torch.rand([5120, 64])  # (N, C)
    edge_index = torch.from_numpy(IHPSphere(4).edge_index)
    data = Data(edge_index=edge_index, features=x)
    # Test network
    GCNUnet = IHPGCNUnet(in_channels=64, out_channels=1,
                         levels=[4, 3, 2, 1, 0])

    y: torch.Tensor = GCNUnet(data)
    label = torch.zeros([5120, 1])

    loss = torch.mean((y - label) ** 2)
    loss.backward()
    breakpoint()  # DEBUG: Check forward and backward propagation
