# Various convolutional blocks (trimmed: only the blocks IHPGCNUnet needs).
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNConvBNReLU(nn.Module):
    """ 1D convolution with batch normalization and activation. Use GCN. """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        self_loops: bool = True,
        bias: bool = True,
    ):
        """ Construct GCN Convolution, Batch normalization and activation
        block.

        Args:
            in_channels(int): Input feature channel length.
            out_channels(int): Output feature channel length.
            self_loops(bool): Add self-loops in GCN Convolution or not. Default
                              to True.
            bias(bool): Add bias or not. Default to True.
        """

        super().__init__()
        self.conv = GCNConv(in_channels, out_channels,
                            add_self_loops=self_loops, bias=bias)
        self.actv = nn.ReLU()
        self.norm = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:

        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = self.actv(x)

        return x


class GCNUnetEncBlock(nn.Module):
    """ Two concat Conv + BN + Activation blocks in encoder. """

    def __init__(self, in_channels: int, out_channels: int):
        """ Construct GCN UNet Conv block.

        Args:
            in_channels(int): Input feature channel length.
            out_channels(int): Output feature channel length.
        """

        super().__init__()
        mid_channels = out_channels
        self.block1 = GCNConvBNReLU(in_channels, mid_channels)
        self.block2 = GCNConvBNReLU(mid_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:

        x = self.block1(x, edge_index)
        x = self.block2(x, edge_index)

        return x


class GCNUnetDecBlock(nn.Module):
    """ Two concat Conv + BN + Activation blocks in decoder. """

    def __init__(self, in_channels: int, out_channels: int):
        """ Construct GCN UNet Conv block.

        Args:
            in_channels(int): Input feature channel length.
            out_channels(int): Output feature channel length.
        """

        super().__init__()
        mid_channels = in_channels // 2
        self.block1 = GCNConvBNReLU(in_channels, mid_channels)
        self.block2 = GCNConvBNReLU(mid_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:

        x = self.block1(x, edge_index)
        x = self.block2(x, edge_index)

        return x
