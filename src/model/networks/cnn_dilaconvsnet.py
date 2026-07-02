import torch
import torch.nn as nn


class conv_block(nn.Module):

    def __init__(self, chann_in, chann_out, k_size, stride, p_size, dilation=1):

        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=chann_in, out_channels=chann_out,
                      kernel_size=k_size, stride=stride, padding=p_size,
                      dilation=dilation),
            nn.BatchNorm3d(chann_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        x = self.conv(x)
        return x


class CNNDilatedTracker(nn.Module):
    """
    Implementation of original CNN-tracker convolutional networks. Adapted
    from https://github.com/BubblyYi/Coronary-Artery-Tracking-via-3D-CNN-Classification/blob/master/models/centerline_net.py
    """

    def __init__(
        self,
        channels: int = 32,
        out_channels: int = 500,
    ):
        super(CNNDilatedTracker, self).__init__()

        self.layer1 = conv_block(1, channels, 3, stride=1, p_size=0)
        self.layer2 = conv_block(channels, channels, 3, stride=1, p_size=0)
        self.layer3 = conv_block(
            channels, channels, 3, stride=1, p_size=0, dilation=2)
        self.layer4 = conv_block(
            channels, channels, 3, stride=1, p_size=0, dilation=4)
        self.layer5 = conv_block(channels, 2 * channels, 3, stride=1, p_size=0)
        self.layer6 = conv_block(
            2 * channels, 2 * channels, 1, stride=1, p_size=0)
        self.n_classes = out_channels
        self.layer7 = nn.Conv3d(2 * channels, out_channels + 1,
                                kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)

        return out


class CNNDilatedTrackerLargePatch(nn.Module):
    """
    Implementation of adapted CNN-tracker convolutional networks for large patch.
    Adapted from
    https://github.com/BubblyYi/Coronary-Artery-Tracking-via-3D-CNN-Classification/blob/master/models/centerline_net.py
    """

    def __init__(
        self,
        channels: int = 32,
        out_channels: int = 500,
    ):
        super(CNNDilatedTrackerLargePatch, self).__init__()

        self.layer1 = conv_block(1, channels, 3, stride=2, p_size=0)
        self.layer2 = conv_block(channels, channels, 3, stride=2, p_size=0)
        self.layer3 = conv_block(
            channels, channels, 3, stride=1, p_size=0, dilation=2)
        self.layer4 = conv_block(
            channels, channels, 3, stride=1, p_size=0, dilation=4)
        self.layer5 = conv_block(channels, 2 * channels, 3, stride=1, p_size=0)
        self.layer6 = conv_block(
            2 * channels, 2 * channels, 1, stride=1, p_size=0)
        self.n_classes = out_channels
        self.layer7 = nn.Conv3d(2 * channels, out_channels + 1,
                                kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)

        return out
