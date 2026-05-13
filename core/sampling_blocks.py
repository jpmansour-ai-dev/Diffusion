
import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsample(nn.Module):
    """
        Input:  (B, C, H, W)
        → strided convolution (stride=2): kernel jumps 2 pixels at a time → halves H and W
        Output: (B, C, H/2, W/2)
        For info, the general formula for output size of a convolution is:
        output_size = floor((input_size + 2*padding - kernel_size) / stride) + 1
    """

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """
        Input:  (B, C, H, W)
        → nearest-neighbor interpolation: each pixel copied to a 2x2 block → doubles H and W
        → convolution on upsampled feature map
        Output: (B, C, 2*H, 2*W)
    """

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        # The conv layer is what makes the upsampling process learnable.
        # The upsampling here is just nearest-neighbor interpolation
        # it copies each pixel into a 2x2 block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.upsample(x))
