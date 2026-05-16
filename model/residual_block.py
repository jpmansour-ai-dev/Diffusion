from pathlib import Path
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F


_cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config" / "config.yaml"))
channels_per_group = _cfg["model"]["channels_per_group"]

def num_groups(channels: int) -> int:
    return channels // channels_per_group


class ResBlock(nn.Module):
    """One residual block: two conv layers + time injection + skip connection.

    Inputs
    ------
    x_t : (B, in_channels,  H, W)   — feature map at timestep t
    t   : (B, time_dim)             — one timestep vector per sample

    Output
    ------
        : (B, out_channels, H, W)   
    """

    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()

        self.norm1 = nn.GroupNorm(num_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Linear map converting (B, time_dim) tensors to (B, out_channels) tensors 
        self.time_projection = nn.Linear(time_dim, out_channels) 

        self.norm2 = nn.GroupNorm(num_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # skip: 1x1 conv to match channels for the residual addition, identity if already equal
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)


    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x_t))) 

        # t_proj is (B, out_channels, 1, 1) — one scalar per channel,
        # broadcasted by + across every pixel of h (B, out_channels, H, W)
        t_proj = self.time_projection(t)           
        h = h + t_proj[:, :, None, None]       
        
        # Here, the convolution learns how features behave at different noise levels.
        h = self.conv2(F.silu(self.norm2(h)))  

        return h + self.skip(x_t)      
