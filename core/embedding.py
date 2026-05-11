import math

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):

    def __init__(self, dim: int) -> None:
        assert dim % 2 == 0, f"Embedding dimension must be even, got {dim}"
        super().__init__()
        
        half = dim // 2
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(
                start=0,
                end=half,
                step=1,
                dtype=torch.float32,
            )
            / (half - 1)
        )
        self.register_buffer("frequencies", frequencies) 

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = t[:, None].float() * self.frequencies[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
