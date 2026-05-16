import math

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):

    def __init__(self, sinusoidal_dim: int) -> None:
        assert sinusoidal_dim % 2 == 0, f"Embedding dimension must be even, got {sinusoidal_dim}"
        super().__init__()

        half = sinusoidal_dim // 2
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


# Below is a two-layer MLP: Linear → SiLU → Linear  
# It projects the fixed sinusoidal encoding (128d) into a learned space (512d)
# Which is large enough to always project DOWN into any channel count in the network (max 256d)
class TimeEmbedding(nn.Module):

    def __init__(self, sinusoidal_dim: int) -> None:
        super().__init__()
        self.sinusoidal = SinusoidalEmbedding(sinusoidal_dim)
        self.MLP = nn.Sequential(
            nn.Linear(sinusoidal_dim, 4 * sinusoidal_dim),
            nn.SiLU(),
            nn.Linear(4 * sinusoidal_dim, 4 * sinusoidal_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.MLP(self.sinusoidal(t))
