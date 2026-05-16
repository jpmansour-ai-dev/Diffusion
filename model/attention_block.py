import torch
import torch.nn as nn

from model.residual_block import num_groups


class AttentionBlock(nn.Module):
    """
        Input: x   (B, C, H, W) - Here, either HxW = 16x16 or 8x8
        → Group normalization
        → reshape to (B, H*W, C) -  H*W vectors (tokens) of size C, per sample. Each pixel becomes one token
        → self-attention    
        → reshape to (B, C, H, W)
        → residual addition (adding the original input to the output of attention block)
        Output: x   (B, C, H, W)
    """

    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()

        self.norm = nn.GroupNorm(num_groups(channels), channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # (B, C, H, W) → (B, C, H*W) → (B, H*W, C)
        tokens = self.norm(x).reshape(B, C, H * W).transpose(1, 2)
        # To understand why we did not reshape directly from (B, C, H, W) to (B, H*W, C), 
        # and the choice of the embedding space,
        # Take an example of B=1, C=2, H=2, W=2. 
        # Starting with the reshape call:
        # channel 0: [[a, b], [c, d]]  becomes  [a, b, c, d]
        # channel 1: [[e, f], [g, h]]  becomes  [e, f, g, h]
        # Now, for channel 0, we have [a, b, c, d] and for channel 1, we have [e, f, g, h].
        # Think of channel as index i, and the pixel as j.
        # By transposing (i.e. swapping the last two dimensions C and H*W), we get:
        # token 0: [a, e]  
        # token 1: [b, f]  
        # token 2: [c, g] 
        # token 3: [d, h] 
        # Which is exactly the correct choice of tokens
        # In the same token, the idea is to group together the same pixel across all feature maps 
        # to take into account all of the characteristics, after convolutional layer.

        # self-attention : Q = K = V = tokens
        # Output : Each token is replaced by a weighted combination of all other tokens
        out, _ = self.attention(tokens, tokens, tokens, need_weights=False)

        # (B, H*W, C) → (B, C, H*W) → (B, C, H, W)
        out = out.transpose(1, 2).reshape(B, C, H, W)

        return x + out
