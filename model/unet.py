from pathlib import Path
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.embedding import TimeEmbedding
from model.residual_block import ResBlock, num_groups
from model.attention_block import AttentionBlock
from model.sampling_blocks import Downsample, Upsample


_cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config" / "config.yaml"))

sinusoidal_dim     = _cfg["embedding"]["dimension"]
channels_per_level = _cfg["model"]["channels_per_level"]
num_res_per_level  = _cfg["model"]["num_res_per_level"]
attn_resolutions   = _cfg["model"]["attention_resolutions"]
num_heads          = _cfg["model"]["num_heads"]
img_channels       = _cfg["model"]["image_channels"]
resolution         = _cfg["model"]["image_size"]


class UNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        # ── Time embedding ────────────────────────────────────────────────────
        self.time_embedding = TimeEmbedding(sinusoidal_dim)
        mlp_dim = sinusoidal_dim * 4                        # 512

        # ── Input conv ───────────────────────────────────────────────────────
        # Getting 32 feature maps of size 64x64 from the 3 input channels (RGB):
        self.input_conv = nn.Conv2d(img_channels, channels_per_level[0], kernel_size=3, padding=1)

        # ── Encoder ──────────────────────────────────────────────────────────
        # Initializing:
        self.res_blocks  = nn.ModuleList()    # will hold Residual Blocks
        self.attn_blocks   = nn.ModuleList()    # will hold AttentionBlocks (or Identity)
        self.down_blocks = nn.ModuleList()    # will hold Downsample blocks

        in_channels  = channels_per_level[0]   # Initializing the nb of channels at level 0
        encoder_output_channels = [in_channels]   #  Storing


        for level, out_channels in enumerate(channels_per_level):
            level_resolution = resolution // (2 ** level)   # 64, 32, 16, 8

            for _ in range(num_res_per_level):
                self.res_blocks.append(ResBlock(in_channels, out_channels, mlp_dim))

                # Add self-attention at low spatial resolutions (e.g. 16×16, 8×8)
                if level_resolution in attn_resolutions:
                    self.attn_blocks.append(AttentionBlock(out_channels, num_heads))
                else:
                    self.attn_blocks.append(nn.Identity())

                in_channels = out_channels
                encoder_output_channels.append(in_channels)

            # Downsample on every level except the last one
            num_levels = len(channels_per_level)
            if level < num_levels - 1:
                self.down_blocks.append(Downsample(in_channels))
                encoder_output_channels.append(in_channels)


        # ── Bottleneck ───────────────────────────────────────────────────────
        # At the deepest level (8×8, 256 channels): ResBlock → Attention → ResBlock
        mid_channels = channels_per_level[-1]    # 256
        self.mid_res_1 = ResBlock(mid_channels, mid_channels, mlp_dim)
        self.mid_attn  = AttentionBlock(mid_channels, num_heads)
        self.mid_res_2 = ResBlock(mid_channels, mid_channels, mlp_dim)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec_res_blocks  = nn.ModuleList()    # will hold Residual Blocks
        self.dec_attn_blocks = nn.ModuleList()    # will hold AttentionBlocks (or Identity)
        self.up_blocks       = nn.ModuleList()    # will hold Upsample blocks

        for level in reversed(range(num_levels)):
            out_channels     = channels_per_level[level]
            level_resolution = resolution // (2 ** level)   # 8, 16, 32, 64

            for _ in range(num_res_per_level + 1):
                skip_channels = encoder_output_channels.pop()
                self.dec_res_blocks.append(
                    ResBlock(in_channels + skip_channels, out_channels, mlp_dim)
                )

                if level_resolution in attn_resolutions:
                    self.dec_attn_blocks.append(AttentionBlock(out_channels, num_heads))
                else:
                    self.dec_attn_blocks.append(nn.Identity())

                in_channels = out_channels

            # Upsample on every level except the last one
            if level > 0:
                self.up_blocks.append(Upsample(in_channels))

        # ── Output conv ──────────────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(num_groups(in_channels), in_channels)
        self.out_act  = nn.SiLU()
        self.out_conv = nn.Conv2d(in_channels, img_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        # Time embedding
        t_emb = self.time_embedding(t)                  # (B, 512)

        # Input convolution
        h = self.input_conv(x)                          # (B, 32, 64, 64)

        # Encoder
        skips    = [h]
        enc_idx  = 0
        down_idx = 0

        for i, _ in enumerate(channels_per_level):
            for _ in range(num_res_per_level):
                h = self.res_blocks[enc_idx](h, t_emb)
                h = self.attn_blocks[enc_idx](h)
                skips.append(h)
                enc_idx += 1
            if i < len(channels_per_level) - 1:
                h = self.down_blocks[down_idx](h)
                skips.append(h)
                down_idx += 1

        # 4. Bottleneck
        h = self.mid_res_1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_res_2(h, t_emb)

        # 5. Decoder
        dec_idx = 0
        up_idx  = 0

        for level in reversed(range(len(channels_per_level))):
            for _ in range(num_res_per_level + 1):
                skip = skips.pop()
                h    = torch.cat([h, skip], dim=1)
                h    = self.dec_res_blocks[dec_idx](h, t_emb)
                h    = self.dec_attn_blocks[dec_idx](h)
                dec_idx += 1

            if level > 0:
                h = self.up_blocks[up_idx](h)
                up_idx += 1

        # 6. Output convolution
        h = self.out_norm(h)
        h = self.out_act(h)
        h = self.out_conv(h)
        return h
