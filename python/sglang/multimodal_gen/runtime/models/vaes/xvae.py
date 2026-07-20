# SPDX-License-Identifier: Apache-2.0
"""XVAE — 3D Video Autoencoder for JoyO V2.

Ported from diffusers AutoencoderKLXVAE implementation.
Key features:
- XVAERMSNorm (channel-first L2 normalize * scale)
- CausalConv3d with temporal causal padding
- Patchify/Unpatchify at pixel level (patch_size=2)
- Feature caching for temporal-tiled decode/encode
- z_dim=128, spatial_compress=32x, temporal_compress=4x
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from sglang.multimodal_gen.configs.models.vaes.xvae import XVAEConfig
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.vaes.common import (
    DiagonalGaussianDistribution,
    ParallelTiledVAE,
)

CACHE_T = 2


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class CausalConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self._causal_padding = (
            self.padding[2], self.padding[2],
            self.padding[1], self.padding[1],
            2 * self.padding[0], 0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x: Tensor, cache_x: Optional[Tensor] = None) -> Tensor:
        padding = list(self._causal_padding)
        if cache_x is not None and self._causal_padding[-2] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[-2] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


class XVAERMSNorm(nn.Module):
    def __init__(self, dim: int, channel_first: bool = True, images: bool = False, bias: bool = False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        normed = F.normalize(x, dim=1 if self.channel_first else -1) * self.scale * self.gamma
        if self.bias is not None:
            normed = normed + self.bias
        return normed


class FP32Upsample(nn.Upsample):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type_as(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = XVAERMSNorm(in_channels, channel_first=True, images=False)
        self.q = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        b, c, t, h, w = x.shape
        residual = x

        x = self.norm(x)
        q = rearrange(self.q(x), "b c t h w -> (b t) 1 (h w) c")
        k = rearrange(self.k(x), "b c t h w -> (b t) 1 (h w) c")
        v = rearrange(self.v(x), "b c t h w -> (b t) 1 (h w) c")

        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "(b t) 1 (h w) c -> b c t h w", b=b, t=t, h=h, w=w)
        x = self.proj_out(x)

        return residual + x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = XVAERMSNorm(in_channels, channel_first=True, images=False)
        self.conv1 = CausalConv3d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = XVAERMSNorm(self.out_channels, channel_first=True, images=False)
        self.conv2 = CausalConv3d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = CausalConv3d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = None

    def forward(self, x: Tensor, feat_cache=None, feat_idx=None) -> Tensor:
        shortcut = x
        x = swish(self.norm1(x))

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:].clone()
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1:], cache_x], dim=2)
            x = self.conv1(x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        x = swish(self.norm2(x))

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:].clone()
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1:].to(cache_x.device), cache_x], dim=2)
            x = self.conv2(x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x)

        if self.nin_shortcut is not None:
            shortcut = self.nin_shortcut(shortcut)

        return x + shortcut


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, temporal_downsample: bool):
        super().__init__()
        self.spatial = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.temporal = None
        if temporal_downsample:
            self.temporal = CausalConv3d(
                out_channels, out_channels, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )
        self.group_size = (8 if temporal_downsample else 4) * in_channels // out_channels

    def shortcut(self, x: Tensor) -> Tensor:
        if self.temporal is not None:
            r1, group_size, pad_t = 2, self.group_size, x.shape[2] % 2
        else:
            r1, group_size, pad_t = 1, self.group_size, 0
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        x = rearrange(x, "b c (t r1) (h r2) (w r3) -> b (r1 r2 r3 c) t h w", r1=r1, r2=2, r3=2)
        b, c, t, h, w = x.shape
        return x.view(b, c // group_size, group_size, t, h, w).mean(dim=2)

    def forward(self, x: Tensor, feat_cache=None, feat_idx=None) -> Tensor:
        b, c, t, h, w = x.shape
        shortcut = self.shortcut(x)

        x = F.pad(x, (0, 1, 0, 1, 0, 0))
        x = self.spatial(x)

        if self.temporal is not None:
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:].clone()
                    x = self.temporal(torch.cat([feat_cache[idx][:, :, -1:], x], dim=2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
            else:
                x0 = x[:, :, :1]
                if t > 1:
                    x_ = self.temporal(x)
                    x = torch.cat([x0, x_], dim=2)
                else:
                    dummy = F.pad(x, (0, 0, 0, 0, 2, 0))
                    x = x0 + 0.0 * self.temporal(dummy)

        return x + shortcut


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, temporal_upsample: bool):
        super().__init__()
        self.temporal = None
        if temporal_upsample:
            self.temporal = CausalConv3d(
                in_channels, in_channels * 2, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)
            )
        self.spatial = nn.Sequential(
            FP32Upsample(scale_factor=(1, 2, 2), mode="nearest-exact"),
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
        )
        self.repeats = (8 if temporal_upsample else 4) * out_channels // in_channels

    def shortcut(self, x: Tensor, first_chunk: bool = False) -> Tensor:
        if self.temporal is not None:
            r1, repeats, skip = 2, self.repeats, 1 if first_chunk else 0
        else:
            r1, repeats, skip = 1, self.repeats, 0
        x = x.repeat_interleave(repeats=repeats, dim=1)
        x = rearrange(x, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
        x = x[:, :, skip:, :, :]
        return x

    def forward(self, x: Tensor, feat_cache=None, feat_idx=None, first_chunk: bool = False) -> Tensor:
        b, c, t, h, w = x.shape
        shortcut = self.shortcut(x, first_chunk=bool(feat_cache is None or first_chunk))

        if self.temporal is not None:
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:].clone()
                    if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                        cache_x = torch.cat([feat_cache[idx][:, :, -1:], cache_x], dim=2)
                    x = self.temporal(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = rearrange(x, "b (r c) t h w -> b c (t r) h w", r=2)
            else:
                x0 = x[:, :, :1]
                if t > 1:
                    x_ = self.temporal(x)[:, :, 1:]
                    x_ = rearrange(x_, "b (r c) t h w -> b c (t r) h w", r=2)
                    x = torch.cat([x0, x_], dim=2)
                else:
                    dummy = self.temporal(x)
                    dummy = rearrange(dummy, "b (r c) t h w -> b c (t r) h w", r=2)[:, :, :1]
                    x = x0 + 0.0 * dummy

        x = self.spatial(x)
        return x + shortcut


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------


class XVAEEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        z_channels: int,
        num_res_blocks: int,
        block_in_channels: tuple[int, ...],
        temporal_downsample: tuple[bool, ...],
        channel_doubling: bool = True,
    ):
        super().__init__()
        self.conv_in = CausalConv3d(in_channels, block_in_channels[0], kernel_size=3, stride=1, padding=1)

        self.down_blocks = nn.ModuleList()
        for i_level, block_in in enumerate(block_in_channels):
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock(in_channels=block_in, out_channels=block_in))
            if i_level != len(block_in_channels) - 1:
                out_ch = block_in * 2 if channel_doubling else block_in_channels[i_level + 1]
                self.down_blocks.append(DownsampleBlock(block_in, out_ch, temporal_downsample[i_level]))

        block_in = block_in_channels[-1]
        self.mid_blocks = nn.ModuleList([
            ResidualBlock(in_channels=block_in, out_channels=block_in),
            AttnBlock(block_in),
            ResidualBlock(in_channels=block_in, out_channels=block_in),
        ])

        self.norm_out = XVAERMSNorm(block_in, channel_first=True, images=False)
        self.conv_out = CausalConv3d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor, feat_cache=None, feat_idx=None) -> Tensor:
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:].clone()
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1:], cache_x], dim=2)
            x = self.conv_in(x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        for block in self.down_blocks:
            x = block(x, feat_cache=feat_cache, feat_idx=feat_idx)

        for block in self.mid_blocks:
            if isinstance(block, ResidualBlock):
                x = block(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = block(x)

        x = swish(self.norm_out(x))

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:].clone()
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1:], cache_x], dim=2)
            x = self.conv_out(x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)

        return x


class XVAEDecoder(nn.Module):
    def __init__(
        self,
        z_channels: int,
        out_channels: int,
        num_res_blocks: int,
        block_in_channels: tuple[int, ...],
        temporal_upsample: tuple[bool, ...],
        channel_doubling: bool = True,
    ):
        super().__init__()
        block_in = block_in_channels[0]
        self.conv_in = CausalConv3d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid_blocks = nn.ModuleList([
            ResidualBlock(in_channels=block_in, out_channels=block_in),
            AttnBlock(block_in),
            ResidualBlock(in_channels=block_in, out_channels=block_in),
        ])

        self.up_blocks = nn.ModuleList()
        for i_level, block_in in enumerate(block_in_channels):
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(ResidualBlock(in_channels=block_in, out_channels=block_in))
            if i_level != len(block_in_channels) - 1:
                out_ch = block_in // 2 if channel_doubling else block_in_channels[i_level + 1]
                self.up_blocks.append(UpsampleBlock(block_in, out_ch, temporal_upsample[i_level]))

        block_in = block_in_channels[-1]
        self.norm_out = XVAERMSNorm(block_in, channel_first=True, images=False)
        self.conv_out = CausalConv3d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor, feat_cache=None, feat_idx=None, first_chunk: bool = False) -> Tensor:
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:].clone()
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1:], cache_x], dim=2)
            x = self.conv_in(x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        for block in self.mid_blocks:
            if isinstance(block, ResidualBlock):
                x = block(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = block(x)

        for block in self.up_blocks:
            if isinstance(block, ResidualBlock):
                x = block(x, feat_cache=feat_cache, feat_idx=feat_idx)
            elif isinstance(block, UpsampleBlock):
                x = block(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)

        x = swish(self.norm_out(x))

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:].clone()
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1:], cache_x], dim=2)
            x = self.conv_out(x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)

        return x


# ---------------------------------------------------------------------------
# Top-level AutoencoderKL
# ---------------------------------------------------------------------------


class AutoencoderKLXVAE(ParallelTiledVAE):
    """XVAE — 3D Video Autoencoder for JoyO V2.

    Spatial compression: patch_size(2) × 2^4 = 32x.
    Temporal compression: 2^2 = 4x.
    Latent channels: 128.
    """

    layer_names = ["encoder.down_blocks", "decoder.up_blocks"]

    def __init__(self, config: XVAEConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        arch = config.arch_config

        self.patch_size = arch.patch_size
        self.latent_channels = arch.z_dim
        self.use_feature_caching = config.use_feature_cache

        self.encoder = XVAEEncoder(
            in_channels=arch.in_channels * (arch.patch_size ** 2),
            z_channels=arch.z_dim,
            num_res_blocks=arch.num_res_blocks,
            block_in_channels=arch.block_in_channels,
            temporal_downsample=arch.temporal_downsample,
            channel_doubling=arch.channel_doubling,
        )
        self.decoder = XVAEDecoder(
            z_channels=arch.z_dim,
            out_channels=arch.out_channels * (arch.patch_size ** 2),
            num_res_blocks=arch.num_res_blocks,
            block_in_channels=tuple(reversed(arch.block_in_channels)),
            temporal_upsample=arch.temporal_downsample,
            channel_doubling=arch.channel_doubling,
        )

        # Latent normalization buffers (loaded from checkpoint)
        if arch.latents_mean is not None:
            self.register_buffer(
                "latents_mean_buf",
                torch.tensor(arch.latents_mean).view(1, -1, 1, 1, 1),
            )
        else:
            self.latents_mean_buf = None

        if arch.latents_std is not None:
            self.register_buffer(
                "latents_inv_std_buf",
                1.0 / torch.tensor(arch.latents_std, dtype=torch.float32).view(1, -1, 1, 1, 1),
            )
        else:
            self.latents_inv_std_buf = None

    @staticmethod
    def _patchify(x: Tensor, patch_size: int) -> Tensor:
        if patch_size == 1:
            return x
        return rearrange(x, "b c t (h r1) (w r2) -> b (c r1 r2) t h w", r1=patch_size, r2=patch_size)

    @staticmethod
    def _unpatchify(x: Tensor, patch_size: int) -> Tensor:
        if patch_size == 1:
            return x
        return rearrange(x, "b (r1 r2 c) t h w -> b c t (h r1) (w r2)", r1=patch_size, r2=patch_size)

    def _count_conv_modules(self):
        enc_num = sum(isinstance(m, CausalConv3d) for m in self.encoder.modules())
        dec_num = sum(isinstance(m, CausalConv3d) for m in self.decoder.modules())
        return enc_num, dec_num

    def _encode(self, x: Tensor) -> Tensor:
        x = self._patchify(x, self.patch_size)

        if not self.use_feature_caching:
            return self.encoder(x)

        enc_num, _ = self._count_conv_modules()
        feat_cache = [None] * enc_num
        feat_idx = [0]

        ffactor_temporal = self.config.arch_config.scale_factor_temporal
        out = []
        num_iters = 1 + (x.shape[2] - 1) // ffactor_temporal
        for i in range(num_iters):
            feat_idx[0] = 0
            if i == 0:
                h = self.encoder(x[:, :, :1], feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                h = self.encoder(
                    x[:, :, 1 + (i - 1) * ffactor_temporal: 1 + i * ffactor_temporal],
                    feat_cache=feat_cache, feat_idx=feat_idx,
                )
            out.append(h)

        return torch.cat(out, dim=2)

    def _decode(self, z: Tensor) -> Tensor:
        if not self.use_feature_caching:
            decoded = self.decoder(z)
            return self._unpatchify(decoded, self.patch_size)

        _, dec_num = self._count_conv_modules()
        feat_cache = [None] * dec_num
        feat_idx = [0]

        decoded = []
        for i in range(z.shape[2]):
            feat_idx[0] = 0
            out = self.decoder(
                z[:, :, i: i + 1],
                feat_cache=feat_cache,
                feat_idx=feat_idx,
                first_chunk=(i == 0),
            )
            decoded.append(out)

        decoded = torch.cat(decoded, dim=2)
        return self._unpatchify(decoded, self.patch_size)

    def normalize_latents(self, latents: Tensor) -> Tensor:
        if self.latents_mean_buf is not None and self.latents_inv_std_buf is not None:
            return (latents - self.latents_mean_buf.to(latents)) * self.latents_inv_std_buf.to(latents)
        return latents

    def denormalize_latents(self, latents: Tensor) -> Tensor:
        if self.latents_mean_buf is not None and self.latents_inv_std_buf is not None:
            return latents / self.latents_inv_std_buf.to(latents) + self.latents_mean_buf.to(latents)
        return latents
