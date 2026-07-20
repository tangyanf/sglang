# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class XVAEArchConfig(VAEArchConfig):
    """Architecture config for XVAE (JoyO V2 video autoencoder).

    Key differences from WanVAE:
    - z_dim=128 (vs 16)
    - patch_size=2 (pixel-space patchify before encoder)
    - 5-level downsample (vs 4)
    - XVAERMSNorm (vs GroupNorm)
    - scale_factor_spatial=32 (vs 8)
    - No quant_conv/post_quant_conv
    """

    z_dim: int = 128
    patch_size: int = 2
    in_channels: int = 3
    out_channels: int = 3
    block_in_channels: tuple[int, ...] = (160, 320, 640, 1280, 1280)
    temporal_downsample: tuple[bool, ...] = (False, True, True, False, False)
    num_res_blocks: int = 2
    channel_doubling: bool = False

    temporal_compression_ratio: int = 4
    spatial_compression_ratio: int = 32
    scale_factor_temporal: int = 4
    scale_factor_spatial: int = 32

    # Latent normalization (128-dim, loaded from checkpoint buffer)
    latents_mean: tuple[float, ...] | None = None
    latents_std: tuple[float, ...] | None = None


@dataclass
class XVAEConfig(VAEConfig):
    arch_config: XVAEArchConfig = field(default_factory=XVAEArchConfig)
    use_feature_cache: bool = True

    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False
    use_parallel_decode: bool = False

    def auto_parallel_decode_prefers_spatial_shard(self) -> bool:
        return True

    def get_vae_scale_factor(self):
        return self.arch_config.scale_factor_spatial
