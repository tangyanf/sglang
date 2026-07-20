# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class JoyOV2ArchConfig(DiTArchConfig):
    """Architecture config for JoyO V2 DiT.

    Single-stream packed transformer with MoE, GQA, DiT modulation.
    51 layers, hidden=6144, 48Q/16KV heads, 48 experts Token-Chosen MoE.
    """

    hidden_size: int = 6144
    num_attention_heads: int = 48
    num_kv_heads: int = 16
    attention_head_dim: int = 128
    num_layers: int = 51
    num_encoder_layers: int = 2

    # MoE
    num_experts: int = 48
    moe_hidden_size: int = 4096
    share_expert_dim: int = 8192
    top_p: float = 0.125
    routed_scaling_factor: float = 10.0
    scale_weights_by_expert_count: bool = True
    expert_count_scaling_power: float = 0.5

    # Dense FFN (for layer 0 and refiners)
    ffn_hidden_size: int = 16384

    # Input/output
    latent_channels: int = 128
    text_dim: int = 5120
    patch_size: tuple[int, int, int] = (1, 1, 1)
    num_channels_latents: int = 128

    # RoPE
    rope_theta: float = 10000.0
    max_position_embeddings: int = 8192
    mrope_section: tuple[int, ...] = (16, 24, 24)
    mrope_interleaved: bool = True

    # Norm
    norm_eps: float = 1e-6
    sandwich_norm: bool = True

    # MoE layer list: layer 0 is dense, layers 1-50 are MoE
    moe_layer_list: tuple[int, ...] = None

    param_names_mapping: dict = field(default_factory=lambda: {
        # === Top-level embedders ===
        r"^x_embedder\.(.*)$": r"pixel_embedder.\1",
        r"^context_embedder\.norm\.(.*)$": r"cap_embedder.0.\1",
        r"^context_embedder\.proj\.(.*)$": r"cap_embedder.1.\1",
        r"^time_embedder\.linear_1\.(.*)$": r"t_embedder.0.\1",
        r"^time_embedder\.linear_2\.(.*)$": r"t_embedder.2.\1",
        # === Output projections ===
        r"^proj_out\.norm\.(.*)$": r"vis_proj_out.0.\1",
        r"^proj_out\.linear\.(.*)$": r"vis_proj_out.1.\1",
        r"^audio_proj_out\.norm\.(.*)$": r"audio_proj_out.0.\1",
        r"^audio_proj_out\.linear\.(.*)$": r"audio_proj_out.1.\1",
        # === Main blocks: transformer_blocks.{i} -> layers.{i} ===
        r"^transformer_blocks\.(\d+)\.norm1\.(.*)$": r"layers.\1.attention_norm.\2",
        r"^transformer_blocks\.(\d+)\.norm1_post\.(.*)$": r"layers.\1.attention_norm2.\2",
        r"^transformer_blocks\.(\d+)\.norm2\.(.*)$": r"layers.\1.ffn_norm.\2",
        r"^transformer_blocks\.(\d+)\.norm2_post\.(.*)$": r"layers.\1.ffn_norm2.\2",
        r"^transformer_blocks\.(\d+)\.mod\.(.*)$": r"layers.\1.modulation_model.\2",
        r"^transformer_blocks\.(\d+)\.attn\.to_qkv\.(.*)$": r"layers.\1.attention.wqkv.\2",
        r"^transformer_blocks\.(\d+)\.attn\.to_out\.(.*)$": r"layers.\1.attention.wo.\2",
        r"^transformer_blocks\.(\d+)\.attn\.norm_q\.(.*)$": r"layers.\1.attention.q_norm.\2",
        r"^transformer_blocks\.(\d+)\.attn\.norm_k\.(.*)$": r"layers.\1.attention.k_norm.\2",
        r"^transformer_blocks\.(\d+)\.ff\.w1\.(.*)$": r"layers.\1.feed_forward.w1.\2",
        r"^transformer_blocks\.(\d+)\.ff\.w2\.(.*)$": r"layers.\1.feed_forward.w2.\2",
        r"^transformer_blocks\.(\d+)\.ff\.gate\.(.*)$": r"layers.\1.feed_forward.gate.\2",
        r"^transformer_blocks\.(\d+)\.ff\.experts\.(.*)$": r"layers.\1.feed_forward.experts.\2",
        r"^transformer_blocks\.(\d+)\.ff\.shared_expert\.w1\.(.*)$": r"layers.\1.feed_forward.share_expert_w1.\2",
        r"^transformer_blocks\.(\d+)\.ff\.shared_expert\.w2\.(.*)$": r"layers.\1.feed_forward.share_expert_w2.\2",
        r"^transformer_blocks\.(\d+)\.ff\.shared_expert\.gate\.(.*)$": r"layers.\1.feed_forward.shared_expert_gate.\2",
        # === Refiners (prefix stays, internal names change) ===
        # noise_refiner
        r"^noise_refiner\.(\d+)\.norm1\.(.*)$": r"noise_refiner.\1.attention_norm.\2",
        r"^noise_refiner\.(\d+)\.norm1_post\.(.*)$": r"noise_refiner.\1.attention_norm2.\2",
        r"^noise_refiner\.(\d+)\.norm2\.(.*)$": r"noise_refiner.\1.ffn_norm.\2",
        r"^noise_refiner\.(\d+)\.norm2_post\.(.*)$": r"noise_refiner.\1.ffn_norm2.\2",
        r"^noise_refiner\.(\d+)\.mod\.(.*)$": r"noise_refiner.\1.modulation_model.\2",
        r"^noise_refiner\.(\d+)\.attn\.to_qkv\.(.*)$": r"noise_refiner.\1.attention.wqkv.\2",
        r"^noise_refiner\.(\d+)\.attn\.to_out\.(.*)$": r"noise_refiner.\1.attention.wo.\2",
        r"^noise_refiner\.(\d+)\.attn\.norm_q\.(.*)$": r"noise_refiner.\1.attention.q_norm.\2",
        r"^noise_refiner\.(\d+)\.attn\.norm_k\.(.*)$": r"noise_refiner.\1.attention.k_norm.\2",
        r"^noise_refiner\.(\d+)\.ff\.w1\.(.*)$": r"noise_refiner.\1.feed_forward.w1.\2",
        r"^noise_refiner\.(\d+)\.ff\.w2\.(.*)$": r"noise_refiner.\1.feed_forward.w2.\2",
        # context_text_refiner
        r"^context_text_refiner\.(\d+)\.norm1\.(.*)$": r"context_text_refiner.\1.attention_norm.\2",
        r"^context_text_refiner\.(\d+)\.norm1_post\.(.*)$": r"context_text_refiner.\1.attention_norm2.\2",
        r"^context_text_refiner\.(\d+)\.norm2\.(.*)$": r"context_text_refiner.\1.ffn_norm.\2",
        r"^context_text_refiner\.(\d+)\.norm2_post\.(.*)$": r"context_text_refiner.\1.ffn_norm2.\2",
        r"^context_text_refiner\.(\d+)\.attn\.to_qkv\.(.*)$": r"context_text_refiner.\1.attention.wqkv.\2",
        r"^context_text_refiner\.(\d+)\.attn\.to_out\.(.*)$": r"context_text_refiner.\1.attention.wo.\2",
        r"^context_text_refiner\.(\d+)\.attn\.norm_q\.(.*)$": r"context_text_refiner.\1.attention.q_norm.\2",
        r"^context_text_refiner\.(\d+)\.attn\.norm_k\.(.*)$": r"context_text_refiner.\1.attention.k_norm.\2",
        r"^context_text_refiner\.(\d+)\.ff\.w1\.(.*)$": r"context_text_refiner.\1.feed_forward.w1.\2",
        r"^context_text_refiner\.(\d+)\.ff\.w2\.(.*)$": r"context_text_refiner.\1.feed_forward.w2.\2",
        # audio_refiner
        r"^audio_refiner\.(\d+)\.norm1\.(.*)$": r"audio_refiner.\1.attention_norm.\2",
        r"^audio_refiner\.(\d+)\.norm1_post\.(.*)$": r"audio_refiner.\1.attention_norm2.\2",
        r"^audio_refiner\.(\d+)\.norm2\.(.*)$": r"audio_refiner.\1.ffn_norm.\2",
        r"^audio_refiner\.(\d+)\.norm2_post\.(.*)$": r"audio_refiner.\1.ffn_norm2.\2",
        r"^audio_refiner\.(\d+)\.mod\.(.*)$": r"audio_refiner.\1.modulation_model.\2",
        r"^audio_refiner\.(\d+)\.attn\.to_qkv\.(.*)$": r"audio_refiner.\1.attention.wqkv.\2",
        r"^audio_refiner\.(\d+)\.attn\.to_out\.(.*)$": r"audio_refiner.\1.attention.wo.\2",
        r"^audio_refiner\.(\d+)\.attn\.norm_q\.(.*)$": r"audio_refiner.\1.attention.q_norm.\2",
        r"^audio_refiner\.(\d+)\.attn\.norm_k\.(.*)$": r"audio_refiner.\1.attention.k_norm.\2",
        r"^audio_refiner\.(\d+)\.ff\.w1\.(.*)$": r"audio_refiner.\1.feed_forward.w1.\2",
        r"^audio_refiner\.(\d+)\.ff\.w2\.(.*)$": r"audio_refiner.\1.feed_forward.w2.\2",
    })
    reverse_param_names_mapping: dict = field(default_factory=dict)

    _fsdp_shard_conditions: list = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        # transformer/config.json carries `patch_size: null` / `mrope_section: null`
        # / `moe_layer_list: null`. update_model_arch setattr's them unconditionally
        # over the dataclass defaults, so re-apply the intended defaults when the
        # HF config left them blank.
        if self.patch_size is None:
            self.patch_size = (1, 1, 1)
        if self.mrope_section is None:
            self.mrope_section = (16, 24, 24)
        if self.moe_layer_list is None:
            self.moe_layer_list = tuple(range(1, self.num_layers))


@dataclass
class JoyOV2DiTConfig(DiTConfig):
    arch_config: JoyOV2ArchConfig = field(default_factory=JoyOV2ArchConfig)
    prefix: str = "JoyOV2"
