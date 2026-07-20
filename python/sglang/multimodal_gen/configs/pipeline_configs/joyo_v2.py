# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.joyo_v2 import JoyOV2DiTConfig
from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.qwen3vl import Qwen3VLConfig
from sglang.multimodal_gen.configs.models.vaes.xvae import XVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)


def joyo_v2_postprocess_text(
    outputs: BaseEncoderOutput, _text_inputs
) -> torch.Tensor:
    """Extract last hidden state from Qwen3-VL outputs.

    The standard text encoding stage calls this after running the encoder.
    For JoyO V2 the custom text encoding stage (JoyOV2TextEncodingStage) handles
    extraction directly and is the only path used at inference, so this function
    is not on the live path.

    Note: sglang's Qwen3VLTextModel collects all_hidden_states before the final
    RMSNorm, so outputs.hidden_states[-1] is the pre-norm residual. Joytron GT
    uses the post-norm output; the custom stage reapplies the encoder's final
    norm. If this passthrough is ever wired up, it must do the same.
    """
    last_hidden_state = outputs.hidden_states[-1]
    return last_hidden_state


@dataclass
class JoyOV2T2VConfig(PipelineConfig):
    """Pipeline configuration for JoyO V2 T2V inference."""

    task_type: ModelTaskType = ModelTaskType.T2V

    dit_config: DiTConfig = field(default_factory=JoyOV2DiTConfig)

    vae_config: VAEConfig = field(default_factory=XVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    flow_shift: float = 4.0

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Qwen3VLConfig(),)
    )
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = (
        field(default_factory=lambda: (joyo_v2_postprocess_text,))
    )

    precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))

    prompt_template: str = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    devide_denominator: int = 8

    def __post_init__(self):
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True

    def get_latent_dtype(self, prompt_dtype: torch.dtype) -> torch.dtype:
        # Joytron samples the initial noise in fp32 (torch.randn on CPU) and only
        # casts to bf16 inside the denoise loop. Sampling latents in fp32 here
        # (instead of following prompt_embeds' bf16) lets seed=12345 reproduce the
        # Joytron GT noise bit-exactly rather than drawing from the bf16 RNG stream.
        return torch.float32

    def get_decode_scale_and_shift(self, device, dtype, vae):
        # XVAE denormalize: latents / inv_std + mean  (== latents * std + mean).
        # sglang decode does `latents / scale + shift`, so map (scale, shift) =
        # (inv_std, mean). Both buffers are already shape (1, C, 1, 1, 1) on the VAE.
        inv_std = getattr(vae, "latents_inv_std_buf", None)
        mean = getattr(vae, "latents_mean_buf", None)
        if isinstance(inv_std, torch.Tensor) and isinstance(mean, torch.Tensor):
            return (
                inv_std.to(device=device, dtype=dtype),
                mean.to(device=device, dtype=dtype),
            )
        return 1.0, None

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            auto_dit_layerwise_offload=True,
        )
