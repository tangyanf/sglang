# SPDX-License-Identifier: Apache-2.0
"""JoyO V2 custom denoising stage.

Implements Joytron's packed denoising loop:
- CFG via packed pos+neg in one forward pass
- SD3 timeshift schedule
- Euler step with CFG Zero Star
- rope_meta construction matching PreCalRopeMeta
"""

from __future__ import annotations

import torch
from einops import rearrange

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.joyo_v2 import (
    build_position_ids,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


def sd3_timeshift(t: torch.Tensor, shift: float) -> torch.Tensor:
    """SD3-style time-shift: sigma(t) = 1 - (shift * t) / (1 + (shift - 1) * t)."""
    return 1 - (shift * t) / (1 + (shift - 1) * t)


def build_rope_meta(
    text_seqlens: torch.Tensor,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    device: torch.device,
) -> dict:
    """Build rope_meta dict matching Joytron PreCalRopeMeta for T2V (no cond frames).

    Args:
        text_seqlens: (n_samples,) int32 — valid text token count per sample
        latent_t, latent_h, latent_w: latent dimensions after patchify
        device: target device
    Returns:
        rope_meta dict with text/pixel/mix entries
    """
    n_samples = text_seqlens.shape[0]
    pixel_seqlen = latent_t * latent_h * latent_w

    # --- Text ---
    text_pos_ids_list = []
    for i in range(n_samples):
        t_len = int(text_seqlens[i].item())
        t_pos = torch.arange(t_len, device=device, dtype=torch.int32).unsqueeze(0).expand(3, -1)
        text_pos_ids_list.append(t_pos)
    text_position_ids = torch.cat(text_pos_ids_list, dim=1)  # (3, S_text_total)
    text_seqlens_i32 = text_seqlens.to(torch.int32)
    text_cu_seqlens = torch.zeros(n_samples + 1, dtype=torch.int32, device=device)
    text_cu_seqlens[1:] = torch.cumsum(text_seqlens_i32, dim=0)

    # --- Pixel ---
    pixel_seqlens_tensor = torch.full(
        (n_samples,), pixel_seqlen, dtype=torch.int32, device=device
    )
    pixel_cu_seqlens = torch.zeros(n_samples + 1, dtype=torch.int32, device=device)
    pixel_cu_seqlens[1:] = torch.cumsum(pixel_seqlens_tensor, dim=0)

    # NOTE: pixel.position_id has NO text-offset (unlike mix.position_id below).
    # Verified against Joytron PreCalRopeMeta dump: single-modal pixel positions
    # start at 0; only the pixel slice inside mix.position_id is shifted by text_len.
    pixel_pos_ids_list = []
    for _ in range(n_samples):
        p_pos = build_position_ids(latent_t, latent_h, latent_w, device, pre_len=0)
        pixel_pos_ids_list.append(p_pos)
    pixel_position_ids = torch.cat(pixel_pos_ids_list, dim=1)  # (3, S_pixel_total)

    # --- Mix (text + pixel per sample) ---
    mix_seqlens_list = []
    mix_pos_ids_list = []
    for i in range(n_samples):
        t_len = int(text_seqlens[i].item())
        mix_len = t_len + pixel_seqlen
        mix_seqlens_list.append(mix_len)

        t_pos = torch.arange(t_len, device=device, dtype=torch.int32).unsqueeze(0).expand(3, -1)
        p_pos = build_position_ids(latent_t, latent_h, latent_w, device, pre_len=t_len)
        mix_pos_ids_list.append(torch.cat([t_pos, p_pos], dim=1))

    mix_seqlens = torch.tensor(mix_seqlens_list, dtype=torch.int32, device=device)
    mix_cu_seqlens = torch.zeros(n_samples + 1, dtype=torch.int32, device=device)
    mix_cu_seqlens[1:] = torch.cumsum(mix_seqlens, dim=0)
    mix_position_ids = torch.cat(mix_pos_ids_list, dim=1)

    rope_meta = {
        "text": {
            "cu_seqlens": text_cu_seqlens,
            "seqlens": text_seqlens_i32,
            "seqlens_tuple": tuple(text_seqlens_i32.tolist()),
            "position_id": text_position_ids,
            "max_seq_len": int(text_seqlens_i32.max().item()),
        },
        "pixel": {
            "cu_seqlens": pixel_cu_seqlens,
            "seqlens": pixel_seqlens_tensor,
            "seqlens_tuple": tuple(pixel_seqlens_tensor.tolist()),
            "position_id": pixel_position_ids,
            "max_seq_len": pixel_seqlen,
            "cond_seqlens": torch.zeros(n_samples, dtype=torch.int32, device=device),
        },
        "mix": {
            "cu_seqlens": mix_cu_seqlens,
            "seqlens": mix_seqlens,
            "seqlens_tuple": tuple(mix_seqlens.tolist()),
            "position_id": mix_position_ids,
            "max_seq_len": int(mix_seqlens.max().item()),
            "cond_seqlens": torch.zeros(n_samples, dtype=torch.int32, device=device),
        },
    }
    return rope_meta


class JoyOV2DenoisingStage(PipelineStage):
    """JoyO V2 packed denoising stage implementing Joytron's eval loop."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    @property
    def parallelism_type(self):
        return StageParallelismType.REPLICATED

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        return [
            ComponentUse(
                stage_name=stage_name,
                component_name="transformer",
                phase="transformer",
                preferred_ready_after_request=True,
                memory_intensive=True,
            )
        ]

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        pipeline_config = server_args.pipeline_config
        device = batch.latents.device
        target_dtype = PRECISION_TO_TYPE.get(pipeline_config.precision, torch.bfloat16)

        do_cfg = batch.do_classifier_free_guidance
        guidance_scale = batch.guidance_scale
        num_steps = batch.num_inference_steps
        timeshift = batch.timeshift
        zero_cfg_star_step = batch.zero_cfg_star_step

        # Latent dimensions
        vae_cfg = pipeline_config.vae_config.arch_config
        latent_h = batch.height // vae_cfg.scale_factor_spatial
        latent_w = batch.width // vae_cfg.scale_factor_spatial
        latent_t = (batch.num_frames - 1) // vae_cfg.scale_factor_temporal + 1
        latent_channels = vae_cfg.z_dim

        # --- 1. Prepare latents ---
        latents = batch.latents  # (B, C, T, H, W) from latent prep stage
        batch_size = latents.shape[0]

        if do_cfg:
            latents = torch.cat([latents, latents], dim=0)  # (2B, C, T, H, W)

        num_samples = latents.shape[0]

        # Flatten: (N, C, T, H, W) → (S_pixel_total, C)
        # Joytron patchfy: rearrange (C, T, H, W) → (T*H*W, C) for patch_size=(1,1,1)
        hidden_states = rearrange(latents, "n c t h w -> (n t h w) c").to(target_dtype)

        # --- 2. Get text embeddings ---
        text_embeddings = batch.prompt_embeds[0].to(target_dtype)  # (S_text_total, D)
        text_seqlens = batch.joyo_text_seqlens  # (num_samples,) int32

        # --- 3. Build rope_meta ---
        rope_meta = build_rope_meta(text_seqlens, latent_t, latent_h, latent_w, device)

        # --- 4. Compute sigma schedule ---
        sigmas = sd3_timeshift(
            torch.linspace(1, 0, num_steps + 1, device=device), timeshift
        )

        # --- 5. Transformer use ---
        transformer_use = ComponentUse(
            self.__class__.__name__,
            "transformer",
            phase="transformer",
            preferred_ready_after_request=True,
            memory_intensive=True,
        )
        manager = self._component_residency_manager
        manager.begin_use(transformer_use, module=self.transformer)

        # --- 6. Denoising loop ---
        for step_idx in range(num_steps):
            timestep = sigmas[step_idx: step_idx + 1].expand(num_samples)

            with set_forward_context(
                current_timestep=sigmas[step_idx],
                forward_batch=batch,
                attn_metadata=None,
            ):
                noise_pred = self.transformer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=text_embeddings,
                    timestep=timestep,
                    rope_meta=rope_meta,
                )

            # Handle tuple output (pixel_pred, audio_pred)
            if isinstance(noise_pred, tuple):
                noise_pred = noise_pred[0]

            # denoise_update (matches Joytron GaussianDenoiser.denoise_update)
            hidden_states = self._denoise_update(
                hidden_states, noise_pred, sigmas, step_idx,
                guidance_scale, zero_cfg_star_step, do_cfg,
            )

        manager.end_use(transformer_use)

        # --- 7. Extract pos half and reshape ---
        if do_cfg:
            # hidden_states is (2 * S_pixel_per_sample, C), take first half
            s_per_sample = latent_t * latent_h * latent_w
            hidden_states = hidden_states[: batch_size * s_per_sample]

        # Reshape: (B*T*H*W, C) → (B, C, T, H, W)
        batch.latents = rearrange(
            hidden_states.float(),
            "(n t h w) c -> n c t h w",
            n=batch_size, t=latent_t, h=latent_h, w=latent_w,
        )

        return batch

    @staticmethod
    def _denoise_update(
        hidden_states: torch.Tensor,
        noise_pred: torch.Tensor,
        sigmas: torch.Tensor,
        step_idx: int,
        guidance_scale: float,
        zero_cfg_star_step: int,
        do_cfg: bool,
    ) -> torch.Tensor:
        """Joytron GaussianDenoiser.denoise_update — CFG + Euler step."""
        dt = sigmas[step_idx + 1] - sigmas[step_idx]

        hidden_states = hidden_states.float()
        noise_pred = noise_pred.float()

        if do_cfg and guidance_scale > 1.0:
            hs_pos, hs_neg = hidden_states.chunk(2)
            np_pos, np_neg = noise_pred.chunk(2)

            if zero_cfg_star_step >= 0:
                dot_product = torch.sum(np_neg * np_pos)
                squared_norm = torch.sum(np_neg ** 2) + 1e-8
                st_star = dot_product / squared_norm
            else:
                st_star = 1.0

            combined = np_neg * st_star + guidance_scale * (np_pos - np_neg * st_star)

            if step_idx >= zero_cfg_star_step:
                hs_pos = hs_pos + combined * dt
            # else: hs_pos unchanged (skip update before zero_cfg_star_step)

            hidden_states = torch.cat([hs_pos, hs_pos], dim=0)
        else:
            if step_idx >= zero_cfg_star_step:
                hidden_states = hidden_states + noise_pred * dt

        return hidden_states.to(torch.bfloat16)
