# SPDX-License-Identifier: Apache-2.0
"""JoyO V2 custom text encoding stage.

Implements Joytron's text encoding flow:
1. Wrap prompts with chat template
2. Duplicate for CFG (pos + neg packed as 2 samples)
3. Tokenize with padding
4. safe_devide_append (pad to SP-divisible length)
5. Run Qwen3-VL encoder
6. Extract valid tokens per-sample → flat text_embeddings
"""

from __future__ import annotations

import torch

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
    safe_devide_append,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class JoyOV2TextEncodingStage(PipelineStage):
    """Encode text prompts for JoyO V2 packed inference."""

    def __init__(self, text_encoder, tokenizer):
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

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
                component_name="text_encoder",
                phase="text_encoder",
                preferred_ready_after_request=True,
                memory_intensive=True,
            )
        ]

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        pipeline_config = server_args.pipeline_config
        device = torch.device("cuda")
        dtype = PRECISION_TO_TYPE.get(
            pipeline_config.text_encoder_precisions[0], torch.bfloat16
        )

        prompt_template = pipeline_config.prompt_template
        devide_denominator = pipeline_config.devide_denominator
        do_cfg = batch.guidance_scale > 1.0

        # 1. Wrap prompts with template
        prompt = batch.prompt if isinstance(batch.prompt, str) else batch.prompt
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = list(prompt)

        formatted = [prompt_template.format(p) for p in prompts]

        # 2. CFG: append negative prompts
        if do_cfg:
            neg = batch.negative_prompt
            if neg is None:
                neg = ""
            if isinstance(neg, str):
                neg_prompts = [neg] * len(formatted)
            else:
                neg_prompts = list(neg)
            formatted_neg = [prompt_template.format(n) for n in neg_prompts]
            captions = formatted + formatted_neg
        else:
            captions = formatted

        # 3. Tokenize
        max_length = 4096
        text_inputs = self.tokenizer(
            captions,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        # 4. safe_devide_append
        # Compute S_pixel for the extra_lens calculation
        vae_cfg = pipeline_config.vae_config.arch_config
        latent_h = batch.height // vae_cfg.scale_factor_spatial
        latent_w = batch.width // vae_cfg.scale_factor_spatial
        latent_t = (batch.num_frames - 1) // vae_cfg.scale_factor_temporal + 1
        num_pixels = latent_t * latent_h * latent_w
        num_samples = len(captions)
        extra_lens = (num_pixels * num_samples) % devide_denominator

        input_ids, attention_mask = safe_devide_append(
            input_ids,
            attention_mask,
            num_pixels_per_sample=num_pixels,
            denominator=devide_denominator,
        )

        # 5. Encode with Qwen3-VL
        text_encoder_use = ComponentUse(
            self.__class__.__name__,
            "text_encoder",
            phase="text_encoder",
            preferred_ready_after_request=True,
            memory_intensive=True,
        )
        manager = self._component_residency_manager
        manager.begin_use(text_encoder_use, module=self.text_encoder)

        with torch.no_grad():
            with set_forward_context(
                current_timestep=0,
                attn_metadata=None,
                forward_batch=batch,
            ):
                outputs = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
        # Joytron extracts the encoder's post-final-norm output as text
        # embeddings. sglang's Qwen3VLTextModel collects all_hidden_states
        # *before* the final RMSNorm, so outputs.hidden_states[-1] is the
        # un-normalized residual (std ~30, vs Joytron GT std ~3.09). The top-level
        # Qwen3VLForConditionalGeneration output drops last_hidden_state, so we
        # reapply the encoder's own final norm here. Verified: applying this norm
        # to the raw hidden_states[-1] matches Joytron GT within allclose(1e-2).
        final_norm = self.text_encoder.model.language_model.norm
        last_hidden_state = final_norm(outputs.hidden_states[-1]).to(dtype)

        manager.end_use(text_encoder_use)

        # 6. Extract valid tokens per-sample → flat packed text_embeddings
        mask_bool = attention_mask.bool()
        text_seqlens = mask_bool.sum(dim=1).to(torch.int32)
        text_embeddings = last_hidden_state[mask_bool]  # (S_text_total, D)

        # --- DEBUG: dump text encoding intermediates on rank0 ---
        import os
        _dump_dir = os.environ.get("JOYO_V2_DUMP_DIR", "")
        if _dump_dir:
            try:
                import torch.distributed as dist
                _rank = dist.get_rank() if dist.is_initialized() else 0
            except Exception:
                _rank = 0
            if _rank == 0:
                os.makedirs(_dump_dir, exist_ok=True)
                _dump_path = os.path.join(_dump_dir, "sglang_text_stage_output.pt")
                torch.save(
                    {
                        "captions": captions,
                        "input_ids": input_ids.detach().cpu(),
                        "attention_mask": attention_mask.detach().cpu(),
                        "text_seqlens": text_seqlens.detach().cpu(),
                        "text_embeddings": text_embeddings.detach().cpu(),
                        "last_hidden_state_shape": list(last_hidden_state.shape),
                        "last_hidden_state_dtype": str(last_hidden_state.dtype),
                        "do_cfg": do_cfg,
                        "num_pixels": num_pixels,
                        "num_samples": num_samples,
                        "extra_lens": extra_lens,
                        "devide_denominator": devide_denominator,
                    },
                    _dump_path,
                )
                logger.info(f"[JOYO_V2_DUMP] saved text-stage output to {_dump_path}")

        # Store results in batch
        batch.prompt_embeds = [text_embeddings]
        batch.joyo_text_seqlens = text_seqlens
        batch.joyo_num_captions = num_samples
        batch.do_classifier_free_guidance = do_cfg
        batch.is_prompt_processed = True

        return batch
