# SPDX-License-Identifier: Apache-2.0
"""JoyO V2 video generation pipeline.

Single-stream packed architecture following Joytron's eval loop:
- Qwen3-VL text encoding with chat template + safe_devide_append
- Packed CFG (pos+neg as 2 samples in one forward)
- SD3 timeshift Euler denoising with CFG Zero Star
- XVAE decoding
"""

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.joyo_v2_denoising import (
    JoyOV2DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.joyo_v2_text import (
    JoyOV2TextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class JoyOV2Pipeline(LoRAPipeline, ComposedPipelineBase):
    """JoyO V2 video generation pipeline with packed CFG denoising."""

    pipeline_name = "JoyOV2Pipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def initialize_pipeline(self, server_args: ServerArgs):
        pass

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(InputValidationStage())
        self.add_stage(
            JoyOV2TextEncodingStage(
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
            ),
            "joyo_v2_text_encoding",
        )
        self.add_standard_latent_preparation_stage()
        self.add_stage(
            JoyOV2DenoisingStage(
                transformer=self.get_module("transformer"),
            ),
            "joyo_v2_denoising",
        )
        self.add_standard_decoding_stage()


EntryClass = JoyOV2Pipeline
