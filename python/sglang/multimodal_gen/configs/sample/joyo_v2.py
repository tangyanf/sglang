# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class JoyOV2T2VSamplingParams(SamplingParams):
    """Default sampling parameters for JoyO V2 T2V."""

    height: int = 480
    width: int = 832
    num_frames: int = 161
    fps: int = 24

    guidance_scale: float = 4.0
    num_inference_steps: int = 50
    timeshift: float = 4.0
    zero_cfg_star_step: int = 0

    negative_prompt: str = (
        "一段充满卡顿与形体崩坏的低画质抽象生成视频。 视频中有1个无法辨识的畸变对象。 "
        "（畸变人形）：一个形体残缺、面部模糊扭曲的插画风格轮廓，四肢比例失调，边缘布满极度锐化的锯齿。 "
        "视频有1个混乱背景。背景是由粗糙的颗粒纹理与毫无规律的密集条纹拼接而成的平面，毫无深度感。"
        "画面上方叠加着类似报错代码或残缺Logo的水印。 "
        "低画质（Low Quality）、早期CGI崩坏风、极高对比度、极度锐化处理、严重的数字颗粒与噪点污染。 "
        "视频包含1个毫无逻辑的镜头。 混乱脱节的抽搐镜头。画面中央的 处于极不稳定的卡顿移动状态，"
        "动作缺乏连贯的物理惯性，呈现出抽帧式的原位颤动。镜头伴随着毫无轨迹可寻的剧烈抖动与突兀跳帧，"
        "景别在特写与全景之间发生逻辑断层的来回抽拉，构图重心完全丧失，充斥着严重的画面撕裂与数字伪影。"
    )

    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (832, 480),
            (480, 832),
            (1280, 720),
            (720, 1280),
        ]
    )
