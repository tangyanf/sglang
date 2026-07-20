"""JoyO V2: 走标准 DiffGenerator.generate 链路出 mp4.

对标 Layer 4b (`test_joyo_v2_pipeline_e2e_mp4.py`)：Layer 4b 是手动装 pipeline
逐 stage 调用 + rank0 手动 VAE decode；这个脚本走的是官方 `DiffGenerator.generate`
全链路（含 pipeline 发现 / VAELoader / DecodingStage / _sample_to_uint8_frames），
用于验证 `JoyOV2T2VConfig.get_decode_scale_and_shift` 和 `post_decoding` 两个 override
是否让 pipeline 一次跑通。

**不使用 dump 的 raw_pixel**（那要求手动喂 pipeline 内部张量），仅靠一个真实 prompt
+ seed 生成，然后与 Layer 4b 的 mp4 做**目视对比**（不是 bit-exact，因为 seed / noise
生成路径两侧不同）。

Launch:
    cd /pfs/tangyanfei/sglang
    /pfs/tangyanfei/miniconda/envs/joytron/bin/python \\
        scripts/joyo_v2/test_joyo_v2_generate_e2e.py

如果需要指定 GPU：
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/joyo_v2/test_joyo_v2_generate_e2e.py
"""

import os
import sys

from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator


MODEL_PATH = "/pfs/tangyanfei/joyo_v2_diffusers"

OUT_DIR = "/pfs/tangyanfei/sglang/eval_output"
# Allow overriding the output file name via env so two runs don't overwrite each
# other (used to diff run-to-run determinism, e.g. with JOYO_V2_DETERMINISTIC=1).
OUT_FILE_NAME = os.environ.get(
    "JOYO_V2_OUT_NAME", "sglang_joyo_v2_generate_e2e.mp4"
)

# 与 Joytron eval dump 里的 pos caption 完全一致（去掉 Qwen chat template 之后的原文），
# 保证目视/数值对比的可比性。Joytron dump 见:
#   /pfs/tangyanfei/Joytron/eval_output/joytron_layer3_raw_inputs_rank0.pt (caption[0])
PROMPT = (
    "在健身房内，一男一女正在使用哑铃进行手臂力量训练。 视频中有2个核心对象。\n"
    "         ID_A （前景男子）：下身穿着黑色紧身运动长裤，脚穿黑色带白底的运动鞋，"
    "上身穿蓝色短袖T恤。身材健壮，手臂肌肉线条明显。\n"
    "         ID_B （背景女子）：下身穿着黑色七分运动紧身裤，脚穿红黑相间的运动鞋，"
    "上身穿红色短袖上衣。 视频有1个背景。这是一个室内健身房场景。地面铺设着灰黑色带有细碎斑点"
    "的橡胶地垫。背景正前方是一面巨大的墙面镜，反射出窗户的光线以及健身房内的有氧器械"
    "（如椭圆机）。画面右侧后方放置着一个多层的哑铃架，上面整齐地摆放着多排黑色哑铃。"
    "室内照明充足，整体色调偏向冷灰色和金属质感。画面具有较浅的景深，焦点主要集中在前景人物上，"
    "背景略显模糊。 近景平视向上摇摄镜头。画面初始对准前方  ID_A  的腿部和双脚，"
    "他双脚微微分开平稳地站在橡胶地垫上； ID_B  站在其右后方，同样双脚分开站立。"
    "随着镜头缓慢向上摇摄，可见  ID_A  双手各握着一个侧面印有红色字样的黑色大哑铃，"
    "自然下垂于身体前方。随后， ID_A  微微屈膝，上半身稍稍前倾，接着双臂发力，"
    "将双手中的哑铃向身体两侧逐渐向上平举，手臂肌肉随动作紧绷隆起。镜头持续向上移动，"
    "逐渐呈现出  ID_A  穿着蓝色上衣的健壮躯干和部分下巴轮廓。在后方中景处， "
    "ID_B  也是双手握着哑铃置于大腿前方。画面结束时， ID_A  正在执行哑铃侧平举的动作状态。"
)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)

    # Force deterministic algorithms so two runs with the same prompt+seed produce
    # bit-exact output. Must be set BEFORE from_pretrained() spawns the 8 workers —
    # spawn children inherit os.environ, and gpu_worker.run_scheduler_process reads
    # JOYO_V2_DETERMINISTIC to enable torch.use_deterministic_algorithms there.
    # Without it, the MoE index_add_ (atomicAdd) accumulation order varies per run,
    # producing "detail-level" differences after 50 denoise steps.
    os.environ["JOYO_V2_DETERMINISTIC"] = "1"

    generator = DiffGenerator.from_pretrained(
        model_path=MODEL_PATH,
        num_gpus=8,
        tp_size=8,
        attention_backend="fa",
        local_mode=True,
    )

    try:
        # 分辨率对齐 Joytron eval 的 `noise_shape=(1, 3, 161, 192, 352)`
        # (T, H, W)：H=192, W=352 → latent (41, 6, 11)，与 Layer 4b 用的 dump
        # raw_pixel 尺寸完全一致。
        # fps 对齐 Joytron `GaussianDenoiser.dump_pixel` 的默认 fps=16
        # (fm_trainer.py:404 未显式传 fps)。
        # negative_prompt / timeshift / zero_cfg_star_step 不传：走 registry 命中
        # JoyOV2T2VSamplingParams 的默认值（已与 eval_stage2p1_8gpu.py 对齐）。
        result = generator.generate(
            sampling_params_kwargs=dict(
                prompt=PROMPT,
                height=192,
                width=352,
                num_frames=161,
                fps=16,
                num_inference_steps=50,
                guidance_scale=4.0,
                seed=12345,
                # Joytron samples the initial noise on CPU (torch.manual_seed →
                # CPU generator). Match it so seed=12345 reproduces GT noise
                # bit-exactly; combined with JoyOV2T2VConfig.get_latent_dtype
                # returning fp32, the drawn noise == Joytron raw_pixel.
                generator_device="cpu",
                output_path=OUT_DIR,
                output_file_name=OUT_FILE_NAME,
                save_output=True,
                return_frames=False,
            )
        )
    finally:
        generator.shutdown()

    if result is None:
        print("[FAIL] generator.generate returned None", flush=True)
        return 1

    if isinstance(result, list):
        result = result[0]

    print("\n===== DiffGenerator.generate result =====", flush=True)
    print(f"  output_file_path : {getattr(result, 'output_file_path', None)}", flush=True)
    print(f"  duration         : {getattr(result, 'duration', None):.2f}s"
          if getattr(result, "duration", None) is not None else "  duration: n/a",
          flush=True)

    expected = os.path.join(OUT_DIR, OUT_FILE_NAME)
    if os.path.isfile(expected):
        size_mb = os.path.getsize(expected) / (1024 * 1024)
        print(f"[OK] mp4 saved: {expected} ({size_mb:.2f} MiB)", flush=True)
        print("\nCompare against Layer 4b:\n"
              "  sglang generate : " + expected + "\n"
              "  sglang Layer 4b : /pfs/tangyanfei/sglang/eval_output/sglang_joyo_v2_e2e.mp4\n"
              "  Joytron GT      : /pfs/tangyanfei/Joytron/eval_output/results/"
              "rebuild_dance_highmotion_sport/eval_stage2p1_8gpu/"
              "world_size-8/iter-322001/eval-0-dp-000-sap-0.mp4",
              flush=True)
        return 0

    print(f"[FAIL] expected mp4 not found at {expected}", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
