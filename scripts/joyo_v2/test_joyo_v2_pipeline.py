"""Layer 3: JoyO V2 端到端 pipeline 验证（含真实 Qwen3-VL text encoding）.

区别于 Layer 2（`test_joyo_v2_denoise_e2e.py` 使用 dump 的 text_embeddings）：
本脚本从 dump 的 raw caption 出发，走 Qwen3-VL encoder 拿到 text_embeddings，
再喂到 sglang 封装的 denoise 逻辑，逐 stage 与 Joytron dump 对比。

关键消除变量：
  - caption：从 `joytron_layer3_raw_inputs_rank0.pt` 读（消除 prompt jsonl 不可达）
  - raw_pixel：同上（消除 seed=12345 CPU generator）
  ⇒ sglang 与 Joytron 唯一可能的差异 = 封装模块本身的正确性

对齐断言（rank 0）：
  S1. text_embeddings vs dump.kwargs['text_embeddings']  （bf16 tolerance）
  S2. text_seqlens vs dump.rope_meta['text']['seqlens']  （bit-exact）
  S3. packed init hidden_states vs dump.kwargs['hidden_states'] （bf16 tolerance）
  S4. final latent vs joytron_final_latent_rank0.pt  （bf16 tolerance）

Launch:
    cd /pfs/tangyanfei/sglang
    /pfs/tangyanfei/miniconda/envs/joytron/bin/torchrun --nproc_per_node=8 \\
        scripts/joyo_v2/test_joyo_v2_pipeline.py
"""

import glob
import os
import time

import torch
import torch.distributed as dist
from einops import rearrange

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])

torch.cuda.set_device(LOCAL_RANK)
device = torch.device(f"cuda:{LOCAL_RANK}")


def rprint(*a, **k):
    if RANK == 0:
        print(*a, **k, flush=True)


# =========================================================================
# 1. Init dist
# =========================================================================
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)

init_distributed_environment(
    world_size=WORLD_SIZE, rank=RANK, local_rank=LOCAL_RANK, device_id=device
)
initialize_model_parallel(
    tensor_parallel_degree=WORLD_SIZE,
    sequence_parallel_degree=1,
    ulysses_degree=1,
    ring_degree=1,
    data_parallel_size=1,
    pipeline_parallel_degree=1,
    classifier_free_guidance_degree=1,
)
rprint(f"[init] TP={WORLD_SIZE}, SP=1, device={device}")

from sglang.multimodal_gen.configs.models.dits.joyo_v2 import JoyOV2DiTConfig
from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.joyo_v2 import (
    JoyOV2Transformer3DModel,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.joyo_v2 import (
    safe_devide_append,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.joyo_v2_denoising import (
    JoyOV2DenoisingStage,
    build_rope_meta,
    sd3_timeshift,
)
from sglang.multimodal_gen.runtime.server_args.server_args import (
    ServerArgs,
    set_global_server_args,
)

MODEL_PATH = "/pfs/tangyanfei/joyo_v2_diffusers"
DUMP_DIR = "/pfs/tangyanfei/Joytron/eval_output"
L3_INPUTS = os.path.join(DUMP_DIR, "joytron_layer3_raw_inputs_rank0.pt")
DIT_IO = os.path.join(DUMP_DIR, f"joytron_dit_io_rank{RANK}.pt")
FINAL_LATENT = os.path.join(DUMP_DIR, "joytron_final_latent_rank0.pt")

NUM_STEPS = 50
TIMESHIFT = 4.0
GUIDANCE = 4.0
ZERO_CFG_STAR_STEP = 0

set_global_server_args(
    ServerArgs(model_path=MODEL_PATH, attention_backend="fa")
)


# =========================================================================
# 2. Load Layer 3 baseline (caption + raw_pixel) — replaces prompt+seed
# =========================================================================
rprint("\n[STEP 1] Load Layer 3 baseline (caption + raw_pixel)...")
l3 = torch.load(L3_INPUTS, map_location="cpu", weights_only=False)
captions = l3["caption"]                    # [pos_with_template, neg_with_template]
raw_pixel = l3["raw_pixel"]                 # Tensor (2, 128, 41, 6, 11) fp32 OR list
num_samples = len(captions)
do_cfg = GUIDANCE > 1.0

if isinstance(raw_pixel, torch.Tensor):
    latent_5d = raw_pixel                   # (N, C, T, H, W)
else:
    latent_5d = torch.stack(list(raw_pixel), dim=0)

_, latent_channels, latent_t, latent_h, latent_w = latent_5d.shape
S_pixel = latent_t * latent_h * latent_w
rprint(f"  captions: {num_samples} samples")
rprint(f"  raw_pixel: shape={tuple(latent_5d.shape)} dtype={latent_5d.dtype}")
rprint(f"  latent dims: t={latent_t} h={latent_h} w={latent_w} S_pixel={S_pixel}")


# =========================================================================
# 3. Load dump baselines for stage-by-stage compare
# =========================================================================
rprint("\n[STEP 2] Load dump baselines for stage compare...")
dump = torch.load(DIT_IO, map_location="cpu", weights_only=False)
kw = dump["kwargs"]
dump_text_embeddings = kw["text_embeddings"]        # (S_text_total, D) bf16
dump_init_hidden = kw["hidden_states"]              # (2*S_pixel, C) bf16
dump_rope_meta = kw["rope_meta"]
dump_text_seqlens = dump_rope_meta["text"]["seqlens"]  # (2,) int32
rprint(f"  dump text_embeddings: {tuple(dump_text_embeddings.shape)} {dump_text_embeddings.dtype}")
rprint(f"  dump init hidden: {tuple(dump_init_hidden.shape)} {dump_init_hidden.dtype}")
rprint(f"  dump text_seqlens: {dump_text_seqlens.tolist()}")


# =========================================================================
# 4. Text encoding (rank 0 only, then broadcast) — mirror JoyOV2TextEncodingStage
# =========================================================================
rprint("\n[STEP 3] Text encoding via Qwen3-VL + safe_devide_append...")

target_dtype = torch.bfloat16
devide_denominator = 8

if RANK == 0:
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(MODEL_PATH, "tokenizer"), trust_remote_code=True
    )
    text_inputs = tokenizer(
        captions, max_length=4096, padding=True, truncation=True, return_tensors="pt"
    )
    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs["attention_mask"].to(device)

    input_ids, attention_mask = safe_devide_append(
        input_ids, attention_mask,
        num_pixels_per_sample=S_pixel,
        denominator=devide_denominator,
    )
    rprint(f"  post-pad input_ids: {tuple(input_ids.shape)}, "
           f"valid_lens={attention_mask.sum(dim=1).tolist()}")

    t0 = time.time()
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        os.path.join(MODEL_PATH, "text_encoder"),
        torch_dtype=target_dtype,
    ).to(device).eval()

    with torch.no_grad():
        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    last_hidden_state = getattr(
        outputs, "last_hidden_state", outputs.hidden_states[-1]
    ).to(target_dtype)

    mask_bool = attention_mask.bool()
    text_embeddings = last_hidden_state[mask_bool]                  # (S_text, D)
    text_seqlens = mask_bool.sum(dim=1).to(torch.int32)             # (num_samples,)

    del text_encoder, outputs
    torch.cuda.empty_cache()
    rprint(f"  encoded in {time.time()-t0:.1f}s: "
           f"text_embeddings={tuple(text_embeddings.shape)} "
           f"seqlens={text_seqlens.tolist()}")

    # ---- [Assert S1+S2] vs dump ----
    print("\n[Layer3-S1] text_seqlens vs dump:", flush=True)
    if text_seqlens.shape == dump_text_seqlens.shape:
        eq = torch.equal(text_seqlens.cpu(), dump_text_seqlens.cpu())
        print(f"  sglang={text_seqlens.tolist()} dump={dump_text_seqlens.tolist()} "
              f"→ {'OK (bit-exact)' if eq else 'MISMATCH'}", flush=True)
    else:
        print(f"  SHAPE MISMATCH sglang={tuple(text_seqlens.shape)} "
              f"vs dump={tuple(dump_text_seqlens.shape)}", flush=True)

    print("\n[Layer3-S2] text_embeddings vs dump:", flush=True)
    sg_t = text_embeddings.cpu().float()
    dp_t = dump_text_embeddings.float()
    if sg_t.shape == dp_t.shape:
        d = (sg_t - dp_t).abs()
        denom = dp_t.abs().mean().item() + 1e-8
        print(f"  shape={tuple(sg_t.shape)} "
              f"max_abs={d.max().item():.4e} mean_abs={d.mean().item():.4e} "
              f"rel_mean={d.mean().item()/denom*100:.4f}%", flush=True)
    else:
        print(f"  SHAPE MISMATCH sglang={tuple(sg_t.shape)} "
              f"vs dump={tuple(dp_t.shape)}", flush=True)
else:
    text_embeddings = None
    text_seqlens = None

dist.barrier()

# Broadcast text_embeddings + text_seqlens to all ranks
if RANK == 0:
    meta_tensor = torch.tensor(
        [text_embeddings.shape[0], text_embeddings.shape[1], num_samples],
        device=device, dtype=torch.long,
    )
else:
    meta_tensor = torch.empty(3, dtype=torch.long, device=device)
dist.broadcast(meta_tensor, src=0)
S_text, D_text, _ = int(meta_tensor[0]), int(meta_tensor[1]), int(meta_tensor[2])

if RANK != 0:
    text_embeddings = torch.empty(S_text, D_text, dtype=target_dtype, device=device)
    text_seqlens = torch.empty(num_samples, dtype=torch.int32, device=device)
dist.broadcast(text_embeddings, src=0)
dist.broadcast(text_seqlens, src=0)
rprint(f"  broadcast done: text_embeddings={tuple(text_embeddings.shape)}")


# =========================================================================
# 5. Latent prep: raw_pixel → packed hidden_states (compare vs dump)
# =========================================================================
rprint("\n[STEP 4] Prepare packed init hidden_states...")

# raw_pixel is (num_samples, C, T, H, W) fp32; Joytron eval takes it as latent directly.
# Match Joytron packed layout: cat all samples along the (T*H*W) sequence axis.
latents = latent_5d.to(device).to(target_dtype)
# Joytron patchfy: (C, T, H, W) → (T*H*W, C), then cat across samples.
hidden_states = rearrange(latents, "n c t h w -> (n t h w) c").contiguous()
rprint(f"  packed init: {tuple(hidden_states.shape)} {hidden_states.dtype}")

if RANK == 0:
    sg_h = hidden_states.cpu().float()
    dp_h = dump_init_hidden.float()
    print("\n[Layer3-S3] packed init hidden vs dump.kwargs['hidden_states']:", flush=True)
    if sg_h.shape == dp_h.shape:
        d = (sg_h - dp_h).abs()
        denom = dp_h.abs().mean().item() + 1e-8
        print(f"  shape={tuple(sg_h.shape)} "
              f"max_abs={d.max().item():.4e} mean_abs={d.mean().item():.4e} "
              f"rel_mean={d.mean().item()/denom*100:.4f}%", flush=True)
    else:
        print(f"  SHAPE MISMATCH sglang={tuple(sg_h.shape)} vs dump={tuple(dp_h.shape)}",
              flush=True)


# =========================================================================
# 6. Build rope_meta via 封装
# =========================================================================
rope_meta = build_rope_meta(text_seqlens, latent_t, latent_h, latent_w, device)
rprint(f"\n[STEP 5] rope_meta built: mix.max_seq_len={rope_meta['mix']['max_seq_len']} "
       f"text.max_seq_len={rope_meta['text']['max_seq_len']}")


# =========================================================================
# 7. Load transformer
# =========================================================================
rprint("\n[STEP 6] Loading transformer...")
transformer_path = os.path.join(MODEL_PATH, "transformer")
safetensors_files = sorted(glob.glob(f"{transformer_path}/*.safetensors"))

t0 = time.time()
config = JoyOV2DiTConfig()
model = maybe_load_fsdp_model(
    model_cls=JoyOV2Transformer3DModel,
    init_params={"config": config, "hf_config": {}, "quant_config": None},
    weight_dir_list=safetensors_files,
    device=device,
    hsdp_replicate_dim=1,
    hsdp_shard_dim=1,
    param_dtype=target_dtype,
    reduce_dtype=torch.float32,
    fsdp_inference=False,
    strict=False,
)
model.eval()
rprint(f"  loaded in {time.time()-t0:.1f}s")


# =========================================================================
# 8. Denoising loop via 封装 _denoise_update
# =========================================================================
rprint(f"\n[STEP 7] Denoising ({NUM_STEPS} steps) via packaged _denoise_update...")
sigmas = sd3_timeshift(
    torch.linspace(1, 0, NUM_STEPS + 1, device=device), TIMESHIFT
)
rprint(f"  sigmas[0]={sigmas[0]:.4f} sigmas[{NUM_STEPS//2}]={sigmas[NUM_STEPS//2]:.4f} "
       f"sigmas[-1]={sigmas[-1]:.4f}")

t0 = time.time()
for step_idx in range(NUM_STEPS):
    t_val = sigmas[step_idx]
    timestep = torch.tensor([t_val, t_val], device=device, dtype=torch.float32)

    with torch.no_grad(), set_forward_context(
        current_timestep=step_idx, attn_metadata=None, forward_batch=None
    ):
        noise_pred = model(
            hidden_states=hidden_states,
            encoder_hidden_states=text_embeddings,
            timestep=timestep,
            rope_meta=rope_meta,
        )
    if isinstance(noise_pred, (tuple, list)):
        noise_pred = noise_pred[0]

    hidden_states = JoyOV2DenoisingStage._denoise_update(
        hidden_states=hidden_states,
        noise_pred=noise_pred,
        sigmas=sigmas,
        step_idx=step_idx,
        guidance_scale=GUIDANCE,
        zero_cfg_star_step=ZERO_CFG_STAR_STEP,
        do_cfg=True,
    )

    if step_idx % 10 == 0 or step_idx == NUM_STEPS - 1:
        hs_pos = hidden_states.chunk(2)[0]
        rprint(f"  step {step_idx:2d}/{NUM_STEPS} sigma={t_val:.4f} "
               f"pos_std={hs_pos.float().std().item():.4f}")

rprint(f"  denoise done in {time.time()-t0:.1f}s")


# =========================================================================
# 9. Final compare
# =========================================================================
final_pos = hidden_states[:S_pixel].contiguous()

if RANK == 0:
    print("\n[Layer3-S4] final latent vs Joytron:", flush=True)
    jf_all = torch.load(FINAL_LATENT, map_location="cpu", weights_only=False)
    jf = jf_all["final_latent"] if isinstance(jf_all, dict) else jf_all
    jf_pos = jf[:S_pixel].float() if jf.shape[0] == 2 * S_pixel else jf.float()
    sg = final_pos.cpu().float()

    print(f"  sglang shape={tuple(sg.shape)} std={sg.std().item():.4f}", flush=True)
    print(f"  joytron pos shape={tuple(jf_pos.shape)} std={jf_pos.std().item():.4f}",
          flush=True)
    if sg.shape == jf_pos.shape:
        d = (sg - jf_pos).abs()
        denom = jf_pos.abs().mean().item() + 1e-8
        print(f"  max_abs = {d.max().item():.4e}", flush=True)
        print(f"  mean_abs = {d.mean().item():.4e}", flush=True)
        print(f"  rel_mean = {d.mean().item()/denom*100:.4f}%", flush=True)

dist.barrier()
dist.destroy_process_group()
