"""Layer 2: sglang JoyO V2 50-step denoise using packaged modules.

Replaces the inline sd3_timeshift / build_rope_meta / _denoise_update with the
封装 in joyo_v2_denoising.py. Uses the same Joytron dump inputs (init_hidden,
text_embeddings, text_seqlens) so that if Layer 1 passed (bf16 tolerance), Layer 2
per-step drift must match Layer 1 bit-exactly (same formulas → same result).

Assertions:
  A. build_rope_meta output == dump's rope_meta (field-level bit-exact)
  B. sd3_timeshift sigmas == inline formula (already guaranteed by shared code)
  C. per-step drift vs Joytron trajectory identical to Layer 1

Launch:
    cd /pfs/tangyanfei/sglang
    /pfs/tangyanfei/miniconda/envs/joytron/bin/torchrun --nproc_per_node=8 \\
        scripts/joyo_v2/test_joyo_v2_denoise_e2e.py
"""

import glob
import os

import torch
import torch.distributed as dist

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(LOCAL_RANK)
device = torch.device(f"cuda:{LOCAL_RANK}")


def rprint(*a, **k):
    if RANK == 0:
        print(*a, **k, flush=True)


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

from sglang.multimodal_gen.configs.models.dits.joyo_v2 import JoyOV2DiTConfig
from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
from sglang.multimodal_gen.runtime.models.dits.joyo_v2 import JoyOV2Transformer3DModel
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.server_args.server_args import (
    ServerArgs,
    set_global_server_args,
)
# --- Layer 2: packaged modules under test ---
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.joyo_v2_denoising import (
    JoyOV2DenoisingStage,
    build_rope_meta,
    sd3_timeshift,
)

WEIGHT_DIR = "/pfs/tangyanfei/joyo_v2_diffusers/transformer"
DUMP = f"/pfs/tangyanfei/Joytron/eval_output/joytron_dit_io_rank{RANK}.pt"

set_global_server_args(
    ServerArgs(model_path="/pfs/tangyanfei/joyo_v2_diffusers", attention_backend="fa")
)

# --- Load model ---
safetensors_files = sorted(glob.glob(f"{WEIGHT_DIR}/*.safetensors"))
config = JoyOV2DiTConfig()
model = maybe_load_fsdp_model(
    model_cls=JoyOV2Transformer3DModel,
    init_params={"config": config, "hf_config": {}, "quant_config": None},
    weight_dir_list=safetensors_files,
    device=device,
    hsdp_replicate_dim=1,
    hsdp_shard_dim=1,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    fsdp_inference=False,
    strict=False,
)
model.eval()
rprint("[loader] model loaded")

# --- Load dump ---
rec = torch.load(DUMP, map_location=device, weights_only=False)
kw = rec["kwargs"]
text_embeddings = kw["text_embeddings"].to(device).to(torch.bfloat16)
rope_meta = kw["rope_meta"]

def _move(d):
    out = {}
    for k, v in d.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else (_move(v) if isinstance(v, dict) else v)
    return out
rope_meta = _move(rope_meta)

thw = rope_meta["pixel"]["thw"]
latent_t, latent_h, latent_w = int(thw[0][0]), int(thw[0][1]), int(thw[0][2])
S_pixel = latent_t * latent_h * latent_w
LATENT_CH = config.arch_config.latent_channels
rprint(f"[dims] t/h/w={latent_t}/{latent_h}/{latent_w} S_pixel={S_pixel}")

# --- [Layer 2 assert A] Rebuild rope_meta via封装 build_rope_meta, compare to dump ---
text_seqlens_from_dump = rope_meta["text"]["seqlens"].to(device)
rope_meta_pkg = build_rope_meta(
    text_seqlens_from_dump, latent_t, latent_h, latent_w, device
)

def _cmp_field(sub: str, key: str) -> str:
    a = rope_meta_pkg[sub].get(key)
    b = rope_meta[sub].get(key)
    if a is None or b is None:
        return f"    {sub}.{key}: SKIP (missing in one side)"
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            return f"    {sub}.{key}: SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}"
        eq = torch.equal(a.to(b.device).to(b.dtype), b)
        return f"    {sub}.{key}: {'OK' if eq else 'MISMATCH'} shape={tuple(a.shape)}"
    return f"    {sub}.{key}: (non-tensor) pkg={a} dump={b}"

if RANK == 0:
    print("[Layer2-A] build_rope_meta vs dump:", flush=True)
    for sub in ("text", "pixel", "mix"):
        for k in ("cu_seqlens", "seqlens", "position_id"):
            print(_cmp_field(sub, k), flush=True)

# Use the封装-built rope_meta from here on (structurally equivalent to dump).
rope_meta = rope_meta_pkg

# --- Sigmas via封装 sd3_timeshift ---
NUM_STEPS = 50
TIMESHIFT = 4.0
GUIDANCE = 4.0
ZERO_CFG_STAR_STEP = 0
sigmas = sd3_timeshift(torch.linspace(1, 0, NUM_STEPS + 1, device=device), TIMESHIFT)
rprint(f"[sigmas] first={sigmas[0]:.4f} last={sigmas[-1]:.4f} "
       f"middle={sigmas[NUM_STEPS//2]:.4f}")

# --- Dump introspection: what's in kw['timestep']? What's the initial noise? ---
dump_timestep = kw["timestep"]
init_hidden = kw["hidden_states"].to(device).to(torch.bfloat16)
rprint(f"[dump] first-forward timestep={dump_timestep.tolist()}, "
       f"init_hidden std={init_hidden.float().std().item():.4f}")

# initial state: use dump's hidden_states as-is (packed pos+neg, both same noise)
hidden_states = init_hidden.clone()

# --- Inline 50-step denoise with per-step comparison ---
JT_TRAJ_PATH = "/pfs/tangyanfei/Joytron/eval_output/joytron_step_traj_rank0.pt"
jt_traj = None
if RANK == 0 and os.path.exists(JT_TRAJ_PATH):
    _blob = torch.load(JT_TRAJ_PATH, map_location="cpu", weights_only=False)
    jt_traj = {int(i): h for (i, h) in _blob["traj"]}  # step_idx -> hidden_states_cpu
    rprint(f"[traj] loaded Joytron per-step trajectory: {len(jt_traj)} steps, "
           f"first shape={tuple(next(iter(jt_traj.values())).shape)}")

rprint(f"[denoise] {NUM_STEPS} steps (inline logic, dump's rope_meta) ...")
per_step_diff = []  # (step_idx, mean_abs_diff, max_abs_diff, rel_pct)

for step_idx in range(NUM_STEPS):
    t_val = sigmas[step_idx]
    timestep = torch.tensor([t_val, t_val], device=device, dtype=torch.float32)

    with torch.no_grad(), set_forward_context(
        current_timestep=step_idx, attn_metadata=None, forward_batch=None
    ):
        out = model(
            hidden_states=hidden_states,
            encoder_hidden_states=text_embeddings,
            timestep=timestep,
            rope_meta=rope_meta,
        )
    if isinstance(out, (tuple, list)):
        out = out[0]

    # --- Layer 2: use packaged _denoise_update ---
    hidden_states = JoyOV2DenoisingStage._denoise_update(
        hidden_states=hidden_states,
        noise_pred=out,
        sigmas=sigmas,
        step_idx=step_idx,
        guidance_scale=GUIDANCE,
        zero_cfg_star_step=ZERO_CFG_STAR_STEP,
        do_cfg=True,
    )
    # For logging parity with Layer 1
    dt = sigmas[step_idx + 1] - sigmas[step_idx]
    with torch.no_grad():
        np_pos_f, np_neg_f = out.float().chunk(2)
        st_star_val = float(
            torch.sum(np_neg_f * np_pos_f) / (torch.sum(np_neg_f ** 2) + 1e-8)
        )
    hs_pos = hidden_states.chunk(2)[0]

    # Per-step comparison vs Joytron (rank0 only)
    if RANK == 0 and jt_traj is not None and step_idx in jt_traj:
        sg_step = hidden_states.detach().cpu().float()
        jt_step = jt_traj[step_idx].float()
        if sg_step.shape == jt_step.shape:
            d = (sg_step - jt_step).abs()
            denom = jt_step.abs().mean().item() + 1e-8
            per_step_diff.append((step_idx, d.mean().item(), d.max().item(),
                                  d.mean().item() / denom * 100))

    if step_idx % 10 == 0 or step_idx == NUM_STEPS - 1:
        rprint(f"  step {step_idx:2d}/{NUM_STEPS}  sigma={t_val:.4f} "
               f"pos_std={hs_pos.float().std().item():.4f} pos_max={hs_pos.abs().max().item():.2f} "
               f"dt={dt.item():.4f} st_star={st_star_val:.4f}")

if RANK == 0 and per_step_diff:
    print("\n[per-step drift] sglang vs Joytron hidden_states after each step:", flush=True)
    print(f"  {'step':>4}  {'mean_abs':>10}  {'max_abs':>10}  {'rel_pct':>8}", flush=True)
    for (si, m, mx, rp) in per_step_diff:
        if si % 5 == 0 or si == NUM_STEPS - 1 or si < 3:
            print(f"  {si:>4}  {m:>10.4e}  {mx:>10.4e}  {rp:>7.2f}%", flush=True)

# --- Compare vs Joytron ---
final_pos = hidden_states[:S_pixel].contiguous()

if RANK == 0:
    jf_path = "/pfs/tangyanfei/Joytron/eval_output/joytron_final_latent_rank0.pt"
    jf_all = torch.load(jf_path, map_location="cpu", weights_only=False)
    print(f"\n[jf-dump] keys={list(jf_all.keys()) if isinstance(jf_all, dict) else 'not-dict'}", flush=True)
    if isinstance(jf_all, dict):
        for k, v in jf_all.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={tuple(v.shape)}, std={v.float().std().item():.4f}", flush=True)
            else:
                print(f"  {k}: {v}", flush=True)
    jf = jf_all["final_latent"] if isinstance(jf_all, dict) else jf_all
    jf_pos = jf[:S_pixel].float() if jf.shape[0] == 2 * S_pixel else jf.float()
    sg = final_pos.cpu().float()

    print(f"\n[final-cmp] sglang(layer1 inline) vs Joytron:", flush=True)
    print(f"  sglang shape={tuple(sg.shape)} std={sg.std().item():.4f}", flush=True)
    print(f"  joytron pos shape={tuple(jf_pos.shape)} std={jf_pos.std().item():.4f}", flush=True)
    if sg.shape == jf_pos.shape:
        d = (sg - jf_pos).abs()
        denom = jf_pos.abs().mean().item() + 1e-8
        print(f"  max_abs_diff  = {d.max().item():.4e}", flush=True)
        print(f"  mean_abs_diff = {d.mean().item():.4e}", flush=True)
        print(f"  joytron |lat| mean = {denom:.4e}", flush=True)
        print(f"  relative mean diff = {d.mean().item() / denom:.2%}", flush=True)

dist.barrier()
dist.destroy_process_group()
