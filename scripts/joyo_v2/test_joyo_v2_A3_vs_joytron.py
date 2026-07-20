"""A3: compare sglang JoyO V2 packed forward vs Joytron dumped I/O (TP=8, SP=1).

Loads the per-rank Joytron DiT forward dump (inputs + output), feeds the SAME
inputs into the sglang packed forward, and reports max/mean abs diff.

Prereq: A2 dump exists at /pfs/tangyanfei/Joytron/eval_output/joytron_dit_io_rank{R}.pt
(one per rank, from the Joytron eval run).

Launch:
    cd /pfs/tangyanfei/sglang
    /pfs/tangyanfei/miniconda/envs/sglang/bin/torchrun \\
        --nproc_per_node=8 --nnodes=1 --node_rank=0 \\
        --master_addr=127.0.0.1 --master_port=29500 \\
        scripts/joyo_v2/test_joyo_v2_A3_vs_joytron.py
"""

import glob
import os
import sys

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


# --- sglang distributed init (TP=8, EP=8, SP=1) ---
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
rprint(f"[boot] TP={WORLD_SIZE}, SP=1")

from sglang.multimodal_gen.configs.models.dits.joyo_v2 import JoyOV2DiTConfig
from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
from sglang.multimodal_gen.runtime.models.dits.joyo_v2 import JoyOV2Transformer3DModel
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.server_args.server_args import (
    ServerArgs,
    set_global_server_args,
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

# --- Load Joytron dump for this rank ---
rec = torch.load(DUMP, map_location=device, weights_only=False)
kw = rec["kwargs"]
joytron_out = rec["output"]
if isinstance(joytron_out, (tuple, list)):
    joytron_out = joytron_out[0]  # pixel prediction
joytron_out = joytron_out.to(device)

hidden_states = kw["hidden_states"].to(device).to(torch.bfloat16)
text_embeddings = kw["text_embeddings"].to(device).to(torch.bfloat16)
timestep = kw["timestep"].to(device)
rope_meta = kw["rope_meta"]

# Move rope_meta tensors to device
def _move(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = _move(v)
        else:
            out[k] = v
    return out

rope_meta = _move(rope_meta)

rprint(f"[input] hidden={tuple(hidden_states.shape)} text={tuple(text_embeddings.shape)} "
       f"timestep={timestep.tolist()}")
rprint(f"[input] mix.cu_seqlens={rope_meta['mix']['cu_seqlens'].tolist()}")

# --- Stage 1: verify timestep embedding (SP-independent, per-sample) ---
joytron_head = rec.get("head_out")
if joytron_head is not None and RANK == 0:
    jt_temb = joytron_head[1].to(device).float()  # (2, 36864)
    with torch.no_grad():
        t_emb_freq = model._timestep_embedding(timestep * 1000, 256).to(torch.bfloat16)
        sg_temb = model.t_embedder(t_emb_freq).float()  # (2, 36864)
    d = (sg_temb - jt_temb).abs()
    print(f"[stage1 t_embedder] sglang={tuple(sg_temb.shape)} joytron={tuple(jt_temb.shape)}", flush=True)
    print(f"[stage1 t_embedder] max_diff={d.max().item():.4e} mean_diff={d.mean().item():.4e}", flush=True)

dist.barrier()

# --- Ablation: inject Joytron's timestep_emb to isolate t_embedder from the rest ---
INJECT_JOYTRON_TEMB = os.environ.get("INJECT_TEMB", "0") == "1"
if INJECT_JOYTRON_TEMB and joytron_head is not None:
    _jt_temb = joytron_head[1].to(device).to(torch.bfloat16)  # (n_samples, factor*D)
    _orig_temb_fwd = model.t_embedder.forward
    def _temb_override(x):
        return _jt_temb
    model.t_embedder.forward = _temb_override
    rprint("[ablation] t_embedder output overridden with Joytron dump")

# --- Per-layer rel-diff: gather Joytron probe layers, hook sglang, compare ---
PROBE_LAYERS = [0, 1, 5, 15, 25, 35, 45, 50]

def _gather_full(sharded):
    """(S_local, 1, D) sharded across ranks -> (S_total, D) full."""
    t = sharded.to(device)
    g = [torch.empty_like(t) for _ in range(WORLD_SIZE)]
    dist.all_gather(g, t.contiguous())
    return torch.cat(g, dim=0).squeeze(1).float()

joytron_full_mix = _gather_full(joytron_head[0])   # after refine+pack (pre L0)
joytron_layers = {}
for li in PROBE_LAYERS:
    key = f"layer{li}_out"
    if rec.get(key) is not None:
        joytron_layers[li] = _gather_full(rec[key])

_captured = {}
def _pre_hook(mod, inp):
    if "mix_in" not in _captured:
        _captured["mix_in"] = inp[0].detach()   # (1, S_total, D)
def _mk_post(i):
    def _hook(mod, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        _captured[i] = t.detach()
    return _hook
_hooks = [model.layers[0].register_forward_pre_hook(_pre_hook)]
for li in PROBE_LAYERS:
    _hooks.append(model.layers[li].register_forward_hook(_mk_post(li)))

# --- sglang forward ---
with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=None):
    sglang_out = model(
        hidden_states=hidden_states,
        encoder_hidden_states=text_embeddings,
        timestep=timestep,
        rope_meta=rope_meta,
        audio_hidden_states=kw.get("audio_hidden_states", None),
    )
if isinstance(sglang_out, (tuple, list)):
    sglang_out = sglang_out[0]

for _h in _hooks:
    _h.remove()

# --- Per-layer rel-diff (rank0): watch how rel grows across depth ---
def _cmp(name, sg, jt):
    if sg is None or jt is None:
        return
    sg = sg.squeeze(0).float() if sg.dim() == 3 else sg.float()
    if sg.shape != jt.shape:
        print(f"[{name}] SHAPE MISMATCH sglang={tuple(sg.shape)} joytron={tuple(jt.shape)}", flush=True)
        return
    d = (sg - jt).abs()
    denom = jt.abs().mean().item() + 1e-8
    print(f"[{name}] max_diff={d.max().item():.3e} mean_diff={d.mean().item():.3e} "
          f"rel={d.mean().item()/denom:.2%}", flush=True)

if RANK == 0:
    _cmp("pre-L0 (packed-mix)", _captured.get("mix_in"), joytron_full_mix)
    for li in PROBE_LAYERS:
        _cmp(f"after-L{li}", _captured.get(li), joytron_layers.get(li))

# --- Compare final output ---
so = sglang_out.float()
jo = joytron_out.float()

# Gather per-rank diff to rank0
diff = (so - jo).abs()
max_diff = diff.max()
mean_diff = diff.mean()
dist.all_reduce(max_diff, op=dist.ReduceOp.MAX)
dist.all_reduce(mean_diff, op=dist.ReduceOp.SUM)
mean_diff = mean_diff / WORLD_SIZE

if RANK == 0:
    print(f"\n=== A3: sglang packed forward vs Joytron (TP=8, SP=1, bf16) ===", flush=True)
    print(f"  sglang_out={tuple(sglang_out.shape)} joytron_out={tuple(joytron_out.shape)}", flush=True)
    print(f"  global max_abs_diff  = {max_diff.item():.4e}", flush=True)
    print(f"  global mean_abs_diff = {mean_diff.item():.4e}", flush=True)
    # relative
    denom = jo.abs().mean().item() + 1e-8
    print(f"  joytron |out| mean   = {denom:.4e}", flush=True)
    print(f"  relative mean diff   = {mean_diff.item()/denom:.2%}", flush=True)
    if max_diff.item() < 5e-2:
        print("  PASS: within bf16 tolerance", flush=True)
    else:
        print("  DIFF too large — investigate", flush=True)

# --- Isolated single-layer test: feed Joytron's layer_{i} output into sglang
#     layer_{i+1}, compare sglang output vs Joytron layer_{i+1}. Same input =>
#     measures the NET per-layer deviation (no accumulation). ---
mix_meta = rope_meta["mix"]
timestep_emb_j = joytron_head[1].to(device).to(torch.bfloat16)  # (2, factor*D)
freqs_cis_full = model.rotary_emb(mix_meta["position_id"])
seqlens_mix = mix_meta["seqlens"]
seqlens_tuple_mix = mix_meta.get("seqlens_tuple")
attn_mask, attn_mask_meta = model._build_attn_mask(mix_meta, device)

def _run_layer(layer_idx, x_full_2d):
    """Run one sglang main layer on a full (S_total, D) input."""
    x_in = x_full_2d.to(torch.bfloat16).unsqueeze(0)  # (1, S, D)
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=None):
        out = model.layers[layer_idx](
            x_in,
            timestep_emb=timestep_emb_j,
            freqs_cis=freqs_cis_full,
            seqlens=seqlens_mix,
            seqlens_tuple=seqlens_tuple_mix,
            attn_mask=attn_mask,
            attn_mask_meta=attn_mask_meta,
        )
    return out.squeeze(0)  # (S, D)

# Isolated tests: input = Joytron layer_{i} output (identical on both sides)
isolate_pairs = [(0, joytron_full_mix, joytron_layers.get(0)),   # dense L0: in=packed-mix
                 (1, joytron_layers.get(0), joytron_layers.get(1))]  # MoE L1: in=L0 out
for li, jt_in, jt_out in isolate_pairs:
    if jt_in is None or jt_out is None:
        continue
    sg_out = _run_layer(li, jt_in).float()
    if RANK == 0:
        d = (sg_out - jt_out).abs()
        denom = jt_out.abs().mean().item() + 1e-8
        print(f"[isolate L{li}] in=Joytron, max_diff={d.max().item():.3e} "
              f"mean_diff={d.mean().item():.3e} rel={d.mean().item()/denom:.3%}", flush=True)

dist.barrier()
dist.destroy_process_group()
