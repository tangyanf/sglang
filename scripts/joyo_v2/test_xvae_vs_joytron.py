"""Verify sglang XVAE decode output matches Joytron XVAE (bit-exact).

Uses XVAE directly (not XvaeCompres32Chanel128 which applies bf16 conversion).

Usage:
    cd /pfs/tangyanfei/sglang
    python scripts/joyo_v2/test_xvae_vs_joytron.py
"""

import sys
sys.path.insert(0, "/pfs/tangyanfei/Joytron")
sys.path.insert(0, "/pfs/tangyanfei/sglang/python")

import types
_mm_stub = types.ModuleType("sglang.multimodal_gen")
_mm_stub.__path__ = ["/pfs/tangyanfei/sglang/python/sglang/multimodal_gen"]
sys.modules["sglang.multimodal_gen"] = _mm_stub

import json
import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def generate_test_latents():
    """Generate deterministic test latents."""
    torch.manual_seed(42)
    z_small = torch.randn(1, 128, 3, 2, 3)
    torch.manual_seed(123)
    z_medium = torch.randn(1, 128, 5, 3, 4)
    return {"small": z_small, "medium": z_medium}


def compare(name, joytron_out, sglang_out):
    print(f"\n[{name}]")
    print(f"  Joytron shape: {joytron_out.shape}, range=[{joytron_out.min():.4f}, {joytron_out.max():.4f}]")
    print(f"  sglang  shape: {sglang_out.shape}, range=[{sglang_out.min():.4f}, {sglang_out.max():.4f}]")

    if joytron_out.shape != sglang_out.shape:
        print(f"  SHAPE MISMATCH!")
        return False

    max_diff = (joytron_out - sglang_out).abs().max().item()
    mean_diff = (joytron_out - sglang_out).abs().mean().item()
    is_exact = torch.equal(joytron_out, sglang_out)

    print(f"  max_diff:  {max_diff:.2e}")
    print(f"  mean_diff: {mean_diff:.2e}")
    print(f"  bit-exact: {is_exact}")
    print(f"  RESULT: {'PASS ✓' if is_exact else 'FAIL ✗'}")
    return is_exact


def main():
    JOYTRON_VAE_PATH = "/pfs3/mgq/public_models/xvae_compress32_ch128_54k.pth"
    SGLANG_VAE_PATH = "/pfs/tangyanfei/joyo_v2_diffusers/vae"

    test_latents = generate_test_latents()

    # ---- Load Joytron XVAE (fp32, no bf16 conversion) ----
    print("=" * 60)
    print("Loading Joytron XVAE (fp32, direct from checkpoint)...")
    from joytron.model.submodule.pixel_vaes.x_vae import XVAE as JoytronXVAE

    ckpt = torch.load(JOYTRON_VAE_PATH, map_location="cpu")
    j_vae = JoytronXVAE(
        in_channels=3, out_channels=3, patch_size=2, latent_channels=128,
        layers_per_block=2, block_in_channels=[160, 320, 640, 1280, 1280],
        temporal_downsample=[False, True, True, False, False],
        channel_doubling=False, enable_feature_caching=True,
    )
    j_vae.load_state_dict(ckpt["vae"], strict=True)
    j_vae.eval().float()
    del ckpt
    print("  Loaded.")

    # ---- Load sglang XVAE ----
    print("\nLoading sglang XVAE...")
    from sglang.multimodal_gen.configs.models.vaes.xvae import XVAEArchConfig, XVAEConfig
    from sglang.multimodal_gen.runtime.models.vaes.xvae import AutoencoderKLXVAE

    with open(f"{SGLANG_VAE_PATH}/config.json") as f:
        vae_cfg = json.load(f)

    arch_config = XVAEArchConfig(
        latents_mean=tuple(vae_cfg["latents_mean"]) if vae_cfg.get("latents_mean") else None,
        latents_std=tuple(vae_cfg["latents_std"]) if vae_cfg.get("latents_std") else None,
    )
    config = XVAEConfig(arch_config=arch_config)
    s_vae = AutoencoderKLXVAE(config)

    state_dict = load_file(f"{SGLANG_VAE_PATH}/diffusion_pytorch_model.safetensors")
    missing, unexpected = s_vae.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys: {missing[:5]}...")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys: {unexpected[:5]}...")
    s_vae.eval().float()
    print("  Loaded.")

    # ---- Weight comparison ----
    print("\n" + "=" * 60)
    print("=== Weight Comparison ===")
    j_sd = j_vae.state_dict()
    s_sd = {k: v for k, v in s_vae.state_dict().items() if not k.startswith("latents_")}

    common_keys = set(j_sd.keys()) & set(s_sd.keys())
    weight_mismatch = []
    for k in sorted(common_keys):
        if not torch.equal(j_sd[k].float(), s_sd[k].float()):
            diff = (j_sd[k].float() - s_sd[k].float()).abs().max().item()
            weight_mismatch.append((k, diff))

    if not weight_mismatch:
        print(f"  All {len(common_keys)} weights are BIT-EXACT ✓")
    else:
        print(f"  {len(weight_mismatch)} weights DIFFER:")
        for k, d in weight_mismatch[:5]:
            print(f"    {k}: max_diff={d:.2e}")

    # ---- Decode comparison (no feature caching) ----
    print("\n" + "=" * 60)
    print("=== Decode Comparison (no feature caching) ===")
    j_vae.use_feature_caching = False
    s_vae.use_feature_caching = False

    all_pass = True
    for name, z in test_latents.items():
        with torch.no_grad():
            j_out = j_vae.decode(z.clone(), return_dict=False)[0]
            s_out = s_vae._decode(z.clone())
        passed = compare(f"{name} (no-cache)", j_out, s_out)
        if not passed:
            all_pass = False

    # ---- Decode comparison (with feature caching) ----
    print("\n" + "=" * 60)
    print("=== Decode Comparison (with feature caching) ===")
    j_vae.use_feature_caching = True
    s_vae.use_feature_caching = True

    for name, z in test_latents.items():
        with torch.no_grad():
            j_out = j_vae.decode(z.clone(), return_dict=False)[0]
            s_out = s_vae._decode(z.clone())
        passed = compare(f"{name} (cached)", j_out, s_out)
        if not passed:
            all_pass = False

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"Overall: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")


if __name__ == "__main__":
    main()
