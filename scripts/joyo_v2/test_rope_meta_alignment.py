# SPDX-License-Identifier: Apache-2.0
"""rope_meta alignment test: sglang build_rope_meta vs Joytron PreCalRopeMeta.

Verifies that given the same inputs (text_seqlens, latent dimensions),
sglang's build_rope_meta produces identical cu_seqlens, seqlens, position_id
as Joytron's PreCalRopeMeta.forward.

Run: python scripts/joyo_v2/test_rope_meta_alignment.py
"""

import sys
sys.path.insert(0, "/pfs/tangyanfei/sglang/python")
sys.path.insert(0, "/pfs/tangyanfei/Joytron")

import torch


def build_joytron_rope_meta(text_seqlens_list, latent_t, latent_h, latent_w):
    """Reproduce Joytron PreCalRopeMeta.forward logic for T2V (no cond, no audio).

    Directly implements the logic from:
    joytron/model/submodule/text_vae_timestep_encoder.py L392-527
    with spatial_patch_size=1, temporal_patch_size=1 (actual effective values).
    """
    device = torch.device("cpu")
    n_samples = len(text_seqlens_list)

    # Joytron uses patch_size effectively = (1,1,1) based on weight shapes
    temporal_patch_size = 1
    spatial_patch_size = 1

    pixel_seqlen = (latent_t // temporal_patch_size) * (latent_h // spatial_patch_size) * (latent_w // spatial_patch_size)
    nt = latent_t // temporal_patch_size
    nh = latent_h // spatial_patch_size
    nw = latent_w // spatial_patch_size

    # --- Text ---
    txt_pos_ids_list = []
    for t_len in text_seqlens_list:
        pos = torch.arange(0, t_len, device=device, dtype=torch.int32)
        txt_pos_ids_list.append(pos.unsqueeze(0).expand(3, t_len))

    text_position_ids = torch.cat(txt_pos_ids_list, dim=1)
    text_seqlens = torch.tensor(text_seqlens_list, dtype=torch.int32, device=device)
    text_cu_seqlens = torch.zeros(n_samples + 1, dtype=torch.int32, device=device)
    text_cu_seqlens[1:] = torch.cumsum(text_seqlens, dim=0)

    # --- Pixel ---
    vis_pos_ids_list = []
    for _ in range(n_samples):
        t_pos = torch.arange(0, nt, device=device, dtype=torch.int32).view(-1, 1, 1).expand(nt, nh, nw).flatten()
        h_pos = torch.arange(nh, device=device, dtype=torch.int32).view(1, -1, 1).expand(nt, nh, nw).flatten()
        w_pos = torch.arange(nw, device=device, dtype=torch.int32).view(1, 1, -1).expand(nt, nh, nw).flatten()
        vis_pos_ids_list.append(torch.stack([t_pos, h_pos, w_pos], dim=0))

    pixel_position_ids = torch.cat(vis_pos_ids_list, dim=1)
    pixel_seqlens_tensor = torch.full((n_samples,), pixel_seqlen, dtype=torch.int32, device=device)
    pixel_cu_seqlens = torch.zeros(n_samples + 1, dtype=torch.int32, device=device)
    pixel_cu_seqlens[1:] = torch.cumsum(pixel_seqlens_tensor, dim=0)

    # --- Mix ---
    mix_pos_ids_list = []
    mix_seqlens_list = []
    for i, t_len in enumerate(text_seqlens_list):
        # text pos
        t_pos = torch.arange(0, t_len, device=device, dtype=torch.int32).unsqueeze(0).expand(3, t_len)
        # pixel pos with pre_len=t_len (temporal offset)
        pt_pos = torch.arange(t_len, t_len + nt, device=device, dtype=torch.int32).view(-1, 1, 1).expand(nt, nh, nw).flatten()
        ph_pos = torch.arange(nh, device=device, dtype=torch.int32).view(1, -1, 1).expand(nt, nh, nw).flatten()
        pw_pos = torch.arange(nw, device=device, dtype=torch.int32).view(1, 1, -1).expand(nt, nh, nw).flatten()
        p_pos = torch.stack([pt_pos, ph_pos, pw_pos], dim=0)
        mix_pos_ids_list.append(torch.cat([t_pos, p_pos], dim=1))
        mix_seqlens_list.append(t_len + pixel_seqlen)

    mix_position_ids = torch.cat(mix_pos_ids_list, dim=1)
    mix_seqlens = torch.tensor(mix_seqlens_list, dtype=torch.int32, device=device)
    mix_cu_seqlens = torch.zeros(n_samples + 1, dtype=torch.int32, device=device)
    mix_cu_seqlens[1:] = torch.cumsum(mix_seqlens, dim=0)

    return {
        "text": {
            "cu_seqlens": text_cu_seqlens,
            "seqlens": text_seqlens,
            "position_id": text_position_ids,
            "max_seq_len": int(text_seqlens.max().item()),
            "seqlens_tuple": tuple(text_seqlens.tolist()),
        },
        "pixel": {
            "cu_seqlens": pixel_cu_seqlens,
            "seqlens": pixel_seqlens_tensor,
            "position_id": pixel_position_ids,
            "max_seq_len": pixel_seqlen,
            "seqlens_tuple": tuple(pixel_seqlens_tensor.tolist()),
            "cond_seqlens": torch.zeros(n_samples, dtype=torch.int32, device=device),
        },
        "mix": {
            "cu_seqlens": mix_cu_seqlens,
            "seqlens": mix_seqlens,
            "position_id": mix_position_ids,
            "max_seq_len": int(mix_seqlens.max().item()),
            "seqlens_tuple": tuple(mix_seqlens.tolist()),
            "cond_seqlens": torch.zeros(n_samples, dtype=torch.int32, device=device),
        },
    }


def build_sglang_rope_meta(text_seqlens_list, latent_t, latent_h, latent_w):
    """Call sglang's build_rope_meta."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.joyo_v2_denoising import (
        build_rope_meta,
    )
    device = torch.device("cpu")
    text_seqlens = torch.tensor(text_seqlens_list, dtype=torch.int32, device=device)
    return build_rope_meta(text_seqlens, latent_t, latent_h, latent_w, device)


def compare_rope_meta(joytron_meta, sglang_meta, prefix=""):
    """Compare two rope_meta dicts, return True if all match."""
    all_ok = True
    for mod_key in ["text", "pixel", "mix"]:
        if mod_key not in joytron_meta or mod_key not in sglang_meta:
            print(f"  {prefix}{mod_key}: MISSING in one side")
            all_ok = False
            continue
        jm = joytron_meta[mod_key]
        sm = sglang_meta[mod_key]
        for field in ["cu_seqlens", "seqlens", "position_id", "max_seq_len", "seqlens_tuple"]:
            if field not in jm or field not in sm:
                continue
            jv = jm[field]
            sv = sm[field]
            if isinstance(jv, torch.Tensor):
                if not torch.equal(jv, sv):
                    print(f"  {prefix}{mod_key}.{field}: MISMATCH")
                    print(f"    joytron: {jv[:10]}...")
                    print(f"    sglang:  {sv[:10]}...")
                    all_ok = False
                else:
                    pass  # match
            else:
                if jv != sv:
                    print(f"  {prefix}{mod_key}.{field}: MISMATCH: {jv} vs {sv}")
                    all_ok = False
    return all_ok


def main():
    print("=" * 60)
    print("rope_meta alignment test: sglang vs Joytron (T2V, no cond)")
    print("=" * 60)

    # Test case 1: Single sample (no CFG)
    print("\n[Test 1] Single sample, text_len=256, latent=(41, 6, 11)")
    text_seqlens_1 = [256]
    latent_t, latent_h, latent_w = 41, 6, 11
    j1 = build_joytron_rope_meta(text_seqlens_1, latent_t, latent_h, latent_w)
    s1 = build_sglang_rope_meta(text_seqlens_1, latent_t, latent_h, latent_w)
    ok1 = compare_rope_meta(j1, s1)
    print(f"  Result: {'PASS' if ok1 else 'FAIL'}")
    print(f"  pixel seq_len = {latent_t * latent_h * latent_w}")
    print(f"  mix seq_len = {text_seqlens_1[0] + latent_t * latent_h * latent_w}")

    # Test case 2: CFG (2 samples with different text lengths)
    print("\n[Test 2] CFG: 2 samples, text_lens=[312, 89], latent=(41, 6, 11)")
    text_seqlens_2 = [312, 89]
    j2 = build_joytron_rope_meta(text_seqlens_2, latent_t, latent_h, latent_w)
    s2 = build_sglang_rope_meta(text_seqlens_2, latent_t, latent_h, latent_w)
    ok2 = compare_rope_meta(j2, s2)
    print(f"  Result: {'PASS' if ok2 else 'FAIL'}")
    print(f"  text cu_seqlens = {j2['text']['cu_seqlens']}")
    print(f"  mix cu_seqlens = {j2['mix']['cu_seqlens']}")

    # Test case 3: Larger resolution
    print("\n[Test 3] Single sample, text_len=500, latent=(41, 23, 40) [720p]")
    text_seqlens_3 = [500]
    lt, lh, lw = 41, 23, 40  # 720x1280 / 32
    j3 = build_joytron_rope_meta(text_seqlens_3, lt, lh, lw)
    s3 = build_sglang_rope_meta(text_seqlens_3, lt, lh, lw)
    ok3 = compare_rope_meta(j3, s3)
    print(f"  Result: {'PASS' if ok3 else 'FAIL'}")

    # Test case 4: position_id content spot check
    print("\n[Test 4] Position ID content check (single sample, latent 3x2x2)")
    text_seqlens_4 = [4]
    lt4, lh4, lw4 = 3, 2, 2
    j4 = build_joytron_rope_meta(text_seqlens_4, lt4, lh4, lw4)
    s4 = build_sglang_rope_meta(text_seqlens_4, lt4, lh4, lw4)
    ok4 = compare_rope_meta(j4, s4)
    print(f"  Result: {'PASS' if ok4 else 'FAIL'}")
    print(f"  text position_id:\n    {j4['text']['position_id']}")
    print(f"  pixel position_id:\n    {j4['pixel']['position_id']}")
    print(f"  mix position_id:\n    {j4['mix']['position_id']}")

    print("\n" + "=" * 60)
    all_pass = ok1 and ok2 and ok3 and ok4
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
