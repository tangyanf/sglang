# SPDX-License-Identifier: Apache-2.0
"""JoyO V2 Transformer DiT Model for sglang.

Single-stream packed architecture with:
- GQA (48Q/16KV heads, QK norm, headwise gate)
- Expert-Chosen MoE (48 experts, sigmoid router, EP=8)
- Shared expert (TP-sharded)
- DiT modulation (6-factor learnable table)
- Sandwich norm
- 3D MRoPE (interleaved)

Ported from Joytron's causal_video_model.py + grouped_query_attention.py + moe_block.py.
Uses sglang parallel primitives for TP/SP/EP.
"""

import os
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func
from einops import rearrange

from sglang.multimodal_gen.configs.models.dits.joyo_v2 import JoyOV2DiTConfig
from sglang.multimodal_gen.runtime.distributed import (
    divide,
    get_sp_group,
    get_sp_world_size,
    get_tp_group,
    get_tp_world_size,
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.weight_attrs import set_weight_attrs

logger = init_logger(__name__)

_MODULATION_FACTOR = 6

# Attention backend: "sdpa" (default, matches Joytron eval's SDPA core for
# numerical alignment) or "flash" (flash_attn_varlen, faster). Override via
# JOYO_ATTN_BACKEND=flash once SDPA alignment is confirmed.
_JOYO_ATTN_BACKEND = os.environ.get("JOYO_ATTN_BACKEND", "sdpa").lower()


# ---------------------------------------------------------------------------
# RMSNorm (bit-aligned with Joytron / diffusers JoyOV2RMSNorm)
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """RMSNorm matching Joytron's rms_norm.py and diffusers JoyOV2RMSNorm.

    Order matters for bf16 parity: normalize in fp32, cast back to input dtype,
    THEN multiply by weight — i.e. ``weight * x_normed.to(dtype)``. This differs
    from sglang's stock RMSNorm which multiplies weight in fp32 before casting.
    Since the diffusers checkpoint (bf16) was produced with this exact order,
    we replicate it here to load those weights with maximum numerical fidelity.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        # Cast the normalized activation back to the input dtype first, then
        # apply weight, matching Joytron / diffusers JoyOV2RMSNorm. Cast weight
        # too so a weight that loaded as fp32 can't silently upcast the output.
        return self.weight.to(input_dtype) * x.to(input_dtype)


# ---------------------------------------------------------------------------
# Multimodal packing (sample-outer, modality-inner) — matches Joytron
# MultiModalPack / MultiModalUnpack
# ---------------------------------------------------------------------------


def multimodal_pack(
    tensors: list[torch.Tensor], cu_seqlens_list: list[torch.Tensor]
) -> torch.Tensor:
    """Pack per-modality sequences into one packed sequence.

    Layout matches Joytron MultiModalPack: outer loop over samples, inner loop
    over modalities → ``[s0_m0, s0_m1, ..., s1_m0, s1_m1, ...]``.

    Args:
        tensors: list of (S_modality_total, D), one per modality (text, pixel, ...)
        cu_seqlens_list: list of (n_samples+1,) cu_seqlens, one per modality
    Returns:
        (S_total, D) packed
    """
    n_samples = len(cu_seqlens_list[0]) - 1
    chunks = []
    for i in range(n_samples):
        for tensor, cu in zip(tensors, cu_seqlens_list):
            chunks.append(tensor[int(cu[i]): int(cu[i + 1])])
    return torch.cat(chunks, dim=0)


def multimodal_unpack(
    mix: torch.Tensor, cu_seqlens_list: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Inverse of multimodal_pack. Returns one (S_modality_total, D) per modality."""
    n_samples = len(cu_seqlens_list[0]) - 1
    num_modalities = len(cu_seqlens_list)
    parts = [[] for _ in range(num_modalities)]
    offset = 0
    for i in range(n_samples):
        for j, cu in enumerate(cu_seqlens_list):
            seq_len = int(cu[i + 1]) - int(cu[i])
            parts[j].append(mix[offset: offset + seq_len])
            offset += seq_len
    return [torch.cat(chunks, dim=0) for chunks in parts]


# ---------------------------------------------------------------------------
# 3D MRoPE (Interleaved)
# ---------------------------------------------------------------------------


class JoyOV2RotaryEmbedding(nn.Module):
    """3D Multi-modal Rotary Embedding with interleaved layout.

    Matches Joytron's MRotaryEmbedding + mrope_native.
    mrope_section=[16, 24, 24] means: of head_dim//2=64 positions,
    T occupies 16 dims, H occupies 24, W occupies 24.
    Interleaved: T at positions 0::3, H at 1::3, W at 2::3 within each section.
    """

    def __init__(
        self,
        head_dim: int = 128,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        mrope_section: tuple[int, ...] = (16, 24, 24),
        interleaved: bool = True,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.mrope_section = mrope_section
        self.interleaved = interleaved

        # Build on CPU (bit-exact with Joytron's MRotaryEmbedding._compute_cos_sin_cache);
        # will be moved / rebuilt on first forward if the model was created on meta.
        self.register_buffer(
            "cos_sin_cache", self._build_cache(), persistent=False
        )

    def _build_cache(self) -> torch.Tensor:
        """Build cos/sin cache on CPU in fp32 (bit-exact with Joytron)."""
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        return torch.cat([freqs.cos(), freqs.sin()], dim=-1)

    def _ensure_cache(self, device: torch.device) -> None:
        """Materialize cache on ``device`` when meta-loaded or on a different device."""
        cache = self.cos_sin_cache
        if cache.is_meta or cache.device != device:
            self.cos_sin_cache = self._build_cache().to(device)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Compute freqs_cis from position_ids.

        Args:
            position_ids: (3, S) — [T, H, W] position indices
        Returns:
            freqs_cis: (S, head_dim) — packed [cos, sin] with interleaved sections
        """
        self._ensure_cache(position_ids.device)
        # Index cache: (3, S, head_dim)
        cos_sin = self.cos_sin_cache[position_ids]  # (3, S, head_dim)
        cos, sin = cos_sin.chunk(2, dim=-1)  # each (3, S, head_dim//2)

        if self.interleaved:
            cos = self._interleave(cos)
            sin = self._interleave(sin)
        else:
            cos = self._concat_sections(cos)
            sin = self._concat_sections(sin)

        return torch.cat([cos, sin], dim=-1)  # (S, head_dim)

    def _interleave(self, x: torch.Tensor) -> torch.Tensor:
        """Interleave T/H/W dimensions. x: (3, S, head_dim//2)."""
        result = x[0].clone()
        sec = self.mrope_section
        result[..., 1: sec[1] * 3: 3] = x[1, ..., 1: sec[1] * 3: 3]
        result[..., 2: sec[2] * 3: 3] = x[2, ..., 2: sec[2] * 3: 3]
        return result

    def _concat_sections(self, x: torch.Tensor) -> torch.Tensor:
        """Chunked MRoPE: select each axis's chunk."""
        return torch.cat(
            [m[i] for i, m in enumerate(x.split(list(self.mrope_section), dim=-1))],
            dim=-1,
        )


# ---------------------------------------------------------------------------
# Modulation
# ---------------------------------------------------------------------------


class JoyOV2Modulate(nn.Module):
    """Wan-style learnable modulation table. Matches Joytron ModulateWan.

    Storage: modulate_table shape (1, factor * hidden_size) — flat, not 3D.
    Forward input x has the same layout; output splits into `factor` shards
    of shape (B, hidden_size) each.
    """

    def __init__(self, hidden_size: int, factor: int = 6, dtype=None, device=None):
        super().__init__()
        self.factor = factor
        self.hidden_size = hidden_size
        self.modulate_table = nn.Parameter(
            torch.zeros(1, factor * hidden_size, dtype=dtype, device=device)
            / hidden_size**0.5,
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # x: (B, factor * hidden_size)
        return list((self.modulate_table + x).chunk(self.factor, dim=-1))


# ---------------------------------------------------------------------------
# GQA Attention (TP-sharded)
# ---------------------------------------------------------------------------


class JoyOV2Attention(nn.Module):
    """GQA with QK Norm, Headwise Gate, and USP Attention."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        norm_eps: float = 1e-6,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        tp_size = get_tp_world_size()
        self.local_num_heads = divide(num_attention_heads, tp_size)
        self.local_num_kv_heads = divide(num_kv_heads, tp_size)

        # QKV + headwise gate fused into one ColumnParallel projection
        # Output dims: [q_dim, kv_dim, gate_dim]
        q_dim = head_dim * num_attention_heads
        kv_dim = head_dim * 2 * num_kv_heads
        gate_dim = num_attention_heads  # one scalar per Q head

        self.wqkv = MergedColumnParallelLinear(
            hidden_size,
            [q_dim, kv_dim, gate_dim],
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wqkv",
        )

        self.wo = RowParallelLinear(
            head_dim * num_attention_heads,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.wo",
        )

        # QK Norm
        self.q_norm = RMSNorm(head_dim, eps=norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=norm_eps)

        # Attention core: standard flash_attn varlen (GQA + variable-length
        # native support). Joytron eval also uses flash_attn varlen, so this
        # maximizes numerical alignment. SP=1 assumed (no Ulysses all-to-all);
        # SP>1 support (head-exchange a2a) is deferred.
        self.softmax_scale = head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        attn_mask_meta: dict | None = None,
        num_replicated_suffix: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, S_local, D) — B=1 packed for JoyO V2
            freqs_cis: precomputed cos/sin for RoPE (S, D) packed
            attn_mask: (B, S) key mask for varlen attention (per-sample isolation)
            attn_mask_meta: varlen FA metadata (cu_seqlens/indices/...) built by caller
            num_replicated_suffix: for SP text suffix replication
        """
        B, S, _ = hidden_states.shape

        qkv_out, _ = self.wqkv(hidden_states)

        # Split into local Q, KV, gate
        local_q_dim = self.head_dim * self.local_num_heads
        local_kv_dim = self.head_dim * 2 * self.local_num_kv_heads
        local_gate_dim = self.local_num_heads

        q, kv, gate_weight = qkv_out.split(
            [local_q_dim, local_kv_dim, local_gate_dim], dim=-1
        )

        q = q.view(B, S, self.local_num_heads, self.head_dim)
        kv = kv.view(B, S, self.local_num_kv_heads, 2 * self.head_dim)
        k, v = kv.chunk(2, dim=-1)
        # v is a strided view of kv (last-dim stride = 2*head_dim); the attention
        # kernel needs a contiguous last dim. q/k become contiguous after norm+RoPE.
        v = v.contiguous()

        # QK Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE (applied to GQA layout: q has local_num_heads, k has local_num_kv_heads)
        if freqs_cis is not None:
            cos, sin = freqs_cis.chunk(2, dim=-1)
            q = self._apply_rotary(q, cos, sin)
            k = self._apply_rotary(k, cos, sin)

        # Attention: samples isolated by cu_seqlens. Two backends:
        #  - "sdpa" (default): per-sample SDPA, matches Joytron eval's
        #    AttentionCore exactly (Joytron eval leaves configure_optimizable
        #    commented out, so it uses the default SDPA core, NOT flash-attn).
        #  - "flash": flash_attn_varlen_func (faster; enable once aligned).
        assert attn_mask_meta is not None, "JoyO V2 attention requires cu_seqlens meta"
        cu_seqlens = attn_mask_meta["cu_seqlens"].to(torch.int32)
        max_seqlen = int(attn_mask_meta["max_seqlen"])

        if _JOYO_ATTN_BACKEND == "flash":
            q_flat = q.reshape(B * S, self.local_num_heads, self.head_dim).contiguous()
            k_flat = k.reshape(B * S, self.local_num_kv_heads, self.head_dim).contiguous()
            v_flat = v.reshape(B * S, self.local_num_kv_heads, self.head_dim).contiguous()
            attn_output = flash_attn_varlen_func(
                q_flat, k_flat, v_flat,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=0.0,
                softmax_scale=self.softmax_scale,
                causal=False,
            )
            attn_output = attn_output.reshape(B, S, self.local_num_heads, self.head_dim)
        else:
            attn_output = self._sdpa_varlen(q, k, v, cu_seqlens)

        # Headwise gate: (B, S, local_heads, 1)
        gate = gate_weight.view(B, S, self.local_num_heads, 1).sigmoid()
        attn_output = attn_output * gate

        # Reshape and output projection
        attn_output = attn_output.reshape(B, S, -1)
        output, _ = self.wo(attn_output)
        return output

    def _sdpa_varlen(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        """Per-sample SDPA matching Joytron AttentionCore (varlen path).

        q: (B=1, S, Hq, D), k/v: (1, S, Hkv, D). GQA is handled by expanding
        k/v to Hq heads (repeat_interleave), same as Joytron _maybe_expand_kv.
        Each sample's [cu[i]:cu[i+1]] slice runs an independent full-attention
        SDPA (is_causal=False), then results are concatenated.
        """
        B, S, Hq, D = q.shape
        Hkv = k.shape[2]
        # Expand GQA -> MHA (Joytron _maybe_expand_kv uses repeat_interleave)
        if Hkv != Hq:
            repeat = Hq // Hkv
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)

        # Flatten batch (B=1): (S, H, D)
        q_flat = q.reshape(S, Hq, D)
        k_flat = k.reshape(S, Hq, D)
        v_flat = v.reshape(S, Hq, D)

        cu = cu_seqlens.tolist()
        outs = []
        for i in range(len(cu) - 1):
            s, e = cu[i], cu[i + 1]
            # (1, H, seq, D)
            qs = q_flat[s:e].transpose(0, 1).unsqueeze(0)
            ks = k_flat[s:e].transpose(0, 1).unsqueeze(0)
            vs = v_flat[s:e].transpose(0, 1).unsqueeze(0)
            o = F.scaled_dot_product_attention(qs, ks, vs, is_causal=False)
            outs.append(o.squeeze(0).transpose(0, 1))  # (seq, H, D)
        attn_output = torch.cat(outs, dim=0)  # (S, H, D)
        return attn_output.reshape(B, S, Hq, D)

    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply neox-style rotary embedding. x: (B, S, H, D).

        Matches Joytron mrope_native: cos/sin are cast to x's dtype and the
        result is returned in x's dtype (so bf16 in -> bf16 out, keeping q/k/v
        dtypes aligned for the attention kernel).
        """
        orig_dtype = x.dtype
        cos = cos.to(orig_dtype).unsqueeze(0).unsqueeze(2)  # (1, S, 1, D//2)
        sin = sin.to(orig_dtype).unsqueeze(0).unsqueeze(2)
        x1, x2 = x.chunk(2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat([o1, o2], dim=-1)


# ---------------------------------------------------------------------------
# Expert-Chosen MoE (EP-sharded)
# ---------------------------------------------------------------------------


class JoyOV2ExpertDispatcher:
    """Expert-Chosen dispatcher using all_to_all_single.

    Implements the same logic as Joytron's ExpertDispatcher:
    - dispatch: route tokens to expert-owning ranks via all-to-all
    - combine: reverse all-to-all + index_add for multi-expert accumulation
    """

    def __init__(self, group, num_experts: int):
        self.group = group
        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.world_size

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        expert_token_ids: torch.Tensor,
        expert_token_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            hidden_states: (S, D) — local tokens
            expert_token_ids: (E, C) — global token indices per expert
            expert_token_weights: (E, C) — routing weights
        Returns:
            expert_input: (E_loc, C, D)
            expert_weights: (E_loc, C)
            handle: dict for combine
        """
        S, D = hidden_states.shape
        E, C = expert_token_ids.shape
        E_loc = self.num_local_experts
        P = self.world_size
        device = expert_token_ids.device

        ranks = torch.arange(P, device=device)

        # Map negative ids (sentinel) to non-real rank bucket
        cids = torch.where(expert_token_ids < 0, P * S, expert_token_ids)
        flat_cids = cids.reshape(E * C)

        # Send plan: bucket by destination rank (which rank owns the expert)
        dst_rank = torch.arange(E * C, device=device) // (C * E_loc)
        owned_by_me = flat_cids // S == self.rank
        send_bucket = torch.where(owned_by_me, dst_rank, P)
        send_counts = (send_bucket.unsqueeze(0) == ranks.unsqueeze(1)).sum(dim=1)

        # Recv plan: bucket by source rank (where tokens come from)
        my_cids = cids[self.rank * E_loc: (self.rank + 1) * E_loc].reshape(E_loc * C)
        recv_bucket = my_cids // S
        recv_counts = (recv_bucket.unsqueeze(0) == ranks.unsqueeze(1)).sum(dim=1)

        # Stable sort for deterministic ordering
        perm_send = torch.argsort(send_bucket, stable=True)
        perm_recv = torch.argsort(recv_bucket, stable=True)

        # Host readback for variable-length all-to-all
        counts_cpu = torch.cat([send_counts, recv_counts]).cpu()
        send_sizes = counts_cpu[:P].tolist()
        recv_sizes = counts_cpu[P:].tolist()
        total_send = sum(send_sizes)
        total_recv = sum(recv_sizes)

        # Gather tokens to send
        send_token_indices = (flat_cids[perm_send[:total_send]] % S).to(torch.int64)
        send_tensor = torch.index_select(hidden_states, 0, send_token_indices)

        # All-to-all
        recv_tensor = hidden_states.new_empty((total_recv, D))
        dist.all_to_all_single(
            recv_tensor, send_tensor,
            output_split_sizes=recv_sizes,
            input_split_sizes=send_sizes,
            group=self.group,
        )

        # Place received tokens into (E_loc, C, D)
        place_flat = perm_recv[:total_recv]
        place_expert = place_flat // C
        place_slot = place_flat % C
        expert_input = hidden_states.new_zeros((E_loc, C, D))
        expert_input[place_expert, place_slot] = recv_tensor

        # Local expert weights
        expert_weights = expert_token_weights[self.rank * E_loc: (self.rank + 1) * E_loc]

        handle = {
            "send_sizes": send_sizes,
            "recv_sizes": recv_sizes,
            "send_token_indices": send_token_indices,
            "place_expert": place_expert,
            "place_slot": place_slot,
            "num_local_tokens": S,
        }
        return expert_input, expert_weights, handle

    def combine(self, expert_output: torch.Tensor, handle: dict) -> torch.Tensor:
        """Reverse dispatch: scatter expert outputs back to source tokens."""
        D = expert_output.shape[-1]

        # Gather results from placed positions
        send_back = expert_output[handle["place_expert"], handle["place_slot"]]

        # Reverse all-to-all
        recv_back = expert_output.new_empty((sum(handle["send_sizes"]), D))
        dist.all_to_all_single(
            recv_back, send_back,
            output_split_sizes=handle["send_sizes"],
            input_split_sizes=handle["recv_sizes"],
            group=self.group,
        )

        # Accumulate into output (tokens chosen by multiple experts get summed)
        output = expert_output.new_zeros((handle["num_local_tokens"], D))
        output.index_add_(0, handle["send_token_indices"], recv_back)
        return output


class JoyOV2MoEFeedForward(nn.Module):
    """Expert-Chosen MoE with EP dispatch and shared expert.

    Gate is replicated. Experts are EP-sharded. Shared expert is TP-sharded.
    """

    def __init__(
        self,
        hidden_size: int,
        moe_hidden_size: int,
        num_experts: int,
        top_p: float,
        share_expert_dim: int,
        routed_scaling_factor: float,
        scale_weights_by_expert_count: bool = True,
        expert_count_scaling_power: float = 0.5,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_p = top_p
        self.routed_scaling_factor = routed_scaling_factor
        self.scale_weights_by_expert_count = scale_weights_by_expert_count
        self.expert_count_scaling_power = expert_count_scaling_power

        tp_group = get_tp_group()
        self.ep_group = tp_group.device_group  # EP=TP in JoyO V2 eval
        self.ep_size = get_tp_world_size()
        self.ep_rank = tp_group.rank_in_group
        self.num_local_experts = num_experts // self.ep_size

        # Gate: replicated across all ranks
        self.gate = ReplicatedLinear(
            hidden_size, num_experts, bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
        )

        # Local experts — keys: {prefix}.experts.w1, {prefix}.experts.w2
        # Use a sub-module to match the "experts.w1" / "experts.w2" key structure.
        # Diffusers stores all 48 experts as [48, ...]; each EP rank keeps 6.
        self.experts = nn.Module()
        self.experts.w1 = nn.Parameter(
            torch.empty(self.num_local_experts, moe_hidden_size * 2, hidden_size)
        )
        self.experts.w2 = nn.Parameter(
            torch.empty(self.num_local_experts, hidden_size, moe_hidden_size)
        )
        set_weight_attrs(
            self.experts.w1,
            {"is_expert": True, "weight_loader": self.expert_weight_loader},
        )
        set_weight_attrs(
            self.experts.w2,
            {"is_expert": True, "weight_loader": self.expert_weight_loader},
        )

        # Shared expert (TP-sharded)
        self.share_expert_w1 = MergedColumnParallelLinear(
            hidden_size,
            [share_expert_dim, share_expert_dim],
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.share_expert_w1",
        )
        self.share_expert_w2 = RowParallelLinear(
            share_expert_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.share_expert_w2",
        )
        self.shared_expert_gate = ReplicatedLinear(
            hidden_size, 1, bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.shared_expert_gate",
        )

        # Dispatcher (lazy init)
        self._dispatcher = None

    @property
    def dispatcher(self):
        if self._dispatcher is None:
            self._dispatcher = JoyOV2ExpertDispatcher(
                self.ep_group, self.num_experts
            )
        return self._dispatcher

    def _shared_expert_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Shared expert: SwiGLU + sigmoid gate."""
        gate_up, _ = self.share_expert_w1(x)
        gate, up = gate_up.chunk(2, dim=-1)
        h = F.silu(gate) * up
        out, _ = self.share_expert_w2(h)
        gate_val, _ = self.shared_expert_gate(out)
        return torch.sigmoid(gate_val) * out

    def _expert_chosen_routing(
        self, gate_prob: torch.Tensor, num_tokens: int,
        seqlens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expert-Chosen routing: each expert selects top_p fraction of tokens.

        Top-k per sample uses ``ceil(seqlens_i * top_p)`` (clamped to seqlens_i)
        to match Joytron's ``top_p_kept_column_indices`` bit-for-bit. Global path
        uses ``ceil(num_tokens * top_p)`` for the same reason.
        """
        import math

        gate_prob_t = gate_prob.transpose(0, 1)  # (E, S)

        if seqlens is not None and seqlens.shape[0] > 1:
            # Sample-level segmented sort
            device = gate_prob_t.device
            seqlens_list = seqlens.tolist()

            # ceil, clamped to sample length — parity with Joytron
            # (torch.ceil on fp32 tensor, see joytron/utils/seqlen_meta.py)
            top_ks_f32 = torch.ceil(
                torch.tensor(seqlens_list, dtype=torch.float32) * self.top_p
            ).long().tolist()
            top_ks = [min(k, s) for k, s in zip(top_ks_f32, seqlens_list)]

            seg_starts = [0]
            for s in seqlens_list:
                seg_starts.append(seg_starts[-1] + s)
            kept_cols = []
            for start, s, k in zip(seg_starts, seqlens_list, top_ks):
                kept_cols.extend(range(start, start + k))
            kept_indices = torch.tensor(kept_cols, dtype=torch.long, device=device)

            sample_ids = torch.repeat_interleave(
                torch.arange(len(seqlens_list), device=device),
                seqlens, output_size=num_tokens
            )
            sort_key = gate_prob_t - sample_ids.unsqueeze(0) * 2.0
            _, expert_token_ids = torch.sort(sort_key, dim=1, descending=True, stable=True)
            expert_token_ids = expert_token_ids.index_select(1, kept_indices)
            expert_token_weights = torch.gather(gate_prob_t, 1, expert_token_ids)
        else:
            # Global path: ceil to match Joytron's `math.ceil(num_token * top_p)`
            tokens_per_expert = math.ceil(num_tokens * self.top_p)
            expert_token_weights, expert_token_ids = torch.topk(
                gate_prob_t, tokens_per_expert, dim=1, sorted=True
            )

        # Normalize weights per token
        token_weight_sums = torch.zeros(
            num_tokens, dtype=expert_token_weights.dtype, device=expert_token_weights.device
        )
        token_weight_sums.scatter_add_(0, expert_token_ids.flatten(), expert_token_weights.flatten())
        gathered_sums = torch.clamp(token_weight_sums[expert_token_ids], min=1e-9)
        expert_token_weights = expert_token_weights / gathered_sums

        # Scale by expert count
        if self.scale_weights_by_expert_count:
            token_counts = torch.zeros(
                num_tokens, dtype=expert_token_weights.dtype, device=expert_token_weights.device
            )
            token_counts.scatter_add_(
                0, expert_token_ids.flatten(),
                torch.ones_like(expert_token_ids.flatten(), dtype=expert_token_weights.dtype)
            )
            gathered_counts = token_counts[expert_token_ids]
            expert_token_weights = expert_token_weights * gathered_counts.pow(self.expert_count_scaling_power)

        return expert_token_ids, expert_token_weights

    def _local_expert_compute(
        self, expert_input: torch.Tensor, expert_weights: torch.Tensor
    ) -> torch.Tensor:
        """Run local experts via bmm. Input: (E_loc, C, D), weights: (E_loc, C).

        `expert_weights` is fp32 (routing runs in fp32). Joytron's EcGroupedExperts
        multiplies the weight then casts the result back to the input dtype
        (`out = out.to(x.dtype)`), so downstream stays bf16. Match that here.
        """
        in_dtype = expert_input.dtype
        h = torch.bmm(expert_input, self.experts.w1.transpose(1, 2))
        gate, up = h.chunk(2, dim=-1)
        h = F.silu(gate) * up
        out = torch.bmm(h, self.experts.w2.transpose(1, 2))
        out = out * expert_weights.unsqueeze(-1)
        return out.to(in_dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        seqlens: Optional[torch.Tensor] = None,
        seqlens_tuple: Optional[tuple] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, S, D) — B=1 for packed sequences
            seqlens: per-sample token counts
        """
        B, S, D = hidden_states.shape
        x = hidden_states.reshape(-1, D)  # (S_total, D) — full packed sequence
        num_tokens = x.shape[0]

        # This block runs on the FULL packed sequence (not SP-sharded). Gate
        # routing sees all tokens. For EP, the expert dispatcher requires the
        # Joytron token layout: each EP rank owns a contiguous slice of S_total
        # (id // local_len == rank). So we slice x into per-rank locals for
        # dispatch, run experts, combine, then all-gather back to full.
        x_full = x
        num_tokens_full = num_tokens

        # Gate (replicated, fp32) — on full sequence
        gate_logits, _ = self.gate(x_full)
        gate_prob = torch.sigmoid(gate_logits.float())

        # Expert-Chosen routing — global token ids over full sequence
        expert_token_ids, expert_token_weights = self._expert_chosen_routing(
            gate_prob, num_tokens_full, seqlens=seqlens
        )

        # EP dispatch → local expert compute → EP combine
        if self.ep_size > 1:
            # Slice full sequence into this rank's contiguous local tokens so the
            # dispatcher's `id // local_len == rank` ownership test holds (Joytron
            # sequence-parallel token layout).
            assert num_tokens_full % self.ep_size == 0, (
                f"packed tokens {num_tokens_full} not divisible by ep_size {self.ep_size}"
            )
            local_len = num_tokens_full // self.ep_size
            x_local = x_full[self.ep_rank * local_len: (self.ep_rank + 1) * local_len].contiguous()

            expert_input, expert_weights, handle = self.dispatcher.dispatch(
                x_local, expert_token_ids, expert_token_weights
            )
            expert_output = self._local_expert_compute(expert_input, expert_weights)
            routed_local = self.dispatcher.combine(expert_output, handle)  # (local_len, D)

            # All-gather local outputs back to the full packed sequence
            gathered = [torch.empty_like(routed_local) for _ in range(self.ep_size)]
            dist.all_gather(gathered, routed_local.contiguous(), group=self.ep_group)
            routed_output = torch.cat(gathered, dim=0)  # (S_total, D)
        else:
            # Single rank: all experts local. Preserve Joytron's index_add
            # accumulation order for bf16-consistent output (no reordering).
            expert_input = x_full[expert_token_ids]
            expert_output = self._local_expert_compute(expert_input, expert_token_weights)
            flat_ids = expert_token_ids.flatten()
            flat_output = expert_output.reshape(-1, D).to(x_full.dtype)
            routed_output = torch.zeros_like(x_full)
            routed_output.index_add_(0, flat_ids, flat_output)

        routed_output = routed_output * self.routed_scaling_factor

        # Shared expert (on full sequence, TP-sharded internally)
        shared_output = self._shared_expert_forward(x)
        routed_output = routed_output + shared_output

        return routed_output.view(B, S, D)

    def expert_weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor
    ) -> None:
        """Load the local EP slice of a full-experts tensor.

        Called by sglang's FSDP loader as `weight_loader(temp_param, full_tensor)`.
        `loaded_weight` has shape `[num_experts, ...]` (all 48 experts from
        diffusers); this rank copies its `[ep_rank*n_loc : (ep_rank+1)*n_loc]`
        slice into `param.data` (shape `[num_local_experts, ...]`).
        """
        assert loaded_weight.shape[0] == self.num_experts, (
            f"Expected full experts dim {self.num_experts}, "
            f"got {loaded_weight.shape[0]}"
        )
        start = self.ep_rank * self.num_local_experts
        end = start + self.num_local_experts
        param.data.copy_(loaded_weight[start:end])


# ---------------------------------------------------------------------------
# Dense Feed Forward (for layer 0 and refiners)
# ---------------------------------------------------------------------------


class JoyOV2FeedForward(nn.Module):
    """SwiGLU FFN, TP-sharded."""

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.w1 = MergedColumnParallelLinear(
            hidden_size,
            [ffn_hidden_size, ffn_hidden_size],
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w1",
        )
        self.w2 = RowParallelLinear(
            ffn_hidden_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.w2",
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        gate_up, _ = self.w1(x)
        gate, up = gate_up.chunk(2, dim=-1)
        out, _ = self.w2(F.silu(gate) * up)
        return out


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------


class JoyOV2TransformerBlock(nn.Module):
    """Single JoyO V2 DiT block: norm → modulate → attn → gate → norm → modulate → ffn → gate."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        ffn_hidden_size: int,
        norm_eps: float,
        sandwich_norm: bool,
        use_moe: bool = False,
        modulation: bool = True,
        # MoE params (only used when use_moe=True)
        moe_hidden_size: int = 4096,
        num_experts: int = 48,
        share_expert_dim: int = 8192,
        top_p: float = 0.125,
        routed_scaling_factor: float = 10.0,
        scale_weights_by_expert_count: bool = True,
        expert_count_scaling_power: float = 0.5,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.sandwich_norm = sandwich_norm
        self.modulation = modulation

        # Norms
        self.attention_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm = RMSNorm(hidden_size, eps=norm_eps)
        if sandwich_norm:
            self.attention_norm2 = RMSNorm(hidden_size, eps=norm_eps)
            self.ffn_norm2 = RMSNorm(hidden_size, eps=norm_eps)

        # Modulation — key: {prefix}.modulation_model.modulate_table
        if modulation:
            self.modulation_model = JoyOV2Modulate(hidden_size, factor=_MODULATION_FACTOR)

        # Attention
        self.attention = JoyOV2Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            norm_eps=norm_eps,
            supported_attention_backends=supported_attention_backends,
            quant_config=quant_config,
            prefix=f"{prefix}.attention",
        )

        # FFN
        if use_moe:
            self.feed_forward = JoyOV2MoEFeedForward(
                hidden_size=hidden_size,
                moe_hidden_size=moe_hidden_size,
                num_experts=num_experts,
                top_p=top_p,
                share_expert_dim=share_expert_dim,
                routed_scaling_factor=routed_scaling_factor,
                scale_weights_by_expert_count=scale_weights_by_expert_count,
                expert_count_scaling_power=expert_count_scaling_power,
                quant_config=quant_config,
                prefix=f"{prefix}.feed_forward",
            )
        else:
            self.feed_forward = JoyOV2FeedForward(
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.feed_forward",
            )

    def forward(
        self,
        x: torch.Tensor,
        timestep_emb: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        seqlens: Optional[torch.Tensor] = None,
        seqlens_tuple: Optional[tuple] = None,
        attn_mask: Optional[torch.Tensor] = None,
        attn_mask_meta: Optional[dict] = None,
        num_replicated_suffix: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, S_local, D) — B=1 packed
            timestep_emb: (n_samples, factor*D) for per-sample modulation
            freqs_cis: (S, D) RoPE cos/sin
            seqlens: (n_samples,) per-sample token counts (modulation/gate/MoE)
            seqlens_tuple: host-side per-sample lengths for MoE sample-level top_p
            attn_mask / attn_mask_meta: varlen attention isolation (per-sample)
        """
        # Modulation
        if self.modulation and timestep_emb is not None:
            scale_msa, gate_msa, shift_msa, scale_mlp, gate_mlp, shift_mlp = \
                self.modulation_model(timestep_emb)
        else:
            scale_msa = gate_msa = shift_msa = None
            scale_mlp = gate_mlp = shift_mlp = None

        # --- Attention path ---
        attn_in = self.attention_norm(x)
        if shift_msa is not None:
            # Modulate: x * (1 + scale) + shift
            # For packed sequences, need repeat_interleave by seqlens
            if seqlens is not None:
                scale_exp = torch.repeat_interleave(scale_msa, seqlens, dim=0).unsqueeze(0)
                shift_exp = torch.repeat_interleave(shift_msa, seqlens, dim=0).unsqueeze(0)
            else:
                scale_exp = scale_msa.unsqueeze(1)
                shift_exp = shift_msa.unsqueeze(1)
            attn_in = attn_in * (1 + scale_exp) + shift_exp

        attn_out = self.attention(
            attn_in,
            freqs_cis=freqs_cis,
            attn_mask=attn_mask,
            attn_mask_meta=attn_mask_meta,
            num_replicated_suffix=num_replicated_suffix,
        )

        if self.sandwich_norm:
            attn_out = self.attention_norm2(attn_out)

        # Gate + residual
        if gate_msa is not None:
            if seqlens is not None:
                gate_exp = torch.repeat_interleave(gate_msa, seqlens, dim=0).unsqueeze(0)
            else:
                gate_exp = gate_msa.unsqueeze(1)
            h = x + attn_out * gate_exp
        else:
            h = x + attn_out

        # --- FFN path ---
        ffn_in = self.ffn_norm(h)
        if shift_mlp is not None:
            if seqlens is not None:
                scale_exp = torch.repeat_interleave(scale_mlp, seqlens, dim=0).unsqueeze(0)
                shift_exp = torch.repeat_interleave(shift_mlp, seqlens, dim=0).unsqueeze(0)
            else:
                scale_exp = scale_mlp.unsqueeze(1)
                shift_exp = shift_mlp.unsqueeze(1)
            ffn_in = ffn_in * (1 + scale_exp) + shift_exp

        ffn_out = self.feed_forward(ffn_in, seqlens=seqlens, seqlens_tuple=seqlens_tuple)

        if self.sandwich_norm:
            ffn_out = self.ffn_norm2(ffn_out)

        # Gate + residual
        if gate_mlp is not None:
            if seqlens is not None:
                gate_exp = torch.repeat_interleave(gate_mlp, seqlens, dim=0).unsqueeze(0)
            else:
                gate_exp = gate_mlp.unsqueeze(1)
            out = h + ffn_out * gate_exp
        else:
            out = h + ffn_out

        return out


# ---------------------------------------------------------------------------
# Top-level Transformer
# ---------------------------------------------------------------------------


class JoyOV2Transformer3DModel(CachableDiT, LayerwiseOffloadableModuleMixin):
    """JoyO V2 DiT: single-stream packed transformer with MoE and GQA."""

    _supports_gradient_checkpointing = True
    _fsdp_shard_conditions = JoyOV2DiTConfig()._fsdp_shard_conditions
    _compile_conditions = JoyOV2DiTConfig()._compile_conditions
    _supported_attention_backends = JoyOV2DiTConfig()._supported_attention_backends
    param_names_mapping = JoyOV2DiTConfig().param_names_mapping
    reverse_param_names_mapping = JoyOV2DiTConfig().reverse_param_names_mapping

    def __init__(
        self,
        config: JoyOV2DiTConfig,
        hf_config: dict[str, Any] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        if hf_config is None:
            hf_config = {}
        super().__init__(config=config, hf_config=hf_config)

        arch = config.arch_config
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.latent_channels
        self.num_layers = arch.num_layers

        # Pixel embedder — key: pixel_embedder.weight/bias
        patch_prod = arch.patch_size[0] * arch.patch_size[1] * arch.patch_size[2]
        self.pixel_embedder = nn.Linear(
            arch.latent_channels * patch_prod, arch.hidden_size, bias=True,
        )

        # Text embedder — keys: cap_embedder.0.weight, cap_embedder.1.weight/bias
        self.cap_embedder = nn.Sequential(
            RMSNorm(arch.text_dim, eps=arch.norm_eps),
            nn.Linear(arch.text_dim, arch.hidden_size, bias=True),
        )

        # Timestep embedder — keys: t_embedder.0.weight, t_embedder.2.weight
        self.t_embedder = nn.Sequential(
            nn.Linear(256, arch.hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(arch.hidden_size, arch.hidden_size * _MODULATION_FACTOR, bias=False),
        )

        # Audio embedder — key: audio_embedder.weight/bias
        self.audio_embedder = nn.Linear(
            arch.latent_channels, arch.hidden_size, bias=True,
        )

        # RoPE (interleaved 3D MRoPE)
        self.sp_size = get_sp_world_size()
        self.rotary_emb = JoyOV2RotaryEmbedding(
            head_dim=arch.attention_head_dim,
            max_position_embeddings=arch.max_position_embeddings,
            base=arch.rope_theta,
            mrope_section=arch.mrope_section,
            interleaved=arch.mrope_interleaved,
        )

        # Refiners
        self.noise_refiner = nn.ModuleList([
            JoyOV2TransformerBlock(
                hidden_size=arch.hidden_size,
                num_attention_heads=arch.num_attention_heads,
                num_kv_heads=arch.num_kv_heads,
                head_dim=arch.attention_head_dim,
                ffn_hidden_size=arch.ffn_hidden_size,
                norm_eps=arch.norm_eps,
                sandwich_norm=arch.sandwich_norm,
                use_moe=False,
                modulation=True,
                supported_attention_backends=self._supported_attention_backends,
                prefix=f"noise_refiner.{i}",
            )
            for i in range(arch.num_encoder_layers)
        ])
        self.context_text_refiner = nn.ModuleList([
            JoyOV2TransformerBlock(
                hidden_size=arch.hidden_size,
                num_attention_heads=arch.num_attention_heads,
                num_kv_heads=arch.num_kv_heads,
                head_dim=arch.attention_head_dim,
                ffn_hidden_size=arch.ffn_hidden_size,
                norm_eps=arch.norm_eps,
                sandwich_norm=arch.sandwich_norm,
                use_moe=False,
                modulation=False,
                supported_attention_backends=self._supported_attention_backends,
                prefix=f"context_text_refiner.{i}",
            )
            for i in range(arch.num_encoder_layers)
        ])
        self.audio_refiner = nn.ModuleList([
            JoyOV2TransformerBlock(
                hidden_size=arch.hidden_size,
                num_attention_heads=arch.num_attention_heads,
                num_kv_heads=arch.num_kv_heads,
                head_dim=arch.attention_head_dim,
                ffn_hidden_size=arch.ffn_hidden_size,
                norm_eps=arch.norm_eps,
                sandwich_norm=arch.sandwich_norm,
                use_moe=False,
                modulation=True,
                supported_attention_backends=self._supported_attention_backends,
                prefix=f"audio_refiner.{i}",
            )
            for i in range(arch.num_encoder_layers)
        ])

        # Main transformer blocks
        self.layers = nn.ModuleList([
            JoyOV2TransformerBlock(
                hidden_size=arch.hidden_size,
                num_attention_heads=arch.num_attention_heads,
                num_kv_heads=arch.num_kv_heads,
                head_dim=arch.attention_head_dim,
                ffn_hidden_size=arch.ffn_hidden_size,
                norm_eps=arch.norm_eps,
                sandwich_norm=arch.sandwich_norm,
                use_moe=(i in arch.moe_layer_list),
                modulation=True,
                moe_hidden_size=arch.moe_hidden_size,
                num_experts=arch.num_experts,
                share_expert_dim=arch.share_expert_dim,
                top_p=arch.top_p,
                routed_scaling_factor=arch.routed_scaling_factor,
                scale_weights_by_expert_count=arch.scale_weights_by_expert_count,
                expert_count_scaling_power=arch.expert_count_scaling_power,
                supported_attention_backends=self._supported_attention_backends,
                prefix=f"layers.{i}",
            )
            for i in range(arch.num_layers)
        ])
        self.layer_names = ["layers"]

        # Output projections — keys: vis_proj_out.0.weight, vis_proj_out.1.weight/bias
        self.vis_proj_out = nn.Sequential(
            RMSNorm(arch.hidden_size, eps=arch.norm_eps),
            nn.Linear(arch.hidden_size, arch.latent_channels, bias=True),
        )
        self.audio_proj_out = nn.Sequential(
            RMSNorm(arch.hidden_size, eps=arch.norm_eps),
            nn.Linear(arch.hidden_size, arch.latent_channels, bias=True),
        )

        self.__post_init__()

    def post_load_weights(self) -> None:
        """Materialize non-persistent buffers on the real device after loading.

        The rotary embedding's ``cos_sin_cache`` is registered non-persistent
        and is created on the "meta" device during ``__init__`` (loader uses
        ``torch.device('meta')`` context). The fsdp_load path checks that no
        param/buffer remains on meta after loading, so we rebuild it here on
        whichever device the model's parameters ended up on.
        """
        device = next(self.parameters()).device
        self.rotary_emb._ensure_cache(device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor] = None,
        timestep: torch.Tensor = None,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance=None,
        **kwargs,
    ) -> torch.Tensor:
        """JoyO V2 packed forward (matches Joytron JoyOBaseModel.forward, PP=1).

        Packed layout: samples are concatenated along the sequence dim, with
        per-sample per-modality segments ordered ``[s0_text, s0_pixel, s1_text,
        s1_pixel, ...]`` (see multimodal_pack). Attention isolates samples via
        cu_seqlens; modulation/gate expand per-sample by seqlens.

        Args:
            hidden_states: (S_pixel_total, latent_channels) — packed pixel latents
            encoder_hidden_states / text_embeddings: (S_text_total, text_dim)
            timestep: (n_samples,) — per-sample diffusion timestep
            rope_meta (kwargs): {"text": {...}, "pixel": {...}, "mix": {...}}
                each with cu_seqlens / seqlens / seqlens_tuple / position_id / max_seq_len
            audio_hidden_states (kwargs): (S_audio_total, latent_channels) or None
        Returns:
            (S_pixel_total, latent_channels) — pixel velocity prediction
        """
        text_embeddings = encoder_hidden_states
        if text_embeddings is not None and not isinstance(text_embeddings, torch.Tensor):
            text_embeddings = text_embeddings[0]
        rope_meta = kwargs.get("rope_meta", None)
        audio_hidden_states = kwargs.get("audio_hidden_states", None)
        assert rope_meta is not None, "JoyO V2 packed forward requires rope_meta"

        device = hidden_states.device

        # --- Timestep embedding (per-sample) ---
        # timestep: (n_samples,) -> t_emb (n_samples, factor*hidden_size)
        t_emb = self._timestep_embedding(timestep * 1000, 256).to(hidden_states.dtype)
        timestep_emb = self.t_embedder(t_emb)

        # --- Embed each modality ---
        pixel_emb = self.pixel_embedder(hidden_states)          # (S_pix, D)
        text_emb = self.cap_embedder(text_embeddings)           # (S_txt, D)
        audio_emb = (
            self.audio_embedder(audio_hidden_states)
            if audio_hidden_states is not None
            else None
        )

        # --- Per-modality RoPE freqs (text / pixel / audio use their own) ---
        text_freqs = self.rotary_emb(rope_meta["text"]["position_id"])
        pixel_freqs = (
            self.rotary_emb(rope_meta["pixel"]["position_id"])
            if rope_meta.get("pixel") is not None
            else None
        )
        audio_freqs = (
            self.rotary_emb(rope_meta["audio"]["position_id"])
            if rope_meta.get("audio") is not None and audio_emb is not None
            else None
        )

        # --- Refine each modality independently ---
        text_seqlens = rope_meta["text"]["seqlens"]
        text_mask, text_meta = self._build_attn_mask(rope_meta["text"], device)
        text_emb = text_emb.unsqueeze(0)  # (1, S_txt, D)
        for block in self.context_text_refiner:   # modulation=False
            text_emb = block(
                text_emb, freqs_cis=text_freqs, seqlens=text_seqlens,
                attn_mask=text_mask, attn_mask_meta=text_meta,
            )
        text_emb = text_emb.squeeze(0)

        if pixel_emb is not None and rope_meta.get("pixel") is not None:
            pixel_seqlens = rope_meta["pixel"]["seqlens"]
            pixel_mask, pixel_meta = self._build_attn_mask(rope_meta["pixel"], device)
            pixel_emb = pixel_emb.unsqueeze(0)
            for block in self.noise_refiner:       # modulation=True
                pixel_emb = block(
                    pixel_emb, timestep_emb=timestep_emb, freqs_cis=pixel_freqs,
                    seqlens=pixel_seqlens, attn_mask=pixel_mask, attn_mask_meta=pixel_meta,
                )
            pixel_emb = pixel_emb.squeeze(0)

        if audio_emb is not None and rope_meta.get("audio") is not None:
            audio_seqlens = rope_meta["audio"]["seqlens"]
            audio_mask, audio_meta = self._build_attn_mask(rope_meta["audio"], device)
            audio_emb = audio_emb.unsqueeze(0)
            for block in self.audio_refiner:       # modulation=True
                audio_emb = block(
                    audio_emb, timestep_emb=timestep_emb, freqs_cis=audio_freqs,
                    seqlens=audio_seqlens, attn_mask=audio_mask, attn_mask_meta=audio_meta,
                )
            audio_emb = audio_emb.squeeze(0)

        # --- Pack modalities: [s0_text, s0_pixel, (s0_audio), s1_text, ...] ---
        modality_tensors = [text_emb]
        modality_cus = [rope_meta["text"]["cu_seqlens"]]
        if pixel_emb is not None and rope_meta.get("pixel") is not None:
            modality_tensors.append(pixel_emb)
            modality_cus.append(rope_meta["pixel"]["cu_seqlens"])
        if audio_emb is not None and rope_meta.get("audio") is not None:
            modality_tensors.append(audio_emb)
            modality_cus.append(rope_meta["audio"]["cu_seqlens"])
        mix = multimodal_pack(modality_tensors, modality_cus)   # (S_total, D)

        # --- Main blocks over packed mix sequence ---
        mix_meta = rope_meta["mix"]
        mix_seqlens = mix_meta["seqlens"]
        mix_seqlens_tuple = mix_meta.get("seqlens_tuple")
        mix_freqs = self.rotary_emb(mix_meta["position_id"])
        mix_mask, mix_attn_meta = self._build_attn_mask(mix_meta, device)

        mix = mix.unsqueeze(0)  # (1, S_total, D)
        for block in self.layers:
            mix = block(
                mix,
                timestep_emb=timestep_emb,
                freqs_cis=mix_freqs,
                seqlens=mix_seqlens,
                seqlens_tuple=mix_seqlens_tuple,
                attn_mask=mix_mask,
                attn_mask_meta=mix_attn_meta,
            )
        mix = mix.squeeze(0)

        # --- Unpack modalities, take pixel (+ audio) ---
        outputs = multimodal_unpack(mix, modality_cus)
        idx = 1  # 0 = text
        pixel_out = None
        if pixel_emb is not None and rope_meta.get("pixel") is not None:
            pixel_out = outputs[idx]
            idx += 1
        audio_out = None
        if audio_emb is not None and rope_meta.get("audio") is not None:
            audio_out = outputs[idx]

        pixel_pred = self.vis_proj_out(pixel_out) if pixel_out is not None else None
        if audio_out is not None:
            audio_pred = self.audio_proj_out(audio_out)
            return pixel_pred, audio_pred
        return pixel_pred

    @staticmethod
    def _build_attn_mask(meta: dict, device: torch.device):
        """Build a (1, S) all-valid key mask + varlen FA meta from cu_seqlens.

        bs=1 packed: the whole sequence is valid, but cu_seqlens carves it into
        per-sample ranges so attention stays within a sample. This yields
        zero-pad packed attention through USPAttention's varlen FA path.
        """
        cu = meta["cu_seqlens"]
        s_total = int(cu[-1])
        attn_mask = torch.ones((1, s_total), dtype=torch.bool, device=device)
        # cu_seqlens already encodes per-sample boundaries; indices are identity
        # (all tokens valid), so pack/scatter are no-ops but FA uses cu_seqlens.
        indices = torch.arange(s_total, dtype=torch.long, device=device)
        max_seqlen = int(meta["max_seq_len"]) if "max_seq_len" in meta else s_total
        attn_mask_meta = {
            "cu_seqlens": cu.to(torch.int32),
            "indices": indices,
            "inv_indices": indices,
            "max_seqlen": max_seqlen,
        }
        return attn_mask, attn_mask_meta

    @staticmethod
    def _timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device, dtype=torch.float32)
            * (2.0 * torch.log(torch.tensor(max_period)) / dim)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


EntryClass = JoyOV2Transformer3DModel
